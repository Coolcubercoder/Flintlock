package flintlock

import (
	"errors"
	"fmt"
	"runtime"
	"runtime/debug"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

// Sentinel errors.
var (
	// ErrClosed is returned by Execute after Close.
	ErrClosed = errors.New("flintlock: sandbox closed")
	// ErrPoolExhausted is returned when every slot is armed and Config.Blocking
	// is false. It is a backpressure signal, not a failure.
	ErrPoolExhausted = errors.New("flintlock: all guarded slots armed")
	// ErrPayloadTooLarge is returned when a payload cannot fit in one slot.
	ErrPayloadTooLarge = errors.New("flintlock: payload exceeds slot capacity")
	// ErrNilFn is returned when Execute is handed a nil parser.
	ErrNilFn = errors.New("flintlock: nil execution function")
)

// Slot lifecycle states. Every transition is a single CAS on slot.state.
//
//	                 CAS Free->Armed
//	   +-------------------------------------------+
//	   |                                           v
//	[Free] <--- scrub+rearm --- [Quarantined] <--- release --- [Armed]
//	                                   ^                          |
//	                                   |      fault + reclaim      |
//	                                   +--------------------------+
//	                                                              |
//	                                          fault, no reclaim   v
//	                                                          [Retired]
const (
	stateFree        uint32 = iota // pages RW, scrubbed, immediately claimable
	stateArmed                     // pages RW, owned by exactly one goroutine
	stateQuarantined               // pages PROT_NONE, cooling down for UAF detection
	stateRetired                   // pages PROT_NONE, permanently withdrawn
)

// cacheLine is the false-sharing granule. 128 rather than 64: Apple silicon and
// modern x86 prefetch in 128-byte pairs, so 64-byte padding still lets two
// slots' atomics contend on adjacent-line prefetch.
const cacheLine = 128

// slot is one guard-flanked sandbox cell.
//
// Every hot field is atomic and the struct is padded to whole cache lines so
// that a CAS storm on slot i cannot invalidate slot i+1's line. With n slots on
// n cores the turnstile degenerates to n independent uncontended CAS sites.
type slot struct {
	state   atomic.Uint32 // one of state*
	_       [4]byte
	epoch   atomic.Uint64 // reuse generation, incremented on rearm
	relTick atomic.Uint64 // turnstile ticket at which it entered quarantine
	faults  atomic.Uint64 // lifetime traps attributed to this cell
	base    uintptr       // first writable byte
	limit   uintptr       // one past last writable byte
	liveOff uintptr       // staged payload offset from base (owner-only)
	liveLen uintptr       // staged payload length         (owner-only)
	dirtyLo uintptr       // scrub range low bound          (owner-only)
	dirtyHi uintptr       // scrub range high bound         (owner-only)
	_       [cacheLine - (4+4+8+8+8+8+8+8+8+8+8)%cacheLine]byte
}

// Compile-time assertion: slot must occupy whole cache lines, or the padding
// arithmetic above is wrong and slots will false-share. A non-zero remainder
// makes this a constant out-of-range index and fails the build.
var _ = [1]struct{}{}[unsafe.Sizeof(slot{})%cacheLine]

// Config parameterizes a Sandbox. The zero value is valid and fully defaulted.
type Config struct {
	// SlotBytes is the usable payload capacity per slot, rounded up to a page.
	// Default 64 KiB.
	SlotBytes int

	// Slots is the number of guard-flanked cells. Concurrency above this
	// number blocks (or fails, per Blocking). Default 2*GOMAXPROCS, min 8.
	Slots int

	// QuarantineDepth is how many turnstile tickets a released slot must wait
	// before reuse, during which its pages stay PROT_NONE. This is what turns a
	// stale pointer into a trap instead of a silent read of another tenant's
	// data. 0 disables use-after-free detection and skips two mprotect calls
	// per Execute. Default 0.
	//
	// Cost: enabling it adds two mprotect syscalls plus a TLB shootdown per
	// slot cycle. Measure before enabling on a hot path; it is off by default
	// precisely because it is the one feature here that is not free.
	QuarantineDepth uint64

	// Blocking makes Execute spin on the turnstile until a slot frees instead
	// of returning ErrPoolExhausted. Default true.
	Blocking bool
	// NonBlocking overrides Blocking to false. Needed because Config's zero
	// value must stay usable while Blocking defaults to true.
	NonBlocking bool

	// SpinBudget is how many full probe sweeps to attempt before yielding to
	// the scheduler. Default 64.
	SpinBudget int

	// ScrubOnRecycle zeroes a slot's dirty extent before handing it to the next
	// tenant, so one payload's residue can never be read by the next. Default
	// true. Disabling it is an information-disclosure risk across tenants.
	ScrubOnRecycle bool
	// NoScrub overrides ScrubOnRecycle to false.
	NoScrub bool

	// ReclaimFaulted returns a slot to service after it traps, instead of
	// retiring it permanently. Default true -- an attacker who can trigger
	// faults on demand would otherwise exhaust the pool. Set false for
	// forensic freezing of the offending cell. Default true.
	ReclaimFaulted bool
	// NoReclaim overrides ReclaimFaulted to false.
	NoReclaim bool

	// AlignMask aligns the payload to this power-of-two boundary. 0 means exact
	// byte alignment: the payload abuts the guard page, so a ONE-byte overstep
	// traps. A mask of 15 gives 16-byte SIMD alignment at the cost of up to 15
	// bytes of undetected slack between the payload and the guard. Default 0.
	AlignMask uintptr

	// AlignLow flushes the payload against the LOWER guard instead of the upper
	// one.
	//
	// A payload smaller than the slot span cannot be byte-exact on both edges
	// at once: one guard abuts the payload and the other is separated from it
	// by span-len bytes of writable slack, which an overstepping pointer walks
	// silently before trapping. Choose which edge is exact:
	//
	//	AlignLow == false (default)   [.... slack ....][= payload =]|GUARD|
	//	                              byte-exact OVERFLOW detection
	//
	//	AlignLow == true              |GUARD|[= payload =][.... slack ....]
	//	                              byte-exact UNDERFLOW detection
	//
	// Overflow is the far more common exploit primitive, so it is the default.
	// A payload sized to exactly SlotCapacity is byte-exact on both edges
	// regardless of this setting.
	AlignLow bool

	// LedgerSize is the fault ring capacity, rounded to a power of two.
	// Default 256.
	LedgerSize int

	// TrapAllPanics converts every panic from fn -- including ordinary ones
	// like index-out-of-range or an explicit panic() -- into an error rather
	// than repropagating it. Default false: only genuine hardware memory faults
	// are converted, so real logic bugs stay loud.
	TrapAllPanics bool

	// OnFault is invoked, on the faulting goroutine, immediately after a trap
	// is classified and the slot is contained. It must not touch sandbox
	// memory and must not panic. May be nil.
	OnFault func(Fault)
}

func (c Config) normalize() Config {
	if c.SlotBytes <= 0 {
		c.SlotBytes = 64 << 10
	}
	if c.Slots <= 0 {
		c.Slots = 2 * runtime.GOMAXPROCS(0)
		if c.Slots < 8 {
			c.Slots = 8
		}
	}
	if c.SpinBudget <= 0 {
		c.SpinBudget = 64
	}
	if c.LedgerSize <= 0 {
		c.LedgerSize = 256
	}
	c.Blocking = !c.NonBlocking
	c.ScrubOnRecycle = !c.NoScrub
	c.ReclaimFaulted = !c.NoReclaim
	if c.AlignMask != 0 && c.AlignMask&(c.AlignMask+1) != 0 {
		// AlignMask must be 2^k - 1; round up to the next such mask.
		m := uintptr(1)
		for m-1 < c.AlignMask {
			m <<= 1
		}
		c.AlignMask = m - 1
	}
	return c
}

// Sandbox is a pool of hardware-guarded execution slots.
//
// It is safe for concurrent use by any number of goroutines. There is no mutex
// anywhere in the data path: admission is an atomic ticket, slot ownership is a
// single CAS, and telemetry is a seqlock ring.
type Sandbox struct {
	cfg   Config
	arena *arena
	slots []slot
	ring  *faultRing

	// turnstile is the monotonic ticket dispenser. A worker's ticket both
	// selects its starting probe index -- which is what spreads n concurrent
	// workers across n distinct cache lines instead of stacking them all on
	// slot 0 -- and serves as the logical clock for quarantine aging.
	turnstile atomic.Uint64

	inflight  atomic.Int64
	closed    atomic.Bool
	executed  atomic.Uint64
	trapped   atomic.Uint64
	contended atomic.Uint64
	retired   atomic.Int64
	rearms    atomic.Uint64
}

// New reserves the arena, installs the guard pages and returns a ready pool.
func New(cfg Config) (*Sandbox, error) {
	cfg = cfg.normalize()

	a, err := mapArena(cfg.SlotBytes, cfg.Slots)
	if err != nil {
		return nil, err
	}

	s := &Sandbox{
		cfg:   cfg,
		arena: a,
		slots: make([]slot, cfg.Slots),
		ring:  newFaultRing(cfg.LedgerSize),
	}
	for i := range s.slots {
		sl := &s.slots[i]
		sl.base = a.slotBase(i)
		sl.limit = sl.base + a.span
		sl.state.Store(stateFree)
	}
	return s, nil
}

// SlotCapacity is the usable bytes per slot after page rounding. It is >= the
// requested Config.SlotBytes.
func (s *Sandbox) SlotCapacity() int { return int(s.arena.span) }

// Slots returns the pool width.
func (s *Sandbox) Slots() int { return s.cfg.Slots }

// PageSize returns the hardware granule the guards were built on.
func (s *Sandbox) PageSize() int { return int(pageSize) }

// ---------------------------------------------------------------------------
// Turnstile
// ---------------------------------------------------------------------------

// acquire claims a slot. Lock-free: no mutex, no channel, no park. Returns the
// slot, its index, and the caller's turnstile ticket.
//
// Routing math: ticket t starts probing at t mod n. Because tickets are handed
// out by a single fetch-and-add, k concurrent workers hold k consecutive
// tickets and therefore begin at k distinct, consecutive slots. In the common
// case each lands on its own cache line with an uncontended CAS and zero
// probes. Contention only produces probing when the pool is genuinely near
// saturation.
func (s *Sandbox) acquire() (*slot, int, uint64, error) {
	n := len(s.slots)
	tk := s.turnstile.Add(1) - 1
	start := int(tk % uint64(n))
	spins := 0

	for {
		// Pass 1: free cells only. This is the fast path and it is pure CAS.
		for probe := 0; probe < n; probe++ {
			i := start + probe
			if i >= n {
				i -= n
			}
			sl := &s.slots[i]
			if sl.state.Load() != stateFree {
				continue
			}
			if sl.state.CompareAndSwap(stateFree, stateArmed) {
				return sl, i, tk, nil
			}
		}

		// Pass 2: quarantined cells whose cooldown has expired. Reviving one
		// costs an mprotect back to RW, so it is deliberately second.
		for probe := 0; probe < n; probe++ {
			i := start + probe
			if i >= n {
				i -= n
			}
			sl := &s.slots[i]
			if sl.state.Load() != stateQuarantined {
				continue
			}
			if tk-sl.relTick.Load() < s.cfg.QuarantineDepth {
				continue
			}
			if sl.state.CompareAndSwap(stateQuarantined, stateArmed) {
				if err := s.rearm(sl, i); err != nil {
					sl.state.Store(stateRetired)
					s.retired.Add(1)
					return nil, -1, tk, err
				}
				return sl, i, tk, nil
			}
		}

		s.contended.Add(1)
		spins++

		// Pass 3: starvation breaker. Every slot is armed or still cooling. If
		// any cell is quarantined, force it back into service regardless of
		// cooldown -- a shortened UAF window is strictly better than a hang.
		if spins > s.cfg.SpinBudget {
			for probe := 0; probe < n; probe++ {
				i := start + probe
				if i >= n {
					i -= n
				}
				sl := &s.slots[i]
				if sl.state.Load() != stateQuarantined {
					continue
				}
				if sl.state.CompareAndSwap(stateQuarantined, stateArmed) {
					if err := s.rearm(sl, i); err != nil {
						sl.state.Store(stateRetired)
						s.retired.Add(1)
						return nil, -1, tk, err
					}
					return sl, i, tk, nil
				}
			}
			if !s.cfg.Blocking {
				return nil, -1, tk, ErrPoolExhausted
			}
			if s.closed.Load() {
				return nil, -1, tk, ErrClosed
			}
			spins = 0
		}

		// Yield rather than burn the core. Gosched keeps this obstruction-free:
		// no ownership is held across the yield, so a preempted acquirer can
		// never block a peer.
		runtime.Gosched()
	}
}

// rearm returns a quarantined cell to writable service.
func (s *Sandbox) rearm(sl *slot, i int) error {
	if s.cfg.QuarantineDepth > 0 || sl.state.Load() == stateRetired {
		if err := s.arena.protect(i, syscall.PROT_READ|syscall.PROT_WRITE); err != nil {
			return err
		}
	}
	s.scrub(sl, i)
	sl.epoch.Add(1)
	s.rearms.Add(1)
	return nil
}

// scrub zeroes the cell's dirty EXTENT -- the [lo,hi) range actually touched by
// the outgoing tenant -- so no residue crosses to the next one.
//
// Tracking a range rather than a high-water mark from offset 0 matters a great
// deal under the default high alignment: the payload sits at the TOP of the
// span, so a zero-based high-water mark would memclr the entire slot to erase a
// few hundred bytes. On a 16 KiB granule that was measured at ~200ns of pure
// waste per Execute.
func (s *Sandbox) scrub(sl *slot, i int) {
	if !s.cfg.ScrubOnRecycle || sl.dirtyHi <= sl.dirtyLo {
		sl.dirtyLo, sl.dirtyHi = 0, 0
		return
	}
	region := s.arena.slotRegion(i)
	lo, hi := sl.dirtyLo, sl.dirtyHi
	if hi > uintptr(len(region)) {
		hi = uintptr(len(region))
	}
	b := region[lo:hi]
	// Recognized by the compiler as a single memclr, not a byte loop.
	for j := range b {
		b[j] = 0
	}
	sl.dirtyLo, sl.dirtyHi = 0, 0
}

// release returns a healthy cell to the pool.
func (s *Sandbox) release(sl *slot, i int, tk uint64) {
	if s.cfg.QuarantineDepth > 0 {
		// Revoke all access before publishing the state change, so a stale
		// pointer held by the just-finished parser faults from this instant on.
		if err := s.arena.protect(i, syscall.PROT_NONE); err != nil {
			sl.state.Store(stateRetired)
			s.retired.Add(1)
			return
		}
		sl.relTick.Store(tk)
		sl.state.Store(stateQuarantined)
		return
	}
	s.scrub(sl, i)
	sl.epoch.Add(1)
	sl.state.Store(stateFree)
}

// contain quarantines or retires a cell that just trapped.
func (s *Sandbox) contain(sl *slot, i int, tk uint64) {
	sl.faults.Add(1)
	if !s.cfg.ReclaimFaulted {
		// Freeze the evidence: pages stay PROT_NONE and the cell never returns.
		_ = s.arena.protect(i, syscall.PROT_NONE)
		sl.state.Store(stateRetired)
		s.retired.Add(1)
		return
	}
	// The parser faulted on a guard page, so the guard itself was never
	// written. The data span may hold attacker-influenced residue, which scrub
	// erases. Recycling is safe and keeps the pool from being exhausted by an
	// attacker who can trigger faults at will.
	sl.dirtyLo, sl.dirtyHi = 0, s.arena.span
	s.release(sl, i, tk)
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

// stage copies the payload into the cell, right-aligned against the upper guard.
//
// Right alignment is the point. Left-aligning the payload at the slot base
// would leave span-len(payload) bytes of slack before the guard page, and a
// linear overflow would have to walk all of that slack -- silently, through
// live writable memory -- before tripping anything. Right-aligned with
// AlignMask 0, the payload's final byte is the final byte before the guard, so
// off-by-one is caught at exactly one byte.
//
//	base                                      limit == guard
//	 |                                             |
//	 v                                             v
//	 [........ unused slack ........][== payload ==]|GUARD|
//	                                               ^
//	                                     one byte past here traps
func (s *Sandbox) stage(sl *slot, i int, payload []byte) unsafe.Pointer {
	region := s.arena.slotRegion(i)
	n := uintptr(len(payload))

	var off uintptr
	if s.cfg.AlignLow {
		off = 0
		if s.cfg.AlignMask != 0 {
			off = (off + s.cfg.AlignMask) &^ s.cfg.AlignMask
		}
	} else {
		off = uintptr(len(region)) - n
		if s.cfg.AlignMask != 0 {
			off &^= s.cfg.AlignMask
		}
	}

	sl.liveOff = off
	sl.liveLen = n
	if sl.dirtyHi == 0 || off < sl.dirtyLo {
		sl.dirtyLo = off
	}
	if end := off + n; end > sl.dirtyHi {
		sl.dirtyHi = end
	}

	if n == 0 {
		// A zero-length payload has no legal byte. Hand back the address at the
		// alignment edge; any dereference of it is out of contract and, at the
		// upper edge, traps on the very first byte.
		if s.cfg.AlignLow {
			return unsafe.Pointer(&region[0])
		}
		return unsafe.Pointer(&region[len(region)-1])
	}
	dst := region[off : off+n : off+n]
	copy(dst, payload)
	return unsafe.Pointer(&dst[0])
}

// Execute stages payload into a guard-flanked slot and runs fn against it.
//
// fn receives a raw pointer to the first payload byte. The len(payload) bytes
// from there are writable. The byte immediately after them is a PROT_NONE guard
// page (with the default AlignMask of 0), as is the page below the slot. Any
// access outside that window traps in hardware and is returned as
// *GuardViolation. The rest of the process is unaffected and never pauses.
//
// Contract that must be honored by fn:
//
//   - fn must not pass the pointer to another goroutine. The fault-trapping
//     flag is per-goroutine; a fault raised on a goroutine that has not set it
//     is an unrecoverable runtime throw that kills the process.
//   - fn must not retain the pointer past its own return. With
//     QuarantineDepth > 0 later use traps; with 0 it silently reads whatever
//     tenant now owns the cell.
//   - fn should not call C code. cgo frames fault outside Go's recovery path.
//
// Execute performs ZERO heap allocations on the success path, so a steady
// stream of clean parses adds nothing to the GC's workload. The trap path does
// allocate (one *GuardViolation carrying the Fault record); that is the cold
// path by construction and is measured in BenchmarkTrap.
func (s *Sandbox) Execute(payload []byte, fn func(unsafe.Pointer)) error {
	if fn == nil {
		return ErrNilFn
	}
	if len(payload) > int(s.arena.span) {
		return fmt.Errorf("%w: %d bytes > %d slot capacity",
			ErrPayloadTooLarge, len(payload), s.arena.span)
	}

	// Register in-flight before re-checking closed. Close observes the
	// counter after setting the flag, so this ordering makes it impossible for
	// Close to unmap the arena while a parser is inside it.
	s.inflight.Add(1)
	if s.closed.Load() {
		s.inflight.Add(-1)
		return ErrClosed
	}
	defer s.inflight.Add(-1)

	sl, idx, tk, err := s.acquire()
	if err != nil {
		return err
	}

	ptr := s.stage(sl, idx, payload)
	return s.run(sl, idx, tk, ptr, fn)
}

// run is the trap frame. It is split out of Execute so the deferred recover has
// the tightest possible scope and so the named-result error rewrite is
// unambiguous.
func (s *Sandbox) run(sl *slot, idx int, tk uint64, ptr unsafe.Pointer, fn func(unsafe.Pointer)) (err error) {
	epoch := sl.epoch.Load()
	started := time.Now()

	// THE trap arming instruction. This sets g.paniconfault on THIS goroutine
	// only, telling runtime.sigpanic to convert a SIGSEGV into a recoverable
	// panic carrying si_addr instead of throwing. It is a single field store;
	// there is no handler installation, no signal mask change, no syscall.
	prev := debug.SetPanicOnFault(true)

	defer func() {
		// Disarm before anything else: everything below runs outside the
		// sandbox and must fault normally if it is itself buggy.
		debug.SetPanicOnFault(prev)

		r := recover()
		if r == nil {
			s.release(sl, idx, tk)
			s.executed.Add(1)
			return
		}

		addr, addrValid, isFault := isMemoryFault(r)
		if !isFault && !s.cfg.TrapAllPanics {
			// An ordinary panic from the parser: not a memory-safety event.
			// Contain the cell, then let it propagate so real logic bugs stay
			// loud instead of being laundered into an error return.
			s.contain(sl, idx, tk)
			panic(r)
		}

		f := s.classify(r, sl, idx, tk, epoch, addr, addrValid, isFault, started)
		s.contain(sl, idx, tk)
		s.ring.push(&f)
		s.trapped.Add(1)
		if s.cfg.OnFault != nil {
			s.cfg.OnFault(f)
		}
		err = &GuardViolation{Fault: f}
	}()

	fn(ptr)
	return nil
}

// classify converts a recovered fault into a forensic record, resolving the
// faulting address against the arena geometry.
func (s *Sandbox) classify(
	r any, sl *slot, idx int, tk, epoch uint64,
	addr uintptr, addrValid, isFault bool, started time.Time,
) Fault {
	f := Fault{
		At:          time.Now(),
		Elapsed:     time.Since(started),
		Ticket:      tk,
		Slot:        idx,
		Epoch:       epoch,
		Addr:        addr,
		AddrValid:   addrValid,
		SlotBase:    sl.base,
		SlotLimit:   sl.limit,
		PayloadBase: sl.base + sl.liveOff,
		PayloadLen:  sl.liveLen,
	}
	if e, ok := r.(error); ok {
		f.Reason = e.Error()
	} else {
		f.Reason = fmt.Sprint(r)
	}
	f.NFrames = captureFrames(&f.Frames)

	switch {
	case !isFault:
		f.Kind = FaultForeign // TrapAllPanics path: a non-memory panic
		return f
	case !addrValid:
		f.Kind = FaultNullDeref
		return f
	}

	f.Offset = int64(addr) - int64(sl.base)
	f.PayloadOffset = int64(addr) - int64(f.PayloadBase)

	owner, _, kind := s.arena.region(addr)

	// Disambiguate the shared guard page. A guard between slot i-1 and slot i
	// is struck as an OVERFLOW when the armed cell is i-1 and as an UNDERFLOW
	// when the armed cell is i. The armed identity is authoritative because
	// only the armed cell was reachable from fn.
	if kind == FaultUnderflow || kind == FaultOverflow {
		switch {
		case f.Offset >= int64(s.arena.span):
			kind = FaultOverflow
		case f.Offset < 0:
			kind = FaultUnderflow
		}
		owner = idx
	}
	if owner != idx && kind != FaultForeign {
		// The address resolves inside the arena but to somebody else's cell:
		// the parser escaped its own sandbox entirely.
		kind = FaultForeign
	}
	f.Kind = kind
	return f
}

// ---------------------------------------------------------------------------
// Introspection and teardown
// ---------------------------------------------------------------------------

// Stats is a point-in-time counter snapshot.
type Stats struct {
	Executed   uint64 // Execute calls that returned cleanly
	Trapped    uint64 // Execute calls that struck a guard
	Contended  uint64 // turnstile sweeps that found no slot
	Rearms     uint64 // quarantined cells returned to service
	Retired    int64  // cells permanently withdrawn
	InFlight   int64  // parsers currently inside the arena
	Tickets    uint64 // turnstile tickets dispensed
	ArenaBytes uint64 // total virtual reservation, guards included
	GuardPages int    // PROT_NONE pages installed
}

// Stats returns current counters. Lock-free; safe during execution.
func (s *Sandbox) Stats() Stats {
	return Stats{
		Executed:   s.executed.Load(),
		Trapped:    s.trapped.Load(),
		Contended:  s.contended.Load(),
		Rearms:     s.rearms.Load(),
		Retired:    s.retired.Load(),
		InFlight:   s.inflight.Load(),
		Tickets:    s.turnstile.Load(),
		ArenaBytes: uint64(s.arena.total),
		GuardPages: s.cfg.Slots + 1,
	}
}

// Faults returns the fault ledger, oldest first.
func (s *Sandbox) Faults() []Fault { return s.ring.snapshot() }

// Reclaim returns retired cells to service. Use after triaging a
// ReclaimFaulted=false freeze. Returns the number revived.
func (s *Sandbox) Reclaim() int {
	revived := 0
	for i := range s.slots {
		sl := &s.slots[i]
		if !sl.state.CompareAndSwap(stateRetired, stateArmed) {
			continue
		}
		if err := s.arena.protect(i, syscall.PROT_READ|syscall.PROT_WRITE); err != nil {
			sl.state.Store(stateRetired)
			continue
		}
		sl.dirtyLo, sl.dirtyHi = 0, s.arena.span
		s.scrub(sl, i)
		sl.epoch.Add(1)
		sl.state.Store(stateFree)
		s.retired.Add(-1)
		revived++
	}
	return revived
}

// Close drains in-flight parsers and releases the entire reservation.
//
// It blocks until every parser has left the arena. Unmapping underneath a live
// parser would convert a contained, recoverable guard trap into an
// unrecoverable process-wide fault, so draining is mandatory, not optional.
func (s *Sandbox) Close() error {
	if !s.closed.CompareAndSwap(false, true) {
		return nil // idempotent
	}
	for s.inflight.Load() > 0 {
		runtime.Gosched()
	}
	return s.arena.unmap()
}
