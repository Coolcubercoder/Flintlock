package flintlock

import (
	"fmt"
	"runtime"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"
)

// FaultKind classifies a trapped memory-safety violation by which boundary the
// offending pointer struck.
type FaultKind uint8

const (
	// FaultOverflow: the access ran off the UPPER edge of the slot into the
	// high guard page. Classic linear buffer overflow / overread.
	FaultOverflow FaultKind = iota + 1
	// FaultUnderflow: the access ran off the LOWER edge into the low guard
	// page. Signature of a negative index or an unsigned length underflow.
	FaultUnderflow
	// FaultUseAfterFree: the access landed inside a data span that is currently
	// revoked (quarantined or retired). A stale pointer outlived its slot.
	FaultUseAfterFree
	// FaultNullDeref: dereference of a null or near-null pointer. The kernel
	// reports these below 0x1000 and the runtime strips the address, so no
	// precise si_addr is available.
	FaultNullDeref
	// FaultForeign: a fault at an address outside this arena entirely. The
	// parser dereferenced a wild pointer with no relationship to its sandbox.
	FaultForeign
)

func (k FaultKind) String() string {
	switch k {
	case FaultOverflow:
		return "GUARD_OVERFLOW"
	case FaultUnderflow:
		return "GUARD_UNDERFLOW"
	case FaultUseAfterFree:
		return "USE_AFTER_FREE"
	case FaultNullDeref:
		return "NULL_DEREF"
	case FaultForeign:
		return "FOREIGN_ADDRESS"
	}
	return "UNKNOWN"
}

// maxFaultFrames caps the captured call stack. Fixed-size so that Fault stays a
// flat, copyable value with no heap indirection on the trap path.
const maxFaultFrames = 24

// Fault is the forensic record of a single trapped violation.
//
// It is a flat value type. Recording one performs no allocation.
type Fault struct {
	// At is the wall-clock instant of capture, microsecond-resolution or better.
	At time.Time
	// Elapsed is time spent inside fn before it faulted.
	Elapsed time.Duration
	// Ticket is the monotonic turnstile ticket of the offending Execute call.
	Ticket uint64
	// Slot is the index of the sandbox slot that was armed, or -1.
	Slot int
	// Epoch is the slot's reuse generation, which distinguishes a fault against
	// the current tenant from a fault against a long-dead one.
	Epoch uint64

	// Addr is the faulting virtual address: si_addr, straight out of the
	// kernel's fault frame via runtime.sigpanic. Zero when unavailable
	// (null-deref path only).
	Addr uintptr
	// AddrValid reports whether Addr carries a real si_addr.
	AddrValid bool

	// SlotBase and SlotLimit are the writable bounds that were in force.
	SlotBase, SlotLimit uintptr
	// PayloadBase and PayloadLen describe the staged payload inside the slot.
	PayloadBase uintptr
	PayloadLen  uintptr

	// Offset is the SIGNED byte displacement of Addr from SlotBase. Negative
	// means below the slot (underflow), >= span means above it (overflow).
	Offset int64
	// PayloadOffset is the signed displacement of Addr from PayloadBase. This
	// is usually the number the developer actually wants: "you read 12 bytes
	// past the end of the message."
	PayloadOffset int64

	// Kind classifies the boundary struck.
	Kind FaultKind
	// Reason is the runtime's own panic text.
	Reason string

	// Frames holds the captured PCs, faulting frame first after the trap
	// machinery is elided.
	Frames  [maxFaultFrames]uintptr
	NFrames int
}

// Stack symbolizes the captured PCs. Allocates; call it off the hot path.
func (f *Fault) Stack() string {
	if f.NFrames == 0 {
		return ""
	}
	var b strings.Builder
	frames := runtime.CallersFrames(f.Frames[:f.NFrames])
	skipping := true
	for {
		fr, more := frames.Next()
		if skipping && trapFrame(fr.Function) {
			if !more {
				break
			}
			continue
		}
		skipping = false
		fmt.Fprintf(&b, "\t%s\n\t\t%s:%d +%#x\n", fr.Function, fr.File, fr.Line, fr.PC-fr.Entry)
		if !more {
			break
		}
	}
	return b.String()
}

// String renders the one-line register dump.
func (f *Fault) String() string {
	addr := "<unavailable>"
	if f.AddrValid {
		addr = fmt.Sprintf("%#016x", f.Addr)
	}
	return fmt.Sprintf(
		"flintlock: %s ticket=%d slot=%d epoch=%d si_addr=%s slot_off=%+d payload_off=%+d "+
			"slot=[%#x,%#x) payload=[%#x+%d) t=%s dwell=%dµs: %s",
		f.Kind, f.Ticket, f.Slot, f.Epoch, addr, f.Offset, f.PayloadOffset,
		f.SlotBase, f.SlotLimit, f.PayloadBase, f.PayloadLen,
		f.At.Format("15:04:05.000000"), f.Elapsed.Microseconds(), f.Reason,
	)
}

// GuardViolation is the error returned by Execute when the payload's parser
// struck a guard page. It carries the full forensic record.
type GuardViolation struct{ Fault Fault }

func (e *GuardViolation) Error() string { return e.Fault.String() }

// Kind exposes the classification for errors.Is-free switching.
func (e *GuardViolation) Kind() FaultKind { return e.Fault.Kind }

// captureFrames records raw PCs. It performs NO symbolization and NO
// allocation, because it runs on the trap path.
//
// It is called from inside the deferred recover, at which point the panicking
// frames have NOT yet been unwound: the physical stack still reads
//
//	<capture> -> <deferred closure> -> runtime.gopanic -> runtime.sigpanic
//	          -> <the instruction that touched the guard page> -> ...
//
// so the offending frame is genuinely captured here, not reconstructed.
// Symbolization and elision of the trap plumbing are deferred to Stack(),
// which callers invoke off the hot path when they actually report a fault.
func captureFrames(dst *[maxFaultFrames]uintptr) int {
	return runtime.Callers(2, dst[:])
}

// trapFrame reports whether a symbolized frame is trap plumbing rather than
// parser code, and so should be elided from a reported stack.
func trapFrame(fn string) bool {
	switch fn {
	case "runtime.gopanic", "runtime.sigpanic", "runtime.panicmem",
		"runtime.panicmemAddr", "runtime.deferreturn", "runtime.Callers":
		return true
	}
	return strings.HasPrefix(fn, "github.com/flintlock/flintlock.captureFrames") ||
		strings.HasPrefix(fn, "github.com/flintlock/flintlock.(*Sandbox).classify") ||
		strings.HasPrefix(fn, "github.com/flintlock/flintlock.(*Sandbox).run")
}

// nilDerefText is the runtime's fixed message for a null-page fault. With
// SetPanicOnFault enabled, runtime.sigpanic still routes addresses below 0x1000
// through panicmem() -- which discards si_addr -- rather than panicmemAddr().
// So a null deref is the one fault that must be identified by its text.
//
// Real guard-page strikes are at arena addresses far above 0x1000 and therefore
// always arrive through panicmemAddr with an exact si_addr; they never depend
// on this string.
const nilDerefText = "invalid memory address or nil pointer dereference"

// isMemoryFault reports whether a recovered panic value is a hardware memory
// fault, as opposed to an ordinary runtime error (index out of range, integer
// divide by zero) or a deliberate panic from the parser.
//
// The discriminator is the Addr() method, which only runtime.errorAddressString
// implements and which the runtime only produces on the paniconfault path.
// Bounds and divide errors also satisfy runtime.Error, so runtime.Error alone
// is NOT a sufficient test -- getting this wrong silently reclassifies ordinary
// logic bugs as memory-safety violations.
func isMemoryFault(r any) (addr uintptr, valid bool, isFault bool) {
	if a, ok := r.(interface{ Addr() uintptr }); ok {
		if _, isRT := r.(runtime.Error); isRT {
			return a.Addr(), true, true
		}
	}
	if rt, ok := r.(runtime.Error); ok {
		if strings.Contains(rt.Error(), nilDerefText) {
			return 0, false, true
		}
	}
	return 0, false, false
}

// ---------------------------------------------------------------------------
// Lock-free fault ledger
// ---------------------------------------------------------------------------

// faultRing is a fixed-capacity, allocation-free MPMC record ledger using a
// per-entry seqlock. Writers never block each other and never block readers;
// a reader that races a writer on the same cell retries or drops that cell
// rather than taking a lock. Overwrites the oldest record when full.
type faultRing struct {
	buf  []Fault
	seq  []atomic.Uint64
	head atomic.Uint64
	mask uint64
	drop atomic.Uint64
}

func newFaultRing(capacity int) *faultRing {
	// Round up to a power of two so index reduction is a mask, not a modulo.
	n := 1
	for n < capacity {
		n <<= 1
	}
	return &faultRing{
		buf:  make([]Fault, n),
		seq:  make([]atomic.Uint64, n),
		mask: uint64(n - 1),
	}
}

// push publishes a record. Wait-free for the writer.
func (r *faultRing) push(f *Fault) {
	i := (r.head.Add(1) - 1) & r.mask
	s := &r.seq[i]
	// Odd sequence == write in flight; readers observing odd must retry.
	s.Add(1)
	r.buf[i] = *f
	s.Add(1)
}

// snapshot returns the currently readable records, oldest first. Cells torn by
// a concurrent writer are skipped and counted, never returned half-written.
func (r *faultRing) snapshot() []Fault {
	head := r.head.Load()
	n := uint64(len(r.buf))
	if head < n {
		n = head
	}
	out := make([]Fault, 0, n)
	for k := head - n; k < head; k++ {
		i := k & r.mask
		s := &r.seq[i]
		before := s.Load()
		if before&1 != 0 {
			continue // writer mid-flight
		}
		f := r.buf[i]
		if s.Load() != before {
			continue // torn under us
		}
		out = append(out, f)
	}
	return out
}

// ensure Fault stays trivially copyable (no pointer chasing on the trap path).
var _ = unsafe.Sizeof(Fault{})

// Overrun returns how many bytes past the END of the payload the access landed.
// 0 means the access touched the first byte immediately after the payload --
// a textbook off-by-one. Negative values mean the access was below the payload
// end (an underflow or an in-bounds fault) and should be read via Offset.
func (f *Fault) Overrun() int64 { return f.PayloadOffset - int64(f.PayloadLen) }

// Underrun returns how many bytes BELOW the start of the payload the access
// landed. 0 means it touched the byte immediately before the payload.
func (f *Fault) Underrun() int64 {
	if f.PayloadOffset >= 0 {
		return -1
	}
	return -f.PayloadOffset - 1
}
