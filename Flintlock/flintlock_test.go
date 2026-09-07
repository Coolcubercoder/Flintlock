package flintlock

import (
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"testing"
	"unsafe"
)

// sink defeats dead-store elimination on deliberate out-of-bounds reads.
var sink byte

func mustNew(t *testing.T, cfg Config) *Sandbox {
	t.Helper()
	s, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// read performs an unchecked load at ptr+off. This is the untrusted-parser
// primitive the whole design exists to contain.
func read(ptr unsafe.Pointer, off int) byte {
	return *(*byte)(unsafe.Add(ptr, off))
}

func TestGeometry(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 1, Slots: 4})
	ps := uintptr(s.PageSize())

	if uintptr(s.SlotCapacity())%ps != 0 {
		t.Fatalf("slot span %d not page-aligned to %d", s.SlotCapacity(), ps)
	}
	if s.arena.base%ps != 0 {
		t.Fatalf("arena base %#x not page-aligned", s.arena.base)
	}
	if got, want := s.arena.stride, s.arena.span+ps; got != want {
		t.Fatalf("stride %d want %d", got, want)
	}
	if got, want := s.arena.total, s.arena.stride*4+ps; got != want {
		t.Fatalf("total %d want %d", got, want)
	}
	// Every slot must be flanked below and above by a guard page.
	for i := 0; i < 4; i++ {
		lo := s.arena.slotBase(i) - ps
		hi := s.arena.slotBase(i) + s.arena.span
		if _, _, k := s.arena.region(lo); k != FaultUnderflow && k != FaultOverflow {
			t.Fatalf("slot %d lower flank %#x not a guard (%v)", i, lo, k)
		}
		if _, _, k := s.arena.region(hi); k != FaultUnderflow && k != FaultOverflow {
			t.Fatalf("slot %d upper flank %#x not a guard (%v)", i, hi, k)
		}
	}
	t.Logf("pagesize=%d span=%d stride=%d total=%d guards=%d",
		ps, s.arena.span, s.arena.stride, s.arena.total, s.Slots()+1)
}

func TestCleanExecution(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 4})
	payload := make([]byte, 1500)
	for i := range payload {
		payload[i] = byte(i * 7)
	}

	var sum uint64
	err := s.Execute(payload, func(p unsafe.Pointer) {
		for i := 0; i < len(payload); i++ {
			sum += uint64(read(p, i))
		}
	})
	if err != nil {
		t.Fatalf("clean execution failed: %v", err)
	}
	var want uint64
	for _, b := range payload {
		want += uint64(b)
	}
	if sum != want {
		t.Fatalf("checksum %d want %d (payload not faithfully staged)", sum, want)
	}
	if st := s.Stats(); st.Executed != 1 || st.Trapped != 0 {
		t.Fatalf("stats %+v", st)
	}
}

// TestOverflowByOneByte is the headline property: with the default alignment a
// single byte past the payload end strikes the guard page.
func TestOverflowByOneByte(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 4})
	payload := make([]byte, 100)

	err := s.Execute(payload, func(p unsafe.Pointer) {
		sink = read(p, len(payload)) // exactly one past the end
	})

	var gv *GuardViolation
	if err == nil {
		t.Fatal("one-byte overflow was NOT trapped")
	}
	gv, ok := err.(*GuardViolation)
	if !ok {
		t.Fatalf("wrong error type %T: %v", err, err)
	}
	if gv.Fault.Kind != FaultOverflow {
		t.Fatalf("kind=%v want GUARD_OVERFLOW", gv.Fault.Kind)
	}
	if !gv.Fault.AddrValid {
		t.Fatal("no si_addr captured")
	}
	if got := gv.Fault.Overrun(); got != 0 {
		t.Fatalf("overrun=%d want 0 (first byte past end)", got)
	}
	if gv.Fault.Addr != gv.Fault.SlotLimit {
		t.Fatalf("si_addr %#x should equal slot limit %#x", gv.Fault.Addr, gv.Fault.SlotLimit)
	}
	t.Logf("%v", &gv.Fault)
	t.Logf("faulting frame:\n%s", gv.Fault.Stack())
}

func TestOverflowDeepRun(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 8192, Slots: 4})
	payload := make([]byte, 64)

	// A runaway strlen-style scan: walks forward until something stops it.
	err := s.Execute(payload, func(p unsafe.Pointer) {
		for i := 0; i < 1<<20; i++ {
			sink |= read(p, i)
		}
	})
	gv, ok := err.(*GuardViolation)
	if !ok {
		t.Fatalf("runaway scan not trapped: %v", err)
	}
	if gv.Fault.Kind != FaultOverflow || gv.Fault.Overrun() != 0 {
		t.Fatalf("expected trap at first byte past end, got %v overrun=%d",
			gv.Fault.Kind, gv.Fault.Overrun())
	}
	t.Logf("runaway scan halted after %d bytes: %v", gv.Fault.PayloadOffset, &gv.Fault)
}

// TestUnderflow uses a full-span payload so both edges are byte-exact.
func TestUnderflow(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 4})
	payload := make([]byte, s.SlotCapacity())

	err := s.Execute(payload, func(p unsafe.Pointer) {
		sink = read(p, -1) // one byte below the slot base
	})
	gv, ok := err.(*GuardViolation)
	if !ok {
		t.Fatalf("underflow not trapped: %v", err)
	}
	if gv.Fault.Kind != FaultUnderflow {
		t.Fatalf("kind=%v want GUARD_UNDERFLOW", gv.Fault.Kind)
	}
	if gv.Fault.Offset != -1 {
		t.Fatalf("offset=%d want -1", gv.Fault.Offset)
	}
	t.Logf("%v", &gv.Fault)
}

// TestUnderflowAlignLow proves the AlignLow mode gives byte-exact underflow
// detection for a short payload.
func TestUnderflowAlignLow(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 4, AlignLow: true})
	payload := make([]byte, 32)

	err := s.Execute(payload, func(p unsafe.Pointer) {
		sink = read(p, -1)
	})
	gv, ok := err.(*GuardViolation)
	if !ok {
		t.Fatalf("aligned-low underflow not trapped: %v", err)
	}
	if gv.Fault.Kind != FaultUnderflow || gv.Fault.Underrun() != 0 {
		t.Fatalf("kind=%v underrun=%d want GUARD_UNDERFLOW/0", gv.Fault.Kind, gv.Fault.Underrun())
	}
	t.Logf("%v", &gv.Fault)
}

// TestUseAfterFree proves the quarantine turns a dangling pointer into a trap.
// The stale pointer is deliberately dereferenced AFTER Execute has returned.
func TestUseAfterFree(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 4, QuarantineDepth: 16})

	var stale unsafe.Pointer
	payload := make([]byte, 128)
	if err := s.Execute(payload, func(p unsafe.Pointer) {
		sink = read(p, 0) // legal while armed
		stale = p         // the bug: pointer escapes its lifetime
	}); err != nil {
		t.Fatalf("clean run failed: %v", err)
	}

	// The slot is now PROT_NONE. Touch the dangling pointer on this goroutine,
	// arming the same per-goroutine trap the sandbox uses internally.
	var faultAddr uintptr
	var caught bool
	func() {
		prev := debug.SetPanicOnFault(true)
		defer func() {
			debug.SetPanicOnFault(prev)
			if r := recover(); r != nil {
				addr, ok, isFault := isMemoryFault(r)
				faultAddr, caught = addr, ok && isFault
			}
		}()
		sink = *(*byte)(stale)
	}()

	if !caught {
		t.Fatal("use-after-free was NOT trapped; quarantine did not revoke access")
	}
	slot, off, kind := s.arena.region(faultAddr)
	if kind != FaultUseAfterFree {
		t.Fatalf("kind=%v want USE_AFTER_FREE", kind)
	}
	t.Logf("UAF trapped: si_addr=%#x slot=%d off=%+d kind=%v", faultAddr, slot, off, kind)
}

// TestSurvivesAndContinues is the core operational claim: a trap isolates one
// worker and every other worker keeps running to completion at full speed.
func TestSurvivesAndContinues(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 8})

	const workers = 64
	const iters = 200
	var clean, trapped, wrong atomic.Int64

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			payload := make([]byte, 64+w)
			for i := range payload {
				payload[i] = byte(w)
			}
			for it := 0; it < iters; it++ {
				malicious := (it+w)%3 == 0
				err := s.Execute(payload, func(p unsafe.Pointer) {
					var sum int
					for j := 0; j < len(payload); j++ {
						sum += int(read(p, j))
					}
					if sum != w*len(payload) {
						wrong.Add(1) // cross-tenant contamination
					}
					if malicious {
						sink = read(p, len(payload)+4096) // deep overflow
					}
				})
				switch {
				case err == nil && !malicious:
					clean.Add(1)
				case err != nil && malicious:
					if _, ok := err.(*GuardViolation); ok {
						trapped.Add(1)
					}
				default:
					t.Errorf("worker %d iter %d: malicious=%v err=%v", w, it, malicious, err)
				}
			}
		}(w)
	}
	wg.Wait()

	if wrong.Load() != 0 {
		t.Fatalf("%d cross-tenant data contaminations", wrong.Load())
	}
	total := int64(workers * iters)
	if clean.Load()+trapped.Load() != total {
		t.Fatalf("accounting: clean=%d trapped=%d want total=%d", clean.Load(), trapped.Load(), total)
	}
	st := s.Stats()
	if st.InFlight != 0 {
		t.Fatalf("leaked in-flight: %d", st.InFlight)
	}
	if st.Trapped != uint64(trapped.Load()) {
		t.Fatalf("ledger trapped=%d observed=%d", st.Trapped, trapped.Load())
	}
	t.Logf("clean=%d trapped=%d contended=%d retired=%d ledger=%d",
		clean.Load(), trapped.Load(), st.Contended, st.Retired, len(s.Faults()))
}

// TestOrdinaryPanicPropagates: a logic bug must stay loud, not be laundered
// into a memory-safety error.
func TestOrdinaryPanicPropagates(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 4})
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("ordinary panic was swallowed")
		}
		if _, ok := r.(*GuardViolation); ok {
			t.Fatalf("ordinary panic misclassified as guard violation")
		}
		t.Logf("propagated as expected: %v", r)
	}()
	_ = s.Execute(make([]byte, 8), func(p unsafe.Pointer) {
		panic("parser logic bug")
	})
	t.Fatal("unreachable")
}

// TestBoundsErrorNotMisclassified guards the discriminator in isMemoryFault:
// index-out-of-range is a runtime.Error but NOT a memory fault.
func TestBoundsErrorNotMisclassified(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 4})
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("bounds error was swallowed as a memory fault")
		}
	}()
	_ = s.Execute(make([]byte, 8), func(p unsafe.Pointer) {
		b := make([]byte, 2)
		i := 5
		sink = b[i]
	})
	t.Fatal("unreachable")
}

func TestPayloadTooLarge(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 2})
	err := s.Execute(make([]byte, s.SlotCapacity()+1), func(unsafe.Pointer) {})
	if err == nil {
		t.Fatal("oversized payload accepted")
	}
	t.Logf("rejected: %v", err)
}

func TestNonBlockingExhaustion(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 1, NonBlocking: true, SpinBudget: 1})
	hold := make(chan struct{})
	armed := make(chan struct{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = s.Execute(make([]byte, 8), func(unsafe.Pointer) {
			close(armed) // the only slot is now genuinely occupied
			<-hold
		})
	}()
	<-armed // without this the pool may not be saturated yet and nothing fails

	var err error
	for i := 0; i < 10000 && err == nil; i++ {
		err = s.Execute(make([]byte, 8), func(unsafe.Pointer) {})
		runtime.Gosched()
	}
	close(hold)
	<-done
	if err != ErrPoolExhausted {
		t.Fatalf("err=%v want ErrPoolExhausted", err)
	}
}

func TestScrubPreventsResidue(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 1})
	secret := make([]byte, 256)
	for i := range secret {
		secret[i] = 0xAA
	}
	if err := s.Execute(secret, func(unsafe.Pointer) {}); err != nil {
		t.Fatal(err)
	}
	// Next tenant is smaller; the bytes above it must be zero, not 0xAA.
	next := make([]byte, 8)
	var residue int
	if err := s.Execute(next, func(p unsafe.Pointer) {
		base := unsafe.Add(p, -1024)
		for i := 0; i < 1024; i++ {
			if *(*byte)(unsafe.Add(base, i)) == 0xAA {
				residue++
			}
		}
	}); err != nil {
		t.Fatal(err)
	}
	if residue != 0 {
		t.Fatalf("%d bytes of previous tenant's data leaked into the next", residue)
	}
}

func TestCloseIsIdempotentAndDrains(t *testing.T) {
	s, err := New(Config{SlotBytes: 4096, Slots: 2})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("second close: %v", err)
	}
	if err := s.Execute(make([]byte, 8), func(unsafe.Pointer) {}); err != ErrClosed {
		t.Fatalf("err=%v want ErrClosed", err)
	}
}

func TestSupervisorLifecycle(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 2})
	sv := Supervise(s)
	sv.SetReraise(false)
	sv.Shutdown()
	if err := s.Execute(make([]byte, 8), func(unsafe.Pointer) {}); err != ErrClosed {
		t.Fatalf("supervisor did not tear down: %v", err)
	}
}

// TestZeroAllocations is the GC claim, measured rather than asserted.
func TestZeroAllocations(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 8})
	payload := make([]byte, 512)
	fn := func(p unsafe.Pointer) { sink = read(p, 0) }

	avg := testing.AllocsPerRun(2000, func() {
		if err := s.Execute(payload, fn); err != nil {
			t.Fatal(err)
		}
	})
	if avg != 0 {
		t.Fatalf("Execute allocates %.2f objects/op; must be 0 for zero GC pressure", avg)
	}
	t.Logf("allocations per Execute: %.0f", avg)
}

func BenchmarkExecuteClean(b *testing.B) {
	s, err := New(Config{SlotBytes: 4096, Slots: runtime.GOMAXPROCS(0) * 2})
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	payload := make([]byte, 512)
	fn := func(p unsafe.Pointer) { sink = read(p, 0) }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := s.Execute(payload, fn); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkExecuteParallel(b *testing.B) {
	s, err := New(Config{SlotBytes: 4096, Slots: runtime.GOMAXPROCS(0) * 4})
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	payload := make([]byte, 512)
	fn := func(p unsafe.Pointer) { sink = read(p, 0) }

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			if err := s.Execute(payload, fn); err != nil {
				b.Fatal(err)
			}
		}
	})
}

func BenchmarkExecuteQuarantined(b *testing.B) {
	s, err := New(Config{SlotBytes: 4096, Slots: 16, QuarantineDepth: 8})
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	payload := make([]byte, 512)
	fn := func(p unsafe.Pointer) { sink = read(p, 0) }

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := s.Execute(payload, fn); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkTrap(b *testing.B) {
	s, err := New(Config{SlotBytes: 4096, Slots: 8})
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	payload := make([]byte, 512)
	fn := func(p unsafe.Pointer) { sink = read(p, 512) }

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := s.Execute(payload, fn); err == nil {
			b.Fatal("expected trap")
		}
	}
}

var _ = fmt.Sprint

// TestQuarantineStarvationBreaker: a cooldown that can never be satisfied must
// degrade to a shortened UAF window, never to a hang. Two slots, a quarantine
// depth far larger than the ticket range, and heavy concurrency.
func TestQuarantineStarvationBreaker(t *testing.T) {
	s := mustNew(t, Config{
		SlotBytes: 4096, Slots: 2, QuarantineDepth: 1 << 40, SpinBudget: 4,
	})
	var done atomic.Int64
	var acc atomic.Uint64
	var wg sync.WaitGroup
	for w := 0; w < 32; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			payload := make([]byte, 64)
			// Goroutine-local sink: writing the package-level one from many
			// goroutines is itself a data race and would mask real findings.
			var local byte
			for i := 0; i < 50; i++ {
				if err := s.Execute(payload, func(p unsafe.Pointer) {
					local = read(p, 0)
				}); err != nil {
					t.Errorf("execute: %v", err)
					return
				}
				done.Add(1)
			}
			acc.Add(uint64(local))
		}()
	}
	wg.Wait()
	if done.Load() != 32*50 {
		t.Fatalf("completed %d of %d", done.Load(), 32*50)
	}
	st := s.Stats()
	if st.Rearms == 0 {
		t.Fatal("starvation breaker never fired")
	}
	t.Logf("no deadlock: completed=%d rearms=%d contended=%d acc=%d",
		done.Load(), st.Rearms, st.Contended, acc.Load())
}

// TestRetireAndReclaim covers the forensic-freeze configuration.
func TestRetireAndReclaim(t *testing.T) {
	s := mustNew(t, Config{SlotBytes: 4096, Slots: 4, NoReclaim: true})
	for i := 0; i < 4; i++ {
		payload := make([]byte, 32)
		err := s.Execute(payload, func(p unsafe.Pointer) { sink = read(p, 32) })
		if _, ok := err.(*GuardViolation); !ok {
			t.Fatalf("iteration %d: %v", i, err)
		}
	}
	if got := s.Stats().Retired; got != 4 {
		t.Fatalf("retired=%d want 4", got)
	}
	// Pool is fully retired; non-blocking admission must now refuse.
	if n := s.Reclaim(); n != 4 {
		t.Fatalf("reclaimed %d want 4", n)
	}
	if err := s.Execute(make([]byte, 32), func(p unsafe.Pointer) { sink = read(p, 0) }); err != nil {
		t.Fatalf("post-reclaim execute: %v", err)
	}
	t.Logf("retired 4 cells, reclaimed 4, pool healthy")
}

// TestFaultLedger verifies the lock-free ring records under concurrency.
func TestFaultLedger(t *testing.T) {
	var seen atomic.Int64
	s := mustNew(t, Config{
		SlotBytes: 4096, Slots: 8, LedgerSize: 64,
		OnFault: func(f Fault) {
			if f.Kind == FaultOverflow {
				seen.Add(1)
			}
		},
	})
	var wg sync.WaitGroup
	for w := 0; w < 16; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 40; i++ {
				_ = s.Execute(make([]byte, 32), func(p unsafe.Pointer) { sink = read(p, 32) })
			}
		}()
	}
	wg.Wait()
	if got := seen.Load(); got != 16*40 {
		t.Fatalf("OnFault fired %d times want %d", got, 16*40)
	}
	led := s.Faults()
	if len(led) == 0 || len(led) > 64 {
		t.Fatalf("ledger holds %d records, want 1..64", len(led))
	}
	for _, f := range led {
		if f.Kind != FaultOverflow || !f.AddrValid {
			t.Fatalf("torn or misclassified record: %v", &f)
		}
	}
	t.Logf("ledger retained %d/%d records, all intact", len(led), 16*40)
}
