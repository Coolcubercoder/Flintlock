package flintlock

import (
	"os"
	"os/signal"
	"sync/atomic"
	"syscall"
)

// Supervisor is the asynchronous kernel signal listener for the sandbox fleet.
//
// # What this does and does not catch
//
// It handles the ASYNCHRONOUS signals -- SIGINT, SIGTERM, SIGQUIT, SIGHUP --
// which os/signal delivers reliably. On receipt it drains every registered
// Sandbox, unmaps its arena, and re-raises the signal with default disposition
// so the process dies with the correct exit status and core-dump behavior.
//
// It does NOT catch SIGSEGV, and no correct Go program can. Per os/signal's
// documentation, a synchronous fault raised by the program itself is converted
// by the runtime into a run-time panic and is never delivered to a Notify
// channel:
//
//	signal.Notify(ch, syscall.SIGSEGV)   // compiles, never fires
//
// Nor may the runtime's handler be displaced with a raw syscall.Sigaction: the
// runtime depends on it for stack growth, async preemption and scheduling, so
// overwriting it turns the first goroutine stack growth into a hard crash.
//
// Guard-page strikes are trapped on the synchronous path instead, by
// runtime/debug.SetPanicOnFault inside Sandbox.run. That is the real hardware
// trap, it carries si_addr, and it is per-goroutine -- which is exactly the
// isolation property wanted here and one a process-wide signal handler could
// not provide.
//
// Why teardown matters enough to want a supervisor at all: an arena is a live
// mapping full of PROT_NONE holes. Leaving it mapped through a slow shutdown
// keeps that VA reserved and keeps stale parser pointers pointing at
// unmapped-but-reserved space. Draining and unmapping on the way out makes late
// accesses fault cleanly.
type Supervisor struct {
	ch      chan os.Signal
	boxes   atomic.Pointer[[]*Sandbox]
	stop    chan struct{}
	stopped atomic.Bool
	// OnSignal is invoked with the received signal before teardown. May be nil.
	OnSignal func(os.Signal)
	// Reraise re-sends the signal with default disposition after teardown so
	// the process exits as the operator expects. Default true.
	reraise bool
}

// Supervise starts the listener over the given sandboxes.
//
// The returned Supervisor keeps running until Stop is called. Signals watched:
// SIGINT, SIGTERM, SIGQUIT, SIGHUP.
func Supervise(boxes ...*Sandbox) *Supervisor {
	sv := &Supervisor{
		ch:      make(chan os.Signal, 4),
		stop:    make(chan struct{}),
		reraise: true,
	}
	list := append([]*Sandbox(nil), boxes...)
	sv.boxes.Store(&list)

	signal.Notify(sv.ch,
		syscall.SIGINT,
		syscall.SIGTERM,
		syscall.SIGQUIT,
		syscall.SIGHUP,
	)
	go sv.loop()
	return sv
}

// Register adds a sandbox to the teardown set. Safe to call concurrently.
func (sv *Supervisor) Register(s *Sandbox) {
	for {
		old := sv.boxes.Load()
		next := append(append([]*Sandbox(nil), (*old)...), s)
		if sv.boxes.CompareAndSwap(old, &next) {
			return
		}
	}
}

// SetReraise controls whether the signal is re-delivered with default
// disposition after teardown. Call before a signal arrives.
func (sv *Supervisor) SetReraise(v bool) { sv.reraise = v }

func (sv *Supervisor) loop() {
	select {
	case sig := <-sv.ch:
		if sv.OnSignal != nil {
			sv.OnSignal(sig)
		}
		sv.teardown()
		if sv.reraise {
			// Restore default disposition and re-raise, so the process exits
			// with the conventional 128+signo status rather than a bare
			// os.Exit that hides how it died.
			signal.Reset(sig.(syscall.Signal))
			_ = syscall.Kill(os.Getpid(), sig.(syscall.Signal))
		}
	case <-sv.stop:
	}
	signal.Stop(sv.ch)
}

// teardown drains and unmaps every registered arena.
func (sv *Supervisor) teardown() {
	for _, s := range *sv.boxes.Load() {
		if s != nil {
			_ = s.Close()
		}
	}
}

// Stop halts the listener without tearing down the sandboxes. Idempotent.
func (sv *Supervisor) Stop() {
	if sv.stopped.CompareAndSwap(false, true) {
		close(sv.stop)
	}
}

// Shutdown halts the listener and closes every registered sandbox.
func (sv *Supervisor) Shutdown() {
	sv.Stop()
	sv.teardown()
}
