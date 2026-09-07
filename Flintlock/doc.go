// Package flintlock implements a hardware-assisted, zero-dependency memory
// sandbox for high-frequency parsing of untrusted bytes in Go.
//
// # Threat model
//
// The payload is untrusted. The parser operating on it (fn) is assumed to be
// fast, native, pointer-arithmetic-heavy code that may contain memory-safety
// defects: linear overreads/overwrites past the end of the buffer, negative
// index underflows before the start of it, and dangling-pointer reuse after the
// buffer's lifetime has ended. Flintlock does not attempt to make such code
// correct. It makes such code *trap deterministically* at the exact instruction
// that oversteps, at zero steady-state cost, and it converts that trap into an
// ordinary Go error on exactly one goroutine while every other goroutine in the
// process keeps running at full speed.
//
// # Enforcement is the MMU, not the CPU
//
// There is no shadow memory, no red-zone poisoning check, no instrumentation
// and no interpreter. Each live slot is flanked, on its immediate lower and
// immediate upper byte boundary, by a page mapped PROT_NONE. Detection is
// performed by the page tables. In the non-faulting path the cost of the entire
// safety property is zero instructions -- the hardware is already doing the
// translation.
//
// # How the SIGSEGV is actually caught
//
// Important, and a common and dangerous misconception:
//
//	signal.Notify(ch, syscall.SIGSEGV)   // <-- WILL NEVER FIRE
//
// The Go runtime installs its own handler for the synchronous fault signals
// (SIGSEGV, SIGBUS, SIGFPE). As os/signal documents, a fault raised by the
// program itself is turned into a run-time panic and is NOT delivered to a
// Notify channel. Replacing the runtime's handler with a raw syscall.Sigaction
// is worse: the runtime relies on that handler for stack growth, async
// preemption and goroutine scheduling, and displacing it crashes the process
// outright.
//
// The supported kernel path is runtime/debug.SetPanicOnFault. It sets a flag on
// the *current goroutine* (g.paniconfault). When the MMU raises SIGSEGV, the
// runtime's handler routes into runtime.sigpanic, observes the flag, and raises
// a recoverable panic whose value carries the faulting virtual address:
//
//	panic value implements interface{ Addr() uintptr }
//
// That address is the register dump. It is siginfo.si_addr, propagated out of
// the kernel's fault frame, and it is the ground truth Flintlock uses to
// compute the signed byte offset of the violation relative to the slot.
//
// Because the flag is per-goroutine, the blast radius of a trap is exactly one
// worker. Goroutines executing outside Execute are unaffected and are never
// stopped, slowed, or serialized.
//
// # Caveat that must be respected
//
// SetPanicOnFault is per-goroutine. If fn hands the sandboxed pointer to a
// *different* goroutine and that goroutine faults, the process dies with an
// unrecoverable "unexpected fault address" throw. fn must not leak the pointer
// off its own goroutine. See Sandbox.Execute.
//
// # Dependencies
//
// Standard library only: syscall, unsafe, sync/atomic, os, os/signal, runtime,
// runtime/debug, time, errors, fmt, strings. No cgo. No build-time codegen.
package flintlock
