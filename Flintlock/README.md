# Flintlock

**Hardware-assisted memory sandboxing for high-frequency parsing of untrusted bytes in Go.**

Flintlock runs untrusted, pointer-arithmetic-heavy parsing code at bare-metal speed while trapping
out-of-bounds access in the MMU. There is no shadow memory, no instrumentation, no interpreter, and
no debugging runtime. Detection costs **zero instructions** in the non-faulting path, because the
page tables are already doing the work.

- Pure Go, standard library only. No cgo, no external modules, no build-time codegen.
- **0 heap allocations** per clean `Execute`, so a steady stream of parses adds nothing to the GC.
- A trap isolates **exactly one goroutine**. The rest of the process never pauses.
- ~96 ns per guarded parse on an Apple M4.

```go
err := sandbox.Execute(untrustedFrame, func(p unsafe.Pointer) {
    parseInPlace(p) // one byte past the payload and this traps in hardware
})

var gv *flintlock.GuardViolation
if errors.As(err, &gv) {
    log.Printf("%s at %+d bytes past end, si_addr=%#x", gv.Fault.Kind, gv.Fault.Overrun(), gv.Fault.Addr)
}
```

---

## The problem

Parsing untrusted wire data quickly in Go means dropping to `unsafe.Pointer` and doing your own
pointer arithmetic. The moment you do, you have given up every guarantee the language provides: a
bad length field becomes a buffer overflow, and a retained pointer becomes a use-after-free. The
usual answers each cost you the reason you dropped to `unsafe` in the first place — bounds-checked
copies, ASAN-style shadow memory, or a sandboxing runtime.

Flintlock takes the third option: let the hardware do it. Every live buffer is flanked by pages the
kernel has been told are unreachable. Overstepping is a page fault, and a page fault is free until
it happens.

## How it works

One `mmap` reservation, carved into uniform-stride slots, each flanked above and below by a
`PROT_NONE` page:

```
 base
  |
  v
 +-------+-------------+-------+-------------+-------+-------------+-------+
 |   G   |   slot 0    |   G   |   slot 1    |   G   |   slot 2    |   G   |
 +-------+-------------+-------+-------------+-------+-------------+-------+
  <-page-> <-- span --> <-page->
  <------ stride ------>

 stride = span + pageSize        guards = slots + 1        total = stride*slots + pageSize
```

Adjacent slots share a physical guard page: the upper guard of slot *i* is the lower guard of slot
*i+1*. That is not a weakening — the page is `PROT_NONE` from both directions, so a one-byte
overflow off slot *i* and a one-byte underflow off slot *i+1* both fault — and it halves guard
address-space consumption from 2n to n+1.

Uniform stride is what lets an arbitrary faulting address be inverted back to its owning slot with
two integer divisions instead of a search.

The payload is staged **flush against the upper guard**, so a one-byte overstep traps immediately
rather than walking silently through slack:

```
 base                                       limit == guard
  |                                              |
  v                                              v
  [........ unused slack ........][== payload ==]|GUARD|
                                                ^
                                      one byte past here traps
```

### Trapping the fault

This is the part most designs get wrong, so it is worth stating plainly:

```go
signal.Notify(ch, syscall.SIGSEGV)   // compiles, looks right, NEVER FIRES
```

The Go runtime installs its own handler for the synchronous fault signals. As `os/signal`
documents, a fault raised by the program itself is converted into a run-time panic and is **never
delivered to a Notify channel**. Replacing that handler with a raw `syscall.Sigaction` is worse: the
runtime depends on it for stack growth, async preemption and scheduling, so displacing it turns the
first goroutine stack growth into a hard crash.

The supported path is `runtime/debug.SetPanicOnFault`, which sets a flag on the **current
goroutine**. When the MMU raises `SIGSEGV`, `runtime.sigpanic` observes the flag and raises a
recoverable panic carrying the faulting address:

```go
panic value implements interface{ Addr() uintptr }   // this is si_addr, from the kernel fault frame
```

Because the flag is per-goroutine, the blast radius of a trap is exactly one worker. That is a
property a process-wide signal handler could not provide.

`Fault.Addr` is the real `si_addr`, not a reconstruction. The test suite asserts it equals the slot
limit exactly on a one-byte overflow.

## Install

```
go get github.com/flintlock/flintlock
```

Requires Go 1.22+ and a POSIX platform (Linux, macOS, BSD). See [Portability](#portability).

## Usage

```go
sb, err := flintlock.New(flintlock.Config{
    SlotBytes: 64 << 10,
    Slots:     runtime.GOMAXPROCS(0) * 2,
})
if err != nil {
    return err
}
defer sb.Close()

// Tear the arena down cleanly on SIGINT/SIGTERM/SIGQUIT/SIGHUP.
sv := flintlock.Supervise(sb)
defer sv.Stop()

for frame := range incoming {
    err := sb.Execute(frame, func(p unsafe.Pointer) {
        handle(p, len(frame))
    })
    if err != nil {
        metrics.Malformed.Inc()   // contained; the process keeps serving
    }
}
```

A complete worked example of an attacker-controlled length field being trapped is in
[example_test.go](example_test.go).

## The contract

`Execute` hands `fn` a raw pointer. Three rules, and the first one is sharp:

1. **`fn` must not pass the pointer to another goroutine.** The trap flag is per-goroutine. A fault
   on a goroutine that has not armed it is an unrecoverable `throw` that kills the process. This
   cannot be enforced at compile time.
2. **`fn` must not retain the pointer past its own return.** With `QuarantineDepth > 0` later use
   traps; with `0` it silently reads whatever tenant now owns the cell.
3. **`fn` should not call C code.** cgo frames fault outside Go's recovery path.

Ordinary panics from `fn` — `panic("...")`, index-out-of-range, divide-by-zero — are **not**
laundered into errors. They propagate, so real logic bugs stay loud. Only genuine hardware memory
faults become `*GuardViolation`. Set `TrapAllPanics` to change this.

## API

| Symbol | Purpose |
|---|---|
| `New(Config) (*Sandbox, error)` | Reserve the arena and install guard pages |
| `(*Sandbox) Execute(payload []byte, fn func(unsafe.Pointer)) error` | Stage and run under guard |
| `(*Sandbox) Close() error` | Drain in-flight parsers, unmap |
| `(*Sandbox) Stats() Stats` | Lock-free counter snapshot |
| `(*Sandbox) Faults() []Fault` | Fault ledger, oldest first |
| `(*Sandbox) Reclaim() int` | Return retired cells to service |
| `(*Sandbox) SlotCapacity() int` | Usable bytes per slot after page rounding |
| `Supervise(...*Sandbox) *Supervisor` | Async signal listener for clean teardown |
| `GuardViolation` | Error type carrying the `Fault` record |

Errors: `ErrClosed`, `ErrPoolExhausted`, `ErrPayloadTooLarge`, `ErrNilFn`.

### Config

| Field | Default | Meaning |
|---|---|---|
| `SlotBytes` | 64 KiB | Payload capacity per slot, rounded up to a page |
| `Slots` | `2*GOMAXPROCS`, min 8 | Pool width. Size to peak concurrency |
| `QuarantineDepth` | `0` (off) | Tickets a released slot stays `PROT_NONE`. Enables use-after-free detection |
| `AlignLow` | `false` | Flush payload against the lower guard instead of the upper |
| `AlignMask` | `0` | Payload alignment, e.g. `15` for 16-byte SIMD |
| `NonBlocking` | `false` | Return `ErrPoolExhausted` instead of spinning when saturated |
| `SpinBudget` | `64` | Probe sweeps before yielding |
| `NoScrub` | `false` | Disable inter-tenant zeroing (**information-disclosure risk**) |
| `NoReclaim` | `false` | Retire faulted cells permanently for forensics |
| `TrapAllPanics` | `false` | Convert ordinary panics to errors too |
| `LedgerSize` | `256` | Fault ring capacity |
| `OnFault` | `nil` | Callback on the faulting goroutine after containment |

The zero `Config` is valid and fully defaulted.

### Fault record

`Fault` carries `Addr` (raw `si_addr`), `Offset` (signed displacement from the slot base),
`PayloadOffset`, `Kind`, the slot geometry that was in force, timing, and captured PCs.
`Overrun()` gives bytes past the payload end — `0` is a textbook off-by-one. `Stack()` symbolizes
lazily, off the trap path, beginning at the faulting instruction with trap plumbing elided:

```
flintlock: GUARD_OVERFLOW ticket=0 slot=0 epoch=0 si_addr=0x000000010674c000 slot_off=+16384
payload_off=+100 slot=[0x106748000,0x10674c000) payload=[0x10674bf9c+100) dwell=17µs
```

Kinds: `GUARD_OVERFLOW`, `GUARD_UNDERFLOW`, `USE_AFTER_FREE`, `NULL_DEREF`, `FOREIGN_ADDRESS`.

## Performance

Apple M4, Go 1.27, darwin/arm64, 16 KiB pages, 512-byte payloads.

| Benchmark | ns/op | B/op | allocs/op |
|---|---:|---:|---:|
| `ExecuteClean` | 96–102 | 0 | **0** |
| `ExecuteParallel` (10P) | 213 | 0 | **0** |
| `ExecuteQuarantined` | ~1,100 | 0 | **0** |
| `Trap` (fault + classify + record) | ~2,900 | 440 | 3 |

The trap path allocates one `*GuardViolation` carrying the record. That is the cold path by
construction.

### Scaling: read this before sizing a deployment

Parallel throughput is flat to 4 cores and then regresses:

| GOMAXPROCS | 1 | 2 | 4 | 8 | 10 |
|---|---:|---:|---:|---:|---:|
| ns/op | 110 | 119 | 115 | 208 | 213 |

This is **cache-line contention, not core heterogeneity**. A calibration benchmark on the same
machine confirms it: unshared work scales 3.8× from 1→4 cores (56.7 → 14.7 ns), while a single
contended atomic fetch-add degrades **21×** under parallelism (2.7 → 56.8 ns).

`Execute` currently touches four shared atomics per call: the turnstile dispenser, `inflight`
(increment and decrement), and `executed`. Only the turnstile is inherent to the design. The other
three are bookkeeping and could be sharded across padded stripes indexed by the ticket that is
already computed — a known, unimplemented optimization.

Until then: **Flintlock scales well to roughly 4 concurrent parsers and flattens beyond that.** For
higher core counts, run several independent `Sandbox` instances and shard work across them, which
avoids the shared line entirely.

Also watch `Stats().Contended`. Under-sizing `Slots` is expensive — 64 workers on 8 slots logged
83,770 contended sweeps in the test suite. Correct and deadlock-free, but you are paying `Gosched`.

## Design trade-offs

**A short payload cannot be byte-exact on both edges at once.** With one guard pair and a payload
smaller than the page-rounded span, whichever edge you do not flush against leaves writable slack
that an overstepping pointer walks silently. Overflow is the dominant exploit primitive, so
right-alignment is the default; `AlignLow` inverts it. A payload sized to exactly `SlotCapacity()`
is byte-exact both ways.

**Use-after-free detection is not free**, which is why `QuarantineDepth` defaults to off. Revoking a
released slot's pages costs two `mprotect` calls plus a TLB shootdown: 102 ns → 1,169 ns per
`Execute`. Enable it deliberately.

**Virtual address cost is `(span + pageSize) × slots + pageSize`.** At a 16 KiB granule a 1 KiB slot
still consumes 32 KiB of VA. Guard pages are never faulted in, so RSS stays near zero — but VA is
not free at high slot counts.

**Page size is resolved at runtime, never hardcoded.** Linux/amd64 is 4096; darwin/arm64 is 16384;
linux/arm64 ships in 4K, 16K and 64K configurations. A hardcoded 4096 on a 16K host yields
`mprotect(EINVAL)` at best, and a guard page silently overlapping live data at worst.

## Guarantees and non-guarantees

Flintlock catches:

- Linear overflow and overread past the payload end (byte-exact by default)
- Underflow below the slot base
- Use-after-free through a stale pointer, when `QuarantineDepth > 0`
- Wild pointers landing anywhere in the arena, classified as `FOREIGN_ADDRESS`
- Inter-tenant residue disclosure, via scrub-on-recycle

Flintlock does **not** catch:

- Intra-payload corruption. Writing byte 3 when you meant byte 300 is in-bounds and invisible
- Oversteps into slack under a non-exact alignment, until they reach the guard
- Faults on goroutines that have not armed the per-goroutine trap flag (these kill the process)
- Anything in cgo frames
- Wild pointers that happen to land on unrelated valid mappings elsewhere in the process

It is a containment boundary for memory-safety *bugs*, not a capability sandbox for hostile *code*.
Untrusted code that can choose its own target addresses is out of scope.

## Portability

`syscall.Mmap`, `Mprotect` and `Munmap` here are POSIX. Verified on darwin/arm64. Linux (amd64 and
arm64, all page granules) is supported by the same code path; `MAP_ANON` is defined on both. A
Windows port needs `VirtualAlloc`/`VirtualProtect`.

## Testing

19 tests, 4 benchmarks, 1 runnable example. The suite deliberately oversteps in every direction and
asserts the hardware catches it — it does not mock the fault path.

```
go test ./...                       # correctness
go test -race -count=3 ./...        # lock-free claims
go test -run=NONE -bench=. -benchmem ./...
```

Verified clean under `-race` at GOMAXPROCS 1, 2, 10 and 16, and `go vet`/`gofmt` clean. Notable
coverage: byte-exact overflow and underflow, runaway scans, use-after-free through quarantine,
64-worker concurrent mixed clean/hostile load with cross-tenant contamination checks, ordinary
panics propagating unmolested, bounds errors not being misclassified as memory faults, starvation
breaking under an unsatisfiable quarantine cooldown, and a `testing.AllocsPerRun` assertion that
`Execute` allocates zero.

## License

Not yet specified.
