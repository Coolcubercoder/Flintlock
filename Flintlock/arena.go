package flintlock

import (
	"fmt"
	"syscall"
	"unsafe"
)

// pageSize is the kernel's hardware page granule, resolved once at init.
//
// It is deliberately NOT a constant. linux/amd64 is 4096; darwin/arm64 is
// 16384; linux/arm64 ships in 4K, 16K and 64K kernel configurations. A
// hardcoded 4096 on a 16K-granule machine produces mprotect(EINVAL) at best
// and, if the arithmetic is merely misaligned rather than rejected, a guard
// page that silently overlaps live data at worst.
var pageSize = uintptr(syscall.Getpagesize())

// alignUp rounds v up to the next multiple of a. a must be a power of two,
// which every hardware page granule is.
func alignUp(v, a uintptr) uintptr { return (v + a - 1) &^ (a - 1) }

// arena is one contiguous anonymous mapping carved into guard-flanked slots.
//
// Byte layout, for slots=3 (G = PROT_NONE guard page, D = PROT_READ|PROT_WRITE
// data span):
//
//	base
//	 |
//	 v
//	+-------+-------------+-------+-------------+-------+-------------+-------+
//	|   G   |   D slot 0  |   G   |   D slot 1  |   G   |   D slot 2  |   G   |
//	+-------+-------------+-------+-------------+-------+-------------+-------+
//	 <-page-> <- span   -> <-page->
//	 <------- stride ----->
//
//	stride = span + pageSize
//	total  = stride*slots + pageSize
//	guards = slots + 1
//
// Each slot is flanked on BOTH its immediate lower and immediate upper byte
// boundary by a PROT_NONE page, as required. Adjacent slots share one physical
// guard page: the upper guard of slot i is the lower guard of slot i+1. This is
// not a weakening of the invariant -- the page is PROT_NONE from both
// directions, so a one-byte overflow off slot i and a one-byte underflow off
// slot i+1 both fault -- and it halves guard VA consumption from 2n to n+1.
//
// The whole extent is a single mmap so that the slots are stride-uniform, which
// is what lets faultLocate() invert an arbitrary faulting address back to an
// owning slot with two integer divisions instead of a search.
type arena struct {
	// raw is the exact slice returned by syscall.Mmap. Go's syscall package
	// keys its internal mmap bookkeeping off this slice, so Munmap must be
	// handed back this precise value -- not a reslice of it.
	//
	// raw spans PROT_NONE pages. Nothing may ever read or range over it. It is
	// held only for address arithmetic, subslicing, and teardown.
	raw []byte

	base   uintptr // uintptr(&raw[0]); page-aligned by mmap contract
	limit  uintptr // base + total
	total  uintptr // bytes mapped, including all guards
	span   uintptr // usable bytes per slot, page-multiple
	stride uintptr // span + pageSize
	slots  int
}

// mapArena reserves and protects the backing store for a pool of `slots`
// sandboxes, each holding at least slotBytes of payload.
func mapArena(slotBytes, slots int) (*arena, error) {
	if slots <= 0 {
		return nil, fmt.Errorf("flintlock: arena needs >=1 slot, got %d", slots)
	}
	if slotBytes <= 0 {
		return nil, fmt.Errorf("flintlock: arena needs >=1 slot byte, got %d", slotBytes)
	}

	// The data span must itself be a whole number of pages. mprotect operates
	// at page granularity, so a slot sized to a sub-page boundary would force
	// the upper guard to share a page with live data -- which mprotect would
	// then make inaccessible, destroying the tail of the slot.
	span := alignUp(uintptr(slotBytes), pageSize)
	stride := span + pageSize
	total := stride*uintptr(slots) + pageSize

	// Overflow guard on the size arithmetic itself before it reaches the kernel.
	if span < uintptr(slotBytes) || stride < span || total < stride {
		return nil, fmt.Errorf("flintlock: arena geometry overflows uintptr")
	}
	if uintptr(int(total)) != total {
		return nil, fmt.Errorf("flintlock: arena of %d bytes exceeds int", total)
	}

	// fd == -1 with MAP_ANON: demand-zero pages, no file backing, invisible to
	// the Go heap and therefore never scanned, swept or moved by the GC.
	// MAP_PRIVATE keeps a fork()ed child's writes from being visible here.
	raw, err := syscall.Mmap(
		-1, 0, int(total),
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_ANON|syscall.MAP_PRIVATE,
	)
	if err != nil {
		return nil, fmt.Errorf("flintlock: mmap %d bytes: %w", total, err)
	}

	a := &arena{
		raw:    raw,
		base:   uintptr(unsafe.Pointer(&raw[0])),
		total:  total,
		span:   span,
		stride: stride,
		slots:  slots,
	}
	a.limit = a.base + total

	// mmap is contractually page-aligned; assert rather than assume, because
	// every offset computation below depends on it.
	if a.base&(pageSize-1) != 0 {
		syscall.Munmap(raw)
		return nil, fmt.Errorf("flintlock: mmap returned unaligned base %#x", a.base)
	}

	// Install the guards. Revoke ALL access: not read-only, not no-execute --
	// PROT_NONE. A read-only guard would silently absorb overreads, which are
	// precisely how heartbleed-class disclosure bugs exfiltrate adjacent slots.
	for i := 0; i <= slots; i++ {
		off := uintptr(i) * stride
		guard := raw[off : off+pageSize : off+pageSize]
		if err := syscall.Mprotect(guard, syscall.PROT_NONE); err != nil {
			syscall.Munmap(raw)
			return nil, fmt.Errorf("flintlock: mprotect guard %d PROT_NONE: %w", i, err)
		}
	}
	return a, nil
}

// slotRegion returns the writable data span of slot i as a full-slice-expression
// bounded slice. The cap is pinned to the span so that append() on a derived
// slice can never grow across the upper guard page.
func (a *arena) slotRegion(i int) []byte {
	off := uintptr(i)*a.stride + pageSize
	return a.raw[off : off+a.span : off+a.span]
}

// slotBase returns the first writable address of slot i.
func (a *arena) slotBase(i int) uintptr { return a.base + uintptr(i)*a.stride + pageSize }

// protect changes the page permissions of slot i's data span.
func (a *arena) protect(i, prot int) error {
	if err := syscall.Mprotect(a.slotRegion(i), prot); err != nil {
		return fmt.Errorf("flintlock: mprotect slot %d prot=%#x: %w", i, prot, err)
	}
	return nil
}

// unmap releases the entire reservation, guards included.
func (a *arena) unmap() error {
	if a.raw == nil {
		return nil
	}
	raw := a.raw
	a.raw = nil
	// Restore RW across the whole extent first. munmap does not require it,
	// but leaving PROT_NONE holes in the address space during teardown makes
	// any late, buggy dangling access land in freshly recycled VA instead of
	// faulting -- so we normalize before releasing.
	_ = syscall.Mprotect(raw, syscall.PROT_READ|syscall.PROT_WRITE)
	if err := syscall.Munmap(raw); err != nil {
		return fmt.Errorf("flintlock: munmap: %w", err)
	}
	return nil
}

// region classifies an arbitrary faulting virtual address against the arena
// geometry. This is the inverse of the layout above and is the whole reason the
// arena is one uniform-stride mapping.
//
// Returns the owning slot index, the signed byte offset of addr relative to
// that slot's data base, and what kind of boundary was struck.
func (a *arena) region(addr uintptr) (slot int, off int64, kind FaultKind) {
	if addr < a.base || addr >= a.limit {
		return -1, 0, FaultForeign
	}
	rel := addr - a.base
	idx := int(rel / a.stride)
	pos := rel % a.stride

	if pos < pageSize {
		// Struck a guard page. It is simultaneously the upper guard of slot
		// idx-1 and the lower guard of slot idx. Attribute it to whichever is
		// a real slot; when both are, the trailing-guard case (idx == slots)
		// has already been excluded, so it is genuinely ambiguous and the
		// caller's armed-slot identity disambiguates. Report the underflow
		// side here and let Sandbox.classify() override with the executing
		// slot when it owns one of the two.
		if idx >= a.slots {
			// Final guard of the arena: can only be an overflow off the last slot.
			return a.slots - 1, int64(pos) + int64(a.span), FaultOverflow
		}
		if idx == 0 {
			return 0, int64(pos) - int64(pageSize), FaultUnderflow
		}
		return idx, int64(pos) - int64(pageSize), FaultUnderflow
	}

	// Inside a data span. Reachable only when that span is currently PROT_NONE,
	// i.e. the slot is quarantined or retired -- which is the signature of a
	// use-after-free through a stale pointer.
	return idx, int64(pos) - int64(pageSize), FaultUseAfterFree
}
