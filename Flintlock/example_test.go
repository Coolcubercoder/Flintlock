package flintlock_test

import (
	"encoding/binary"
	"fmt"
	"unsafe"

	"github.com/flintlock/flintlock"
)

// parseFrame is a deliberately unsafe, bare-metal parser of the kind Flintlock
// exists to contain: it trusts a length field from the wire.
//
//	[0:2] payload length, big endian
//	[2:]  payload bytes
func parseFrame(p unsafe.Pointer, declared int) uint64 {
	var sum uint64
	for i := 0; i < declared; i++ {
		sum += uint64(*(*byte)(unsafe.Add(p, 2+i)))
	}
	return sum
}

// Example_maliciousLengthField shows a classic attacker-controlled length
// overflow being trapped by hardware and returned as an ordinary Go error,
// while the process keeps serving.
func Example_maliciousLengthField() {
	sb, err := flintlock.New(flintlock.Config{SlotBytes: 4096, Slots: 8})
	if err != nil {
		panic(err)
	}
	defer sb.Close()

	frame := make([]byte, 2+8)
	copy(frame[2:], []byte{1, 2, 3, 4, 5, 6, 7, 8})

	// Honest frame: declares 8 bytes, carries 8 bytes.
	binary.BigEndian.PutUint16(frame, 8)
	var sum uint64
	if err := sb.Execute(frame, func(p unsafe.Pointer) {
		sum = parseFrame(p, int(binary.BigEndian.Uint16(frame)))
	}); err != nil {
		fmt.Println("unexpected:", err)
	} else {
		fmt.Println("honest frame parsed, checksum =", sum)
	}

	// Hostile frame: declares 60000 bytes, still carries 8.
	binary.BigEndian.PutUint16(frame, 60000)
	err = sb.Execute(frame, func(p unsafe.Pointer) {
		sum = parseFrame(p, int(binary.BigEndian.Uint16(frame)))
	})

	var gv *flintlock.GuardViolation
	if gv, _ = err.(*flintlock.GuardViolation); gv != nil {
		fmt.Printf("hostile frame trapped: %s at %+d bytes past payload end\n",
			gv.Fault.Kind, gv.Fault.Overrun())
	}

	// The process is unharmed and still serving at full speed.
	binary.BigEndian.PutUint16(frame, 8)
	if err := sb.Execute(frame, func(p unsafe.Pointer) {
		sum = parseFrame(p, 8)
	}); err == nil {
		fmt.Println("still serving after the trap, checksum =", sum)
	}

	st := sb.Stats()
	fmt.Printf("executed=%d trapped=%d guard_pages=%d\n", st.Executed, st.Trapped, st.GuardPages)

	// Output:
	// honest frame parsed, checksum = 36
	// hostile frame trapped: GUARD_OVERFLOW at +0 bytes past payload end
	// still serving after the trap, checksum = 36
	// executed=2 trapped=1 guard_pages=9
}
