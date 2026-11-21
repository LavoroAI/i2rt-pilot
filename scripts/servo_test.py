#!/usr/bin/env python3
"""Test a Feetech STS3215 servo and record its 0 / max positions.

Talks the Feetech STS protocol directly over pyserial (no SDK needed).

    python3 servo_test.py scan               # find the servo (ID + baud)
    python3 servo_test.py monitor            # live position / voltage / temp / load
    python3 servo_test.py record             # torque off, move by hand, record 0 and max
    python3 servo_test.py move               # torque on, sweep 0 -> max -> middle (from limits file)
    python3 servo_test.py move --to 2048     # go to one position
    python3 servo_test.py center             # shift the servo's offset so the recorded range sits mid-turn
    python3 servo_test.py off                # torque off (servo goes limp)

Defaults: port /dev/ttyACM0, baud 1000000, ID 1. Positions are raw encoder
steps, 0..4095 for one turn (4096 steps = 360 deg).
"""

import argparse
import json
import select
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import serial

DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0183172-if00"
LIMITS_FILE = Path.home() / "i2rt/i2rt/flow_base/head_servo_limits.json"  # read by head_tilt_controller.py
BAUDS = [1_000_000, 500_000, 250_000, 128_000, 115_200, 76_800, 57_600, 38_400]

# Instructions
PING, READ, WRITE = 0x01, 0x02, 0x03

# STS3215 control table
MIN_ANGLE_LIMIT = 9  # 2 bytes
MAX_ANGLE_LIMIT = 11  # 2 bytes
POSITION_OFFSET = 31  # 2 bytes, sign bit 11 (EEPROM): present = actual - offset
TORQUE_ENABLE = 40  # 1 byte
ACCELERATION = 41  # 1 byte, then goal position / time / speed follow
GOAL_POSITION = 42  # 2 bytes
PRESENT_POSITION = 56  # 2 bytes, then speed (58), load (60), voltage (62), temp (63)
LOCK = 55  # 1 byte, 0 = EEPROM writable
MOVING = 66  # 1 byte

STEPS_PER_TURN = 4096


def signed(value: int, sign_bit: int) -> int:
    """Feetech uses sign-magnitude: one bit is the sign, the rest the magnitude."""
    mag = value & ((1 << sign_bit) - 1)
    return -mag if value & (1 << sign_bit) else mag


class Bus:
    def __init__(self, port: str, baud: int, timeout: float = 0.05) -> None:
        self.ser = serial.Serial(port, baud, timeout=timeout)

    def close(self) -> None:
        self.ser.close()

    def _send(self, sid: int, instr: int, params: Sequence[int] = ()) -> None:
        body = [sid, len(params) + 2, instr, *params]
        self.ser.reset_input_buffer()
        self.ser.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))

    def _receive(self, sid: int) -> bytes | None:
        """Return the parameter bytes of the status packet from `sid`, or None."""
        deadline = time.monotonic() + self.ser.timeout * 2
        buf = b""
        while time.monotonic() < deadline:
            buf += self.ser.read(max(1, self.ser.in_waiting))
            i = buf.find(b"\xff\xff")
            while i != -1 and len(buf) >= i + 4:
                rid, length = buf[i + 2], buf[i + 3]
                if len(buf) < i + 4 + length:
                    break
                pkt = buf[i + 2 : i + 4 + length]  # id, len, error, params..., checksum
                if rid == sid and (~sum(pkt[:-1])) & 0xFF == pkt[-1]:
                    error = pkt[2]
                    if error:
                        print(f"  [servo {sid} status error 0x{error:02x}]", file=sys.stderr)
                    return bytes(pkt[3:-1])
                i = buf.find(b"\xff\xff", i + 1)
        return None

    def ping(self, sid: int) -> bool:
        self._send(sid, PING)
        return self._receive(sid) is not None

    def read(self, sid: int, addr: int, n: int) -> bytes:
        self._send(sid, READ, [addr, n])
        data = self._receive(sid)
        if data is None or len(data) != n:
            raise IOError(f"no reply from servo {sid} reading address {addr}")
        return data

    def write(self, sid: int, addr: int, data: Sequence[int]) -> None:
        self._send(sid, WRITE, [addr, *data])
        if self._receive(sid) is None:
            raise IOError(f"no reply from servo {sid} writing address {addr}")

    # --- helpers -------------------------------------------------------------

    def state(self, sid: int) -> dict:
        d = self.read(sid, PRESENT_POSITION, 8)
        return {
            "position": signed(d[0] | d[1] << 8, 15) % STEPS_PER_TURN,
            "speed": signed(d[2] | d[3] << 8, 15),
            "load": signed(d[4] | d[5] << 8, 10) / 10.0,  # % of max torque
            "voltage": d[6] / 10.0,
            "temp": d[7],
        }

    def position(self, sid: int) -> int:
        # With torque off the servo keeps counting past 4095 / below 0; fold into one turn
        d = self.read(sid, PRESENT_POSITION, 2)
        return signed(d[0] | d[1] << 8, 15) % STEPS_PER_TURN

    def torque(self, sid: int, on: bool) -> None:
        self.write(sid, TORQUE_ENABLE, [1 if on else 0])

    def goto(self, sid: int, pos: float, speed: int, acc: int) -> None:
        pos = max(0, min(STEPS_PER_TURN - 1, int(pos)))
        # acceleration, goal position (2), goal time (2), goal speed (2)
        self.write(sid, ACCELERATION, [acc, pos & 0xFF, pos >> 8, 0, 0, speed & 0xFF, speed >> 8])


def controller_running() -> int | None:
    """PID of a running head_tilt_controller.py (it owns the servo), or None."""
    for proc in Path("/proc").iterdir():
        try:
            if proc.name.isdigit() and b"head_tilt_controller" in (proc / "cmdline").read_bytes():
                return int(proc.name)
        except OSError:
            pass
    return None


def open_bus(args: argparse.Namespace) -> Bus:
    pid = controller_running()
    if pid:
        sys.exit(f"head_tilt_controller.py is running (PID {pid}) and owns the servo; stop it first.")
    try:
        return Bus(args.port, args.baud)
    except serial.SerialException as e:
        msg = str(e)
        if "Permission denied" in msg:
            msg += (
                "\n\nYour user cannot open the serial port. Fix once with:\n"
                "    sudo usermod -aG dialout $USER\n"
                "then log out and back in (or run `newgrp dialout` in this shell)."
            )
        sys.exit(msg)


def key_pressed() -> bool:
    """True if Enter was pressed (non-blocking)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


# --- commands ----------------------------------------------------------------


def cmd_scan(args: argparse.Namespace) -> None:
    if controller_running():
        open_bus(args)  # prints which process owns the servo and exits
    bauds = [args.baud] if args.baud_given else BAUDS
    found = []
    for baud in bauds:
        try:
            bus = Bus(args.port, baud, timeout=0.01)
        except serial.SerialException as e:
            args.baud = baud
            open_bus(args)  # prints the permission hint and exits
            sys.exit(str(e))
        print(f"baud {baud}: ", end="", flush=True)
        ids = [sid for sid in range(0, 254) if bus.ping(sid)]
        bus.close()
        print(ids or "-")
        found += [(baud, sid) for sid in ids]
    if not found:
        print("\nNo servo answered. Check the servo power supply (7.4 V) and the cable.")
    else:
        baud, sid = found[0]
        print(f"\nUse: --baud {baud} --id {sid}")


def cmd_monitor(args: argparse.Namespace) -> None:
    bus = open_bus(args)
    print("Ctrl+C to stop.")
    try:
        while True:
            s = bus.state(args.id)
            deg = s["position"] * 360.0 / STEPS_PER_TURN
            print(
                f"\rpos {s['position']:5d} ({deg:6.1f} deg)  speed {s['speed']:5d}  "
                f"load {s['load']:5.1f}%  {s['voltage']:4.1f} V  {s['temp']:3d} C   ",
                end="",
                flush=True,
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    finally:
        bus.close()


def capture(bus: Bus, sid: int, label: str) -> tuple[int, bool]:
    """Show the live position until Enter; return it and whether it wrapped past 0/4095."""
    print(f"\nMove the servo by hand to the {label} position, then press Enter.")
    last = bus.position(sid)
    wrapped = False
    while True:
        pos = bus.position(sid)
        if abs(pos - last) > STEPS_PER_TURN // 2:
            wrapped = True
        last = pos
        print(f"\r  position {pos:5d}  ({pos * 360.0 / STEPS_PER_TURN:6.1f} deg)   ", end="", flush=True)
        if key_pressed():
            print(f"\n  -> {label} = {pos}")
            return pos, wrapped
        time.sleep(0.03)


def cmd_record(args: argparse.Namespace) -> None:
    bus = open_bus(args)
    try:
        bus.torque(args.id, False)
        print("Torque OFF: the servo is free to turn by hand.")
        zero, wrapped0 = capture(bus, args.id, "0 (start)")
        maximum, wrapped = capture(bus, args.id, "max (end)")
        wrapped = wrapped or wrapped0
    finally:
        bus.close()

    span = maximum - zero
    print(f"\n0 = {zero}, max = {maximum}, span = {span} steps ({span * 360.0 / STEPS_PER_TURN:.1f} deg)")
    if wrapped:
        print(
            "WARNING: the position passed through the 0/4095 wrap point between the two captures.\n"
            "The raw numbers do not describe that range; `move` would go the other way round.\n"
            "Run `center` to shift the servo's offset so the range sits in the middle of the turn."
        )
    if zero > maximum:
        print("Note: 0 is above max, so this joint counts down from 0 to max.")

    limits = {
        "id": args.id,
        "zero": zero,
        "max": maximum,
        "wrapped": wrapped,
        "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(args.limits).write_text(json.dumps(limits, indent=2) + "\n")
    print(f"Saved to {args.limits}")


def wait_until(bus: Bus, sid: int, target: int, timeout: float = 10.0, tol: int = 15) -> bool:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = bus.state(sid)
        print(f"\r  pos {s['position']:5d} -> {target:5d}  load {s['load']:5.1f}%   ", end="", flush=True)
        if abs(s["position"] - target) <= tol and not bus.read(sid, MOVING, 1)[0]:
            print("  ok")
            return True
        time.sleep(0.03)
    print("  TIMEOUT (blocked, or target outside the angle limits?)")
    return False


def cmd_move(args: argparse.Namespace) -> None:
    if args.to is not None:
        targets = [("target", args.to)]
    else:
        try:
            limits = json.loads(Path(args.limits).read_text())
        except FileNotFoundError:
            sys.exit(f"{args.limits} not found; run `record` first or pass --to <position>.")
        if limits.get("wrapped") or not all(0 <= limits[k] < STEPS_PER_TURN for k in ("zero", "max")):
            sys.exit("The recorded range crosses the 0/4095 wrap point; run `center` (or re-record) first.")
        lo, hi = limits["zero"], limits["max"]
        targets = [("0", lo), ("max", hi), ("middle", (lo + hi) // 2)]

    bus = open_bus(args)
    try:
        d = bus.read(args.id, MIN_ANGLE_LIMIT, 4)
        lim_lo, lim_hi = d[0] | d[1] << 8, d[2] | d[3] << 8
        print(f"Servo angle limits: {lim_lo}..{lim_hi}")
        start = bus.position(args.id)
        bus.goto(args.id, start, args.speed, args.acc)  # hold where it is before enabling torque
        bus.torque(args.id, True)
        for name, pos in targets:
            print(f"Moving to {name} ({pos}) at speed {args.speed}")
            bus.goto(args.id, pos, args.speed, args.acc)
            wait_until(bus, args.id, pos)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if not args.hold:
            bus.torque(args.id, False)
            print("Torque OFF.")
        bus.close()


def cmd_center(args: argparse.Namespace) -> None:
    """Set the servo's position offset so the recorded range is centred on 2048, and update the limits file."""
    limits = json.loads(Path(args.limits).read_text())
    zero, maximum = limits["zero"], limits["max"]
    if abs(maximum - zero) > STEPS_PER_TURN // 2:  # recorded across a jump: unwrap max next to zero
        maximum -= STEPS_PER_TURN if maximum > zero else -STEPS_PER_TURN
    if abs(maximum - zero) >= STEPS_PER_TURN - 200:
        sys.exit("The recorded range is almost a full turn; it cannot be placed inside 0..4095.")

    bus = open_bus(args)
    try:
        d = bus.read(args.id, POSITION_OFFSET, 2)
        old = signed(d[0] | d[1] << 8, 11)
        delta = round((zero + maximum) / 2) - STEPS_PER_TURN // 2
        new = (old + delta + STEPS_PER_TURN // 2) % STEPS_PER_TURN - STEPS_PER_TURN // 2  # -2048..2047
        new = max(-2047, new)
        delta = new - old
        before = bus.position(args.id)
        print(f"Offset {old} -> {new} (positions shift by {-delta})")

        bus.torque(args.id, False)
        bus.write(args.id, LOCK, [0])
        mag = abs(new)
        raw = mag | (0x800 if new < 0 else 0)
        bus.write(args.id, POSITION_OFFSET, [raw & 0xFF, raw >> 8])
        bus.write(args.id, LOCK, [1])
        time.sleep(0.1)
        after = bus.position(args.id)
    finally:
        bus.close()

    new_zero, new_max = (zero - delta) % STEPS_PER_TURN, (maximum - delta) % STEPS_PER_TURN
    expected = (before - delta) % STEPS_PER_TURN
    print(f"Present position {before} -> {after} (expected {expected})")
    if abs(after - expected) > 5:
        print("WARNING: the position did not shift as expected; check before using the limits.")
    limits.update(zero=new_zero, max=new_max, wrapped=False, offset=new, centered=time.strftime("%Y-%m-%d %H:%M:%S"))
    Path(args.limits).write_text(json.dumps(limits, indent=2) + "\n")
    print(f"0 = {new_zero}, max = {new_max}; saved to {args.limits}")


def cmd_off(args: argparse.Namespace) -> None:
    bus = open_bus(args)
    bus.torque(args.id, False)
    bus.close()
    print("Torque OFF.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["scan", "monitor", "record", "move", "center", "off"])
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--baud", type=int, default=None, help="default 1000000 (scan tries all)")
    p.add_argument("--id", type=int, default=1)
    p.add_argument("--limits", default=str(LIMITS_FILE), help="where record saves / move reads")
    p.add_argument("--to", type=int, help="move: go to this one position instead of the sweep")
    p.add_argument("--speed", type=int, default=400, help="move: steps/s (max ~3400)")
    p.add_argument("--acc", type=int, default=20, help="move: acceleration (0..254, 0 = max)")
    p.add_argument("--hold", action="store_true", help="move: keep torque on afterwards")
    args = p.parse_args()
    args.baud_given = args.baud is not None
    args.baud = args.baud or 1_000_000

    {
        "scan": cmd_scan,
        "monitor": cmd_monitor,
        "record": cmd_record,
        "move": cmd_move,
        "center": cmd_center,
        "off": cmd_off,
    }[args.command](args)


if __name__ == "__main__":
    main()
