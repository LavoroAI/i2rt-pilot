"""FlowBase + linear rail controller that also tilts the head servo with the rail height.

Same as flow_base_controller.py (gamepad driving, portal server on port 11323), plus a
Feetech STS3215 head servo on a USB serial adapter. The servo angle follows the rail height,
measured as motor radians above the lower limit switch (rail_bottom, taken at homing).

Taught curve (used when head_tilt_points.json exists): run with --teach, drive the rail with
the gamepad, set the head by hand (servo is limp) and press Enter in the terminal at each
height. The points (height_rad, servo) are saved after every press; normal runs follow a
smooth monotone curve through them (PCHIP, monotone_cubic), holding the end values beyond the first / last point.

    --teach keys: Enter = record point, u + Enter = undo last, p + Enter = print points

Without taught points it falls back to a power curve between the recorded servo limits:

    fraction = (rail_pos - rail_bottom) / rail_travel            # 0 at bottom, 1 at top
    servo    = servo_zero + fraction ** curve * (servo_max - servo_zero)

servo_zero / servo_max come from `servo_test.py record`. rail_travel is learned the first time
the upper limit switch is triggered and saved to the rail travel file.
"""

import atexit
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, NoReturn, Optional, Tuple

import numpy as np
import portal
import serial

from i2rt.flow_base.flow_base_controller import BASE_DEFAULT_PORT, LinearRailVehicle

logger = logging.getLogger("HeadTiltController")

HERE = Path(__file__).resolve().parent
DEFAULT_SERVO_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB0183172-if00"
DEFAULT_SERVO_LIMITS = HERE / "head_servo_limits.json"
DEFAULT_RAIL_TRAVEL_FILE = HERE / "rail_travel.json"
DEFAULT_POINTS_FILE = HERE / "head_tilt_points.json"
HEAD_CONTROL_FREQ = 50


class FeetechServo:
    """Minimal Feetech STS protocol driver for one servo (STS3215)."""

    TORQUE_ENABLE = 40
    ACCELERATION = 41  # followed by goal position (42), goal time (44), goal speed (46)
    PRESENT_POSITION = 56

    def __init__(self, port: str, servo_id: int = 1, baud: int = 1_000_000) -> None:
        self.servo_id = servo_id
        self.ser = serial.Serial(port, baud, timeout=0.02)

    def close(self) -> None:
        self.ser.close()

    def _transact(self, instr: int, params: Tuple[int, ...] = ()) -> bytes:
        body = [self.servo_id, len(params) + 2, instr, *params]
        self.ser.reset_input_buffer()
        self.ser.write(bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF]))
        deadline = time.monotonic() + 0.05
        buf = b""
        while time.monotonic() < deadline:
            buf += self.ser.read(max(1, self.ser.in_waiting))
            i = buf.find(b"\xff\xff")
            if i != -1 and len(buf) >= i + 4 and len(buf) >= i + 4 + buf[i + 3]:
                pkt = buf[i + 2 : i + 4 + buf[i + 3]]  # id, len, error, params..., checksum
                if pkt[0] == self.servo_id and (~sum(pkt[:-1])) & 0xFF == pkt[-1]:
                    return bytes(pkt[3:-1])
                buf = buf[i + 2 :]
        raise IOError(f"no reply from head servo {self.servo_id}")

    def read_position(self) -> int:
        d = self._transact(0x02, (self.PRESENT_POSITION, 2))
        value = d[0] | d[1] << 8
        value = -(value & 0x7FFF) if value & 0x8000 else value
        return value % 4096  # with torque off the servo keeps counting past 4095; fold into one turn

    def torque_enabled(self) -> bool:
        return bool(self._transact(0x02, (self.TORQUE_ENABLE, 1))[0])

    def set_torque(self, on: bool) -> None:
        self._transact(0x03, (self.TORQUE_ENABLE, 1 if on else 0))

    def set_goal(self, position: int, speed: int, acc: int) -> None:
        position = int(np.clip(position, 0, 4095))
        self._transact(0x03, (self.ACCELERATION, acc, position & 0xFF, position >> 8, 0, 0, speed & 0xFF, speed >> 8))


def monotone_cubic(xs: np.ndarray, ys: np.ndarray) -> Callable[[float], float]:
    """Smooth curve through the points without overshoot (PCHIP, Fritsch-Carlson slopes).

    Pure numpy on purpose: importing scipy here stalls the process for seconds on the Pi, long enough
    for the DM motors to time out and the vehicle to go compliant.
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    h = np.diff(xs)
    d = np.diff(ys) / h
    m = np.empty_like(ys)
    m[0], m[-1] = d[0], d[-1]
    for k in range(1, len(xs) - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0  # local extremum: flat, so the curve does not overshoot
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])

    def f(x: float) -> float:
        k = int(np.clip(np.searchsorted(xs, x, side="right") - 1, 0, len(xs) - 2))
        t = (x - xs[k]) / h[k]
        h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
        h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
        return h00 * ys[k] + h10 * h[k] * m[k] + h01 * ys[k + 1] + h11 * h[k] * m[k + 1]

    return f


class HeadTiltController:
    """Thread that sets the head servo from the linear rail height."""

    def __init__(
        self,
        vehicle: LinearRailVehicle,
        servo_port: str,
        servo_limits_file: Path,
        rail_travel_file: Path,
        rail_travel: Optional[float] = None,
        curve: float = 2.0,
        speed: int = 1500,
        acc: int = 50,
        deadband: int = 3,
        points_file: Path = DEFAULT_POINTS_FILE,
        teach: bool = False,
    ) -> None:
        self.vehicle = vehicle
        self.servo_port = servo_port
        self.rail_travel_file = Path(rail_travel_file)
        self.curve = curve
        self.speed = speed
        self.acc = acc
        self.deadband = deadband

        limits = self.load_limits(servo_limits_file)
        self.servo_id = int(limits.get("id", 1))
        self.servo_zero = int(limits["zero"])
        self.servo_max = int(limits["max"])

        # Rail travel: command line > saved file > learned at the upper limit switch
        if rail_travel is None and self.rail_travel_file.exists():
            rail_travel = json.loads(self.rail_travel_file.read_text())["rail_travel"]
        self.rail_travel = rail_travel
        self.rail_bottom: Optional[float] = None

        # Taught curve: [(height above rail bottom in rad, servo position), ...]
        self.points_file = Path(points_file)
        self.teach = teach
        self.points: list = []
        if self.points_file.exists():
            self.points = [tuple(pt) for pt in json.loads(self.points_file.read_text())["points"]]
        self._curve_fn = None
        self._backed_up = False
        self._recorded: list = []  # teach mode: points in the order they were recorded (for undo)
        if teach:
            self.points = []  # start a fresh curve; the old file is backed up on the first new point
        self._rebuild_curve()

        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {}
        self.servo: Optional[FeetechServo] = None
        self.running = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

        logger.info(
            f"Head servo id {self.servo_id}: zero {self.servo_zero}, max {self.servo_max}; "
            f"rail travel {self.rail_travel if self.rail_travel is not None else 'unknown (drive to the top once)'}; "
            + (
                f"teach mode (servo limp, Enter records a point to {self.points_file})"
                if teach
                else f"taught curve, {len(self.points)} points"
                if self.points
                else f"power curve {self.curve}"
            )
        )

    @staticmethod
    def load_limits(servo_limits_file: Path) -> Dict[str, Any]:
        limits = json.loads(Path(servo_limits_file).read_text())
        if limits.get("wrapped") or not all(0 <= limits[k] <= 4095 for k in ("zero", "max")):
            raise ValueError(
                f"{servo_limits_file}: recorded servo range crosses 0/4095; run `servo_test.py center` first"
            )
        return limits

    def start(self) -> None:
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.servo is not None:
            self.servo.close()  # torque stays on: the head holds its last position
            self.servo = None

    def _rebuild_curve(self) -> None:
        self.points.sort()
        self._curve_fn = None
        if len(self.points) >= 3:
            xs, ys = zip(*self.points, strict=True)
            self._curve_fn = monotone_cubic(xs, ys)

    def target_for(self, rail_pos: float) -> Tuple[float, int]:
        """Rail position -> (height fraction 0..1, servo goal position)."""
        height = 0.0 if self.rail_bottom is None else max(0.0, rail_pos - self.rail_bottom)
        fraction = float(np.clip(height / self.rail_travel, 0.0, 1.0)) if self.rail_travel else 0.0
        if self.points:
            xs, ys = zip(*self.points, strict=True)
            h = float(np.clip(height, xs[0], xs[-1]))
            goal = self._curve_fn(h) if self._curve_fn is not None else np.interp(h, xs, ys)
            return fraction, round(float(goal))
        if self.rail_bottom is None or not self.rail_travel:
            fraction = 0.0
        tilt = fraction**self.curve
        return fraction, round(self.servo_zero + tilt * (self.servo_max - self.servo_zero))

    def record_point(self) -> str:
        """Teach mode: store (current rail height, current servo position) and save the curve."""
        with self._lock:
            state = dict(self._state)
        if self.rail_bottom is None or state.get("rail_position") is None:
            return "Not recorded: rail bottom not known yet (rail must home to the lower limit first)"
        if state.get("servo_position") is None:
            return "Not recorded: no reading from the head servo"
        height = round(state["rail_position"] - self.rail_bottom, 3)
        servo = int(state["servo_position"])
        self.points = [pt for pt in self.points if abs(pt[0] - height) > 0.5]  # replace a point at ~same height
        self.points.append((height, servo))
        self._recorded.append((height, servo))
        self._rebuild_curve()
        self._save_points()
        return f"Recorded height {height:.2f} rad -> servo {servo}. " + self.describe_points()

    def undo_point(self) -> str:
        """Teach mode: remove the most recently recorded point."""
        while self._recorded and self._recorded[-1] not in self.points:
            self._recorded.pop()  # already replaced by a later point at the same height
        if not self._recorded:
            return "No points to undo"
        removed = self._recorded.pop()
        self.points.remove(removed)
        self._rebuild_curve()
        self._save_points()
        return f"Removed ({removed[0]:.1f} rad, {removed[1]}). " + self.describe_points()

    def describe_points(self) -> str:
        return f"{len(self.points)} points: " + ", ".join(f"({h:.1f} rad, {s})" for h, s in self.points)

    def _save_points(self) -> None:
        if not self._backed_up and self.points_file.exists():
            backup = self.points_file.with_suffix(f".{time.strftime('%Y%m%d-%H%M%S')}.bak.json")
            backup.write_text(self.points_file.read_text())
            logger.info(f"Previous taught curve backed up to {backup}")
        self._backed_up = True
        self.points_file.write_text(
            json.dumps(
                {
                    "points": [list(pt) for pt in self.points],
                    "rail_travel": self.rail_travel,
                    "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
            + "\n"
        )

    def _connect(self) -> None:
        self.servo = FeetechServo(self.servo_port, self.servo_id)
        current = self.servo.read_position()
        self.last_goal = current
        if self.teach:
            self.servo.set_torque(False)  # limp: move the head by hand
        else:
            self.servo.set_goal(current, self.speed, self.acc)  # hold where it is before enabling torque
            self.servo.set_torque(True)
        logger.info(f"Head servo connected on {self.servo_port}, at {current}")

    def _update_rail_reference(self, rail: Dict[str, Any]) -> None:
        pos = rail["position"]["motor"]
        vel = rail["velocity"]["motor"]
        # Session-local home reference: latch once after startup homing settles.
        # A limit can remain asserted over some release travel; continuously
        # sampling it would make the zero follow the moving carriage.
        # Re-home/restart deliberately to establish a new reference. Never infer
        # a raw encoder zero from the saved relative-height tilt calibration.
        if (
            self.rail_bottom is None
            and rail.get("initialized", False)
            and rail.get("lower_limit_triggered", False)
            and not rail.get("upper_limit_triggered", False)
            and np.isfinite(pos)
            and np.isfinite(vel)
            and abs(vel) <= 0.2
        ):
            self.rail_bottom = float(pos)
            logger.info(f"Head tilt: rail bottom latched for this session at {pos:.3f} rad")
        if rail.get("upper_limit_triggered") and self.rail_bottom is not None:
            travel = pos - self.rail_bottom
            if self.rail_travel is None or abs(travel - self.rail_travel) > 0.05:
                self.rail_travel = travel
                self.rail_travel_file.write_text(json.dumps({"rail_travel": travel}, indent=2) + "\n")
                logger.info(f"Head tilt: rail travel learned = {travel:.3f} rad (saved to {self.rail_travel_file})")

    def _loop(self) -> None:
        period = 1.0 / HEAD_CONTROL_FREQ
        last_error_log = 0.0
        last_torque_check = 0.0
        while self.running:
            t0 = time.time()
            try:
                if self.servo is None:
                    self._connect()
                # Something else (servo_test.py, a power dip) may switch torque off; switch it back on
                if not self.teach and time.time() - last_torque_check > 1.0:
                    last_torque_check = time.time()
                    if not self.servo.torque_enabled():
                        logger.warning("Head servo torque was off; re-enabling")
                        self.servo.set_goal(self.servo.read_position(), self.speed, self.acc)
                        self.last_goal = -10000  # resend the goal below
                        self.servo.set_torque(True)
                rail = self.vehicle.get_linear_rail_state()
                if "position" in rail:
                    self._update_rail_reference(rail)
                    rail_pos = rail["position"]["motor"]
                    fraction, goal = self.target_for(rail_pos)
                    servo_position = None
                    if self.teach:
                        servo_position = self.servo.read_position()
                    elif abs(goal - self.last_goal) >= self.deadband:
                        self.servo.set_goal(goal, self.speed, self.acc)
                        self.last_goal = goal
                    with self._lock:
                        self._state = {
                            "rail_position": rail_pos,
                            "rail_bottom": self.rail_bottom,
                            "rail_travel": self.rail_travel,
                            "height_fraction": fraction,
                            "servo_goal": self.last_goal,
                            "servo_position": servo_position,
                        }
            except Exception as e:  # keep driving the base even if the head servo is unplugged
                if self.servo is not None:
                    self.servo.close()
                    self.servo = None
                if time.time() - last_error_log > 5.0:
                    logger.error(f"Head servo error (retrying): {e}")
                    last_error_log = time.time()
                time.sleep(1.0)
            time.sleep(max(0.0, period - (time.time() - t0)))

    def get_state(self, input_dict: Dict[str, Any] | None = None) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state, connected=self.servo is not None)


if __name__ == "__main__":
    import argparse
    import select
    import sys

    import pygame

    from i2rt.utils.gamepad_utils import Gamepad

    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, default="can0")
    parser.add_argument("--servo-port", default=DEFAULT_SERVO_PORT)
    parser.add_argument("--servo-limits", default=str(DEFAULT_SERVO_LIMITS), help="from servo_test.py record")
    parser.add_argument("--rail-travel-file", default=str(DEFAULT_RAIL_TRAVEL_FILE))
    parser.add_argument("--rail-travel", type=float, default=None, help="rail travel bottom->top in rad")
    parser.add_argument("--curve", type=float, default=2.0, help="tilt = height_fraction ** curve (2 = quadratic)")
    parser.add_argument("--head-speed", type=int, default=1500, help="servo speed limit, steps/s")
    parser.add_argument("--points-file", default=str(DEFAULT_POINTS_FILE), help="taught curve")
    parser.add_argument("--teach", action="store_true", help="servo limp; Enter records (rail height, head) points")

    # Initialize pygame and joystick
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick/gamepad connected!")
        exit()

    joy = pygame.joystick.Joystick(0)
    CALIBRATION_RETRY_DELAY = 1
    DEADZONE = 0.05  # Deadzone for base control (x, y, theta)
    args = parser.parse_args()

    gamepad = Gamepad()
    while not gamepad.centered(DEADZONE):
        logger.warning("Center both sticks before starting Linearbot")
        time.sleep(CALIBRATION_RETRY_DELAY)

    max_vel = np.array([0.8, 0.8, 3.0])
    max_accel = np.array([0.8, 0.8, 3.0])
    lift_max_vel = 14.0  # Maximum velocity for linear rail (rad/s)

    HeadTiltController.load_limits(args.servo_limits)  # fail before homing the rail if the limits are bad

    vehicle = LinearRailVehicle(
        vehicle_max_vel=max_vel,
        vehicle_max_accel=max_accel,
        lift_max_vel=lift_max_vel,
        channel=args.channel,
        auto_home=True,
        enable_linear_rail=True,
    )

    head = HeadTiltController(
        vehicle,
        servo_port=args.servo_port,
        servo_limits_file=args.servo_limits,
        rail_travel_file=args.rail_travel_file,
        rail_travel=args.rail_travel,
        curve=args.curve,
        speed=args.head_speed,
        points_file=args.points_file,
        teach=args.teach,
    )
    head.start()
    if args.teach:
        print(
            "\nTEACH MODE: drive the rail with the gamepad, set the head by hand, then press Enter here.\n"
            "Start at the bottom, end at the top, 5-8 heights (more where the tilt changes fast).\n"
            "Enter = record, u + Enter = undo last, p + Enter = print points, Ctrl+C = quit (points are saved).\n"
        )

    # Register cleanup function to ensure brake is engaged on exit
    def close_vehicle() -> None:
        try:
            head.stop()
            vehicle.close()
        except Exception as e:
            logger.error(f"Error during atexit close: {e}")

    atexit.register(close_vehicle)

    # Closing the terminal (SIGHUP) or kill (SIGTERM) runs the same cleanup as Ctrl-C
    import signal

    def _raise_interrupt(signum: int, frame: object) -> NoReturn:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_interrupt)
    signal.signal(signal.SIGHUP, _raise_interrupt)

    class TimeoutRemoteCommand:
        """Unified remote command handler for LinearRailVehicle (base + linear rail)"""

        def __init__(self, timeout: float = 0.2) -> None:
            self.timeout = timeout
            self.last_update_time = time.time() - 1000000
            self.command = np.zeros(4)  # Support 4D: [x, y, theta, linear_rail]
            self.frame = "local"
            self._lock = threading.Lock()

        def is_command_valid(self) -> bool:
            return time.time() - self.last_update_time < self.timeout

        def remote_set_target_velocity(self, input_dict: Dict[str, Any]) -> None:
            """Set target velocity for base (and optionally linear rail)"""
            target_velocity = input_dict["target_velocity"]
            frame = input_dict["frame"]
            with self._lock:
                if len(target_velocity) == 3:
                    self.command[:3] = target_velocity
                else:
                    self.command = target_velocity
                self.frame = frame
                self.last_update_time = time.time()

        def get_command(self) -> Tuple[np.ndarray, str]:
            """Get base command [x, y, theta, linear_rail] and frame"""
            with self._lock:
                return self.command, self.frame

    remote_command = TimeoutRemoteCommand()

    from navigation_telemetry import NavigationTelemetry, stamp

    navigation_capture = NavigationTelemetry()

    # setup server for remote calls
    server = portal.Server(BASE_DEFAULT_PORT)
    server.bind("get_odometry", vehicle.get_odometry)
    server.bind("reset_odometry", vehicle.reset_odometry)
    server.bind("set_target_velocity", remote_command.remote_set_target_velocity)
    server.bind("get_linear_rail_state", vehicle.get_linear_rail_state)
    server.bind("get_head_tilt_state", head.get_state)
    server.bind("capture_clock", navigation_capture.clock)
    server.bind("capture_events", navigation_capture.read)
    server.start(block=False)

    print(f"Joystick Name: {joy.get_name()}")
    print(f"Number of Axes: {joy.get_numaxes()}")
    print(f"Number of Buttons: {joy.get_numbuttons()}")

    gamepad_command_frame = "local"
    gamepad_command_override = True

    last_gampad_mode_togged = False
    count = 0
    last_rail_log_time = time.time()
    RAIL_LOG_INTERVAL = 1.0  # Log linear rail position and head tilt every 1 second
    try:
        while True:
            gamepad_cmd = gamepad.get_user_cmd()  # 3D: [x, y, theta]
            gamepad_button = gamepad.get_button_reading()

            if gamepad_button["key_mode"] and not last_gampad_mode_togged:
                last_gampad_mode_togged = True
                gamepad_command_frame = "global" if gamepad_command_frame == "local" else "local"
            elif not gamepad_button["key_mode"]:
                last_gampad_mode_togged = False

            if gamepad_button["key_left_1"]:
                vehicle.reset_odometry()

            lift_vel = gamepad.get_rail_cmd()  # D-pad up/down

            cmd_4d = np.append(gamepad_cmd, lift_vel)

            is_remote_command_valid = remote_command.is_command_valid()

            if is_remote_command_valid:
                user_cmd, user_frame = remote_command.get_command()
                gamepad_command_override = False

                if gamepad_button["key_left_2"]:
                    gamepad_command_override = True
            else:
                gamepad_command_override = True
            if not vehicle.running():
                print("Motor interface is not running, exiting...")
                print("Please check the E stop or the motor connection. ")
                break
            if gamepad_command_override:
                cmd = cmd_4d
                frame = gamepad_command_frame
            else:
                cmd = user_cmd
                frame = user_frame
            if count % 20 == 0:
                sys.stdout.write(f"\rframe: {frame} cmd: {cmd[0]:.1f} {cmd[1]:.1f} {cmd[2]:.1f} rail: {cmd[3]:.1f}")
                sys.stdout.flush()

            if args.teach and select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.readline().strip().lower()
                try:  # a teach-mode error must never stop the vehicle
                    if key == "u":
                        print("\n" + head.undo_point())
                    elif key == "p":
                        print("\n" + head.describe_points())
                    else:
                        print("\n" + head.record_point())
                except Exception as e:
                    logger.error(f"Teach mode error (point not recorded): {e}")

            current_time = time.time()
            if current_time - last_rail_log_time >= RAIL_LOG_INTERVAL:
                h = head.get_state()
                if h.get("rail_position") is not None:
                    height = h["rail_position"] - (h["rail_bottom"] or 0.0)
                    head_text = (
                        f"head (by hand) {h['servo_position']}" if args.teach else f"-> head servo {h['servo_goal']}"
                    )
                    print(
                        f"Linear rail - pos: {h['rail_position']:.4f} rad, height {height:.2f} rad "
                        f"({100 * h['height_fraction']:.0f}%) {head_text}"
                        + ("" if h["connected"] else " (servo NOT connected)")
                    )
                else:
                    print(f"Head tilt: waiting for rail state (servo connected: {h['connected']})")
                last_rail_log_time = current_time

            count += 1

            if len(cmd) == 4:
                base_cmd = cmd[:3] * max_vel
                # Rail command is m/s; keep the old motor speed (fraction of lift_max_vel rad/s).
                meters_per_rad = vehicle.linear_rail.meters_per_rad if vehicle.linear_rail is not None else None
                rail_cmd = cmd[3] * lift_max_vel * meters_per_rad if meters_per_rad else 0.0
                physical_command = np.append(base_cmd, rail_cmd)
            else:
                physical_command = cmd * max_vel
            submit_start = stamp()
            vehicle.set_target_velocity(physical_command, frame=frame)
            submit_end = stamp()
            observed_command = physical_command if len(physical_command) == 4 else np.append(physical_command, 0.0)
            navigation_capture.publish(
                observed_command,
                frame,
                "gamepad" if gamepad_command_override else "remote",
                submit_start,
                submit_end,
                vehicle,
                head,
                gamepad_button["key_left_1"],
            )

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        try:
            head.stop()
            vehicle.close()
        except Exception as e:
            logger.error(f"Error during close: {e}")
        pygame.quit()
