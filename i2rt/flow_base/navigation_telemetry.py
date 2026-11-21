"""Read-only bounded telemetry for HeadTiltVehicle. Standard library only."""

import json
import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def stamp() -> dict[str, int]:
    before = time.monotonic_ns()
    wall = time.time_ns()
    after = time.monotonic_ns()
    return {"mono_ns": (before + after) // 2, "wall_ns": wall, "clock_read_uncertainty_ns": (after - before) // 2}


class NavigationTelemetry:
    def __init__(self, capacity: int = 15000) -> None:
        self.session = str(uuid.uuid4())
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        self.events = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.seq = 0
        self.errors = 0

    def clock(self, request: dict) -> dict[str, Any]:
        received = stamp()
        # timesyncd creates this only after a successful synchronization this boot.
        synced = Path("/run/systemd/timesync/synchronized").exists()
        return {
            "session": self.session,
            "boot_id": self.boot_id,
            "receive": received,
            "send": stamp(),
            "ntp_synchronized_this_boot": synced,
        }

    def read(self, request: dict) -> dict[str, Any]:
        after = int(request.get("after_seq", 0))
        with self.lock:
            oldest = self.events[0]["seq"] if self.events else self.seq + 1
            out = []
            if not request.get("latest_only", False):
                for event in reversed(self.events):
                    if event["seq"] <= after:
                        break
                    out.append(event)
                out = list(reversed(out))[:500]
            result = {
                "session": self.session,
                "oldest_seq": oldest,
                "latest_seq": self.seq,
                "lost_events": max(0, oldest - after - 1),
                "observer_errors": self.errors,
            }
        # Serialize outside the motor-observer lock. A single byte buffer also
        # avoids Portal writev's IOV_MAX limit on large nested batches.
        result["events_json"] = json.dumps(out, default=lambda v: v.item()).encode()
        return result

    def publish(
        self,
        command: Sequence[float],
        frame: str,
        source: str,
        submit_start: dict,
        submit_end: dict,
        vehicle: Any,
        head: Any,
        reset_requested: bool,
    ) -> None:
        # Never propagate observer failure into the motor loop.
        try:
            odom = vehicle.get_odometry()
            rail = vehicle.get_linear_rail_state()
            h = head.get_state()
            state_stamp = stamp()
            command = [float(x) for x in command]
            yaw = float(odom["position"]["rotation"])
            body = command.copy()
            if frame == "global":
                c, s = math.cos(yaw), math.sin(yaw)
                body[:2] = [c * command[0] + s * command[1], -s * command[0] + c * command[1]]
            e = {
                "session": self.session,
                "submit_start": submit_start,
                "submit_end": submit_end,
                "state_read": state_stamp,
                "command": command,
                "command_body": body,
                "frame": frame,
                "source": source,
                "command_semantics": "submitted_setpoint_not_confirmed_actuator_application",
                "odometry_reset_requested": bool(reset_requested),
                "odometry": {"translation": [float(x) for x in odom["position"]["translation"][:2]], "yaw": yaw},
                "rail": {
                    k: rail.get(k)
                    for k in (
                        "position",
                        "velocity",
                        "initialized",
                        "brake_on",
                        "upper_limit_triggered",
                        "lower_limit_triggered",
                    )
                },
                "head": {
                    k: h.get(k) for k in ("rail_bottom", "rail_travel", "servo_goal", "servo_position", "connected")
                },
                "state_sample_age": "unknown: legacy driver exposes cached state without acquisition timestamps",
            }
            with self.lock:
                self.seq += 1
                e["seq"] = self.seq
                self.events.append(e)
        except Exception:
            with self.lock:
                self.errors += 1
