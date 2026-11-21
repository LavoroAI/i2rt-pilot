#!/usr/bin/env python3
import os

import numpy as np
import pygame


class Gamepad:
    def __init__(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        # Initialize pygame and joystick
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("No joystick/gamepad connected!")
            exit()
        else:
            print(f"Detected {pygame.joystick.get_count()} joystick(s).")

        # Initialize the joystick
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        xbox = self.joy.get_name().lower() in {"xbox 360 controller", "microsoft x-box 360 pad"}
        self.turn_axis = 3 if xbox else 2
        self.mode_button = 7 if xbox else 12
        self.override_button = 4 if xbox else 8
        self.reset_button = 6
        if (
            self.joy.get_numaxes() <= self.turn_axis
            or self.joy.get_numbuttons() <= max(self.mode_button, self.override_button, self.reset_button)
            or self.joy.get_numhats() < 1
        ):
            raise ValueError("Unsupported gamepad layout; refusing motor startup")
        print(f"Joystick Name: {self.joy.get_name()}")
        print(f"Number of Axes: {self.joy.get_numaxes()}")
        print(f"Number of Buttons: {self.joy.get_numbuttons()}")

    def get_button_reading(self) -> dict[str, int]:
        pygame.event.pump()
        key_mode = self.joy.get_button(self.mode_button)
        key_left_2 = self.joy.get_button(self.override_button)
        key_left_1 = self.joy.get_button(self.reset_button)
        return dict(
            key_mode=key_mode,
            key_left_2=key_left_2,
            key_left_1=key_left_1,
        )

    def get_user_cmd(self) -> np.ndarray:
        pygame.event.pump()
        # Read inputs
        x = self.joy.get_axis(1)  # Left stick Y-axis
        y = self.joy.get_axis(0)  # Left stick X-axis
        th = self.joy.get_axis(self.turn_axis)  # Right stick X-axis

        user_cmd = np.array([-x, -y, -th])
        # if < 0.05 force to zero
        user_cmd[np.abs(user_cmd) < 0.05] = 0
        return user_cmd

    def get_rail_cmd(self) -> float:
        """Rail command from the D-pad: up = +1, down = -1, else 0."""
        pygame.event.pump()
        return float(self.joy.get_hat(0)[1])

    def centered(self, deadzone: float = 0.05) -> bool:
        pygame.event.pump()
        return all(abs(self.joy.get_axis(i)) < deadzone for i in (0, 1, self.turn_axis))

    def close(self) -> None:
        pygame.quit()


if __name__ == "__main__":
    pad = Gamepad()
    try:
        while True:
            cmd = pad.get_user_cmd()
            buttons = pad.get_button_reading()
            print(f"user_cmd = {cmd}  |  buttons = {buttons}")
    except KeyboardInterrupt:
        pad.close()
