from pyrlhook import GameInterface, PixelFormat, PixelDataType
import cv2
import os
from time import sleep
from subprocess import Popen


class Recorder:
    def __init__(self, record=True):
        self.frames = None
        self.record = record

    def start(self):
        self.frames = []
        self.record = True

    def stop(self):
        self.record = False

    def add_frame(self, frame):
        self.frames.append(frame[::-1])

    def save(self, filename, fps, codec='FFV1', file_ending='.mp4'):
        # ffmpeg -i input -c:v h264 -b 1000k output
        fourcc = cv2.VideoWriter_fourcc(*codec)
        height, width, _ = self.frames[0].shape
        vw = cv2.VideoWriter(filename + file_ending, fourcc, fps / 2, (width, height))
        for f in self.frames:
            vw.write(f[:, :, ::-1])
        for _ in range(len(self.frames) // 50):
            vw.write(self.frames[-1][:, :, ::-1])
        vw.release()


class SuperHexagonInterface:

    frame_size = 60, 60  # h, w
    frame_size_cropped = 60, 60  # h, w
    n_actions = 3
    n_levels = 6

    def __init__(
            self,
            frame_skip=4,
            super_hexagon_path='C:\\Program Files (x86)\\Steam\\steamapps\\common\\Super Hexagon\\superhexagon.exe',
            run_afap=True,
            record=False,
            allow_game_restart=False
    ):
        self.game_process = None
        self.game_process = None
        self.game = None
        self.super_hexagon_path = super_hexagon_path
        self.run_afap = run_afap
        self.allow_game_restart = allow_game_restart
        self.recorder = Recorder(record)
        self.steps_alive = 0
        self.simulated_steps = 0
        self.frame_skip = frame_skip

        if allow_game_restart:
            self._restart_game()
        else:
            self._attach_game()

        self.level = self._get_level()

    def _esc(self, down):
        self.game.write_byte('superhexagon.exe', [0x00294B00, 0x42877], 1 if down else 0)

    def _press_esc(self):
        self.game.step(False)
        self._esc(True)
        for _ in range(10):
            self.game.step(False)
        self._esc(False)
        self.game.step(False)

    def _left(self, down):
        self.game.write_byte('superhexagon.exe', [0x00294B00, 0x428BD], 1 if down else 0)

    def _right(self, down):
        self.game.write_byte('superhexagon.exe', [0x00294B00, 0x428C0], 1 if down else 0)

    def _space(self, down):
        self.game.write_byte('superhexagon.exe', [0x00294B00, 0x4287C], 1 if down else 0)

    def _get_level(self):
        t = self.game.read_byte('superhexagon.exe', [0x00294B00, 0x54CC])
        return (-t) % 6

    def _get_main_menu_selection(self):
        return self.game.read_byte('superhexagon.exe', [0x00294B00, 0x111EC])

    def _get_current_menu(self):
        return self.game.read_byte('superhexagon.exe', [0x00294B00, 0x48])

    def get_triangle_angle(self):
        return self.game.read_dword('superhexagon.exe', [0x00294B00, 0x2958])

    def get_world_angle(self):
        return self.game.read_dword('superhexagon.exe', [0x00294B00, 0x1AC])

    def get_n_survived_frames(self):
        return self.game.read_dword('superhexagon.exe', [0x00294B00, 0x2988])

    def _reset_rotation(self):
        self.game.write_dword('superhexagon.exe', [0x00294B00, 0x1AC], 1)

    def _restart_game(self):
        assert self.allow_game_restart, "in order to restart the game 'allow_game_restart' must be set to True"
        self.game = None
        sleep(1)
        if self.game_process is not None:
            self.game_process.terminate()
        self.game_process = Popen([self.super_hexagon_path], cwd=os.path.dirname(self.super_hexagon_path))
        sleep(10)
        self._attach_game()

    def _attach_game(self):
        self.game = GameInterface('superhexagon.exe', PixelFormat.RGB, PixelDataType.UINT8)

        if self.run_afap:
            self.game.run_afap(62.5)

        for _ in range(10):
            self.game.step(False)
        self._left(False)
        self._right(False)
        self._space(False)
        self._esc(False)
        for _ in range(10):
            self.game.step(False)

        self._goto_level_select_menu()

    def _goto_level_select_menu(self):
        while True:
            m = self._get_current_menu()
            if m == 1:
                break
            elif m == 3 or m == 0:
                self._press_esc()
            elif m == 2:

                f = self.game.step(True)
                if f[240, 384, 0] == 255:  # main menu

                    while self._get_main_menu_selection() != 0:  # rotate to 'start game'
                        self.game.step(False)
                        self._right(True)
                        for _ in range(5):
                            self.game.step(False)
                        self._right(False)
                        self.game.step(False)

                    self._space(True)
                    for i in range(10):
                        self.game.step(False)
                    self._space(False)
                    self.game.step(False)

                else:  # game over screen
                    self._press_esc()

    def select_level(self, level: int):

        assert 0 <= level <= 5, f"'level' has to be between 0 and 5 but was {level}"

        # otherwise the game will sometimes crash for some unknown reason
        for _ in range(60):
            self.game.step(False)

        self._goto_level_select_menu()

        # otherwise the game will sometimes crash for some unknown reason
        for _ in range(60):
            self.game.step(False)

        for _ in range(6):
            if self._get_level() == level:
                break

            # select next level
            self._right(True)
            for _ in range(5):
                self.game.step(False)
            self._right(False)

            # wait for the next level to be selected
            for _ in range(30):
                self.game.step(False)
        else:
            self._restart_game()
            self.select_level(level)
            return

        self.level = level

        for _ in range(30):
            self.game.step(False)

    def _postprocess_frame(self, frame):
        f = cv2.cvtColor(cv2.resize(frame[:, 144:624], self.frame_size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2GRAY)
        fc = cv2.cvtColor(cv2.resize(frame[150:330, 294:474], self.frame_size_cropped, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2GRAY)
        center_color = f[self.frame_size[0] // 2, self.frame_size[1] // 2]
        if center_color > 200:
            return f < 200, fc < 200
        elif center_color > 90:
            thresh = center_color + 40
            return f > thresh, fc > thresh
        else:
            thresh = center_color + 28
            return f > thresh, fc > thresh

    def reset(self):

        for _ in range(30):
            self.game.step(False)

        # go to level select
        if self._get_current_menu() != 1:
            self._press_esc()

        for _ in range(30):
            self.game.step(False)

        # press space for the game to start and do nothing for the first 70 time steps
        self._space(True)
        for i in range(70):
            frame = self.game.step(self.recorder.record or i == 69)
            if self.recorder.record:
                self.recorder.add_frame(frame)
        self._space(False)

        self.steps_alive = self.get_n_survived_frames()
        self.simulated_steps = 0

        return self._postprocess_frame(frame)

    def step(self, action):

        # sleep(.05)
        if action == 1:
            self._left(True)
        elif action == 2:
            self._right(True)

        steps_alive_old = self.steps_alive
        for i in range(self.frame_skip):
            frame = self.game.step(self.recorder.record or i == (self.frame_skip - 1))
            if self.recorder.record:
                self.recorder.add_frame(frame)
        self.steps_alive = self.get_n_survived_frames()

        is_game_over = self.steps_alive < steps_alive_old + self.frame_skip

        frame, frame_cropped = self._postprocess_frame(frame)

        if action == 1:
            self._left(False)
        elif action == 2:
            self._right(False)

        self.steps_alive = self.get_n_survived_frames()
        self.simulated_steps += 1

        return (frame, frame_cropped), -1 if is_game_over else 0, is_game_over
