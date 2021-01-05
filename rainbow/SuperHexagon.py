from RLHook import GameInterface, PixelFormat, PixelDataType
import numpy as np
import cv2
from cv2 import resize, INTER_AREA
from time import sleep


class Recorder:
    def __init__(self):
        self.frames = None
        self.record = False
        self.skip = False

    def start(self):
        self.frames = []
        self.record = True

    def stop(self):
        self.record = False

    def add_frame(self, frame):
        if self.skip:
            self.skip = False
        else:
            self.frames.append(frame[::-1])
            self.skip = True

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

    def __init__(self, frame_skip=4):  # , level
        self.game = GameInterface('superhexagon.exe', PixelFormat.RGB, PixelDataType.UINT8)
        self.game.run_afap(62.5)
        self.recorder = Recorder()
        self.steps_alive = 0
        self.frame_skip = frame_skip

    def _is_game_over(self, frame):
        # return self.game.read_byte(self._is_alive_pointer) == 0
        # print(frame[-1, -1, 0], frame[-1, -1, 1], frame[-1, -1, 2])
        return frame[-1, -1, 0] > 254 and frame[-1, -1, 1] > 254 and frame[-1, -1, 2] > 254

    def _left(self, down):
        self.game.write_byte('superhexagon.exe', [0x00294B00, 0x428BD], 1 if down else 0)

    def _right(self, down):
        self.game.write_byte('superhexagon.exe', [0x00294B00, 0x428C0], 1 if down else 0)

    def _space(self, down):
        self.game.write_byte('superhexagon.exe', [0x00294B00, 0x4287C], 1 if down else 0)

    def get_triangle_angle(self):
        return self.game.read_dword('superhexagon.exe', [0x00294B00, 0x2958])

    def get_world_angle(self):
        return self.game.read_dword('superhexagon.exe', [0x00294B00, 0x1AC])

    def get_n_survived_frames(self):
        return self.game.read_dword('superhexagon.exe', [0x00294B00, 0x2988])

    def _reset_rotation(self):
        self.game.write_dword('superhexagon.exe', [0x00294B00, 0x1AC], 1)

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

        # wait for the game to show the game over screen
        for _ in range(90):
            self.game.step(False)

        # press space for the game to start and do nothing for the first 70 time steps
        self._space(True)
        for i in range(70):
            frame = self.game.step(self.recorder.record or i == 69)
            if self.recorder.record:
                self.recorder.add_frame(frame)
        self._space(False)

        self.steps_alive = self.get_n_survived_frames()

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

        return (frame, frame_cropped), -1 if is_game_over else 0, is_game_over
