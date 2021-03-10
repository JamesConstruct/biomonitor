import time

import cv2 as cv

from config import CameraConfig


class CameraInput:

    def __init__(self, max_fps, buffer_size, sim=False):
        self.max_fps = max_fps
        self.buffer_size = buffer_size
        self.sim = sim
        self.cap = None
        self.prep_cap()

    def prep_cap(self):
        if self.sim:
            self.cap = cv.VideoCapture(CameraConfig.simulation_source)
            self.cap.set(cv.CAP_PROP_POS_FRAMES, CameraConfig.simulation_begin)
            if not self.cap.isOpened():
                raise NameError("Failed opening " + CameraConfig.simulation_source) # BaseException.Exception.OSError.FileNotFoundError
        else:
            self.cap = cv.VideoCapture(CameraConfig.index)
            if not self.cap.isOpened():
                raise Exception("Failed opening camera. Index: " + str(CameraConfig.index))

        if CameraConfig.resolution:
            self.cap.set(cv.CAP_PROP_FRAME_WIDTH, CameraConfig.resolution[0])
            self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, CameraConfig.resolution[1])

    def get(self):
        buffer = []
        t = time.time()
        for _ in range(self.buffer_size):
            ret, frame = self.cap.read()
            if ret:
                if CameraConfig.gray:
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

                if self.sim and CameraConfig.resolution:
                    frame = cv.resize(frame, CameraConfig.resolution, interpolation=cv.INTER_AREA)

                buffer.append(frame)

            else:
                return False

            time.sleep(max(0, 1/self.max_fps-(time.time()-t)))
            t = time.time()

        return buffer

    def terminate(self):
        self.cap.release()
