import numpy as np

import camera_input as Cam
from classifier import Classifier
from config import MainConfig
from object_isolator import ObjectIsolator

if __name__ == '__main__':
    cam = Cam.CameraInput(MainConfig.fps, MainConfig.buffer_size, sim=True)
    ai = Classifier()

    dataset = []
    buffer = cam.get()
    i = 0
    while buffer:
        op = ObjectIsolator.isolate(buffer)
        preped = ai.prepare(op)
        dataset += preped

        buffer = cam.get()

    np.save('Datasets/dataset.npy', np.asarray(dataset))

