import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import camera_input as Cam
from classifier import Classifier
from config import MainConfig
from object_isolator import ObjectIsolator
from rating import Rating

if __name__ == '__main__':
    cam = Cam.CameraInput(MainConfig.fps, MainConfig.buffer_size, sim=True)
    ai = Classifier()
    ai.load()

    buffer = cam.get()
    timeline = []
    ratings = []
    while buffer:
        op = ObjectIsolator.isolate(buffer)
        preped = ai.prepare(op)

        predictions = []
        for one in preped:
            predictions.append(ai.predict(one, False))
        timeline.append(predictions)

        r = Rating.evaluate(predictions)
        ratings.append(r)

        buffer = cam.get()

    n = [len(x) for x in timeline]
    cont = [sum([y.argmax() for y in x]) for x in timeline]
    ok = np.array(n) - np.array(cont)
    df = pd.DataFrame({'n': n, 'clean': ok, 'contaminated': cont, 'rating': ratings})
    df.plot.line()
    plt.grid()
    plt.show()

    cam.terminate()
