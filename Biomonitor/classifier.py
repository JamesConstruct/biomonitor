import numpy as np
import tensorflow as tf

from config import ClassifierConfig, MainConfig


class Classifier:

    def __init__(self):
        self.bins = {
            0: 'clean',
            1: 'contaminated'
        }

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Flatten(input_shape=(MainConfig.buffer_size * 2,)))  # input layer
        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(len(self.bins), activation=tf.nn.softmax))  # output layer, #softmax

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ClassifierConfig.learning_rate),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # or binary
                           metrics=['accuracy'])

    def save(self):
        self.model.save_weights(ClassifierConfig.model_path)

    def load(self):
        self.model.load_weights(ClassifierConfig.model_path)

    def train(self, data, epochs, split):
        self.model.fit(data['x'], data['y'], epochs=epochs, validation_split=split)

    def predict(self, data, verbose=True):
        p = self.model.predict([data])
        state = self.bins[p[0].argmax()]
        if verbose:
            print("Prediction: {:.2f}% \t{}".format(max(p[0] * 100), state))
        return p

    def prepare(self, data, n=-1):
        vec = []
        for one in data:
            if len(data[one]) >= ClassifierConfig.trajectory_min_length:
                v = self._vectorize(np.array(data[one]))
                vec.append(v)

        if n != -1:  # do not sort, if it isn't required
            vec = sorted(vec, key=len, reverse=True)[0:min(len(vec), n)]

        for i in range(len(vec)):
            length = len(vec[i])
            for ii in range(MainConfig.buffer_size - length):
                vec[i].append(vec[i][ii % length])  # padding

            vec[i] = np.array(vec[i]).reshape(1, MainConfig.buffer_size * 2)

        return vec

    def translate(self, data):
        if type(data) is int:
            return self.bins[data]
        elif type(data) is str:
            return self.bins.values().index(data)

    def _vectorize(self, points):
        v = []  # vectors
        for i in range(1, len(points), 1):
            vec = points[i] - points[i - 1]
            v.append(vec)

        m = np.max(v)
        if m != 0:
            v = [x / m for x in v]  # normalize vectors

        return v


tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
