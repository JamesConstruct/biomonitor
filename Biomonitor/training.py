import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from classifier import Classifier

epochs = 500
validation_split = 0.05
test_split = 0.1

if __name__ == '__main__':
    ai = Classifier()
    ai.load()

    dataset_clean = np.concatenate((
        np.load('Datasets/dataset-clean-1.npy'),
        np.load('Datasets/dataset-clean-2.npy'),
        np.load('Datasets/dataset-clean-3.npy'),
        np.load('Datasets/dataset-clean-4.npy'),
        np.load('Datasets/dataset-clean-5.npy')
    ))
    dataset_contaminated = np.concatenate((
        np.load('Datasets/dataset-contaminated-1.npy'),
        np.load('Datasets/dataset-contaminated-2.npy'),
        np.load('Datasets/dataset-contaminated-3.npy'),
        np.load('Datasets/dataset-contaminated-4.npy'),
        np.load('Datasets/dataset-contaminated-5.npy')
    ))

    print("Clean: ", len(dataset_clean))
    print("Contaminated: ", len(dataset_contaminated))

    dataset = np.concatenate((
        np.array(dataset_clean),
        np.array(dataset_contaminated)
    ))
    dataset_x = np.array(dataset).reshape(len(dataset), 300)
    dataset_y = np.concatenate((
                    np.zeros((len(dataset_clean), 1)),
                    np.ones((len(dataset_contaminated), 1))
                    # np.full((len(dataset_dead), 1), 2)
                 ))

    sm = SMOTE(sampling_strategy=1.0, random_state=57)
    dataset_x, dataset_y = sm.fit_resample(dataset_x, dataset_y)

    x_train, x_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=test_split)

    data = {
        'x': x_train,
        'y': y_train
    }

    try:
        ai.train(data, epochs, validation_split)
    except KeyboardInterrupt:
        pass

    print('TESTS:')
    for i, one in enumerate([dataset_clean, dataset_contaminated]):
        test = {'x': np.array(one).reshape(len(one), 300),
                'y': np.full((len(one), 1), i)}
        ai.model.evaluate(test['x'], test['y'], batch_size=128)
    ai.model.evaluate(x_test, y_test)

    if input("Save? ").upper() == "YES":
        ai.save()
