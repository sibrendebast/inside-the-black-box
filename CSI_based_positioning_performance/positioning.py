import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input
import neural_nets
import numpy as np
from keras.utils import multi_gpu_model
from data_generator import DataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
num_gpus = 1
total_gpus = 8
start_gpu = 0
cuda = ""
for i in range(num_gpus):
    cuda += str((start_gpu + i) % total_gpus) + ","
print("Adding visible CUDA devices:", cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
tf.logging.set_verbosity(tf.logging.ERROR)


def true_dist(y_true, y_pred):
    return np.sqrt(
        np.square(np.abs(y_pred[:, 0] - y_true[:, 0]))
        + np.square(np.abs(y_pred[:, 1] - y_true[:, 1]))
    )


class positioner():

    def __init__(self, num_antennas=64, num_subcarriers=100, num_gpus=1):
        self.cnn = neural_nets.build_cnn()
        self.fully_connected = neural_nets.build_fully_connected()
        self.label = neural_nets.build_label()

        input = Input((num_antennas, num_subcarriers, 2))
        cnned = self.cnn(input)
        fced = self.fully_connected(cnned)
        label = self.label(fced)
        self.model = Model(inputs=input, outputs=label)
        self.model.summary()
        if num_gpus > 1:
            self.model = multi_gpu_model(self.model, num_gpus)
        self.model.compile(optimizer='Adam', loss='mse',
                           metrics=[neural_nets.dist])


num_scenarios = 3
room = ["boardroom", "lab", "lab"]
los = ["LoS", "LoS", "nLoS"]
topology = "URA"
num_samples = 252004
num_antennas = 64
num_sub = 100
labels = np.load('../data/labels.npy')

# Training size
trainings_size = 0.85                    # 85% training set
validation_size = 0.05                     # 10% validation set
test_size = 0.1                          # 5% test set

batch_size = 64
num_epochs = 200


for scenario in range(num_scenarios):
    name = room[scenario] + "_" + topology + "_" + los[scenario]
    dataset = "../data/" + name + "/CSI_samples/"
    bad_samples = np.load("../data/" + name + "/bad_channels.npy")
    # buils array with all valid channel indices
    IDs = []
    for x in range(num_samples):
        if x not in bad_samples:
            IDs.append(x)
    IDs = np.array(IDs)
    # shuffle the indices with fixed seed
    np.random.seed(64)
    np.random.shuffle(IDs)
    # get the number of channels
    actual_num_samples = IDs.shape[0]
    # distributed the samples over the train, validation and test set
    train_IDs = IDs[:int(trainings_size*actual_num_samples)]  # first 85% of the data
    val_IDs = IDs[int(trainings_size*actual_num_samples):int((trainings_size + validation_size) * actual_num_samples)]
    test_IDs = IDs[-int(test_size * actual_num_samples):]  # last 5% of the data

    val_generator = DataGenerator(dataset, val_IDs, labels)
    test_generator = DataGenerator(dataset, test_IDs, labels)
    train_generator = DataGenerator(dataset, train_IDs, labels)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint('./checkpoints/positioning_' + name + '.hdf5', monitor='val_dist',
                         mode='min', verbose=1, save_best_only=True)

    pos = positioner(num_gpus=num_gpus)
    nn = pos.model

    nn.fit_generator(train_generator, epochs=num_epochs, validation_data=val_generator, callbacks=[es, mc])

    nn = load_model("./checkpoints/positioning_" + name + ".hdf5",
                    custom_objects={"tf": tf, "dist": neural_nets.dist})

    test_pred = nn.predict_generator(test_generator)
    test_length = test_pred.shape[0]

    test_errors = true_dist(labels[test_IDs[:test_length]], test_pred)
    mean_test_error = np.mean(np.abs(test_errors))

    print("Results for the " + name + " positioning task:\n")
    print('Mean error on the Test set: ', mean_test_error, 'mm')
    result_file = open('results.txt', 'a')
    result_file.write("Results for the " + name + " positioning task:\n")
    result_file.write('\033[1m{:<40}{:.4f}\033[0m'.format('Mean error on the Test set: ', mean_test_error) + 'mm\n')

    plt.figure()
    plt.hist(test_errors, bins=128, range=(0, 500))
    plt.ylabel('Number of occurence')
    plt.xlabel('Distance error [mm]')
    plt.savefig('./plots/error_histogram' + name + '.png', bbox_inches='tight', pad_inches=0)
