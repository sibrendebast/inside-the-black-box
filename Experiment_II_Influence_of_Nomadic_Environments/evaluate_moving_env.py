import tensorflow as tf
from keras.models import load_model
# from keras.layers import Input
import neural_nets
import numpy as np
# from keras.utils import multi_gpu_model
# from data_generator import DataGenerator
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint
# import matplotlib.pyplot as plt
import os
num_gpus = 1
total_gpus = 8
start_gpu = 1
cuda = ""
for i in range(num_gpus):
    cuda += str((start_gpu + i) % total_gpus) + ","
print("Adding visible CUDA devices:", cuda)
os.environ["CUDA_VISIBLE_DEVICES"] = cuda
tf.logging.set_verbosity(tf.logging.ERROR)


room = "lab"
topology = "URA"
loss = ["LoS", "nLoS"]
num_samples = 240
num_antennas = 64
num_sub = 100
num_scenarios = 7
num_users = 4
scenarios = ['0: Reference', '1: Back', '2: Left', '3: Right', '4: Front', '5: Left to Right', '6: Front to Back']
labels = np.load('../data/labels.npy')


def true_dist(y_true, y_pred):
    return np.sqrt(
        np.square(np.abs(y_pred[0] - y_true[0]))
        + np.square(np.abs(y_pred[1] - y_true[1]))
    )


for los in loss:
    name = room + "_" + topology + "_" + los
    dataset = "../data/" + name + "_nomadic/CSI_samples/"
    nn = load_model("../CSI_based_positioning_performance/best_models/positioning_" + name + ".h5",
                    custom_objects={"tf": tf, "dist": neural_nets.dist})

    results = np.empty((num_scenarios, num_users, num_samples, 2))
    error = np.empty((num_scenarios, num_users, num_samples))
    for scenario in range(num_scenarios):
        for user in range(num_users):
            for sample in range(num_samples):
                id = scenario * 10000 + user * 1000 + sample
                channel = np.load(dataset + "channel_measurement_" + str(id).zfill(6) + '.npy')
                X = np.empty((1, num_antennas, num_sub, 2))
                X[0, :, :, 0] = channel.real
                X[0, :, :, 1] = channel.imag
                loc_pred = nn.predict([X])
                results[scenario, user, sample, :] = loc_pred[0]
                dist = true_dist(loc_pred[0], labels[user*63001 + 31500])
                error[scenario, user, sample] = dist

    np.save("moving_results_" + name, results)
    np.save("moving_errors_" + name, error)
