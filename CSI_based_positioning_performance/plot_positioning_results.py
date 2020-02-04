import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import tensorflow as tf
import neural_nets
from data_generator import DataGenerator
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


def true_dist(y_true, y_pred):
    return np.sqrt(
        np.square(np.abs(y_pred[:, 0] - y_true[:, 0]))
        + np.square(np.abs(y_pred[:, 1] - y_true[:, 1]))
    )


num_samples = 252004
num_scenarios = 3
topology = "URA"
room = ["boardroom", "lab", "lab"]
los = ["LoS", "LoS", "nLoS"]
figure_labels = ["LoS, Boardroom", "LoS, Lab", "nLoS, Lab"]
styles = [':', '-.', '--', '-']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
labels = np.load('/home/sdebast/data/mamimo_measurements/labels.npy')

errors = []
for scenario in range(num_scenarios):
    name = room[scenario] + "_" + topology + "_" + los[scenario]
    print(name)
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
    test_size = 0.10
    test_IDs = IDs[-int(test_size * actual_num_samples):]  # last 5% of the data

    nn = load_model("./best_models/positioning_" + name + ".h5",
                    custom_objects={"tf": tf, "dist": neural_nets.dist})

    test_generator = DataGenerator(dataset, test_IDs, labels, shuffle=False)

    test_pred = nn.predict_generator(test_generator)
    test_length = test_pred.shape[0]

    test_errors = true_dist(labels[test_IDs[:test_length]], test_pred)
    mean_test_error = np.mean(np.abs(test_errors))
    print("Mean error on testset:", mean_test_error)
    errors.append(test_errors)

plt.figure()
plt.title("CDF of the Positioning Error")
plt.ylabel("F(X)")
plt.xlabel('Positioning error [mm]')
for scenario in range(num_scenarios):
    plt.hist(errors[scenario], density=True, cumulative=True, linestyle=styles[scenario],
             histtype='step', bins=100, range=(0, 100), color=colors[scenario],
             label=figure_labels[scenario])
plt.grid()
plt.yticks([0, 0.25, 0.5, 0.75, 1])
plt.legend(loc='lower right')
plt.savefig('plots/cdf_positioning.eps', bbox_inches='tight', pad_inches=0)
plt.savefig('plots/cdf_positioning.png', bbox_inches='tight', pad_inches=0)


plt.figure()
# plt.title("CDF of the SINR for different path planning algorithms.")
for scenario in range(num_scenarios):
    # print(i)
    data = np.array(errors[i])
    data = np.sort(data)
    average = sum(data)/len(data)
    # print(labels[i], average)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    # length = len(data)
    # nb_samples = 200
    # step = length / nb_samples
    # idx = [i*step for i in range(nb_samples-1)]
    # data = np.take(data, idx)
    curve_x = [0]
    curve_x.extend(data)
    curve_x.extend([100])
    curve_y = [0]
    curve_y.extend(p)
    curve_y.extend([1])
    # print(len(data))
    plt.plot(curve_x, curve_y, label=figure_labels[scenario], linestyle=styles[scenario])
    # print(len(sinrs[i]))
    # plt.hist(sinrs[i], density=True, cumulative=True,
    #          label=labels[i], histtype='step', bins=250)
# print("Histograms created")
font_size = 10
plt.title("CDF of the Positioning Error")
plt.ylabel("F(X)")
plt.xlabel('Positioning error [mm]')
plt.xticks(fontsize=font_size)
plt.yticks([0, 0.25, 0.5, 0.75, 1], fontsize=font_size)
plt.legend(loc='lower right')
plt.grid(linestyle=':', linewidth=1)
plt.axis([0, 100, -0.1, 1.1])
# plt.show()
# print("Saving plots")
plt.savefig('plots/cdf_positioning_fix.eps',
            bbox_inches='tight', pad_inches=0)
plt.savefig('plots/cdf_positioning_fix.png',
            bbox_inches='tight', pad_inches=0)
# print(".png done")
