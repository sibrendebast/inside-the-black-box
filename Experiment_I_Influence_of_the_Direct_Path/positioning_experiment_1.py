import tensorflow as tf
from keras.models import load_model
import neural_nets
import numpy as np
from data_generator import DataGenerator
import matplotlib.pyplot as plt
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


# calculate the euclidian distance between two points
def true_dist(y_true, y_pred):
    return np.sqrt(
        np.square(np.abs(y_pred[:, 0] - y_true[:, 0]))
        + np.square(np.abs(y_pred[:, 1] - y_true[:, 1]))
    )


room = "lab"
los = "LoS"
topology = "URA"
name = room + "_" + topology + "_" + los
dataset = "../data/" + name + "/CSI_samples/"
labels = np.load('../data/labels.npy')

num_samples = 252004
num_antennas = 64
num_sub = 100

# take 10% of the data a testset
test_size = 0.1

#  ########### load lab URA LoS dataset #################
# Get bad samples
bad_samples = np.load("../data/" + name + "/bad_channels.npy")
# builds array with all valid channel indices
IDs = []
for x in range(num_samples):
    if x not in bad_samples:
        IDs.append(x)
IDs = np.array(IDs)
# shuffle the indices with fixed seed
np.random.seed(64)
np.random.shuffle(IDs)
# get the number of CSI samples
actual_num_samples = IDs.shape[0]
# get the test set IDs
test_IDs = IDs[-int(test_size * actual_num_samples):]  # last 5% of the data

# get a datagenerator for the test set from the lab URA LoS dataset
test_generator = DataGenerator(dataset, test_IDs, labels, shuffle=False)


# load board room model
nn = load_model("../CSI_based_positioning_performance/best_models/positioning_boardroom_URA_LoS.h5",
                custom_objects={"tf": tf, "dist": neural_nets.dist})

# estimate the position of the test set samples
test_pred = nn.predict_generator(test_generator)
# get the number of estimated positions
test_length = test_pred.shape[0]

# calculate the distance between the ground thruth and estimates
test_errors = true_dist(labels[test_IDs[:test_length]], test_pred)
# calculate the mean positioning error
mean_test_error = np.mean(np.abs(test_errors))
median_test_error = np.median(np.abs(test_errors))

# print the results
print("Results for experiment I:")
print('\033[1m{:<40}{:.4f}\033[0m'.format('Mean error on Test area: ', mean_test_error), 'mm')
print('\033[1m{:<40}{:.4f}\033[0m'.format('Median error on Test area: ', median_test_error), 'mm')
result_file = open('results.txt', 'a')
result_file.write("Results for experiment I:\n")
result_file.write('\033[1m{:<40}{:.4f}\033[0m'.format('Mean error on Test area: ', mean_test_error) + 'mm\n')
result_file.write('\033[1m{:<40}{:.4f}\033[0m'.format('Median error on Test area: ', median_test_error) + 'mm\n')

deviation = labels[test_IDs[:test_length]] - test_pred
mean_deviation = np.mean(deviation, axis=0)
median_deviation = np.median(deviation, axis=0)
print("Mean deviation of the error:", mean_deviation)
print("Median deviation of the error:", median_deviation)

fig = plt.figure()
cm = plt.cm.get_cmap('jet')
plt.scatter(x=labels[test_IDs[:test_length]][:, 0],
            y=labels[test_IDs[:test_length]][:, 1],
            c=test_errors,
            cmap=cm,
            s=(72./fig.dpi)**2)
plt.colorbar()
plt.savefig('plots/error_scatter_experiment_I.png', bbox_inches='tight', pad_inches=0)
plt.savefig('plots/error_scatter_experiment_I.eps', bbox_inches='tight', pad_inches=0)

plt.figure()
plt.hist(test_errors, bins=128, range=(0, 2000))
plt.ylabel('Number of occurence')
plt.xlabel('Distance error [mm]')
plt.savefig('plots/error_histogram_experiment_I.png', bbox_inches='tight', pad_inches=0)
plt.savefig('plots/error_histogram_experiment_I.eps', bbox_inches='tight', pad_inches=0)
