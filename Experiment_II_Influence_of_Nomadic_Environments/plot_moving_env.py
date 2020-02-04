import numpy as np
import matplotlib.pyplot as plt

num_samples = 240
num_antennas = 64
num_sub = 100
num_scenarios = 7
num_users = 4
scenarios = ['Reference', 'Back', 'Left', 'Right', 'Front', 'Left to Right', 'Front to Back']

room = ["lab", "lab"]
topology = ["URA", "URA"]
los = ["LoS", "nLoS"]

for i in range(len(room)):
    name = room[i] + "_" + topology[i] + "_" + los[i]
    print(name)
    # results = np.load("moving_results_LoS.npy")
    # errors = np.load("moving_errors_LoS.npy")
    results = np.load("moving_results_" + name + ".npy")
    errors = np.load("moving_errors_" + name + ".npy")

    mean_ref_position = np.empty((num_users, 2))
    corrected_results = np.empty((num_scenarios, num_users, num_samples, 2))
    for j in range(num_users):
        mean_ref_position[j, 0] = np.mean(results[0, j, :, 0])
        mean_ref_position[j, 1] = np.mean(results[0, j, :, 1])
        corrected_results[:, j, :, :] = results[:, j, :, :] - mean_ref_position[j]

    print(errors.shape)
    mean_ref_error = np.mean(errors[0, :, :], axis=1)
    print(mean_ref_position)
    print(mean_ref_error)

    corrected_errors = np.sqrt(np.square(np.abs(corrected_results[:, :, :, 0]))
                               + np.square(np.abs(corrected_results[:, :, :, 1])))

    mean_errors = np.empty((num_scenarios, num_users))
    std_errors = np.empty((num_scenarios, num_users))
    for i, scenario in enumerate(scenarios):
        print(scenario)
        plt.figure()
        plt.plot(corrected_errors[i, :, :].T)
        plt.savefig('plots/moving_test_' + name + '.png')
        for user in range(num_users):
            mean_errors[i, user] = np.mean(corrected_errors[i, user, :])
            std_errors[i, user] = np.std(corrected_errors[i, user, :])
            # print(np.mean(corrected_errors[i, user, :]))

    num_samples_to_plot = 121
    time_step = 0.5
    time = [i*time_step for i in range(num_samples_to_plot)]
    plt.figure(figsize=(6.4, 5.4))
    # plt.title("")
    plt.subplots_adjust(hspace=0.5, bottom=0.1, top=0.9)
    plt.subplot(3, 1, 1)
    plt.title("Moving from left to right in the back")
    plt.plot(time, corrected_errors[1, 0, :num_samples_to_plot].T, label="User 1")
    plt.plot(time, corrected_errors[1, 1, :num_samples_to_plot].T, label="User 2")
    plt.plot(time, corrected_errors[1, 2, :num_samples_to_plot].T, label="User 3")
    plt.plot(time, corrected_errors[1, 3, :num_samples_to_plot].T, label="User 4")
    plt.legend(ncol=4)
    plt.axis([0, time[-1], 0, 2000])
    plt.subplot(3, 1, 2)
    plt.title("Moving from left to right in the middle")
    plt.plot(time, corrected_errors[5, :, :num_samples_to_plot].T)
    plt.ylabel("Positioning error [mm]")
    plt.axis([0, time[-1], 0, 2000])
    plt.subplot(3, 1, 3)
    plt.plot(time, corrected_errors[4, :, :num_samples_to_plot].T)
    plt.title("Moving from left to right in the front")
    plt.xlabel("Time [s]")
    plt.axis([0, time[-1], 0, 2000])
    plt.savefig('plots/moving_comparison_' + name + '.png')
    plt.savefig('plots/moving_comparison_' + name + '.eps',
                bbox_inches='tight', pad_inches=0)

    print('Mean deviation:')
    print(mean_errors)
    print('std deviation:')
    print(std_errors)
