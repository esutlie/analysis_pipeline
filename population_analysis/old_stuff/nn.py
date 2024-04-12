from create_bins_df import create_precision_df
import backend
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import metrics
from population_analysis.old_stuff.isomap import get_phase, get_block
from thread_function import get_x
from sklearn.model_selection import train_test_split


def main():
    files = backend.get_session_list()
    for session in files:
        [normalized_spikes, convolved_spikes, _, original_spikes], interval_ids, intervals_df = create_precision_df(
            session, regenerate=False)
        if normalized_spikes is None:
            continue
        blocks = intervals_df.block.unique()
        blocks.sort()
        phases = get_phase(interval_ids, intervals_df)
        phase_filter = np.where((phases == 1) | (phases == 2))[0]
        normalized_spikes = normalized_spikes[:, phase_filter]
        interval_ids = interval_ids[phase_filter]
        block = get_block(interval_ids, intervals_df)
        high_fr = np.where(np.mean(original_spikes, axis=1) * 1000 > 1)
        times, longest = get_x(interval_ids)
        data = normalized_spikes[high_fr]

        b0 = block == blocks[0]
        b1 = block == blocks[1]
        time_net_b0 = time_nn(data[:, b0], times[b0])
        time_net_b1 = time_nn(data[:, b1], times[b1])
        pretty_plot_single_block([time_net_b0, time_net_b1], [b0, b1], times, data, session, blocks)

        # block_net = block_nn(np.hstack(data, times), block)


def time_nn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y, test_size=0.33, random_state=42)
    model_nn = Sequential()
    model_nn.add(Dense(12, input_shape=(len(X),), activation='relu'))
    model_nn.add(Dense(8, activation='relu'))
    model_nn.add(Dense(1, activation="linear"))
    model_nn.compile(loss='mean_squared_error', optimizer='adam',
                     metrics=[metrics.MeanSquaredError()])
    model_nn.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test))
    return model_nn


def block_nn(X, y):  # change to binary classifier
    X_train, X_test, y_train, y_test = train_test_split(
        X.T, y, test_size=0.33, random_state=42)
    model_nn = Sequential()
    model_nn.add(Dense(12, input_shape=(len(X),), activation='relu'))
    model_nn.add(Dense(8, activation='relu'))
    model_nn.add(Dense(1, activation="linear"))
    model_nn.compile(loss='mean_squared_error', optimizer='adam',
                     metrics=[metrics.MeanSquaredError()])
    model_nn.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test))
    return model_nn


def pretty_plot_single_block(models, booleans, times, data, title, blocks):
    color_sets = backend.get_color_sets()
    for i, model_i in enumerate(models):
        fig, ax = plt.subplots()
        for j, b in enumerate(booleans):
            prediction = model_i.predict(data[:, b].T)
            mean, std, t = get_mean(prediction, times[b])
            ax.plot(t, mean, c=color_sets['set2'][j])
        ax.set_title(f'{title} fit to {blocks[i]}')
        ax.legend(blocks)
        backend.save_fig(fig, f'{title}_b{i}', sub_folder='nns')


def get_mean(pred, x):
    times = np.unique(x)
    times.sort()
    mean_neural_time = np.zeros(len(times))
    std_neural_time = np.zeros(len(times))
    for j, x_i in enumerate(times):
        mean_neural_time[j] = np.mean(pred[np.where(x == x_i)[0]])
        std_neural_time[j] = np.std(pred[np.where(x == x_i)[0]])
    return mean_neural_time, std_neural_time, times


def pretty_plot(interval_ids, intervals_df, x, prediction):
    color_sets = backend.get_color_sets()
    blocks = intervals_df.block.unique()
    blocks.sort()
    phases = get_phase(interval_ids, intervals_df)
    x_blocks = get_block(interval_ids, intervals_df)
    phase_filter = np.where((phases == 1) | (phases == 2))[0]
    for c, block in enumerate(blocks):
        i = x_blocks == block
        plt.scatter(x[i], prediction[i], s=1, c=color_sets['set2'][c])
    plt.xlabel('real time')
    plt.ylabel('predicted time')
    plt.show()
    plt.plot()
    y_max = []
    for c, block in enumerate(blocks):
        i = x_blocks == block
        block_predictions = prediction[i]
        block_x = x[i]
        times = np.unique(block_x)
        times.sort()
        plt.plot(times, times, c=[.8, .8, .8])
        mean_neural_time = np.zeros(len(times))
        std_neural_time = np.zeros(len(times))
        for j, x_i in enumerate(times):
            mean_neural_time[j] = np.mean(block_predictions[np.where(block_x == x_i)[0]])
            std_neural_time[j] = np.std(block_predictions[np.where(block_x == x_i)[0]])
        plt.plot(times, mean_neural_time, c=color_sets['set2'][c])
        y_max.append(max(mean_neural_time))
    plt.xlabel('real time')
    plt.ylabel('predicted time')
    plt.legend(['true'] + list(blocks))
    plt.ylim([-.3, max(y_max) + .3])
    plt.show()


if __name__ == '__main__':
    main()
