import numpy as np
import random
import math
import matplotlib.pyplot as plt
import backend


class NearestLines:
    def __init__(self, x, k=3):
        self.color_set = backend.get_color_sets()['set2']
        self.color_set_dark = backend.get_color_sets()['set2_med_dark']
        self.max_iter = 50
        self.x = x
        self.k = k
        self.cluster_lines = []
        self.assignments = np.zeros([len(x)]).astype(int)
        self.distances = []

        self.history = []
        self.variance = []

    def reset(self):
        self.cluster_lines = []
        self.assignments = np.zeros([len(self.x)]).astype(int)
        self.distances = []
        self.history = []
        self.variance = []

    def elbow(self, k_max=6):
        elbow_variance = []
        for k in range(2, k_max):
            self.k = k
            assignments, variance = self.fit()
            elbow_variance.append(variance)
            self.reset()
        elbow_variance_sums = [np.sum(l) for l in elbow_variance]
        plt.plot(range(2, k_max), elbow_variance_sums)
        plt.show()

    def fit(self):
        ind = list(range(len(self.x)))
        for i in range(self.max_iter):
            selected = random.sample(ind, self.k)
            self.cluster_lines = [self.x[select] for select in selected]
            self.distances = np.zeros([len(self.x), len(self.cluster_lines)])
            # self.plot()
            while True:
                new_assignments = self.cluster()
                if np.all(new_assignments == self.assignments):
                    # self.plot()
                    self.variance.append(
                        [np.sum([self.distances[ind, j] ** 2 for ind in np.where(j == self.assignments)[0]]) for j in
                         range(self.k)])
                    self.history.append(new_assignments)
                    break
                self.assignments = new_assignments
                # self.plot()
                for j in range(len(self.cluster_lines)):
                    self.cluster_lines = [np.mean(np.stack([self.x[ind] for ind in np.where(j == self.assignments)[0]]),
                                                  axis=0) for j in range(self.k)]
                # self.plot()
        self.plot_variance()
        select = np.argmin(np.sum(np.array(self.variance), axis=1))
        return self.history[select], self.variance[select]

    def cluster(self):
        return np.array([np.argmin(row) for row in self.measure()])

    def measure(self):
        for i in range(len(self.x)):
            for j in range(len(self.cluster_lines)):
                self.distances[i, j] = np.sum(np.sqrt(np.sum((self.x[i] - self.cluster_lines[j]) ** 2, axis=0)))
        return self.distances

    def plot(self):
        colors = [self.color_set[c] for c in self.assignments]
        for j, c in enumerate(colors):
            plt.plot(self.x[j, 0].T, self.x[j, 1].T, c=c)
        for j in range(self.k):
            plt.plot(self.cluster_lines[j][0].T, self.cluster_lines[j][1].T, linewidth=3,
                     c=self.color_set_dark[j])
        plt.show()

    def plot_variance(self):
        fig, ax = plt.subplots()
        bottom = np.zeros(len(self.variance))
        # if self.k == 1:
        #     variance = np.expand_dims(np.array([l[0] for l in self.variance]), axis=1)
        # else:
        variance = np.array(self.variance)
        ax.bar(np.arange(len(self.variance)), variance[:, 0], .5, bottom=bottom)
        for i in range(self.k - 1):
            bottom += variance[:, i]
            ax.bar(np.arange(len(self.variance)), variance[:, i + 1], .5, bottom=bottom)
        plt.show()


if __name__ == '__main__':
    sim_x = np.random.random([20, 5, 10])
    base_x = np.arange(10) / 2
    base_y = np.exp(base_x / 2)
    sim_x = np.array([sim_x[:, i] + base for i, base in enumerate([base_x, base_y, base_x, base_y, base_x])])
    sim_x = np.swapaxes(sim_x, 0, 1)
    for i in range(len(sim_x)):
        sim_x[i, 1] = sim_x[i, 1] + i / 2
    clusters = NearestLines(sim_x).fit()
