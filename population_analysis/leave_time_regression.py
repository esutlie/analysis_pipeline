import backend
import numpy as np
from create_bins_df import create_bins_df, create_precision_df
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def leave_time_regression():
    files = backend.get_session_list()
    for session in files:
        if session != 'ES029_2022-09-14_bot72_0_g0':
            continue
        session_df = create_bins_df(session)
        if type(session_df) == bool or session_df.spike_rates.iloc[0].size < 10:
            continue
        blocks = np.unique(session_df.block)
        blocks.sort()
        x_all = np.vstack(session_df.spike_rates.values)
        x_all_whiten, transform_function = backend.whiten(x_all)
        # unit_max = np.max(x_all, axis=0)
        # x_all_norm = x_all / unit_max
        y = session_df.time_from_reward.values
        # trial = session_df.trial.values
        pca_on_average_pre_whitened(x_all_whiten, session_df)
        # fit_pca(x_all_whiten, y)
        # fit_tsne(x_all_norm, y)
        # pca, transform_function, average_pca, time_bins = pca_on_average(session_df)
        # x_pca = pca.transform(transform_function(x_all))
        # fig = plt.figure(figsize=(12, 12))
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(average_pca[:, 0], average_pca[:, 1], c=time_bins, cmap="Blues_r")
        # ind = np.where((trial == 20) & (y < 2))[0]
        # ax.scatter(x_pca[ind, 0], x_pca[ind, 1], x_pca[ind, 2], c=y[ind], cmap="Reds_r")
        # plt.show()
        for block in blocks:
            # block_df = session_df[session_df.block == block]
            # x = np.vstack(block_df.spike_rates.values)
            # x_whiten = transform_function(x)
            x_block_whiten = x_all_whiten[session_df.block == block]
            # x_block_pca = pca.transform(transform_function(x))
            # x_norm = x / unit_max
            y_block = y[session_df.block == block]
            regressor = SVR(kernel='rbf')
            # cv = LeaveOneOut()
            # regressor = LinearRegression()
            # scores = cross_val_score(regressor, x_norm, y, scoring='neg_mean_absolute_error',
            #                          cv=cv, n_jobs=-1)
            # rmse = np.sqrt(np.mean(np.absolute(scores)))

            regressor.fit(x_block_whiten, y_block)
            session_df['predict_time_from_reward'] = regressor.predict(x_all_whiten)
            compare_plot(session_df, 'time_from_reward', 'predict_time_from_reward', title=f'{session}_{block}')


def compare_plot(session_df, key1, key2, mouse=None, title=None, save_plot=False):
    color_sets = backend.get_color_sets()
    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    sns.scatterplot(session_df, x=key1, y=key2, hue='block', palette='Set2')
    # sns.scatterplot(mouse_df[key1], mouse_df[key2], hue='block', data=mouse_df, palette='Set2')
    blocks = session_df.block.unique()
    blocks.sort()
    for i, block in enumerate(blocks):
        block_df = session_df[session_df.block == block]
        regression = LinearRegression()
        regression.fit(np.expand_dims(block_df[key1].values, axis=1), np.expand_dims(block_df[key2].values, axis=1))
        x = np.linspace(0, block_df[key1].max())
        y = regression.predict(np.expand_dims(x, axis=1))
        ax.plot(x, y, '-', c=color_sets['set2_med_dark'][i])
    ax.set_xlabel('Actual Time')
    ax.set_ylabel('Predicted Time')
    ax.set_aspect('equal', adjustable='box')
    if title:
        ax.set_title(title)
    plt.show()

    #
    # if save_plot:
    #     backend.save_fig(fig, f'{mouse}_compare_plot.png')
    # else:
    #     plt.show()


def pca_on_average(session_df):
    time_bins = range(round(session_df.time_from_reward.max() * 10))
    x_all_norm_mean = np.vstack([np.mean(np.vstack(session_df[(session_df.time_from_reward > t / 10) & (
            session_df.time_from_reward < t / 10 + .1)].spike_rates.values), axis=0) for t in time_bins])
    x_whitened, transform_function = backend.whiten(x_all_norm_mean, method='pca')
    pca = PCA(5)
    pca.fit(x_whitened)
    x_pca = pca.transform(x_whitened)
    plt.scatter(x_pca[:40, 0], x_pca[:40, 1], c=np.array(time_bins)[:40] / 10, cmap="plasma")
    plt.show()
    return pca, transform_function, x_pca[:40], np.array(time_bins)[:40] / 10


def pca_on_average_pre_whitened(x_whitened, session_df):
    """
    This function serves to run pca on the average spike rates between intervals
    :param x_whitened: feature matrix. Needs to already have been whitened
    :param session_df: Complete data matrix to to get bins from
    :return: None
    """
    time_bins = range(round(session_df.time_from_reward.max() * 10))
    x_whitened_mean = np.vstack([np.mean(np.vstack(x_whitened[(session_df.time_from_reward > t / 10) & (
            session_df.time_from_reward < t / 10 + .1)]), axis=0) for t in time_bins])
    pca = PCA(5)
    pca.fit(x_whitened_mean)
    x_pca = pca.transform(x_whitened_mean)
    plt.scatter(x_pca[:40, 0], x_pca[:40, 1], c=np.array(time_bins)[:40] / 10, cmap="plasma")
    plt.show()
    # return pca, transform_function, x_pca[:40], np.array(time_bins)[:40] / 10


def tsne_on_average(session_df):
    time_bins = range(round(session_df.time_from_reward.max() * 10))
    x_all_norm_mean = np.vstack([np.mean(np.vstack(session_df[(session_df.time_from_reward > t / 10) & (
            session_df.time_from_reward < t / 10 + .1)].spike_rates.values), axis=0) for t in time_bins])
    tsne = TSNE(3)
    x_tsne = tsne.fit_transform(x_all_norm_mean)
    plt.scatter(x_tsne[:40, 0], x_tsne[:40, 1], c=np.array(time_bins)[:40] / 10, cmap="plasma")
    plt.show()


def fit_pca(x, y):
    pca = PCA(5)
    pca.fit(x)
    x_pca = pca.transform(x)
    plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y, cmap="plasma")
    plt.show()


def fit_tsne(x, y):
    tsne = TSNE(3)
    x_tsne = tsne.fit_transform(x)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, cmap="plasma")
    plt.xlim([-30, 30])
    plt.ylim([-30, 20])
    plt.show()


if __name__ == '__main__':
    leave_time_regression()
    # linear models arent working so great, maybe we try bayesian somehow or a non linear regression
