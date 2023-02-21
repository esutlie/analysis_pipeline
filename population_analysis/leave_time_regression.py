import backend
import numpy as np
from create_bins_df import create_bins_df
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


def leave_time_regression():
    files = backend.get_session_list()
    to_predict = ['time_from_entry', 'time_from_reward', 'time_before_exit', 'time_in_session']
    for session in files:
        session_df = create_bins_df(session)
        if type(session_df) == bool or session_df.spike_rates.iloc[0].size < 10:
            continue
        blocks = np.unique(session_df.block)
        blocks.sort()
        x_all = np.vstack(session_df.spike_rates.values)
        for block in blocks:
            block_df = session_df[session_df.block == block]
            x = np.vstack(block_df.spike_rates.values)
            y = block_df.time_from_reward.values
            cv = LeaveOneOut()
            model = LinearRegression()
            scores = cross_val_score(model, x, y, scoring='neg_mean_absolute_error',
                                     cv=cv, n_jobs=-1)
            rmse = np.sqrt(np.mean(np.absolute(scores)))
            fit_model = model.fit(x, y)
            session_df['predict_time_from_reward'] = fit_model.predict(x_all)
            compare_plot(session_df, 'time_from_reward', 'predict_time_from_reward', title=f'{session}_{block}')


def compare_plot(mouse_df, key1, key2, mouse=None, title=None, save_plot=False):
    color_sets = backend.get_color_sets()
    fig, ax = plt.subplots(1, 1, figsize=[8, 6])
    sns.scatterplot(mouse_df[key1], mouse_df[key2], hue='block', data=mouse_df, palette='Set2')
    blocks = mouse_df.block.unique()
    blocks.sort()
    for i, block in enumerate(blocks):
        block_df = mouse_df[mouse_df.block == block]
        regression = LinearRegression()
        regression.fit(np.expand_dims(block_df[key1].values, axis=1), np.expand_dims(block_df[key2].values, axis=1))
        x = np.linspace(0, block_df[key1].max())
        y = regression.predict(np.expand_dims(x, axis=1))
        ax.plot(x, y, '-', c=color_sets['set2_med_dark'][i])
    ax.set_xlabel('Actual Time')
    ax.set_ylabel('Predicted Time')
    if title:
        ax.set_title(title)
    plt.show()

    #
    # if save_plot:
    #     backend.save_fig(fig, f'{mouse}_compare_plot.png')
    # else:
    #     plt.show()


if __name__ == '__main__':
    leave_time_regression()
    # linear models arent working so great, maybe we try bayesian somehow or a non linear regression
