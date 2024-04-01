
import numpy as np
import matplotlib.pyplot as plt


def weighted_regression(X, y):


    a = [0,1,2,0,1,2]
    b=[]
    # solution of linear regression
    w_lr = np.linalg.inv(X.T @ X) @ X.T @ y

    # calculate residuals
    res = y - X @ w_lr

    # estimate the covariance matrix
    C = np.diag(res**2)

    # solution of weighted linear regression
    w_wlr = np.linalg.inv(X.T @ np.linalg.inv(C) @ X) @ (X.T @ np.linalg.inv(C) @ y)

    # generate the feature set for plotting
    X_p = np.c_[np.ones(2), np.linspace(X0.min(), X0.max(), 2)]

    # plot the results
    plt.plot(X0, y, 'b.', label='Observations')
    plt.plot(X_p[:,1], X_p @ w_lr, 'r-', label='Linear Regression')
    plt.plot(X_p[:,1], X_p @ w_wlr, 'g-', label='Weighted Linear Regression')
    plt.plot(X_p[:,1], X_p @ [intercept, slope], 'm--', label='Actual Regression')
    plt.grid(linestyle=':')
    plt.ylabel('Response')
    plt.xlabel('Feature')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    weighted_regression()
