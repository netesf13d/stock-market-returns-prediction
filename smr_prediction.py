# -*- coding: utf-8 -*-
"""
Noice
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (mean_squared_error,
                             r2_score,
                             mean_absolute_error,
                             mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   FunctionTransformer)
from sklearn.linear_model import LinearRegression, Lasso, Ridge


# =============================================================================
# 
# =============================================================================

# %% Data loading and preprocessing

stock_returns = np.loadtxt('./X_train_YG7NZSq.csv', delimiter=',', skiprows=1)[:, 1:]
# Y = np.loadtxt('./Y_train_wz11VM6.csv', delimiter=',', skiprows=1)

X = stock_returns


# %% EDA
"""
### Distribution of means and variance
"""

fig1, ax1 = plt.subplots(
    nrows=1, ncols=1, sharey=False, figsize=(5.4, 4.), dpi=100,
    gridspec_kw={'left': 0.15, 'right': 0.96, 'top': 0.88, 'bottom': 0.13, 'wspace': 0.24})
fig1.suptitle("Figure 1: Returns mean and standard deviation",
              x=0.02, ha='left')

ax1.plot(np.mean(stock_returns, axis=1), np.std(stock_returns, axis=1),
         linestyle='', marker='o', markersize=4)

ax1.set_xlim(-0.0016, 0.0016)
ax1.set_xlabel('Average daily return', labelpad=6)
ax1.set_ylim(0, 0.03)
ax1.set_ylabel('Std deviation of daily returns', labelpad=7)
ax1.grid(visible=True, linewidth=0.3)


plt.show()

"""
The daily returns are centered: the mean has probably been substracted to the original values.
"""


# %%
"""
### Evolution of the stock returns
"""

COLORS = [
    '#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000',
    '#95d0fc', '#029386', '#f97306', '#96f97b', '#c20078', '#ffff14',
    '#75bbfd', '#929591', '#0cff0c', '#bf77f6', '#9a0eea', '#033500',
    '#06c2ac', '#c79fef', '#00035b', '#d1b26f', '#00ffff', '#06470c',
    ]


fig2, axs2 = plt.subplots(
    nrows=1, ncols=2, sharey=False, figsize=(8, 3.8), dpi=100,
    gridspec_kw={'left': 0.09, 'right': 0.97, 'top': 0.88, 'bottom': 0.15, 'wspace': 0.28})
fig2.suptitle("Figure 2: Evolution of stock returns",
              x=0.02, ha='left')

# Returns
for i, sr in enumerate(stock_returns):
    axs2[0].plot(sr, linewidth=0.8, color=COLORS[(i+2)%24])

axs2[0].set_xlim(0, 800)
axs2[0].set_xlabel('Date (days)')
axs2[0].set_ylim(-0.25, 0.25)
axs2[0].set_yticks([-0.15, -0.05, 0.05, 0.15], minor=True)
axs2[0].set_ylabel('Daily return')
axs2[0].grid(visible=True, linewidth=0.3)
axs2[0].grid(visible=True, which='minor', linewidth=0.2)

# Cummulative returns
for i, sr in enumerate(np.cumsum(stock_returns, axis=1)):
    axs2[1].plot(sr, linewidth=0.7, color=COLORS[(i+2)%24])

axs2[1].set_xlim(0, 800)
axs2[1].set_xlabel('Date (days)')
axs2[1].set_ylim(-1.25, 1.25)
axs2[1].set_yticks([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25], minor=True)
axs2[1].set_ylabel('Cummulative return')
axs2[1].grid(visible=True, linewidth=0.3)
axs2[1].grid(visible=True, which='minor', linewidth=0.2)


plt.show()

r"""

Most valmues are between -0.05 and 0.05. We also note the presence of extreme events with
very large returns (about $10\sigma$, which is virtually impossible for gaussian processes).
"""

# %%
"""
### Dependence of the returns on the week day
"""

##
k = 5
returns_std = np.std(stock_returns, axis=1)
returns_mod = [X[:, i::k] for i in range(k)]

day_means = [np.mean(np.mean(x, axis=1) / returns_std) for x in returns_mod]
day_stds = [np.mean(np.std(x, axis=1) / returns_std) for x in returns_mod]


##
fig3, ax3 = plt.subplots(
    nrows=1, ncols=1, sharey=True, figsize=(5.4, 3.8), dpi=100,
    gridspec_kw={'left': 0.16, 'right': 0.84, 'top': 0.89, 'bottom': 0.125})
fig3.suptitle("Figure 3: week day dependence of returns",
              x=0.02, ha='left')

# day means
l_mean, = ax3.plot(day_means, linestyle='', marker='o')
ax3.set_xlim(-0.5, 4.5)
ax3.set_xlabel('Day')
ax3.set_ylim(-0.03, 0.03)
# ax3.set_yticks([-0.15, -0.05, 0.05, 0.15], minor=True)
ax3.set_ylabel('Scaled mean return')
ax3.grid(visible=True, axis='y', linewidth=0.3)

# day std deviations
ax3_twin = ax3.twinx()
l_std, = ax3_twin.plot(day_stds, linestyle='', marker='o', color='tab:orange')
ax3_twin.set_xlim(-0.5, 4.5)
ax3_twin.set_xlabel('Day')
ax3_twin.set_ylim(0.97, 1.03)
ax3_twin.set_ylabel('Scaled return std deviation', labelpad=8)


fig3.legend(handles=[l_mean, l_std],
            labels=['mean returns', 'std. dev. of returns'],
            ncols=1, loc=(0.495, 0.755), alignment='center')

plt.show()


"""
No effect
Our convolutional model would be unable to capture such periodicity in returns.
"""

# %% Utils
"""
## Utils

"""

def metric(Y_pred: np.ndarray, Y_true: np.ndarray)-> float:
    """
    TODO doc

    Parameters
    ----------
    Y_pred, Y_true : 2D np.ndarray of shape (N, depth)
        DESCRIPTION.
    """
    norm_pred = np.sqrt(np.sum(Y_pred**2, axis=1))
    norm_true = np.sqrt(np.sum(Y_true**2, axis=1))
    return np.mean(np.sum(Y_true*Y_pred, axis=1)/(norm_true*norm_pred))


def to_csv(vectors: np.ndarray, fname: str)-> None:
    """
    Export 2D array `vectors` to submittable csv format.
    """
    norms = np.sqrt(np.sum(vectors**2, axis=1))
    vectors = (vectors / norms[:, np.newaxis]).ravel()
    
    data = np.concatenate([vectors, norms])
    data = np.stack([np.arange(len(data)), data], axis=1)
    
    # data = np.empty((len(vectors)+len(norms), 2), dtype=float)
    # data[:, 0] = np.arange(len(data))
    # data[:len(vectors)] = vectors
    
    np.savetxt(fname, data, fmt=['%.0f', '%.18e'], delimiter=',',
               header=',0', comments='')
    


##
n_stocks = 50
n_vectors = 10
depth = 250
N = stock_returns.shape[1] - depth
X = np.stack([stock_returns[:, i:i+depth] for i in range(N)])
Y = np.stack([stock_returns[:, i+depth] for i in range(N)])


# %%
"""
## A single vector

!!! calculs pour montrer que ça correspond plus ou moins à du ridge
"""

##
model = Ridge(alpha=1, fit_intercept=False)
cv = KFold(n_splits=5, shuffle=True, random_state=1234)
alphas = np.logspace(-4, 4, 28)
metrics = np.full_like(alphas, -1, dtype=float)
for i, alpha in enumerate(alphas):
    model.alpha = alpha
    Y_pred = np.zeros_like(Y, dtype=float)
    for itr, ival in cv.split(X, Y):
        X_tr = X[itr].reshape(-1, depth)
        Y_tr = Y[itr].ravel()
        model.fit(X_tr, Y_tr)
        
        X_v = X[ival].reshape(-1, depth)
        Y_pred[ival] = model.predict(X_v).reshape(-1, n_stocks)
    metrics[i] = metric(Y_pred, Y)


# %%

## plot
fig4, ax4 = plt.subplots(
    nrows=1, ncols=1, sharey=True, figsize=(5.8, 3.8), dpi=100,
    gridspec_kw={'left': 0.16, 'right': 0.92, 'top': 0.88, 'bottom': 0.14})
fig4.suptitle("Figure 4: metric vs the regularization parameter",
              x=0.02, ha='left')

# day means
ax4.plot(alphas, metrics, linestyle='-', marker='')
ax4.set_xscale('log')
ax4.set_xlim(1e-4, 1e4)
ax4.set_xlabel('alpha')
ax4.set_ylim(-0.014, -0.009)
# ax4.set_yticks([-0.15, -0.05, 0.05, 0.15], minor=True)
ax4.set_ylabel('Metric')
ax4.grid(visible=True, linewidth=0.3)

plt.show()

"""
WXe could select alpha = 10. Unfortunately, it is not possible to submit
zero vectors.
"""


# %% 
"""
## Dimensionality reduction


"""

XX = X.reshape(len(X), -1)
# YY = 















# %%

alpha = 10
model = Ridge(alpha=alpha, fit_intercept=False)
model.fit(X.reshape(-1, depth), Y.ravel())

vectors = np.ones((10, depth), dtype=float)
vectors[0] = model.coef_

norms = np.sqrt(np.sum(vectors**2, axis=1))
vectors[0] /= norms[0]

to_csv(vectors, 'test.csv')



# %%



vectors = np.empty((n_vectors, depth), dtype=float)
XX = X_tr.reshape(-1, depth)
YY = Y_tr.ravel()
for i in range(n_vectors):
    ## get a vector factor
    model = Ridge(alpha=1, fit_intercept=False)
    model.fit(XX, YY)
    vect = model.coef_
    vect_norm = np.sqrt(np.sum(vect**2))
    
    vectors[i] = vect
    ## substract the component to features and targets
    overlap = np.sum(XX * vect, axis=1, keepdims=True)
    XX -= overlap * vect / vect_norm**2
    YY -= overlap.ravel()
    
    print(vect_norm)
    print(overlap)
    



# a = model.coef_

# %%

Y_pred = model.predict(X_val.reshape(-1, depth)).reshape(X_val.shape[:-1])

Y_pred_norm = np.sqrt(np.sum(Y_pred**2, axis=1))
Y_val_norm = np.sqrt(np.sum(Y_val**2, axis=1))

mean_overlap = np.mean(np.sum(Y_val*Y_pred, axis=1)/(Y_pred_norm*Y_val_norm))

print(mean_overlap)















