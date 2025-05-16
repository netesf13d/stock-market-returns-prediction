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
from sklearn.linear_model import LinearRegression, Lasso


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

# %%
"""


"""





