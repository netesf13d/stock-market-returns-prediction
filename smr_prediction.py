# -*- coding: utf-8 -*-
"""
Noice
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import default_rng

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
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor


# =============================================================================
# 
# =============================================================================

# %% Data loading and preprocessing

stock_returns = np.loadtxt('./X_train.csv', delimiter=',', skiprows=1)[:, 1:]
# Y = np.loadtxt('./Y_train_wz11VM6.csv', delimiter=',', skiprows=1)

X = stock_returns


# %% EDA

"""
### Stock returns overview
"""

COLORS = [
    '#7e1e9c', '#15b01a', '#0343df', '#ff81c0', '#653700', '#e50000',
    '#95d0fc', '#029386', '#f97306', '#96f97b', '#c20078', '#ffff14',
    '#75bbfd', '#929591', '#0cff0c', '#bf77f6', '#9a0eea', '#033500',
    '#06c2ac', '#c79fef', '#00035b', '#d1b26f', '#00ffff', '#06470c',
    ]


fig1, axs1 = plt.subplots(
    nrows=1, ncols=2, sharey=False, figsize=(8, 3.8), dpi=100,
    gridspec_kw={'left': 0.09, 'right': 0.97, 'top': 0.88, 'bottom': 0.15, 'wspace': 0.28})
fig1.suptitle("Figure 1: Evolution of stock returns",
              x=0.02, ha='left')

# Returns
for i, sr in enumerate(stock_returns):
    axs1[0].plot(sr, linewidth=0.8, color=COLORS[(i+2)%24])

axs1[0].set_xlim(0, 800)
axs1[0].set_xlabel('Date (days)')
axs1[0].set_ylim(-0.25, 0.25)
axs1[0].set_yticks([-0.15, -0.05, 0.05, 0.15], minor=True)
axs1[0].set_ylabel('Daily return')
axs1[0].grid(visible=True, linewidth=0.3)
axs1[0].grid(visible=True, which='minor', linewidth=0.2)

# Cummulative returns
for i, sr in enumerate(np.cumsum(stock_returns, axis=1)):
    axs1[1].plot(sr, linewidth=0.7, color=COLORS[(i+2)%24])

axs1[1].set_xlim(0, 800)
axs1[1].set_xlabel('Date (days)')
axs1[1].set_ylim(-1.25, 1.25)
axs1[1].set_yticks([-1.25, -0.75, -0.25, 0.25, 0.75, 1.25], minor=True)
axs1[1].set_ylabel('Cummulative return')
axs1[1].grid(visible=True, linewidth=0.3)
axs1[1].grid(visible=True, which='minor', linewidth=0.2)


plt.show()

r"""
We plot in figure 1, for each stock, the returns variation over time (left panel) and the
accumulated values (right panel). Most values are between -0.05 and 0.05, yet we also note
the presence of extreme events with very large returns (both positive and negative).
We can see the price kick caused by some of these events on the right panel
(see for instance the green curve in the bottom right).
"""

#%%
"""
### Price returns distribution
"""

vals = np.logspace([-4], [0], 81)
pos_returns_cdf = np.mean(stock_returns.ravel() >= vals, axis=1)
neg_returns_cdf = np.mean(stock_returns.ravel() <= -vals, axis=1)


fig2, axs2 = plt.subplots(
    nrows=1, ncols=2, sharey=False, figsize=(8.2, 3.8), dpi=100,
    gridspec_kw={'left': 0.085, 'right': 0.97, 'top': 0.88, 'bottom': 0.14, 'wspace': 0.26})
fig2.suptitle("Figure 1: Price returns distribution",
              x=0.02, ha='left')

axs2[0].plot(100*np.mean(stock_returns, axis=1),
             100*np.std(stock_returns, axis=1),
         linestyle='', marker='o', markersize=4)

axs2[0].set_xlim(-0.16, 0.16)
axs2[0].set_xlabel('Average daily return (%)', labelpad=6)
axs2[0].set_ylim(0, 3)
axs2[0].set_ylabel('Std deviation of daily returns (%)', labelpad=7)
axs2[0].grid(visible=True, linewidth=0.3)

axs2[1].plot(vals, pos_returns_cdf, marker='+', linestyle='', markersize=4,
             label='Positive returns')
axs2[1].plot(vals, neg_returns_cdf, marker='+', linestyle='', markersize=4,
             label='Negative returns')
axs2[1].set_xscale('log')
axs2[1].set_xlim(1e-4, 3e-1)
axs2[1].set_yscale('log')
axs2[1].set_ylim(1e-5, 1)
axs2[1].grid(visible=True)
axs2[1].set_xlabel('Absolute daily return')
axs2[1].set_ylabel('Inv. cummulative distribution')
axs2[1].legend()


plt.show()

r"""
We present in figure 2 a scatter plot of the mean and standard deviation of returns
for each stock (left panel), and the inverse cummulative distribution of the returns (right panel).

The daily returns of each stock have zero mean, this indicates a preprocessing of the original values.
The standard deviations for the various stocks are in the range 0.5 - 2.5%, with most values between 0.5 and 1.5%.

The inverse cummulative distribution
corresponds to $P(R \geq x)$ (positive returns) and $P(R \leq -x)$ (negative returns). We plot the values
from the merged stocks. The global returns distribution is almost perfectly symmetric. Most of the returns
fall in the range 0.001-0.01. Extreme returns, larger than 0.1, are not uncommon, and occur with a probability
of about 1/2500. These extreme events can reach an amplitude of $10\sigma$,
which is virtually impossible for gaussian processes.
"""

# %%
"""
### Dependence of the returns on the week day

The daily stock returns correspond to working days. However, we could expect monday
stock returns to have larger variations than the other days. This would be due to
the fact that monday follows the week end, which provides a longer time for market-affecting
events to occur (therefore a higher probability of such events).

In the context of this project, the required convolutional model would be unable
to capture such periodicity in returns.
"""

##
k = 5
returns_std = np.std(stock_returns, axis=1)
returns_mod = [stock_returns[:, i::k] for i in range(k)]

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
Figure 3 presents the mean and standard deviations for each week day of the merged returns.
The data is scaled by the standard deviation across the whole dataset. There seems to be no incidence
of the week day on the returns. The given values certainly correspond to intra-day returns,
which by construction must be insensitive to inter-day variations.
"""

# %%
"""
### Correlations between stock prices
"""

fig4, ax4 = plt.subplots(
    nrows=1, ncols=1, figsize=(7, 6.4), dpi=100,
    gridspec_kw={'left': 0.03, 'right': 0.88, 'top': 0.89, 'bottom': 0.03, 'wspace': 0.24})
cax4 = fig4.add_axes((0.89, 0.06, 0.025, 0.8))
fig4.suptitle("Figure 4: Price returns correlation", x=0.02, ha='left')


ax4.set_aspect('equal')
cmap4 = plt.get_cmap('seismic').copy()
cmap4.set_extremes(under='0.9', over='0.5')
heatmap = ax4.pcolormesh(100*np.corrcoef(stock_returns)[::-1],
                         cmap=cmap4, vmin=-80, vmax=80,
                         edgecolors='0.2', linewidth=0.5)

ax4.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax4.set_xticks([0.5, 9.5, 19.5, 29.5, 39.5, 49.5], [1, 10, 20, 30, 40, 50])
ax4.set_yticks([49.5, 39.5, 29.5, 19.5, 9.5, 0.5], [1, 10, 20, 30, 40, 50])


pos = ax4.get_position().bounds
x, y = pos[0], pos[1] + pos[3]

fig4.colorbar(heatmap, cax=cax4, orientation="vertical", ticklocation="right")
cax4.text(1.4, 1.03, 'Corr.\ncoef. (%)', fontsize=11, ha='center', transform=cax4.transAxes)


plt.show()


"""
We present in figure 4 the correlation matrix of the prices returns for the 50 stocks of the dataset.
The stocks are rather positively correlated, with correlation coefficients as large as 75% for some pairs.
The negative correlations are less pronounced, extending to as low as -20%.
"""


# %% Utils
"""
## Utils

"""

def metric(Y_pred: np.ndarray, Y_true: np.ndarray)-> float:
    """
    Compute the cosine similarity between predicted and true returns.
    The cosine similarity is <Y_pred,Y_true> / (||Y_pred||*||Y_true||).

    Y_pred, Y_true : broadcastable (n+2)D np.ndarray of shape (..., N, n_stocks)
    """
    norm_pred = np.sqrt(np.sum(Y_pred**2, axis=-1))
    norm_true = np.sqrt(np.sum(Y_true**2, axis=-1))
    scalar_product = np.sum(Y_true*Y_pred, axis=-1)
    return np.mean(scalar_product/(norm_true*norm_pred), axis=-1)


def to_csv(vectors: np.ndarray, fname: str)-> None:
    """
    Export 2D array `vectors` to submittable csv format.
    """
    norms = np.sqrt(np.sum(vectors**2, axis=1))
    vectors = (vectors / norms[:, np.newaxis]).ravel()
    
    data = np.concatenate([vectors, norms])
    data = np.stack([np.arange(len(data)), data], axis=1)
    
    np.savetxt(fname, data, fmt=['%.0f', '%.18e'], delimiter=',',
               header=',0', comments='')


def vector_to_vectors(vector: np.ndarray, fname: str)-> np.ndarray:
    """
    Convert a single vector solution to the n_factors = 10 vectors for
    submission.
    """
    vectors = np.zeros(10, 250, dtype=float)
    vectors[np.arange(9), np.arange(9)] = vector[:9]
    vectors[-1, 9:] = vector[9:]
    return vectors


# =============================== Gradients ===================================

def grad_metric(X: np.ndarray,
                norm_Y: np.ndarray,
                weights: np.ndarray,)-> np.ndarray:
    """
    Gradient of the cosine similarity metric evaluated at `weights`.

    Parameters
    ----------
    X : 3D np.ndarray of shape (N, n_stocks=50, depth=250)
        Past returns, the features.
    norm_Y : 2D np.ndarray of shape (N, n_stocks=50)
        Normalized target vectors.
    weights : 1D np.ndarray of shape (depth=250,)
        Wzeights values at which the gradient is evaluated.
    """
    Y_pred = np.sum(X*weights, axis=-1)
    Y_pred_norm = np.sqrt(np.sum(Y_pred**2, axis=1, keepdims=True))
    
    grad1 = np.sum(X * norm_Y[..., None], axis=1)
    grad1 = np.mean(grad1 / Y_pred_norm, axis=0)
    
    proj = np.sum(Y_pred * norm_Y, axis=1, keepdims=True)
    grad2 = np.sum(X * Y_pred[..., None], axis=1)
    grad2 = np.mean(grad2 * proj / Y_pred_norm**2, axis=0)
        
    return grad1 - grad2


def grad_l1_diff(weights: np.ndarray,
                    penalty_factor: float,
                    )-> np.ndarray:
    """
    Gradient of a L1 penalty: lambda * |w[i+1] - w[i]|.
    """
    diff_sign = np.diff(weights)
    l1_grad = np.zeros_like(weights, dtype=float)
    l1_grad[1:] = diff_sign
    l1_grad[:-1] -= diff_sign
    return penalty_factor*l1_grad


def grad_l2_diff(weights: np.ndarray, penalty_factor: float)-> np.ndarray:
    """
    Gradient of a L2 penalty: lambda * (w[i+1] - w[i])^2.
    """
    diff = np.diff(weights)
    l2_grad = np.zeros_like(weights, dtype=float)
    l2_grad[1:] = diff
    l2_grad[:-1] -= diff
    return 2*penalty_factor*l2_grad


def grad_l1_absdiff(weights: np.ndarray,
                    penalty_factor: float,
                    )-> np.ndarray:
    """
    TODO
    Gradient of a L1 difference-of-absolute-values penalty:
        lambda * | |w[i+1]| - |w[i]| |.
    """
    diff_sign = np.diff(weights)
    l1_grad = np.zeros_like(weights, dtype=float)
    l1_grad[1:] = diff_sign
    l1_grad[:-1] -= diff_sign
    return penalty_factor*l1_grad


def grad_l2_absdiff(weights: np.ndarray, penalty_factor: float)-> np.ndarray:
    """
    TODO
    Gradient of a L2 difference-of-absolute-values penalty:
        lambda * ( |w[i+1]| - |w[i]| )^2.
    """
    diff = np.diff(weights)
    l2_grad = np.zeros_like(weights, dtype=float)
    l2_grad[1:] = diff
    l2_grad[:-1] -= diff
    return 2*penalty_factor*l2_grad


# ============================= Weight fitting ================================

class SGDOptimizer():
    """
    Custom implementation of a Nesterov stochastic gradient descent optimizer.
    """
    
    def __init__(self,
                 weights: np.ndarray,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 nesterov: bool = False):
        """
        Parameters
        ----------
        weights : np.ndarray
            Initial weights.
        learning_rate : float, optional
            Learning rate for momentum adjustment.
            The default is 0.01.
        momentum : float, optional
            The factor for momentum persistence across training steps.
            The default is 0.9.
        nesterov : bool, optional
            Enable Nesterov accelerated gradient method.
            The default is False.
        """
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.nesterov: bool = nesterov
        
        self.weights = weights
        self.momentum_vector = np.zeros_like(weights)
    
    
    def reset(self, new_weights: np.ndarray)-> None:
        """
        Reset the optimizer with new weights, and set the momentum to zero.
        """
        self.weights = new_weights
        self.momentum_vector = np.zeros_like(self.weights)
    
    
    def eval_point(self,)-> np.ndarray:
        """
        Return the evaluation point 
        """
        if self.nesterov:
            return self.weights + self.momentum * self.momentum_vector
        else:
            return self.weights
    
    
    def apply_gradients(self, gradients: np.ndarray):
        """
        Update the momentum and weights.
        - The radial component is removed from the momentum before updating
          the weights.
        - The weights are normalized to unit norm.
        1. m <- b * m - lr * grad
        2. m <- m - <m,w> * w
        3. w <- w + m
        4. w <- w / ||w||
        
        `gradients` must be 1D np.ndarray of the same shape as the weights.
        """
        m = (self.momentum * self.momentum_vector
             - self.learning_rate * gradients)
        m = m - np.sum(m*self.weights) * self.weights
        self.momentum_vector = m
        
        w = self.weights + self.momentum_vector
        self.weights = w / np.sqrt(np.sum(w**2))


def sgd_train():
    # TODO
    pass


def sgd_validation():
    # TODO
    pass


##
n_stocks = 50
n_factors = 10
depth = 250
n_steps = stock_returns.shape[1] - depth
X = np.stack([stock_returns[:, i:i+depth] for i in range(n_steps)])
Y = np.stack([stock_returns[:, i+depth] for i in range(n_steps)])


# %%
"""
## Problem reduction
"""

a = np.zeros(250, dtype=float)
a[0:5] = -0.00694934/np.sqrt(5)
a[20:250] = 0.01560642/np.sqrt(230)

Y_pred = np.sum(X*a[::-1], axis=-1)
print(metric(Y_pred, Y))



# %%
"""
## Regularized linear regression

The simplest approach to find a predictor as a linear combination of
past returns is to make a linear regression to find the corresponding weights.
However, the model must be strongly regularized in order to avoid overfitting.
Furthermore, the metric being invariant by a rescaling of the weights vectors (which is what ridge penalizes),
we retain the lasso regularization for the regression.

The targets only appear in the metric in normalized form, it is therefore natural
to normalize the vector prior to training. In theory, the features should not be normalized
(only the predicted vectors are normalized), however, it turns out that doing so improves the results.
"""

## normalize features and targets
YY = Y / np.sqrt(np.sum(Y**2, axis=1, keepdims=True))
XX = X / np.sqrt(np.sum(X**2, axis=1, keepdims=True))

## training metric
model = Lasso(alpha=1, fit_intercept=False)
alphas = np.concatenate([np.logspace(-6, -4, 11), np.logspace(-4, -3, 11)])
tr_metric = np.full_like(alphas, -1, dtype=float)
for i, alpha in enumerate(alphas):
    model.alpha = alpha
    model.fit(XX.reshape(-1, depth), YY.ravel())
    weights = model.coef_
    
    Y_pred = np.sum(X*weights, axis=-1)
    tr_metric[i] = metric(Y_pred, Y)

# %%
## validation metric
cv = KFold(n_splits=10, shuffle=True, random_state=1234)
val_metric = np.full_like(alphas, -1, dtype=float)
for i, alpha in enumerate(alphas):
    model.alpha = alpha
    Y_pred = np.zeros_like(Y, dtype=float)
    for itr, ival in cv.split(XX, YY):
        X_tr = XX[itr].reshape(-1, depth)
        Y_tr = YY[itr].ravel()
        model.fit(X_tr, Y_tr)
        X_v = XX[ival].reshape(-1, depth)
        Y_pred[ival] = model.predict(X_v).reshape(-1, n_stocks)
    val_metric[i] = metric(Y_pred, Y)


# %%

## plot
fig5, ax5 = plt.subplots(
    nrows=1, ncols=1, sharey=True, figsize=(5.8, 3.8), dpi=100,
    gridspec_kw={'left': 0.12, 'right': 0.92, 'top': 0.83, 'bottom': 0.14})
fig5.suptitle("Figure 4: Lasso regression results", x=0.02, ha='left')


l_tr, = ax5.plot(alphas, tr_metric, linestyle='-', marker='')
l_val, = ax5.plot(alphas, val_metric, linestyle='-', marker='')
ax5.set_xscale('log')
ax5.set_xlim(1e-6, 1e-3)
ax5.set_xlabel('alpha')
ax5.set_ylim(-0.02, 0.15)
ax5.set_ylabel('Metric')
ax5.grid(visible=True, linewidth=0.3)
ax5.grid(visible=True, which='minor', linewidth=0.2)

fig5.legend(handles=[l_tr, l_val],
            labels=['Training set', 'Validation set'],
            ncols=2, loc=(0.27, 0.84), alignment='center')

plt.show()

r"""
The model performs poorly on the validation set for $\alpha < 10^{-4}$.
However, the validation score increases past this threshold to reach about 0.025
at $\alpha = 3 \cdot 10^{-4}$, which is similar to the benchmark score.
Further increasing alpha yields a model with vanishing coefficients
(hence a division by zero in the metric computation).
However, the validation score actually depends strongly on the cross-validation splits.
"""

# %%
r"""
### Model interpretation

Let us study the weight vector found in the optimal case $\alpha = 3 \cdot 10^{-4}$.
"""

model = Lasso(alpha=3e-4, fit_intercept=False)
model.fit(XX.reshape(-1, depth), YY.ravel())
model.coef_


"""
The optimal weight vector found is very sparse! Only 19/250 coefficients are non-zero,
and the actual values and positions appear random. We obtained a random sparse vector,
which is quite unsatisfactory.
"""


# %%
"""
## Bruteforce v2

Having reduced the problem, we can now apply the bruteforce approach more efficiently.
"""

rng = default_rng(1234)


niter = 20
best_metric = -1
for i in range(niter):
    vectors = rng.normal(size=(10000, depth))
    Y_pred = np.tensordot(vectors, X, [[1], [2]])
    res = metric(Y_pred, Y)
    
    best_idx = np.argmax(res)
    if res[best_idx] > best_metric:
        best_metric = res[best_idx]
        best_vector = vectors[best_idx]
        print(f'iteration {i}, best metric :', best_metric)

print(best_metric)

"""
Not bad! We get results similar to the benchmark. Two differences though:
- We tried 200k solutions instead of 1000*10 vectors;
- Having the solution in 'merged' form, we could not fit individual contributions.

All in all, this approach seems to have roughly the same efficiency as that of the notebook.
What we gain in sampling speed we lose in optimization freedom.
"""

# %%
"""
## Random projection



"""

# !!!


# %% 
"""
## Gradient descent


With these results, we implemented a Nesterov stochastic gradient descent algorithm
(`SGDOptimizer` in `utils.py`) to find the optimal weights.
"""

w0 = rng.normal(size=depth) # best_vector
w0 = w0 / np.sqrt(np.sum(w0**2)) 

## Set training parameters
n_epochs = 60
batch_size = 16
optimizer = SGDOptimizer(w0, learning_rate=0.05, momentum=0.0, nesterov=False)


## Compute training metric
rng = default_rng(1234)
idx = np.arange(len(X))
tr_metric = np.zeros(n_epochs+1, dtype=float)

Y_pred = np.sum(X * w0, axis=-1)
tr_metric[0] = metric(Y_pred, Y)
print(f'epoch 0/{n_epochs} : train metric {tr_metric[0]:.4f}')

for i in range(n_epochs):
    rng.shuffle(idx)
    for j in range((len(idx)+batch_size-1) // batch_size):
        X_ = X[idx[j*batch_size:(j+1)*batch_size]]
        Y_ = YY[idx[j*batch_size:(j+1)*batch_size]]
        w = optimizer.eval_point()
        grad = -grad_metric(X_, Y_, w)
        optimizer.apply_gradients(grad)
    
    Y_pred = np.sum(X * optimizer.weights, axis=-1)
    tr_metric[i+1] = metric(Y_pred, Y)
    if i % 10 == 9:
        print(f'epoch {i+1}/{n_epochs} : train metric {tr_metric[i]:.4f}')
        
    
# %%

## Compute validation metric
rng = default_rng(1234)
cv = KFold(n_splits=4, shuffle=True, random_state=1234)

val_metric = np.zeros(n_epochs+1, dtype=float)
val_metric[0] = tr_metric[0]
Y_pred = np.zeros((n_epochs,) + Y.shape, dtype=float)

for itr, ival in cv.split(X, YY):
    optimizer.reset(w0)
    X_v = X[ival]
    for i in range(n_epochs):
        rng.shuffle(itr)
        for j in range((len(itr)+batch_size-1) // batch_size):
            X_ = X[itr[j*batch_size:(j+1)*batch_size]]
            Y_ = YY[itr[j*batch_size:(j+1)*batch_size]]
            w = optimizer.eval_point()
            grad = -grad_metric(X_, Y_, w)
            optimizer.apply_gradients(grad)
        Y_pred[i, ival] = np.sum(X_v * optimizer.weights, axis=-1)
val_metric[1:] = metric(Y_pred, Y)


# %%
## plot
fig5, ax5 = plt.subplots(
    nrows=1, ncols=1, sharey=True, figsize=(5.8, 3.8), dpi=100,
    gridspec_kw={'left': 0.125, 'right': 0.94, 'top': 0.83, 'bottom': 0.14})
fig5.suptitle("Figure 5: Metric evolution with SGD training",
              x=0.02, ha='left')


l_tr, = ax5.plot(np.arange(n_epochs+1), tr_metric, linestyle='-', marker='')
l_val, = ax5.plot(np.arange(n_epochs+1), val_metric, linestyle='-', marker='')

ax5.set_xlim(0, n_epochs)
ax5.set_xlabel('epoch')
ax5.set_ylim(-0.01, 0.16)
ax5.set_ylabel('Metric')
ax5.grid(visible=True, linewidth=0.3)

fig5.legend(handles=[l_tr, l_val],
            labels=['Training set', 'Validation set'],
            ncols=2, loc=(0.27, 0.84), alignment='center')

plt.show()

"""
The algorithm is clearly overfitting. The results are actually very similar to those
of the unregulated linear regression. We did not gain anything by implementing the
gradient descent.
"""

# %%
"""
## Using more constrained functions.

Does not work...
"""

X2 = np.empty((n_steps, n_stocks, 10), dtype=float)
X2[..., -3:] = X[..., -3:]
for i in range(1, 8):
    X2[..., -3-i] = np.sum(X[..., -1-2**(i+1):-1-2**i], axis=-1)


# %%
r"""
## A different regularization

When we consider the empirical factors $R_t^{(5)}$ and $R_{t-20}^{(230)}$ traditionally used,
we note that these are mostly constant. It is also not hard to believe that a version of
these factors with smoothed edges will perform equally well. Following this idea, we can require
the weights that we are looking for to be *smooth* functions of the time. This can be enforced
by penalizing the gradient $\nabla_t \mathbf{w}$. Concretely, we think of 4 penalty functions:
- $J_2(\mathbf{w}) = \lambda \sum_{t=0}^{D-1} (w_{t+1} - w_t)^2$, the L2 penalty on the gradient,
which corresponds to the kinetic term in field theories.
- $J_1(\mathbf{w}) = \lambda \sum_{t=0}^{D-1} | w_{t+1} - w_t |$, the L1 penalty on the gradient.
- $J_2^{\mathrm{abs}}(\mathbf{w}) = \lambda \sum_{t=0}^{D-1} (|w_{t+1}| - |w_t|)^2$,
the L2 penalty on the difference between absolute values. The rationale behind this penalty is
that we do not want to penalize factors that correspond to alternating returns ($+-+-+-+-$),
but still enfore a smooth envelope.
- $J_1^{\mathrm{abs}}(\mathbf{w}) = \lambda \sum_{t=0}^{D-1} |\, |w_{t+1}| - |w_t| \, |$, the L1 variant.

Fortunately, the algorithm that we implemented above offers enough flexibility to add
and implement these kind of regularization functions. Routines to evaluate their
gradient can be found in the `utils` module.
"""

# !!!




# %%

w0 = rng.normal(size=depth)
w0 = w0 / np.sqrt(np.sum(w0**2))

## Set training parameters
n_epochs = 100
batch_size = 16
reg_param = 10
optimizer = SGDOptimizer(w0, learning_rate=0.01, momentum=0.8, nesterov=True)


## Training metric
rng = default_rng(1234)
idx = np.arange(len(X))
tr_metric = np.zeros(n_epochs+1, dtype=float)

Y_pred = np.sum(X * w0, axis=-1)
tr_metric[0] = metric(Y_pred, Y)
print(f'epoch 0/{n_epochs} : train metric {tr_metric[0]:.4f}')

for i in range(n_epochs):
    rng.shuffle(idx)
    for j in range((len(idx)+batch_size-1) // batch_size):
        X_ = X[idx[j*batch_size:(j+1)*batch_size]]
        Y_ = YY[idx[j*batch_size:(j+1)*batch_size]]
        w = optimizer.eval_point()
        grad = -grad_metric(X_,Y_, w) + grad_l1_diff(w, reg_param)
        optimizer.apply_gradients(grad)
    
    Y_pred = np.sum(X * optimizer.weights, axis=-1)
    tr_metric[i+1] = metric(Y_pred, Y)
    if i % 10 == 9:
        print(f'epoch {i+1}/{n_epochs} : train metric {tr_metric[i]:.4f}')
        
    
# %%

## Validation metric
rng = default_rng(1234)
cv = KFold(n_splits=4, shuffle=True, random_state=1234)

val_metric = np.zeros(n_epochs+1, dtype=float)
val_metric[0] = tr_metric[0]
Y_pred = np.zeros((n_epochs,) + Y.shape, dtype=float)

for itr, ival in cv.split(X, YY):
    optimizer.reset(w0)
    X_v = X[ival]
    for i in range(n_epochs):
        rng.shuffle(itr)
        for j in range((len(itr)+batch_size-1) // batch_size):
            X_ = X[itr[j*batch_size:(j+1)*batch_size]]
            Y_ = YY[itr[j*batch_size:(j+1)*batch_size]]
            w = optimizer.eval_point()
            grad = -grad_metric(X_, Y_, w) + grad_l1_diff(w, reg_param)
            optimizer.apply_gradients(grad)
        Y_pred[i, ival] = np.sum(X_v * optimizer.weights, axis=-1)
val_metric[1:] = metric(Y_pred, Y)


# %%
## plot
fig5, ax5 = plt.subplots(
    nrows=1, ncols=1, sharey=True, figsize=(5.8, 3.8), dpi=100,
    gridspec_kw={'left': 0.125, 'right': 0.94, 'top': 0.83, 'bottom': 0.14})
fig5.suptitle("Figure 5: Metric evolution with SGD training",
              x=0.02, ha='left')


l_tr, = ax5.plot(np.arange(n_epochs+1), tr_metric, linestyle='-', marker='')
l_val, = ax5.plot(np.arange(n_epochs+1), val_metric, linestyle='-', marker='')

ax5.set_xlim(0, n_epochs)
ax5.set_xlabel('epoch')
ax5.set_ylim(-0.01, 0.16)
ax5.set_ylabel('Metric')
ax5.grid(visible=True, linewidth=0.3)

fig5.legend(handles=[l_tr, l_val],
            labels=['Training set', 'Validation set'],
            ncols=2, loc=(0.27, 0.84), alignment='center')

plt.show()



# %%

from utils import to_csv