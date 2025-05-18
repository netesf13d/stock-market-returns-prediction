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
    Y_pred, Y_true : broadcastable (n+2)D np.ndarray of shape (..., N=504, n_stocks=50)
        DESCRIPTION.
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
    
    # data = np.empty((len(vectors)+len(norms), 2), dtype=float)
    # data[:, 0] = np.arange(len(data))
    # data[:len(vectors)] = vectors
    
    np.savetxt(fname, data, fmt=['%.0f', '%.18e'], delimiter=',',
               header=',0', comments='')
    

def array_to_csv(vectors: np.ndarray, fname: str)-> None:
    """
    Export 2D array `vectors` to submittable csv format.
    """
    norms = np.sqrt(np.sum(vectors**2, axis=1))
    vectors = (vectors / norms[:, np.newaxis]).ravel()
    
    data = np.concatenate([vectors, norms])
    data = np.stack([np.arange(len(data)), data], axis=1)
    
    np.savetxt(fname, data, fmt=['%.0f', '%.18e'], delimiter=',',
               header=',0', comments='')


def vector_to_csv(vector: np.ndarray, fname: str)-> None:
    """
    
    """
    norms = vector[:10]
    norms[-1] = np.sqrt(np.sum(vector[9:]**2))
    
    vectors = np.eye(10, 250, dtype=float)
    vectors[-1, 9:] = vector[9:] / norms[-1]
    
    data = np.concatenate([vectors, norms])
    data = np.stack([np.arange(len(data)), data], axis=1)
    
    np.savetxt(fname, data, fmt=['%.0f', '%.18e'], delimiter=',',
               header=',0', comments='')


def metric_gradient(past_returns: np.ndarray,
                    weights: np.ndarray,
                    norm_tgt_returns: np.ndarray)-> np.ndarray:
    """
    Compute the gradient of the metric.

    Parameters
    ----------
    past_returns : 3D np.ndarray of shape (N, n_stocks=50, depth=250)
        DESCRIPTION.
    weights : 1D np.ndarray of shape (depth=250,)
        DESCRIPTION.
    norm_tgt_returns : 2D np.ndarray of shape (N, n_stocks=50)
        DESCRIPTION.
    """
    pred_returns = np.sum(past_returns*weights, axis=-1)
    pred_returns_norm = np.sqrt(np.sum(pred_returns**2, axis=1, keepdims=True))
    
    grad1 = np.sum(past_returns * norm_tgt_returns[..., None], axis=1)
    grad1 = np.mean(grad1 / pred_returns_norm, axis=0)
    
    proj = np.sum(pred_returns * norm_tgt_returns, axis=1, keepdims=True)
    grad2 = np.sum(past_returns * pred_returns[..., None], axis=1)
    grad2 = np.mean(grad2 * proj / pred_returns_norm**2, axis=0)
    # print(pred_returns_norm.flatten())
        
    return grad1 - grad2



class SGDOptimizer():
    
    def __init__(self,
                 weights: np.ndarray,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 nesterov: bool = False):
        """
        TODO doc

        Parameters
        ----------
        weights : np.ndarray
            Initial weights.
        learning_rate : float, optional
            DESCRIPTION. The default is 0.01.
        momentum : float, optional
            DESCRIPTION. The default is 0.9.
        nesterov : bool, optional
            DESCRIPTION. The default is False.
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
## A single vector
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
fig4, ax4 = plt.subplots(
    nrows=1, ncols=1, sharey=True, figsize=(5.8, 3.8), dpi=100,
    gridspec_kw={'left': 0.12, 'right': 0.92, 'top': 0.83, 'bottom': 0.14})
fig4.suptitle("Figure 4: metric vs the regularization parameter",
              x=0.02, ha='left')


l_tr, = ax4.plot(alphas, tr_metric, linestyle='-', marker='')
l_val, = ax4.plot(alphas, val_metric, linestyle='-', marker='')
ax4.set_xscale('log')
ax4.set_xlim(1e-6, 1e-3)
ax4.set_xlabel('alpha')
ax4.set_ylim(-0.02, 0.15)
ax4.set_ylabel('Metric')
ax4.grid(visible=True, linewidth=0.3)
ax4.grid(visible=True, which='minor', linewidth=0.2)

fig4.legend(handles=[l_tr, l_val],
            labels=['Training set', 'Validation set'],
            ncols=2, loc=(0.27, 0.84), alignment='center')

plt.show()

"""
We get a score of about 0.03 with alpha = 4e-4, this is similar to the benchmark.
This score actually depends strongly on the cross-validation splits.
"""


# %%
"""
## Bruteforce v2

Having reduced the problem, we can now apply the bruteforce approach much more efficiently.
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
## Gradient descent

"""
w0 = rng.normal(size=depth)
w0 = w0 / np.sqrt(np.sum(w0**2)) 

## Set training parameters
n_epochs = 60
batch_size = 16
optimizer = SGDOptimizer(w0, learning_rate=0.05, momentum=0.0, nesterov=False)


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
        grad = -metric_gradient(X_, w, Y_)
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
            grad = -metric_gradient(X_, w, Y_)
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
The algorithm is clearly overfitting.
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
"""
## Add a regularity constraint


"""


def grad_l1_penalty(weights: np.ndarray,
                    penalty_factor: float,
                    )-> np.ndarray:
    """
    Compute the gradient of a L1 penalty: lambda * |w[i+1] - w[i]|.
    """
    diff_sign = np.diff(weights)
    l1_grad = np.zeros_like(weights, dtype=float)
    l1_grad[1:] = diff_sign
    l1_grad[:-1] -= diff_sign
    return penalty_factor*l1_grad


def grad_l2_penalty(weights: np.ndarray, penalty_factor: float)-> np.ndarray:
    """
    Compute the gradient of a L2 penalty: lambda * (w[i+1] - w[i])^2.
    """
    diff = np.diff(weights)
    l2_grad = np.zeros_like(weights, dtype=float)
    l2_grad[1:] = diff
    l2_grad[:-1] -= diff
    return 2*penalty_factor*l2_grad


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
        grad = -metric_gradient(X_, w, Y_) + grad_l1_penalty(w, reg_param)
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
            grad = -metric_gradient(X_, w, Y_) + grad_l1_penalty(w, reg_param)
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

