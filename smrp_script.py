# -*- coding: utf-8 -*-
"""
Python script version of the stock market prediction challenge.

TODO
- SGD training with multiprocessing
- convergence vs reg and init for L2 SGD
"""

import time
from copy import deepcopy
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso


###############################################################################
# %% Data loading and preprocessing

stock_returns = np.loadtxt('./X_train.csv', delimiter=',', skiprows=1)[:, 1:]


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
    gridspec_kw={'left': 0.03, 'right': 0.88, 'top': 0.89, 'bottom': 0.03})
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


def to_csv(vectors: np.ndarray, fname: str)-> np.ndarray:
    """
    Export 2D array `vectors` to submittable csv format.
    """
    norms = np.sqrt(np.sum(vectors**2, axis=1))
    vectors = (vectors / norms[:, np.newaxis]).T.ravel()

    data = np.concatenate([vectors, norms])
    data = np.stack([np.arange(len(data)), data], axis=1)

    np.savetxt(fname, data, fmt=['%.0f', '%.18e'], delimiter=',',
               header=',0', comments='')
    return data


def vector_to_vectors(vector: np.ndarray)-> np.ndarray:
    """
    Convert a single vector solution to the n_factors = 10 vectors for
    submission.
    """
    vectors = np.zeros((10, 250), dtype=float)
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
    Y_pred = X @ weights
    Y_pred_norm = np.sqrt(np.sum(Y_pred**2, axis=1, keepdims=True))
    Y_pred_norm = np.where(Y_pred_norm==0, 1, Y_pred_norm)

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
    diff_sign = np.sign(np.diff(weights))
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
    Gradient of a L1 difference-of-absolute-values penalty:
        lambda * | |w[i+1]| - |w[i]| |.
    """
    sign = np.sign(weights)
    sign_diff_abs = np.sign(np.diff(np.abs(weights)))
    l1_grad = np.zeros_like(weights, dtype=float)
    l1_grad[1:] = sign_diff_abs * sign[1:]
    l1_grad[:-1] -= sign_diff_abs * sign[:-1]
    return penalty_factor*l1_grad


def grad_l2_absdiff(weights: np.ndarray, penalty_factor: float)-> np.ndarray:
    """
    Gradient of a L2 difference-of-absolute-values penalty:
        lambda * ( |w[i+1]| - |w[i]| )^2.
    """
    sign = np.sign(weights)
    diff = np.diff(weights)
    l2_grad = np.zeros_like(weights, dtype=float)
    l2_grad[1:] = diff * sign[1:]
    l2_grad[:-1] -= diff * sign[:-1]
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
        w_norm = np.sqrt(np.sum(w**2))
        self.weights = w / w_norm if w_norm > 0 else w


def sgd_train(X: np.ndarray,
              norm_Y: np.ndarray,
              optimizer: SGDOptimizer,
              gradient_func: Callable,
              batch_size: int = 16,
              n_epochs: int = 200,
              random_state: int | None = None,
              early_stopping: bool = False,
              early_stop_patience: int = 0,
              early_stop_min_delta: float = 0,
              return_best_weights: bool = False,
              verbose: bool = False,
              )-> tuple[np.ndarray, float, int]:
    """
    Wrapper function for stochastic gradient descent training.

    Parameters
    ----------
    X : 3D np.ndarray of shape (N, n_stocks=50, depth=250)
        Past returns, the features.
    norm_Y : 2D np.ndarray of shape (N, n_stocks=50)
        Normalized target vectors.
    optimizer : SGDOptimizer
        The SGD opimizer instance.
    gradient_func : Callable
        A function (X, Y, weights) -> 1D np.ndarray of shape (depth,).
        Gives the gradients of the loss function at the given weight values.
    batch_size : int, optional
        Size of the SGD batches. The default is 16.
    n_epochs : int, optional
        Max number of training epochs. The default is 200.
    random_state : int | None, optional
        Random state for training data shuffling. The default is None.
    early_stopping : bool, optional
        Enable early stopping. The default is False.
    early_stop_patience : int, optional
        Number of epochs with no improvement after which training will be
        stopped. The default is 0.
    early_stop_min_delta : float, optional
        Minimum change in the metric to qualify as an improvement, i.e. an
        absolute change of less than min_delta, will count as no improvement.
        The default is 16.
    return_best_weights : bool, optional
        Whether to return model weights from the epoch with the best value of
        the metric. If False, the model weights obtained at the last step of
        training are returned. The default is False.
    verbose : bool, optional
        Displays the metric after each epoch.
        The default is False.

    Returns
    -------
    weights : 1D np.ndarray of shape (depth,)
        The optimized model weights.
    metric : float
        The corresponding metric.
    epoch : int
        The last training epoch number.

    """
    rng = default_rng(random_state)
    idx = np.arange(len(X))

    best_weights = optimizer.weights
    best_metric = metric(np.sum(X * optimizer.weights, axis=-1), norm_Y)
    best_epoch = 0
    last_metric = best_metric
    last_epoch = 0

    for epoch in range(n_epochs):
        rng.shuffle(idx)

        # epoch train
        for j in range((len(idx)+batch_size-1) // batch_size):
            X_ = X[idx[j*batch_size:(j+1)*batch_size]]
            Y_ = norm_Y[idx[j*batch_size:(j+1)*batch_size]]
            w = optimizer.eval_point()
            optimizer.apply_gradients(gradient_func(X_,Y_, w))

        # update best weights and metric
        curr_metric = metric(np.sum(X * optimizer.weights, axis=-1), norm_Y)
        if return_best_weights and (curr_metric > best_metric):
            best_weights = optimizer.weights
            best_metric = curr_metric
            best_epoch = epoch

        # early stopping
        if early_stopping:
            if curr_metric > last_metric + early_stop_min_delta:
                last_metric = curr_metric
                last_epoch = epoch
            elif epoch - last_epoch > early_stop_patience:
                break

        if verbose:
            print(f'epoch {epoch+1}/{n_epochs} : metric {curr_metric:.4f}')

    if return_best_weights:
        return best_weights, best_metric, best_epoch
    else:
        return optimizer.weights, curr_metric, epoch


def sgd_cv_eval(X: np.ndarray,
                norm_Y: np.ndarray,
                optimizer: SGDOptimizer,
                gradient_func: Callable,
                batch_size: int = 16,
                n_epochs: int = 200,
                random_state: int | None = None,
                cv_splits: int = 4,
                early_stopping: bool = False,
                early_stop_patience: int = 0,
                early_stop_min_delta: float = 0,
                return_best_weights: bool = False,
                verbose: bool = False,
                )-> tuple[np.ndarray, float, float, int]:
    """
    Wrapper function for stochastic gradient descent model evaluation by cross
    validation.

    Parameters
    ----------
    X : 3D np.ndarray of shape (N, n_stocks=50, depth=250)
        Past returns, the features.
    norm_Y : 2D np.ndarray of shape (N, n_stocks=50)
        Normalized target vectors.
    optimizer : SGDOptimizer
        The SGD opimizer instance.
    gradient_func : Callable
        A function (X, Y, weights) -> 1D np.ndarray of shape (depth,).
        Gives the gradients of the loss function at the given weight values.
    batch_size : int, optional
        Size of the SGD batches. The default is 16.
    n_epochs : int, optional
        Max number of training epochs. The default is 200.
    random_state : int | None, optional
        Random state for training data shuffling. The default is None.
    cv_splits : int, optional
        Number of cross-validation splits. The default is 4.
    early_stopping : bool, optional
        Enable early stopping. The default is False.
    early_stop_patience : int, optional
        Number of epochs with no improvement after which training will be
        stopped. The default is 0.
    early_stop_min_delta : float, optional
        Minimum change in the metric to qualify as an improvement, i.e. an
        absolute change of less than min_delta, will count as no improvement.
        The default is 16.
    return_best_weights : bool, optional
        Whether to return model weights from the epoch with the best value of
        the metric. If False, the model weights obtained at the last step of
        training are returned. The default is False.
    verbose : bool, optional
        Displays the metric after each epoch.
        The default is False.

    Returns
    -------
    weights : 2D np.ndarray of shape (cv_splits, depth)
        The optimized model weights for each CV model.
    tr_metric : float
        The metric evaluated on the merged training sets.
    val_metric : float
        The metric evaluated on the merged validation sets.
    epoch : int
        The last training epoch number.
    """
    rng = default_rng(random_state)
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    idx = [[itr, ival] for itr, ival in cv.split(X, norm_Y)]

    optimizers = [deepcopy(optimizer) for _ in range(cv_splits)]
    Y_pred_val = np.empty_like(norm_Y, dtype=float)

    best_weights = [opt.weights for opt in optimizers]
    best_metric = metric(np.sum(X * optimizer.weights, axis=-1), norm_Y)
    best_epoch = 0
    last_metric = best_metric
    last_epoch = 0

    for epoch in range(n_epochs):

        # epoch train for all CV splits
        for i, (itr, ival) in enumerate(idx):
            optim = optimizers[i]
            X_v = X[ival]
            rng.shuffle(itr)

            for j in range((len(itr)+batch_size-1) // batch_size):
                X_ = X[itr[j*batch_size:(j+1)*batch_size]]
                Y_ = norm_Y[itr[j*batch_size:(j+1)*batch_size]]
                w = optim.eval_point()
                optim.apply_gradients(gradient_func(X_,Y_, w))

            Y_pred_val[ival] = np.sum(X_v * optim.weights, axis=-1)

        # update best metric
        curr_metric = metric(Y_pred_val, norm_Y)
        if return_best_weights and (curr_metric > best_metric):
            best_weights = [opt.weights for opt in optimizers]
            best_metric = curr_metric
            best_epoch = epoch

        # early stopping
        if early_stopping:
            if curr_metric > last_metric + early_stop_min_delta:
                last_metric = curr_metric
                last_epoch = epoch
            elif epoch - last_epoch > early_stop_patience:
                break

        if verbose:
            print(f'epoch {epoch+1}/{n_epochs} : metric {curr_metric:.4f}')

    if return_best_weights:
        # compute trainig metric
        Y_pred_tr = np.concatenate([np.sum(X[itr] * w, axis=-1)
                                    for (itr, _), w in zip(idx, best_weights)])
        Y_tr = np.concatenate([norm_Y[itr] for (itr, _) in idx])
        tr_metric = metric(Y_pred_tr, Y_tr)

        return np.array(best_weights), tr_metric, best_metric, best_epoch

    else:
        last_weights = np.array([opt.weights for opt in optimizers])
        # compute trainig metric
        Y_pred_tr = np.concatenate([np.sum(X[itr] * w, axis=-1)
                                    for (itr, _), w in zip(idx, last_weights)])
        Y_tr = np.concatenate([norm_Y[itr] for (itr, _) in idx])
        tr_metric = metric(Y_pred_tr, Y_tr)

        return last_weights, tr_metric, curr_metric, epoch


##
n_stocks = 50
n_factors = 10
depth = 250
n_steps = stock_returns.shape[1] - depth
X = np.stack([stock_returns[:, i:i+depth] for i in range(n_steps)])
Y = np.stack([stock_returns[:, i+depth] for i in range(n_steps)])
YY = Y / np.sqrt(np.sum(Y**2, axis=1, keepdims=True))
# !!!

import sys
sys.exit()

# %%
"""
## Problem reduction
"""

a = np.zeros(250, dtype=float)
a[0:5] = -0.00694934/np.sqrt(5)
a[20:250] = 0.01560642/np.sqrt(230)

Y_pred = X @ a[::-1]
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
    Y_pred = model.predict(XX.reshape(-1, depth)).reshape(-1, n_stocks)
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
    gridspec_kw={'left': 0.14, 'right': 0.93, 'top': 0.83, 'bottom': 0.14})
fig5.suptitle("Figure 5: Lasso regression results", x=0.02, ha='left')


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
at $\alpha = 2.5 \cdot 10^{-4}$, which is similar to the benchmark score.
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
        random_vec = vectors[best_idx]
        print(f'iteration {i}, best metric :', best_metric)

print('Best metric for 200k random vectors: {best_metric:.4f}')

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

Y_pred = X @ w0
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

    Y_pred = X @ optimizer.weights
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
fig6, ax6 = plt.subplots(
    nrows=1, ncols=1, sharey=True, figsize=(5.8, 3.8), dpi=100,
    gridspec_kw={'left': 0.125, 'right': 0.94, 'top': 0.83, 'bottom': 0.14})
fig6.suptitle("Figure 6: Metric evolution with SGD training",
              x=0.02, ha='left')


l_tr, = ax6.plot(np.arange(n_epochs+1), tr_metric, linestyle='-', marker='')
l_val, = ax6.plot(np.arange(n_epochs+1), val_metric, linestyle='-', marker='')

ax6.set_xlim(0, n_epochs)
ax6.set_xlabel('epoch')
ax6.set_ylim(-0.01, 0.16)
ax6.set_ylabel('Metric')
ax6.grid(visible=True, linewidth=0.3)

fig6.legend(handles=[l_tr, l_val],
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

### Finding optimal parameters
"""

## Initial weights
rng = default_rng(1234)
w0 = rng.normal(size=depth)
w0 = w0 / np.sqrt(np.sum(w0**2))

## Grid search
grad_regs = [grad_l1_diff , grad_l2_diff, grad_l1_absdiff, grad_l2_absdiff]
# reg_param_vals = np.logspace(-3, 4, 28)
# !!!
reg_param_vals = np.concatenate([np.logspace(-3, 1, 16, endpoint=False),
                                 np.logspace(1, 2, 10, endpoint=False),
                                 np.logspace(2, 4, 9, endpoint=True)])

sgdreg_tr_scores = np.zeros((4, len(reg_param_vals)), dtype=float)
sgdreg_val_scores = np.zeros((4, len(reg_param_vals)), dtype=float)
for i, grad_reg in enumerate(grad_regs):
    t0 = time.time()
    for j, reg in enumerate(reg_param_vals):
        # print(f'{grad_reg.__name__} ; regularization parameter: {reg:.4f}')
        optim = SGDOptimizer(w0, learning_rate=0.005, momentum=0.9, nesterov=True)
        grad_func = lambda X, Y, w: -grad_metric(X, Y, w) + grad_reg(w, reg)


        _, tr_metric, val_metric, _ = sgd_cv_eval(X, YY, deepcopy(optim),
                                                  grad_func,
                                                  batch_size=16,
                                                  n_epochs=400,
                                                  random_state=1234,
                                                  cv_splits=4,
                                                  early_stopping=True,
                                                  early_stop_patience=40,
                                                  early_stop_min_delta=5e-4,
                                                  return_best_weights=True,
                                                  verbose=False)

        sgdreg_val_scores[i, j] = val_metric
        sgdreg_tr_scores[i, j] = tr_metric
    t1 = time.time()
    print(f'{grad_reg.__name__} : {len(reg_param_vals)} fits in {t1-t0:.2f} s')


# %%

fig7, axs7 = plt.subplots(
    nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 5.8), dpi=100,
    gridspec_kw={'left': 0.085, 'right': 0.93, 'top': 0.86, 'bottom': 0.1,
                 'hspace': 0.1, 'wspace': 0.1})
fig7.suptitle("Figure 7: Regularized SGD scores", x=0.02, ha='left')

# L1 penalty
l_tr, = axs7[0, 0].plot(reg_param_vals, sgdreg_tr_scores[0],
                        linestyle='-', marker='')
l_val, = axs7[0, 0].plot(reg_param_vals, sgdreg_val_scores[0],
                         linestyle='-', marker='')
axs7[0, 0].tick_params(which='both', direction='in',
                       right=True, labelbottom=False,
                       top=True, labeltop=True)
axs7[0, 0].set_xscale('log')
axs7[0, 0].set_xlim(1e-3, 1e4)
axs7[0, 0].set_ylim(-0.01, 0.1)
axs7[0, 0].grid(visible=True, linewidth=0.3)
axs7[0, 0].grid(visible=True, which='minor', linewidth=0.2)
axs7[0, 0].text(0.96, 0.93, r'$\lambda \sum_{t=0}^{D-1} |w_{t+1} - w_t|$',
                ha='right', va='top', transform=axs7[0, 0].transAxes,
                bbox={'boxstyle': 'round', 'facecolor': '0.94'})

# L2 penalty
l_tr, = axs7[0, 1].plot(reg_param_vals, sgdreg_tr_scores[1],
                        linestyle='-', marker='')
l_val, = axs7[0, 1].plot(reg_param_vals, sgdreg_val_scores[1],
                         linestyle='-', marker='')
axs7[0, 1].tick_params(which='both', direction='in',
                       labelleft=False, labelbottom=False,
                       right=True, labelright=True,
                       top=True, labeltop=True)
axs7[0, 1].set_xscale('log')
axs7[0, 1].grid(visible=True, linewidth=0.3)
axs7[0, 1].grid(visible=True, which='minor', linewidth=0.2)
axs7[0, 1].text(0.96, 0.93, r'$\lambda \sum_{t=0}^{D-1} (w_{t+1} - w_t)^2$',
                ha='right', va='top', transform=axs7[0, 1].transAxes,
                bbox={'boxstyle': 'round', 'facecolor': '0.94'})

# L1 penalty on diff of abs
l_tr, = axs7[1, 0].plot(reg_param_vals, sgdreg_tr_scores[2],
                        linestyle='-', marker='')
l_val, = axs7[1, 0].plot(reg_param_vals, sgdreg_val_scores[2],
                         linestyle='-', marker='')
axs7[1, 0].tick_params(which='both', direction='in',
                       right=True, top=True)
axs7[1, 0].set_xscale('log')
axs7[1, 0].grid(visible=True, linewidth=0.3)
axs7[1, 0].grid(visible=True, which='minor', linewidth=0.2)
axs7[1, 0].text(0.96, 0.93, r'$\lambda \sum_{t=0}^{D-1} ||w_{t+1}| - |w_t||$',
                ha='right', va='top', transform=axs7[1, 0].transAxes,
                bbox={'boxstyle': 'round', 'facecolor': '0.94'})

# L2 penalty on diff of abs
l_tr, = axs7[1, 1].plot(reg_param_vals, sgdreg_tr_scores[3],
                        linestyle='-', marker='')
l_val, = axs7[1, 1].plot(reg_param_vals, sgdreg_val_scores[3],
                         linestyle='-', marker='')
axs7[1, 1].tick_params(which='both', direction='in',
                       top=True,
                       left=True, labelleft=False,
                       right=True, labelright=True)
axs7[1, 1].set_xscale('log')
axs7[1, 1].grid(visible=True, linewidth=0.3)
axs7[1, 1].grid(visible=True, which='minor', linewidth=0.2)
axs7[1, 1].text(0.96, 0.93, r'$\lambda \sum_{t=0}^{D-1} (|w_{t+1}| - |w_t|)^2$',
                ha='right', va='top', transform=axs7[1, 1].transAxes,
                bbox={'boxstyle': 'round', 'facecolor': '0.94'})


fig7.text(0.51, 0.025, 'Regularization parameter', fontsize=11,
          ha='center', va='center')
fig7.text(0.02, 0.48, 'Metric', fontsize=11, rotation=90,
          ha='center', va='center')
fig7.legend(handles=[l_tr, l_val],
            labels=['Training metric', 'Validation metric'],
            ncols=2, loc=(0.465, 0.92), fontsize=11)

plt.show()

"""
Figure 7 presents validation and train metric for the 4 penalties we introduced.
They all perform rather equally well, being able to suppress overfitting
for values of the regularization parameter in the range 1 - 100.
The reached validation metric, of about 0.02, are not very high but are certainly robust.
We note that the curves corresponding to the L2 penalty are smoother. This is the result of the
corresponding penalty gradient being continuous, contrary to the other penalty functions.
"""

# %%

"""
### Convergence of the regularized SGD

Despite the not-so-good performance of our models, if they converge to different
weights for different initial conditions/penalty functions, we can expect to make
a more powerful predictor by combining them.

With this in mind, we study in this section the convergence of our models.
More precisely, we try to determine the influence of the initial weights on
the optimum found by the algorithm.
We also question whether models with different penalties but the same initial weights
will converge to the same values.

To do so, we repeatedly fit the models with 10 different inital weight vectors
and record the optimum weights reached by the model. We then study the correlation
between all these vectors.
"""

## Initialize random weight vectors
n_vect = 10
rng = default_rng(1234)
w0 = rng.normal(size=(n_vect, depth))
init_weights = w0 / np.sqrt(np.sum(w0**2, axis=1, keepdims=True))


## parameters chosen based on figure 7
best_reg_params = [5, 5, 10, 10]

## Find optimal weights for each initial weight and model combination
opt_weights = np.empty((4, n_vect, depth))
tr_scores = np.empty((4, n_vect))
for i, (grad_reg, reg) in enumerate(zip(grad_regs, best_reg_params)):
    # print(f'{grad_reg.__name__} ; regularization parameter: {reg:.4f}')
    grad_func = lambda X, Y, w: -grad_metric(X, Y, w) + grad_reg(w, reg)

    for j, wt in enumerate(init_weights):
        optim = SGDOptimizer(wt, learning_rate=0.005, momentum=0.9, nesterov=True)
        opt_wt, tr_metric, _ = sgd_train(X, YY, optim, grad_func,
                                 batch_size=16,
                                 n_epochs=400,
                                 random_state=1234,
                                 early_stopping=True,
                                 early_stop_patience=40,
                                 early_stop_min_delta=5e-4,
                                 return_best_weights=True,
                                 verbose=False)

        opt_weights[i, j] = opt_wt
        tr_scores[i, j] = tr_metric


# %%
"""
!!!
get cv weights for the scanned param
"""

## Initial weights
rng = default_rng(1234)
w0 = rng.normal(size=depth)
w0 = w0 / np.sqrt(np.sum(w0**2))

reg_param_vals = np.logspace(0, 2, 21, endpoint=False)

sgdreg_tr_scores = np.zeros(len(reg_param_vals), dtype=float)
sgdreg_val_scores = np.zeros(len(reg_param_vals), dtype=float)
final_weights = []

t0 = time.time()
for j, reg in enumerate(reg_param_vals):
    print(j)
    optim = SGDOptimizer(w0, learning_rate=0.005, momentum=0.9, nesterov=True)
    grad_func = lambda X, Y, w: -grad_metric(X, Y, w) + grad_l2_diff(w, reg)

    wt, tr_metric, val_metric, _ = sgd_cv_eval(X, YY, deepcopy(optim),
                                              grad_func,
                                              batch_size=16,
                                              n_epochs=500,
                                              random_state=1234,
                                              cv_splits=5,
                                              early_stopping=False,
                                              early_stop_patience=40,
                                              early_stop_min_delta=5e-4,
                                              return_best_weights=True,
                                              verbose=False)

    sgdreg_val_scores[j] = val_metric
    sgdreg_tr_scores[j] = tr_metric
    final_weights.append(wt)
    
t1 = time.time()
print(f'{len(reg_param_vals)} fits in {t1-t0:.2f} s')


# %%
fig8, axs8 = plt.subplots(
    nrows=len(final_weights)+1, ncols=1, sharey=True,
    figsize=(12, 8.5), dpi=200,
    gridspec_kw={'left': 0.01, 'right': 0.99, 'top': 0.9, 'bottom': 0.02,
                 'hspace': 0.4, 'wspace': 0.1})
fig8.suptitle("Figure : CV weights convergence vs reg param (L2 regSGD)",
              x=0.02, ha='left')

axs8[0].annotate('', (0.93, 0.92), (0.83, 0.92), xycoords='figure fraction',
                 arrowprops={'arrowstyle': '->', 'facecolor': 'k', 'linewidth': 2})
fig8.text(0.88, 0.94, 'time', fontsize=11)

axs8[0].imshow(w0[None, ...], cmap='seismic', vmin=-0.4, vmax=0.4)
axs8[0].set_title('Initial vectors')
axs8[0].tick_params(left=False, labelleft=False,
                    bottom=False, labelbottom=False)

for i, w in enumerate(final_weights, start=1):
    axs8[i].imshow(w, cmap='seismic', vmin=-0.4, vmax=0.4)
    axs8[i].tick_params(bottom=False, labelbottom=False,
                        left=False, labelleft=False)
    axs8[i].text(0.02, 2, f'{reg_param_vals[i-1]:.2f}', va='center')

plt.show()




# # !!!
# grad_func = lambda X, Y, w: -grad_metric(X, Y, w) + grad_l2_diff(w, 50)

# for j, wt in enumerate(init_weights):
#     optim = SGDOptimizer(wt, learning_rate=0.005, momentum=0.9, nesterov=True)
#     opt_wt, tr_metric, _ = sgd_train(X, YY, optim, grad_func,
#                              batch_size=16,
#                              n_epochs=400,
#                              random_state=1234,
#                              early_stopping=True,
#                              early_stop_patience=40,
#                              early_stop_min_delta=5e-4,
#                              return_best_weights=True,
#                              verbose=False)

#     opt_weights[1, j] = opt_wt
#     tr_scores[1, j] = tr_metric


# %%

fig8, axs8 = plt.subplots(
    nrows=5, ncols=1, sharex=True, sharey=True, figsize=(12, 4.5), dpi=100,
    gridspec_kw={'left': 0.01, 'right': 0.99, 'top': 0.9, 'bottom': 0.02,
                 'hspace': 0.4, 'wspace': 0.1})
fig8.suptitle("Figure 8: Heatmap representation of the weights",
              x=0.02, ha='left')

axs8[0].annotate('', (0.93, 0.92), (0.83, 0.92), xycoords='figure fraction',
                 arrowprops={'arrowstyle': '->', 'facecolor': 'k', 'linewidth': 2})
fig8.text(0.88, 0.94, 'time', fontsize=11)

axs8[0].imshow(init_weights, cmap='seismic', vmin=-0.4, vmax=0.4)
axs8[0].set_title('Initial vectors')
axs8[0].tick_params(left=False, labelleft=False,
                    bottom=False, labelbottom=False)

titles = ['L1_diff', 'L2_diff', 'L1_absdiff', 'L2_absdiff']
for i, w in enumerate(opt_weights, start=1):
    axs8[i].imshow(w, cmap='seismic', vmin=-0.4, vmax=0.4)
    axs8[i].set_title(rf'{titles[i-1]}, $\lambda = {best_reg_params[i-1]}$')
    axs8[i].tick_params(bottom=False, labelbottom=False,
                        left=False, labelleft=False)

plt.show()

r"""
Figure 8 presents a heatmap representation of the initial weights as well as the
optimum weights found by each model. Note that the coefficients taking into
account more recent times are on the right.
- For the models using regular L1 and L2 penalties $J_1(\mathbf{w})$ and $J_2(\mathbf{w})$,
the regularity and homogeneity of the final weights is striking. these models clearly converge
towards the same optimum independently of the initial weights.
It is interesting to note that this optimum seems to give more importance
to the values far preceding than to the returns of the past days.
The returns of the same period during the previous year are possibly goods indicators
of the current returns.
- For models using using the penalties acting the absolute values of the weights,
$J^{\mathrm{abs}}_1(\mathbf{w})$ and $J^{\mathrm{abs}}_2(\mathbf{w})$, the
final weights also present a form of regularity. However, the optimum depends
strongly on the initial weight values.
Actually, the optimal weights seems to be chosen at random among some specific
(non-gaussian) distribution.
"""

# %%

##
weights = np.concatenate([init_weights[None, ...], opt_weights], axis=0)
weights_corr = 100*np.corrcoef(weights.reshape(-1, depth))[::-1]

##
fig9, ax9 = plt.subplots(
    nrows=1, ncols=1, figsize=(8, 7.4), dpi=100,
    gridspec_kw={'left': 0.03, 'right': 0.88, 'top': 0.89, 'bottom': 0.03})
cax9 = fig9.add_axes((0.89, 0.06, 0.025, 0.8))
fig9.suptitle("Figure 9: Correlation matrix of the weight vectors",
              x=0.02, ha='left')

ax9.set_aspect('equal')
cmap9 = plt.get_cmap('seismic').copy()
cmap9.set_extremes(under='0.9', over='0.5')
heatmap = ax9.pcolormesh(weights_corr,
                         cmap=cmap9, vmin=-100, vmax=100,
                         edgecolors='0.2', linewidth=0.5)

ax9.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,
                left=False, labelleft=False)
for i in range(6):
    ax9.plot(ax9.get_xlim(), [i*n_vect, i*n_vect],
             color='k', lw=2, clip_on=False)
for i in range(6):
    ax9.plot([i*n_vect, i*n_vect], ax9.get_ylim(),
             color='k', lw=2, clip_on=False)

txt = ['Initial\nvectors', 'L1_diff', 'L2_diff', 'L1_absdiff', 'L2_absdiff']
for i, t in enumerate(txt):
    ax9.text(i*0.2+0.1, 1.02, t, transform=ax9.transAxes, ha='center')


fig9.colorbar(heatmap, cax=cax9, orientation="vertical", ticklocation="right")
cax9.text(1.4, 1.03, 'Corr.\ncoef. (%)', fontsize=11, ha='center', transform=cax9.transAxes)


plt.show()

r"""
Figure 9 shows the correlation matrix of all the weight vectors, initial and
optimized by the models. The thick solid black lines split the matrix in blocks
corresponding to the different models.

The initial weight vectors are uncorrelated, both with
each other and with the model solutions, as expected. However, there is a qualitative
difference between the vectors found by the different models:
- The good convergence of the models with regular L1 and L2 penalty translates into
correlation coefficients close to 1.
- For the other model, the final weights are weakly correlated both to each other
and to the inital values. The loss function in this case probably has many
local minima deep enough to trap the SGD optimizer.
"""

# %%
"""
## Submissions to the challenge



### Random vector

!!!
"""

## reverse weights order for submission
factors = vector_to_vectors(random_vec[::-1])
_ = to_csv(factors, './randvec_submission.csv')

print(f'Random vector train metric: {metric(X @ random_vec, YY):.4f}')


"""
!!!
Test metric : 0.00084
"""

# %%
"""
### Optimum found by SGD with L2 penalty

!!!
"""

## reverse weights order for submission
factors = vector_to_vectors(opt_weights[1, 0])
_ = to_csv(factors, './L2regSGD_submission.csv')

print('L2-regularized SGD train metric: '
      f'{metric(X @ opt_weights[1, 0], YY):.4f}')

"""
!!!
"""

# %%
"""
### Linear combination of vectors found by SGD

!!!
With these results, we can try to build explicative factors out of the weights
vectors found above.

We proceed by picking specific vectors from the models and making a linear regression
with this selected features.
We select manually 1 from the regular L1 model and 1 from the regular L2 model,
taking vectors with low correlations.
We pick the 8 other vectors at random among the 2 remaining models.
The selected vectors are represented in figure 10.
"""

## select features
rng = default_rng(1234)
idx = rng.choice(np.arange(20, 40), size=8, replace=False)
A = np.concatenate([opt_weights[[0, 1], [7, 0]],
                    opt_weights[idx//10, idx%10]], axis=0)

##
fig10, ax10 = plt.subplots(
    nrows=1, ncols=1, sharex=True, sharey=True, figsize=(12, 1.2), dpi=100,
    gridspec_kw={'left': 0.01, 'right': 0.99, 'top': 0.9, 'bottom': 0.02,
                 'hspace': 0.4, 'wspace': 0.1})
fig10.suptitle("Figure 10: Selected vectors for model construction",
              x=0.02, y=0.92, ha='left')

ax10.annotate('', (0.93, 0.82), (0.83, 0.82), xycoords='figure fraction',
              arrowprops={'arrowstyle': '->', 'facecolor': 'k', 'linewidth': 2})
fig10.text(0.88, 0.87, 'time', fontsize=11)

ax10.imshow(A, cmap='seismic', vmin=-0.4, vmax=0.4)
ax10.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

plt.show()


# %%
## fit
pred = X.reshape(-1, depth) @ A.T
beta = np.linalg.inv(pred.T @ pred) @ pred.T @ YY.ravel()

##
w = np.dot(beta, A)
print('Linear combination of regularized SGD factors train metric:'
      f'{metric(X @ w, YY):.4f}')

## reverse weights order for submission
factors = vector_to_vectors(w[::-1])
_ = to_csv(factors, './regSGD_submission.csv')

"""
!!!
"""



# %% Conclusion
"""
The metric obtained with this submission is 0.236, lower than the random benchmark.
As we saw, the difficulty of the problem lies in model overfitting. The finacial data
contains a lot of noise, while we have only about 25000 / 250 = 100 observations
per feature dimension. Our strategy of imposing regularity for the features along the
time axis provided regular patterns as possible explicative factors. These factors
were robust to noise. However, mixing them failed to yield a better predictor.

The approach presented here is certainly not the only possibility.
- Other dimension reduction techniques, such as principal components analysis,
could be used to extract common patters in prices variations.
- Instead of learning regular factors from the data, we could use some standard
decompositions such as discrete cosine transform or discrete wavelet transforms
and extract the most relevant coefficients.
- There possibly exist data augmentation techniques, which would improve the
signal-to-noise ratio.
"""

