# -*- coding: utf-8 -*-
"""
Utility functions.
"""

from copy import deepcopy
from typing import Callable

import numpy as np
from numpy.random import default_rng
from sklearn.model_selection import KFold



###############################################################################
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