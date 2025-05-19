# -*- coding: utf-8 -*-
"""
Utility functions.
"""

from typing import Callable

import numpy as np


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
    