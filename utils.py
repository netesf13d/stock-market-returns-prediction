# -*- coding: utf-8 -*-
"""
Utility functions.
"""

from typing import Callable

import numpy as np




# def metric_gradient(past_returns: np.ndarray,
#                     pred_returns: np.ndarray,
#                     norm_curr_returns: np.ndarray)-> np.ndarray:
#     """
#     Compute the gradient of the metric.

#     Parameters
#     ----------
#     past_returns : 3D np.ndarray of shape (N, n_stocks=50, depth=250)
#         DESCRIPTION.
#     pred_returns : 2D np.ndarray of shape (N, n_stocks=50)
#         DESCRIPTION.
#     norm_curr_returns : 2D np.ndarray of shape (N, n_stocks=50)
#         DESCRIPTION.
#     """
#     pred_returns_norm = np.sqrt(np.sum(pred_returns**2, axis=1, keepdims=True))
    
#     grad1 = np.sum(past_returns * norm_curr_returns[..., None], axis=1)
#     grad1 = np.mean(grad1 / pred_returns_norm, axis=0)
    
#     proj = np.sum(pred_returns * norm_curr_returns, axis=1, keepdims=True)
#     grad2 = np.sum(past_returns * pred_returns[..., None], axis=1)
#     grad2 = np.mean(grad2 * proj / pred_returns_norm**2, axis=0)
#     # print(pred_returns_norm.flatten())
        
#     return grad1 - grad2


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
        
        
        # self.momentum_vector = (self.momentum * self.momentum_vector
        #                         - self.learning_rate * gradients)
        # self.weights += self.momentum_vector

    
    
    