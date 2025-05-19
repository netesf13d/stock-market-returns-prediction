# stock-market-returns-prediction

This repository presents various approaches to solve QRT data challenge published in 2022. A detailed description is available [here](https://challengedata.ens.fr/participants/challenges/72/).


## Problem overview

We are given a dataset consisting of daily price returns over three years for a selection of $N=50$ market stocks. The goal is to predict the next-day returns given the past values. Mathematically, denoting $R_{t}$ the vector of returns at time $t$ for the $N$ stocks, we assume a relation 
$$
  R_{t+1} = f\left( \{ R_k \}_{k\leq t}  \right) + \epsilon_{t+1},
$$
where $\epsilon$ is a noise contribution.

In the context of this project, we look for a predictor $f$ that is a linear combination of the past returns. We thus look for explicative factors $F_{i, t}$ such that the predicted returns $\hat{R}_{t+1}$ take the form:
$$
  \hat{R}_{t+1} = \sum_{i=1}^{F} \beta_i \, F_{i, t},
$$
$$
  F_{i, t} = \sum_{j=1}^{D} A_{i, j} \, R_{t+1-j},
$$
with
- $F = 10$, the maximal number of factors;
- $D = 250$, the time depth limit (about 1 year);
- vectors $\{A_i\}_{1\leq i \leq F}$ are orthonormal, ie $\langle A_i,A_j \rangle = \delta_{ij}$.

The metric associated to this problem is the mean cosine similarity between the prediction and the actual value:
$$
  \mathcal{M}(A, \beta) = \frac{1}{T} \sum_{t > D} \frac{\langle \hat{R}_t,R_t \rangle}{\| \hat{R}_t \| \| R_t \|},
$$
where $T$ is the number of predicted returns (here $T = 504$ given that we have the returns for $754$ days).

The vectors $A_i$ can be determined empirically from financial expertise. Such empirical quantities include the 5-day normalized mean returns $R_t^{(5)}$ and the momentum $R_{t-20}^{(230)}$, where
$$
  R_{t}^{(m)} = \frac{1}{\sqrt{m}} \sum_{k=1}^{m} R_{t+1-k}.
$$
The parameters $\beta$ are then fit to market data. These factors have the advantage of being easily interpretable, but are fairly constrained and do not perform very well. The goal of this project is to find explicative factors directly from the data.


## Contents and usage 

This repository is structured as follows.
- The Jupyter notebook `Stock_market_returns_pred.ipynb` presents the results or our study.
- The file `utils.py` contains utility functions used in the notbook.
- The file `requirements.txt` gives the list of the project dependencies. 
- The file `X_train.csv` contains the data for training. The targets are extracted from this file.

The code runs with Python 3.12. To setup the Python environment:
- With `pip`, with python 3.12 installed, run `pip install -r requirements.txt`
- Using `conda`, run `conda create --name <env_name> python=3.12 --file requirements.txt`