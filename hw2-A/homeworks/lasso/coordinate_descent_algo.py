from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    return 2 * np.sum(np.square(X), axis=0)


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    b = 1/X.shape[0] * np.sum(y - np.dot(X, weight))
    for k in range(X.shape[1]):
        c_k = 2*np.dot(X[:, k], (y - (b + np.dot(X, weight) - X[:, k] * weight[k])))
        
        if c_k < -_lambda:
            weight[k] = (c_k + _lambda) / a[k]
        elif c_k > _lambda:
            weight[k] = (c_k - _lambda) / a[k]
        else:
            weight[k] = 0

    return weight, b


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    loss = 0
    norm = 0
    for n in range(X.shape[0]):
        loss += (np.dot(X[n], np.transpose(weight)) + bias - y[n])**2
        norm = np.linalg.norm(weight, 1)
    return loss + _lambda * norm


@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float .

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)
    old_w: Optional[np.ndarray] = None

    b = 0
    while not convergence_criterion(start_weight, old_w, convergence_delta):
        old_w = np.copy(start_weight)
        start_weight, b = step(X, y, start_weight, a, _lambda)

    return start_weight, b


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    if old_w is None:
        return False
    else:
        return np.max(np.abs(old_w - weight)) < convergence_delta





@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    
    w = np.arange(1000) / 100
    w[100 + 1: ] = 0
    X = np.random.normal(0.0, 1.0, (500, 1000))
    y = np.dot(X, w) + np.random.normal(0.0, 1, (500, ))
    

    lam_max = 2*np.max(np.abs(np.dot(y.T - np.mean(y), X)))
    cur_lam = lam_max
    lam_list = []
    W = np.zeros((1000, 1))
    prev_w = None

    FDR = []
    TPR = []
    
    while np.count_nonzero(W[:, -1]) != 1000:
        lam_list.append(cur_lam)
        cur_lam = cur_lam / 2

        w_train, b = train(X, y, _lambda=cur_lam, start_weight=prev_w)
        W = np.concatenate((W, np.expand_dims(w_train, axis=1)), axis=1)
        non_zero_w_train = np.count_nonzero(w_train)
        
        if (non_zero_w_train != 0):
            FDR.append(np.count_nonzero(w_train[100:]) / non_zero_w_train) 
        else:
            FDR.append(0)

        TPR.append(np.count_nonzero(w_train[:100]) / 100)
        prev_w = np.copy(w_train)


# 5.a ------------------------------------------------------------------
    lam_list.append(cur_lam)
    plt.figure(1)
    plt.xscale('log')
    plt.plot(lam_list, np.count_nonzero(W, axis=0))
    plt.xlabel('Lambda')
    plt.ylabel('Nonzeros in w')
    plt.title('5a')
    plt.show()

# 5.b ------------------------------------------------------------------
    plt.figure(2)
    plt.plot(FDR, TPR)
    plt.xlabel('FDR')
    plt.ylabel('TPR')
    plt.title('5b')
    plt.show()



if __name__ == "__main__":
    main()
