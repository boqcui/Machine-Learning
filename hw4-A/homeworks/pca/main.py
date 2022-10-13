from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import load_dataset, problem


@problem.tag("hw4-A")
def reconstruct_demean(uk: np.ndarray, demean_data: np.ndarray) -> np.ndarray:
    """Given a demeaned data, create a recontruction using eigenvectors provided by `uk`.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_vec (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        np.ndarray: Array of shape (n, d).
            Each row should correspond to row in demean_data,
            but first compressed and then reconstructed using uk eigenvectors.
    """
    #raise NotImplementedError("Your Code Goes Here")
    return demean_data.dot(uk).dot(uk.T)


@problem.tag("hw4-A")
def reconstruction_error(uk: np.ndarray, demean_data: np.ndarray) -> float:
    """Given a demeaned data and some eigenvectors calculate the squared L-2 error that recontruction will incur.

    Args:
        uk (np.ndarray): First k eigenvectors. Shape (d, k).
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        float: Squared L-2 error on reconstructed data.
    """
    #raise NotImplementedError("Your Code Goes Here")
    reconstructed = reconstruct_demean(uk, demean_data)
    error = np.sum((demean_data - reconstructed)**2) / demean_data.shape[0]

    return error


@problem.tag("hw4-A")
def calculate_eigen(demean_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given demeaned data calculate eigenvalues and eigenvectors of it.

    Args:
        demean_data (np.ndarray): Demeaned data (centered at 0). Shape (n, d)

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays representing:
            1. Eigenvalues array with shape (d,). Should be in descending order.
            2. Matrix with eigenvectors as columns with shape (d, d)
    """
    #raise NotImplementedError("Your Code Goes Here")
    return(np.linalg.eig(np.dot(demean_data.T, demean_data) / len(demean_data)))


@problem.tag("hw4-A", start_line=2)
def main():
    (x_tr, y_tr), (x_test, _) = load_dataset("mnist")
    """
    Main function of PCA problem. It should load data, calculate eigenvalues/-vectors,
    and then answer all questions from problem statement.

    Part A:
        - Report 1st, 2nd, 10th, 30th and 50th largest eigenvalues
        - Report sum of eigenvalues
    """
    demean_train = x_tr - np.mean(x_tr, 0)
    eigs = calculate_eigen(demean_train)
    le = [0, 1, 9, 29, 49]
    for i in le:
        print("The ", i, " th largest eigenvalue is: ", eigs[0][i])
    print("The sum of eigenvalues is: ", eigs[0].sum())


    """
    Part C:
        - Plot reconstruction error as a function of k (# of eigenvectors used)
            Use k from 1 to 101.
            Plot should have two lines, one for train, one for test.
        - Plot ratio of sum of eigenvalues remaining after k^th eigenvalue with respect to whole sum of eigenvalues.
            Use k from 1 to 101.
    """


    """
    Part D:
        - Visualize 10 first eigenvectors as 28x28 grayscale images.
    """


    """
    Part E:
        - For each of digits 2, 6, 7 plot original image, and images reconstruced from PCA with
            k values of 5, 15, 40, 100.
    """

    



if __name__ == "__main__":
    main()
