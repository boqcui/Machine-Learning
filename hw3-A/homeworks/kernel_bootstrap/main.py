import math
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 4 * np.sin(np.pi * x) * np.cos(6 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return (1 + x_i.reshape(1, -1).T.dot(x_j.reshape(1, -1))) ** d
    


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    n = x_i.shape[0]
    m = x_j.shape[0]
    K = np.ndarray(shape=(n, m))
    for a in range(n):
        for b in range(m):
            K[a][b] = 2.718281828459045**(-gamma*pow(x_i[a] - x_j[b], 2))
            
    return K


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """

    return np.linalg.solve(kernel_function(x, x, kernel_param) + (_lambda * np.eye(x.shape[0])).T, y)
    


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across folds.
    """
    fold_size = len(x) // num_folds
    errors = 0

    for fold in range(num_folds):
        start_index = fold*fold_size
        end_index = (fold+1)*fold_size
        X_test = x[start_index:end_index]
        X_train = np.delete(x, np.arange(start_index, end_index, 1))
        Y_text = y[start_index:end_index]
        Y_train = np.delete(y, np.arange(start_index, end_index, 1))
        alpha_h = train(X_train, Y_train, kernel_function, kernel_param, _lambda)
        errors += np.sum(((np.dot(alpha_h, kernel_function(X_train, X_test, kernel_param)) - Y_text) ** 2)) / fold_size

    return errors/num_folds
    
@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / median(dist(x_i, x_j)^2 for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    #raise NotImplementedError("Your Code Goes Here")
    gamma_median = []
    for i in range(len(x)):
        for j in range (i+1, len(x)):
            gamma_median.append(pow(x[i]-x[j],2))
    gamma_median.sort
    gam_max = (gamma_median[len(x) // 2] + gamma_median[~(len(x) // 2)]) / 2

    lambda_array = 10 ** np.linspace(-5, -1, 30)
    gamma_array = np.linspace(0.0001, 1/gam_max, 30)

    error = 10000000
    best_lam = -10000
    best_gam = -10000
    for lam in lambda_array:
        for gam in gamma_array:
            cur_error = cross_validation(x, y, rbf_kernel, gam, lam, num_folds)
            if error > cur_error:
                error = cur_error
                best_lam = lam
                best_gam = gam
    
    print("rbf_param_search, best_lam: " , best_lam, ", best_gam: ", best_gam, ", error: ", error)
    return best_lam, best_gam

@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - You do not really need to search over gamma. 1 / median((x_i - x_j) for all unique pairs x_i, x_j in x)
            should be sufficient for this problem. That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution {7, 8, ..., 20, 21}
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [7, 8, ..., 20, 21]
    """

    lambda_array = 10 ** np.linspace(-5, -1, 30)

    error = 10000000
    best_lam = -10000
    best_d = -10000
    for lam in lambda_array:
        for d in range(7, 26):
            cur_error = cross_validation(x, y, poly_kernel, d, lam, num_folds)
            if error > cur_error:
                error = cur_error
                best_lam = lam
                best_d = d
    
    print("poly_param_search, best_lam: " , best_lam, ", best_d: ", best_d, ", error: ", error)
    return best_lam, best_d


@problem.tag("hw3-A", start_line=1)
def bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    bootstrap_iters: int = 300,
) -> np.ndarray:
    """Bootstrap function simulation empirical confidence interval of function class.

    For each iteration of bootstrap:
        1. Sample len(x) many of (x, y) pairs with replacement
        2. Train model on these sampled points
        3. Predict values on x_fine_grid (see provided code)

    Lastly after all iterations, calculated 5th and 95th percentiles of predictions for each point in x_fine_point and return them.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        bootstrap_iters (int, optional): [description]. Defaults to 300.

    Returns:
        np.ndarray: A (2, 100) numpy array, where each row contains 5 and 95 percentile of function prediction at corresponding point of x_fine_grid.

    Note:
        - See np.percentile function.
            It can take two percentiles at the same time, and take percentiles along specific axis.
    """
    x_fine_grid = np.linspace(0, 1, 100)
    predictions = np.zeros(shape=(bootstrap_iters, 100))
    res = np.zeros(shape=(2, 100))
    
    for i in range(bootstrap_iters):
        index = np.random.choice(np.arange(len(x)), len(x), replace=True)
        x_boot = x[index]
        y_boot = y[index]
        alpha_h = train(x_boot, y_boot, kernel_function, kernel_param, _lambda)
        predictions[i, :] = np.dot(alpha_h, kernel_function(x_boot, x_fine_grid, kernel_param))
    res[0] = np.array(np.percentile(predictions, 5, axis=0))
    res[1] = np.array(np.percentile(predictions, 95, axis=0))
    return res

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid
        C. For both rbf and poly kernels, plot 5th and 95th percentiles from bootstrap using x_30, y_30 (using the same fine grid as in part B)
        D. Repeat A, B, C with x_300, y_300
        E. Compare rbf and poly kernels using bootstrap as described in the pdf. Report 5 and 95 percentiles in errors of each function.

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    
    # 4.a ------------------------------------------------------------------
    #best_poly_lam, best_d = poly_param_search(x_30, y_30, 30)
    #best_rbf_lam, best_gam = rbf_param_search(x_30, y_30, 30)
    # Above code is very slow, I recorded the output, so don't need to run again.
    best_poly_lam, best_d =  2.592943797404667e-05, 19
    best_rbf_lam, best_gam = 0.002212216291070448 , 43.110969187684965

    
    # 4.b ------------------------------------------------------------------
    x_fine_grid = np.linspace(0, 1, 100)
    f_poly = train(x_30, y_30, poly_kernel, best_d, best_poly_lam)
    f_poly_predictions = np.dot(f_poly, poly_kernel(x_30, x_fine_grid, best_d))
    plt.figure(1)
    plt.title('4.b: f_poly')
    plt.plot(x_30, y_30, 'ko')
    plt.plot(x_fine_grid, f_true(x_fine_grid))
    plt.plot(x_fine_grid, f_poly_predictions)
    plt.legend(['Data', 'True', 'poly_predictions'])
    plt.show()

    f_rbf = train(x_30, y_30, rbf_kernel, best_gam, best_rbf_lam)
    f_rbf_predictions = np.dot(f_rbf, rbf_kernel(x_30, x_fine_grid, best_gam))
    plt.figure(2)
    plt.title('4.b: f_rbf')
    plt.plot(x_30, y_30, 'ko')
    plt.plot(x_fine_grid, f_true(x_fine_grid))
    plt.plot(x_fine_grid, f_rbf_predictions)
    plt.legend(['Data', 'True', 'rbf_predictions'])
    plt.show()


    # 4.c ------------------------------------------------------------------
    x_fine_grid = np.linspace(0, 1, 100)
    boot_poly = bootstrap(x_30, y_30, poly_kernel, best_d, best_poly_lam)
    plt.figure(3)
    plt.title('4.c: boot_poly')
    plt.ylim(-15, 20)
    plt.plot(x_30, y_30, 'ko')
    plt.plot(x_fine_grid, f_true(x_fine_grid))
    plt.plot(x_fine_grid, f_poly_predictions)
    plt.plot(x_fine_grid, boot_poly[0])
    plt.plot(x_fine_grid, boot_poly[1])
    plt.legend(['Data', 'True', 'poly_predictions', '5%', '95%'])
    plt.show()

    rbf_boot = bootstrap(x_30, y_30, rbf_kernel, best_gam, best_rbf_lam)
    plt.figure(4)
    plt.title('4.c: boot_rbf')
    plt.plot(x_30, y_30, 'ko')
    plt.plot(x_fine_grid, f_true(x_fine_grid))
    plt.plot(x_fine_grid, f_rbf_predictions)
    plt.plot(x_fine_grid, rbf_boot[0])
    plt.plot(x_fine_grid, rbf_boot[1])
    plt.legend(['Data', 'True', 'rbf_predictions', '5%', '95%'])
    plt.show()
    


    # 4.d ------------------------------------------------------------------
    #best_poly_lam, best_d = poly_param_search(x_300, y_300, 10)
    #best_rbf_lam, best_gam = rbf_param_search(x_300, y_300, 10)
    # Above code is very very slow, I recorded the output, so don't need to run again.
    best_poly_lam, best_d = 1e-05 , 25
    best_rbf_lam, best_gam = 1e-05 , 8.534305850573523

    x_fine_grid300 = np.linspace(0, 1, 300)
    x_fine_grid = np.linspace(0, 1, 100)
    f_poly = train(x_300, y_300, poly_kernel, best_d, best_poly_lam)
    f_poly_predictions = np.dot(f_poly, poly_kernel(x_300, x_fine_grid300, best_d))
    plt.figure(5)
    plt.title('4.d: f_poly')
    plt.plot(x_300, y_300, 'ko')
    plt.plot(x_fine_grid, f_true(x_fine_grid))
    plt.plot(x_fine_grid300, f_poly_predictions)
    plt.legend(['Data', 'True', 'poly_predictions'])
    plt.show()

    f_rbf = train(x_300, y_300, rbf_kernel, best_gam, best_rbf_lam)
    f_rbf_predictions = np.dot(f_rbf, rbf_kernel(x_300, x_fine_grid300, best_gam))
    plt.figure(6)
    plt.title('4.d: f_rbf')
    plt.plot(x_300, y_300, 'ko')
    plt.plot(x_fine_grid, f_true(x_fine_grid))
    plt.plot(x_fine_grid300, f_rbf_predictions)
    plt.legend(['Data', 'True', 'rbf_predictions'])
    plt.show()

    
    boot_poly = bootstrap(x_300, y_300, poly_kernel, best_d, best_poly_lam)
    plt.figure(7)
    plt.title('4.d: boot_poly')
    plt.plot(x_300, y_300, 'ko')
    plt.plot(x_fine_grid, f_true(x_fine_grid))
    plt.plot(x_fine_grid300, f_poly_predictions)
    plt.plot(x_fine_grid, boot_poly[0])
    plt.plot(x_fine_grid, boot_poly[1])
    plt.legend(['Data', 'True', 'poly_predictions', '5%', '95%'])
    plt.show()

    rbf_boot = bootstrap(x_300, y_300, rbf_kernel, best_gam, best_rbf_lam)
    plt.figure(8)
    plt.title('4.d: boot_rbf')
    plt.plot(x_300, y_300, 'ko')
    plt.plot(x_fine_grid, f_true(x_fine_grid))
    plt.plot(x_fine_grid300, f_rbf_predictions)
    plt.plot(x_fine_grid, rbf_boot[0])
    plt.plot(x_fine_grid, rbf_boot[1])
    plt.legend(['Data', 'True', 'rbf_predictions', '5%', '95%'])
    plt.show()
    


    # 4.e ------------------------------------------------------------------
    x_fine_grid = np.linspace(0, 1, 100)
    bootstrap_iters = 300
    errors = np.zeros(bootstrap_iters)

    for i in range(bootstrap_iters):
        index = np.random.choice(np.arange(len(x_1000)), len(x_1000), replace=True)
        x_boot = x_1000[index]
        y_boot = y_1000[index]
        poly_error = 0
        rbf_error = 0
        for j in range(1000):
            v = np.zeros(shape=(1,))
            v[0] = x_boot[j]
            poly_error += (y_boot[j] - np.dot(f_poly, poly_kernel(x_300, v, best_poly_lam)))**2
            rbf_error += (y_boot[j] - np.dot(f_rbf, rbf_kernel(x_300, v, best_rbf_lam)))**2
        errors[i] = (poly_error - rbf_error) / 1000
    boot_5 = np.array(np.percentile(errors, 5))
    boot_95 = np.array(np.percentile(errors, 95))
    print("4.e: boot_error")
    print("boot_5: ", boot_5, ", boot_95: ", boot_95)

if __name__ == "__main__":
    main()
