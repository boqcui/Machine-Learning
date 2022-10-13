if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    # raise NotImplementedError("Your Code Goes Here")
    y_train = df_train.values[:,0]#.reshape(len(df_train),1)
    X_train = df_train.values[:,1:].reshape(len(df_train),df_train.shape[1]-1)
    y_test = df_test.values[:,0]#.reshape(len(df_test),1)
    X_test = df_test.values[:,1:].reshape(len(df_test),df_test.shape[1]-1)
	
    lambda_max = np.max(np.sum(2*X_train.values*(y_train.values-np.mean(y_train.values))[:, None], axis=0))

	

if __name__ == "__main__":
    main()
