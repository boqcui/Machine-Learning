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
    # get data
    y_train = df_train.values[:,0]#.reshape(len(df_train),1)
    X_train = df_train.values[:,1:].reshape(len(df_train),df_train.shape[1]-1)
    y_test = df_test.values[:,0]#.reshape(len(df_test),1)
    X_test = df_test.values[:,1:].reshape(len(df_test),df_test.shape[1]-1)

    n, d = X_train.shape
    lam_max = 2*np.max(np.abs(np.dot(y_train.T - np.mean(y_train), X_train)))
    cur_lam = lam_max
    lam_list = [cur_lam]
    
    prev_w, b_train = train(X_train, y_train, _lambda=cur_lam)
    error_train = [np.mean(np.square(np.dot(X_train, prev_w) + b_train - y_train))]
    error_test = [np.mean(np.square(np.dot(X_test, prev_w) + b_train - y_test))]
    W = np.expand_dims(prev_w, axis=1)

    while cur_lam >= 0.01:
        lam_list.append(cur_lam)
        cur_lam = cur_lam / 2
        w_train, b_train = train(X_train, y_train, _lambda=cur_lam, start_weight=prev_w)
        error_train.append(np.mean(np.square(np.dot(X_train, prev_w) + b_train - y_train)))
        error_test.append(np.mean(np.square(np.dot(X_test, prev_w) + b_train - y_test)))
        W = np.concatenate((W, np.expand_dims(w_train, axis=1)), axis=1)
        prev_w = np.copy(w_train)

    # 6.c ------------------------------------------------------------------
    plt.figure(1)
    plt.xscale('log')
    plt.plot(lam_list, np.count_nonzero(W, axis=0))
    plt.xlabel('Lambda')
    plt.ylabel('Nonzeros in w')
    plt.title('6.c')
    plt.show()

    # 6.d ------------------------------------------------------------------
    # column index
    agePct12t29 = df_train.columns.get_loc("agePct12t29") - 1
    pctWSocSec = df_train.columns.get_loc("pctWSocSec") - 1
    pctUrban = df_train.columns.get_loc("pctUrban") - 1
    agePct65up = df_train.columns.get_loc("agePct65up") - 1
    householdsize = df_train.columns.get_loc("householdsize") - 1

    plt.figure(2)
    plt.xscale('log')
    plt.plot(lam_list, np.reshape(W[agePct12t29, :], (len(lam_list))))
    plt.plot(lam_list, np.reshape(W[pctWSocSec, :], (len(lam_list))))
    plt.plot(lam_list, np.reshape(W[pctUrban, :], (len(lam_list))))
    plt.plot(lam_list, np.reshape(W[agePct65up, :], (len(lam_list))))
    plt.plot(lam_list, np.reshape(W[householdsize, :], (len(lam_list))))
    plt.legend(["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"])
    plt.xlabel('Lambda')
    plt.ylabel('weight')
    plt.title('6.d')
    plt.show()

    # 6.e ------------------------------------------------------------------
    plt.figure(3)
    plt.xscale('log')
    plt.plot(lam_list, error_train)
    plt.plot(lam_list, error_test)
    plt.legend(["Training Error", "Testing Error"])
    plt.xlabel('Lambda')
    plt.ylabel('Error')
    plt.title('6.e')
    plt.show()

    # 6.f ------------------------------------------------------------------
    w30, b30 = train(X_train, y_train, _lambda=30, start_weight=None)
    largest = np.argmax(w30)
    smallest = np.argmin(w30)
    
    print ("index of larget feature: ", largest, " coefficien: ", w30[largest])
    print ("index of smallest feature: ", smallest, " coefficien: ", w30[smallest])



if __name__ == "__main__":
    main()
