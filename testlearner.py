
    
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it    
import time
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import LinRegLearner as lrl


plt.style.use("fivethirtyeight")
np.random.seed(903430342)






def question_1():
    dataframe = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = dataframe.drop(["date", "EM"], axis=1).values
    Y = dataframe["EM"].values

    train_rows = int(0.6 * X.shape[0])
    trainX = X[:train_rows, :]
    trainY = Y[:train_rows]
    testX = X[train_rows:, :]
    testY = Y[train_rows:]

    train_rmse_values = []
    test_rmse_values = []
    leaf_size_values = np.arange(1, 20+1, dtype=np.uint32)
    for leaf_size in leaf_size_values:
        learner = dt.DTLearner(leaf_size=leaf_size, verbose=False)
        learner.addEvidence(trainX, trainY)
        train_predictionY = learner.query(trainX)
        train_rmse = math.sqrt(((trainY - train_predictionY) ** 2).sum() / trainY.shape[0])
        train_rmse_values.append(train_rmse)
        test_predictionY = learner.query(testX)
        test_rmse = math.sqrt(((testY - test_predictionY) ** 2).sum() / testY.shape[0])
        test_rmse_values.append(test_rmse)

    fig, ax = plt.subplots()
    pd.DataFrame({
        "Train RMSE": train_rmse_values,
        "Test RMSE": test_rmse_values
    }, index=leaf_size_values).plot(
        ax=ax,
        marker="o",
        color = ["black","red"],
        title="1. Error DTLearner vs leaf_size"
    )
    plt.xticks(leaf_size_values,leaf_size_values)
    plt.xlabel("Leaf size")
    plt.ylabel("Root Mean Square Error (RMSE)")
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig("Question_1.png")


def question_2():
    dataframe = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = dataframe.drop(["date", "EM"], axis=1).values
    Y = dataframe["EM"].values

    train_rows = int(0.6 * X.shape[0])
    trainX = X[:train_rows, :]
    trainY = Y[:train_rows]
    testX = X[train_rows:, :]
    testY = Y[train_rows:]

    train_rmse_values = []
    test_rmse_values = []
    
    leaf_size_values = np.arange(1, 20+1, dtype=np.uint32)
    for leaf_size in leaf_size_values:
        learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf_size}, bags=20, boost=False, verbose=False)
        learner.addEvidence(trainX, trainY)
        train_predictionY = learner.query(trainX)
        train_rmse = math.sqrt(((trainY - train_predictionY) ** 2).sum() / trainY.shape[0])
        train_rmse_values.append(train_rmse)
        test_predictionY = learner.query(testX)
        test_rmse = math.sqrt(((testY - test_predictionY) ** 2).sum() / testY.shape[0])
        test_rmse_values.append(test_rmse)

    fig, ax = plt.subplots()
    pd.DataFrame({
        "Train RMSE": train_rmse_values,
        "Test RMSE": test_rmse_values
    }, index=leaf_size_values).plot(
        ax=ax,
        marker="o",
        color = ["black","red"],
        title="2. Error of BagLearner vs leaf_size"
    )
    plt.xticks(leaf_size_values,leaf_size_values)
    plt.xlabel("Leaf size")
    plt.ylabel("Root Mean Square Error (RMSE)")
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig("Question_2.png")






def question_3a():
    dataframe = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = dataframe.drop(["date", "EM"], axis=1).values
    Y = dataframe["EM"].values

    # Time comparison
    training_time = []
    time_values_rt = []
    size_values = np.arange(1, X.shape[0], 30, dtype=np.uint64)
    for size_value in size_values:
        trainX = X[:size_value, :]
        trainY = Y[:size_value]
        # time to train for  DTLearner
        dt_learner = dt.DTLearner(leaf_size=1)
        start = time.time()
        dt_learner.addEvidence(trainX, trainY)
        end = time.time()
        training_time.append(end-start)
        # time to train for  RT
        start = time.time()
        rt_learner = rt.RTLearner(leaf_size=1)
        rt_learner.addEvidence(trainX, trainY)
        end = time.time()
        time_values_rt.append(end-start)

    fig, ax = plt.subplots()
    pd.DataFrame({
        "DTLearner training time": training_time,
        "RTLearner training time": time_values_rt
    }, index=size_values).plot(
        ax=ax,
        marker="o",
        color = ["black","red"],
        title="3a. Training time (DT & RT) on istanbul data"
    )
    plt.xlabel("Training set size")
    plt.ylabel("Time (s)")
    plt.legend(loc=2)
    plt.tight_layout()
    plt.savefig("Question_3a.png")


def question_3b():
    dataframe = pd.read_csv("./Data/Istanbul.csv", header=0)
    X = dataframe.drop(["date", "EM"], axis=1).values
    Y = dataframe["EM"].values

    
    # size by training set
    size_values_dt = []
    size_values_rt = []
    size_values = np.arange(1, X.shape[0], 30, dtype=np.uint64)
    for size_value in size_values:
        trainX = X[:size_value, :]
        trainY = Y[:size_value]
        # size dt
        dt_learner = dt.DTLearner(leaf_size=1)
        dt_learner.addEvidence(trainX, trainY)
        size_values_dt.append(dt_learner.tree.shape[0])
        # size rt
        rt_learner = rt.RTLearner(leaf_size=1)
        rt_learner.addEvidence(trainX, trainY)
        size_values_rt.append(rt_learner.tree.shape[0])

    fig, ax = plt.subplots()
    pd.DataFrame({
        "DT Learner size": size_values_dt,
        "RT Learner size": size_values_rt
    }, index=size_values).plot(
        ax=ax,
        marker="o",
        color = ["black","red"],
        title="3b. tree size by training set (istanbul data)"
    )
    plt.xlabel("Size of training set")
    plt.ylabel("Size of tree (No. nodes and leaves)")
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig("Question3b.png")


if __name__=="__main__":
    #test()
    print("Question 1..."); question_1(); print("ok")
    print("Question 2..."); question_2(); print("ok")
    
    print("Question 3a..."); question_3a(); print("ok") 
    print("Question 3b..."); question_3b(); print("ok.") 
    