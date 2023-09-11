import csv
import sys
import matplotlib.pyplot as plt
import numpy as np

def readFile(filepath):
    dictionaries = []
    with open(filepath, newline='') as file:
        dictRead = csv.DictReader(file)
        for row in dictRead:
            dictionaries.append(row)

    years = np.zeros(len(dictionaries), int)
    days = np.zeros(len(dictionaries), int)

    for index in range(0, len(dictionaries)):
        years[index] = dictionaries[index]['year']
        days[index] = dictionaries[index]['days']

    return years, days


# create matrix X
def Q3a(xArr):
    # create necessary array
    featureVect = np.ones((2, len(xArr)), int)  # creates a 2 x n array
    # fill in feature vectors
    for column in range(0, len(xArr)):
        featureVect[1][column] = xArr[column]
    # transpose the feature vectors and return
    featureVect = np.transpose(featureVect)
    return featureVect


# create vector Y
def Q3b(yArr):
    yVect = np.ones((len(yArr)), int)
    for index in range(0, len(yArr)):
        yVect[index] = yArr[index]
    return yVect


# matrix product of X^TX
def Q3c(X):
    xTrans = np.transpose(X)
    Z = np.dot(xTrans, X)
    return Z


# Inverse of matrix product
def Q3d(Z):
    return np.linalg.inv(Z)


# PI
def Q3e(X):
    return np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X))


# TODO: double check inputs and outputs
# hat  beta
def Q3f(PI, Y):
    return np.dot(PI, Y)


def Q4(beta_0, beta_1, x_test):
    x_test = np.array(x_test)
    y_hat = beta_0 + np.dot(beta_1, x_test)
    return y_hat


def Q5(beta_1):
    a = "Sign"
    b = "Answer"
    if beta_1 == 0:
        a = "="
        b = "There will be a consistent amount of ice days; the Lake Mendota will remain frozen for those days"
    elif beta_1 < 0:
        a = "<"
        b = "There will be a decrease of ice days; Lake Mendota will eventually not freeze"
    elif beta_1 > 0:
        a = ">"
        b = "There will be an increase in ice days; Lake Mendota will remain frozen"
    return a, b


def Q6(beta_hat0, beta_hat1):
    x = beta_hat0*(-1) / beta_hat1
    b = "X is a compelling prediction as data trends for Mendota ice show a general decrease in ice days overall,\nwith spikes and sharp decreases in ice days"
    return x, b


if __name__ == "__main__":
    file = sys.argv[1]  # contains the first argument as string
    years, days = readFile(file)
    X = Q3a(years)
    Y = Q3b(days)
    Z = Q3c(X)
    I = Q3d(Z)
    PI = Q3e(X)
    hat_beta = Q3f(PI, Y)
    y_test = Q4(hat_beta[0], hat_beta[1], 2021)
    Q5a, Q5b = Q5(hat_beta[1])
    Q6a, Q6b = Q6(hat_beta[0], hat_beta[1])

    x_ticks = range(len(years))
    plt.plot(x_ticks, Y)
    plt.xticks(x_ticks, X[:, 1])
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.savefig("plot.png")  # save produced plot as plot.png
    plt.show()

    print("Q3a:")
    print(X)
    print("Q3b:")
    print(Y)
    print("Q3c:")
    print(Z)
    print("Q3d:")
    print(I)
    print("Q3e:")
    print(PI)
    print("Q3f:")
    print(hat_beta)
    print("Q4: " + str(y_test))
    print("Q5a: " + Q5a)
    print("Q5b: " + Q5b)
    print("Q6a:", Q6a)
    print("Q6b: " + Q6b)
