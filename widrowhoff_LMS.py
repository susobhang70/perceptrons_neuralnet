#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np

def augment(w):
    return np.column_stack( [np.array([1]*(len(w))), w])

def normalize(w):
    return -1 * np.array(w)

def LMS(y, b, eta, theta, a, color):
    k = 0
    normalcount = 1
    while True:
        neweta = eta / normalcount
        product = neweta * (b - np.dot(a, y[k]))
        newvector = product * y[k]
        newvalue = np.linalg.norm(newvector)
        # print newvalue
        a = a + newvector
        if newvalue < theta:
            break
        normalcount += 1
        k = (k + 1) % len(y)
    x = np.array([0, 10])
    y = (-a[0] - a[1] * x) / a[2]
    plt.plot(x, y, color=color)
    print a
    return normalcount

def main():
    w1 = np.array([[2, 7], [8, 1], [7, 5], [6, 3], [7, 8], [5, 9], [4, 5]])
    w2 = np.array([[4, 2], [-1, -1], [1, 3], [3, -2], [5, 3.25], [2, 4], [7, 1]])
    b = 2
    eta1 = 2
    eta2 = 0.7
    theta = 0.2
    # a = [1] * 3
    a = [-85, 9, 9]
    # print a

    y1 = []
    y2 = []
    y1 = augment(w1)
    y2 = augment(w2)
    y2 = normalize(y2)
    y = np.append(y1, y2, axis = 0)
    it4 = LMS(y, b, eta2, theta, a, 'r')

    w1 = np.array([[2, 7], [8, 1], [7, 5], [6, 3], [7, 8], [5, 9], [4, 5], [4, 1], [3, -1]])
    w2 = np.array([[4, 2], [-1, -1], [1, 3], [3, -2], [5, 3.25], [2, 4], [7, 1]])
    y1 = augment(w1)
    y2 = augment(w2)
    y2 = normalize(y2)
    y = np.append(y1, y2, axis = 0)
    it4 = LMS(y, b, eta2, theta, a, 'g')

    plt.legend(['LMS without additional points', 'LMS with linearly inseparable data'])
    plotGraph(w1, w2)

def plotGraph(w1, w2):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in w1:
        x1.append(i[0])
        y1.append(i[1])

    for i in w2:
        x2.append(i[0])
        y2.append(i[1])

    plt.scatter(x1, y1, color='r')
    plt.scatter(x2, y2, color='b')
    plt.show()

if __name__ == '__main__':
    main()