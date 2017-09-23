#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np

def augment(w):
    return np.column_stack( [np.array([1]*(len(w))), w])

def normalize(w):
    return -1 * np.array(w)

def simplePerceptron(y, a):
    k = 0
    count = 0
    normalcount = 1
    while count <= len(y):
        # print np.dot(a, y[k])
        if np.dot(a, y[k]) < 0:
            count = 0
            a = a + y[k]
        count += 1
        normalcount += 1
        k = (k + 1) % len(y)
    x = np.array([0, 10])
    y = (-a[0] - a[1] * x) / a[2]
    plt.plot(x, y, color='g')
    print a
    return normalcount

def perceptronWithMargin(y, b, a):
    k = 0
    count = 0
    normalcount = 1
    while count <= len(y):
        # print np.dot(a, y[k])
        if np.dot(a, y[k]) < b:
            count = 0
            a = a + y[k]
        count += 1
        normalcount += 1
        k = (k + 1) % len(y)
    xq = np.array([0, 10])
    yq = (-a[0] - a[1]*xq)/a[2]
    plt.plot(xq, yq, color='b')
    print a
    return normalcount

def perceptronWithMarginAndRelaxation(y, b, eta, a):
    # doesn't converge
    k = 0
    count = 0
    normalcount = 1
    while count <= len(y):
        if np.dot(a, y[k]) < b:
            count = 0
            factor = eta * (b - np.dot(a, y[k]))
            factor = factor / (np.linalg.norm(y[k]) * np.linalg.norm(y[k]))
            a = a + ( factor * y[k] )
        count += 1
        normalcount += 1
        k = (k + 1) % len(y)
    xq = np.array([0, 10])
    yq = (-a[0] - a[1]*xq)/a[2]
    plt.plot(xq, yq, color='r')
    print a
    return normalcount

def LMS(y, b, eta, theta, a):
    k = 0
    normalcount = 1
    while True:
        neweta = eta / normalcount
        product = neweta * (b - np.dot(a, y[k]))
        newvector = product * y[k]
        newvalue = np.linalg.norm(newvector)
        a = a + newvector
        if newvalue < theta:
            break
        normalcount += 1
        k = (k + 1) % len(y)
    x = np.array([0, 10])
    y = (-a[0] - a[1] * x) / a[2]
    plt.plot(x, y, color='y')
    print a
    return normalcount

def main():
    w1 = np.array([[2, 7], [8, 1], [7, 5], [6, 3], [7, 8], [5, 9], [4, 5]])
    w2 = np.array([[4, 2], [-1, -1], [1, 3], [3, -2], [5, 3.25], [2, 4], [7, 1]])
    b = 2
    eta1 = 2
    eta2 = 0.7
    theta = 0.2
    a = [1] * 3
    # a = [-85, 9, 9]
    # a = [100, -10, -10]
    # a = [ 0.68948792, 0.81626883,  0.39491738]
    # a = np.random.rand(3)
    print a

    y1 = []
    y2 = []
    y1 = augment(w1)
    y2 = augment(w2)
    y2 = normalize(y2)
    y = np.append(y1, y2, axis = 0)
    # it4= 0

    it1 = simplePerceptron(y, a)
    it2 = perceptronWithMargin(y, b, a)
    it3 = perceptronWithMarginAndRelaxation(y, b, eta1, a)
    it4 = LMS(y, b, eta2, theta, a)


    # print it1, it2, it3, it4
    plt.legend(['Single Sample Perceptron', 'Perceptron with margin', 'Perceptron with Margin and Relaxation', 'LMS'])
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