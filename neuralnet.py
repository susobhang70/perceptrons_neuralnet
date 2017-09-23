#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
from random import randint

numberPatterns = [1, 3, 6]
input_dimensions = 65
output_dimensions = 2
hidden_layers = 20
np.random.seed()
wji =  np.random.random((input_dimensions, hidden_layers)) - 1
wkj =  np.random.random((hidden_layers + 1, output_dimensions)) - 1
n = 0.7
theta = 0.001

def augment(w):
    temp = np.array([1])
    w = np.append(temp, w)
    return w

def mapping(a):
    if a == 1:
        return np.array([0, 0])
    elif a == 3:
        return np.array([0, 1])
    else:
        return np.array([1, 0])

def sigmoid(z, diff = False):
    value = 1/(1+np.exp(-z))
    if not diff:
        return value
    else:
        return (value* (1 - value))

def preprocess(filename):
    processedData = []
    numbers = []
    count = [0]*8
    with open(filename) as trainingdata:
        content = trainingdata.read().splitlines()
        content = content[21:]
        tempImage = [1]
        for i in range(len(content)):
            if (i % 33) % 4 == 0 and i % 33 != 0:
                for k in range(len(count)):
                    if count[k] >=8:
                        tempImage.append(1)
                    else:
                        tempImage.append(0)
                # print i, len(tempImage)
                count = [0]*8
            if (i - 32) % 33 == 0:
                value = int(content[i])
                # print value
                if value in numberPatterns:
                    numbers.append(value)
                    processedData.append(tempImage)
                tempImage = [1]
                continue
            for j in range(len(content[i])):
                count[j/4] += int(content[i][j])
    
    return numbers, processedData

def check(data, numbers, wji, wkj):
    np.random.seed()
    count = 0
    for it in range(len(data)):
        # randomIndex = randint(0, len(numbers) - 1)
        randomIndex = it
        xi = np.array(data[randomIndex])
        netj = np.dot(xi, wji)
        yj = sigmoid(netj)
        yj = augment(yj)
        netk = np.dot(yj, wkj)
        zk = sigmoid(netk)

        # print zk
        for i in range(len(zk)):
            if zk[i] >= 0.5:
                zk[i] = 1
            else:
                zk[i] = 0

        tk = mapping(numbers[randomIndex])
        if(zk[0] == tk[0] and zk[1] == tk[1]):
            count += 1
        else:
            # print it
            pass
        percent = float(count) * 100 / len(data)
        # print '('+str(it)+')', zk
    print "Correct results: " + str(percent) +"%"


def networkModel(data, numbers, wji, wkj):
    np.random.seed()
    while True:
        randomIndex = randint(0, len(numbers) - 1)
        # print randomIndex
        # randomIndex = it
        xi = np.array(data[randomIndex])
        netj = np.dot(xi, wji)
        yj = sigmoid(netj)
        yj = augment(yj)
        # print yj
        # print wkj
        # break
        netk = np.dot(yj, wkj)
        zk = sigmoid(netk)

        # print zk
        tk = mapping(numbers[randomIndex])
        backupwkj = wkj[:]
        deltak = []
        for k in range(output_dimensions):
            error = tk[k] - zk[k]
            dell = error * sigmoid(netk[k], True)
            deltak.append(dell)
            for j in range(hidden_layers + 1):
                delta = n * dell * yj[j]
                wkj[j][k] = wkj[j][k] + delta


        deltaj = []
        for j in range(hidden_layers):
            sum = 0
            for k in range(output_dimensions):
                # print backupwkj[j][k]
                sum += (backupwkj[j][k] * deltak[k])
                # print sum
            dell = sum * sigmoid(netj[j], True)
            # print dell
            deltaj.append(dell)
            for i in range(input_dimensions):
                delta = n * dell * xi[i]
                wji[i][j] = wji[i][j] + delta

        J = (tk - zk)
        mod = np.linalg.norm(J)
        JW = 0.5 * mod * mod
        if JW < theta:
            # print JW
            break
    return wji, wkj 

def main():
    trainingfile = "optdigits-orig.tra"
    crossvalidationfile = "optdigits-orig.cv"
    numbers, data = preprocess(trainingfile)
    testnumbers, testdata = preprocess(crossvalidationfile)
    # trainingnumbers = numbers[:100]
    # trainingdata = data[:100]
    trainedwji, trainedwkj = networkModel(data, numbers, wji, wkj)
    check(testdata, testnumbers, trainedwji, trainedwkj)
    # for i in trainedwji:
    #     print '[',
    #     for j in i:
    #         print j,
    #     print ']'
    # print
    # for i in trainedwkj:
    #     print '[',
    #     for j in i:
    #         print j,
    #     print ']'

if __name__ == '__main__':
    main()