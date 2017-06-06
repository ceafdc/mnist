#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import mlp

from sys import stderr


def read_mnist_csv(path, n, with_class, with_header=True):

    X = np.empty((n, 28 * 28))  # 28 * 28 is number of pixels
    if with_class:
        Y = np.zeros((n, 10), dtype=int)  # 4 bits to store up to 10

    threshold = 127

    with open(path) as file:
        if with_header:
            file.readline()  # skip first line
        for i, line in enumerate(file):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            line = line.split(',')
            if with_class:
                y = int(line[0])
                Y[i, y] = 1
                line = line[1:]
            c = 0  # will hold number of pixels above threshold
            for j, px in enumerate(line):
                px = int(px)
                if px > threshold:
                    c += 1
                X[i, j] = (255 - px)/255  # normalize and invert color
            # X[i, 28 * 28] = c/(28 * 28)  # normalize

    if with_class:
        return X, Y
    else:
        return X, None


def save_model(model):
    layers = model['layers']

    np.save('hidden.npy', layers['hidden'])
    np.save('output.npy', layers['output'])

if __name__ == '__main__':

    try:
        X = np.load('train_X.npy')  # trying to read in
        Y = np.load('train_Y.npy')  # bynary, which is
        test = np.load('test.npy')  # much faster
        print('finish reading files from binary.',
              X.shape, Y.shape, test.shape,
              file=stderr)
    except:
        X, Y = read_mnist_csv('train.csv', 42000, with_class=True)
        test, _ = read_mnist_csv('test.csv', 28000, with_class=False)
        np.save('train_X.npy', X)  # saving in np bynary
        np.save('train_Y.npy', Y)  # for faster loading
        np.save('test.npy', test)  # next time
        print('finish reading files from csv.',
              X.shape, Y.shape, test.shape,
              file=stderr)

    shuffle_ids = np.arange(X.shape[0])
    np.random.shuffle(shuffle_ids)
    X = X[shuffle_ids]
    Y = Y[shuffle_ids]
    train_X = X[:3000]
    train_Y = Y[:3000]

    model = mlp.MLP([train_X.shape[1], 500, train_Y.shape[1]])

    model.train(train_X, train_Y, threshold=0.005)
    # model = mlp.MLP.load('500.mlp')
    print('finished training', file=stderr)

    kaggle = False  # Kaggle submission
    if kaggle:
        print('ImageId', 'Label', sep=',')
        for idx, x_p in enumerate(test, start=1):
            f_nets, _ = model.solve(x_p)
            o_p = f_nets[-1]
            answer = max(range(len(o_p)), key=lambda x: o_p[x])
            if idx % 100 == 0:
                print('%6.2f%%' % (idx/len(test) * 100), file=stderr)
            print(idx, answer, sep=',')
    else:
        test_X = X[3000:13000]
        test_Y = Y[3000:13000]

        correct = 0
        for idx, x_p, y_p in zip(range(len(test_X)), test_X, test_Y):
            f_nets, _ = model.solve(x_p)
            o_p = f_nets[-1]
            which_max_o = max(range(len(o_p)), key=lambda x: o_p[x])
            which_max_e = max(range(len(y_p)), key=lambda x: y_p[x])
            if which_max_e == which_max_o:
                correct += 1

            if idx % 100 == 0:
                print('testing: %6.2f%%' % (idx/len(test) * 100), file=stderr)

        print(file=stderr)
        print('Result: ', correct/len(test_X) * 100, '%', sep='', file=stderr)
