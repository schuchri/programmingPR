#    Copyright 2016 Stefan Steidl
#    Friedrich-Alexander-Universität Erlangen-Nürnberg
#    Lehrstuhl für Informatik 5 (Mustererkennung)
#    Martensstraße 3, 91058 Erlangen, GERMANY
#    stefan.steidl@fau.de


#    This file is part of the Python Classification Toolbox.
#
#    The Python Classification Toolbox is free software: 
#    you can redistribute it and/or modify it under the terms of the 
#    GNU General Public License as published by the Free Software Foundation, 
#    either version 3 of the License, or (at your option) any later version.
#
#    The Python Classification Toolbox is distributed in the hope that 
#    it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#    warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
#    See the GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with the Python Classification Toolbox.  
#    If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import numpy.matlib


class LinearLogisticRegression(object):

    def __init__(self, learningRate = 0.5, maxIterations = 100):
        self.learning_rate = learningRate
        self.max_iterations = maxIterations
        self.labels = None
        self.input = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of the sigmoid function
    def sigmoid_p(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def fit(self, X, y):
        self.labels = y
        self.input = X
        weight1 = np.random.randn()
        weight2 = np.random.randn()
        print('Labels ->', y)
        print('Input ->', X)
        for i in range(self.max_iterations):
            # pick any random point within the dataset
            ri = np.random.randint(len(X))
            point = X[ri]
            z = point[0] * weight1 + point[1] * weight2
            prediction = self.sigmoid(z)

            # Compare it with what it should have been and calculate the cost.
            target = point[2]
            point_class = 0
            if self.labels[i] == 3:
                point_class = 1
            cost = np.square(prediction - point_class)
            # Take the derivative of each of the parameters.
            dcost_pred = 2 * (prediction - point_class)
            dpred_dz = self.sigmoid_p(z)
            dz_dw1 = point[0]
            dz_dw2 = point[1]

            # Chain the derivatives of the parameters with respect to the cost
            # Partial derivatives.
            dcost_dw1 = dcost_pred * dpred_dz * dz_dw1
            dcost_dw2 = dcost_pred * dpred_dz * dz_dw2

            # Subtract a small fraction of the cost from the parameters w1, w2, b
            w1 = w1 - self.learning_rate * dcost_dw1
            w2 = w2 - self.learning_rate * dcost_dw2


    def gFunc(self, X, theta):
        return None


    def predict(self, X):
        return None


