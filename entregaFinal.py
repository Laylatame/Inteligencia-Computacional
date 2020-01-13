#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 12:56:53 2019

@author: laylatame
A01192934
"""

from scipy._lib._util import check_random_state

import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from jmetal.core.problem import BinaryProblem
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation, SPXCrossover, BitFlipMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.solution import BinarySolution


#Máquina de soporte vectorial

class MultiObjectiveTestSVM(object):

    def __init__(self, Xtrain, Ytrain,  kernel='sigmoid', gamma=0.1, degree=1,
                 C=1, coef0=0, maxEvaluations=100000, populationSize=100, seed=None ):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.seed = check_random_state(seed)
        self.maxEvaluations = maxEvaluations
        self.popsize = populationSize
        self.C = C
        self.coef0 = coef0
        
        self.model = None

    def train(self):
        self.model = SVC(kernel='sigmoid', C=self.C, degree=self.degree, coef0=self.coef0,
                                gamma=self.gamma, random_state=self.seed, verbose=False)

        self.model.fit(self.Xtrain, self.Ytrain)
        return self.model

    def predict(self, Xtest):
        # Obtener las predicciones
        self.y_pred = self.model.predict(Xtest)
        return self.y_pred

    def accuracy(self, Ytest):
        # Obtener la exactitud de las predicciones
        self.accuracy = accuracy_score(Ytest, self.y_pred)
        return self.accuracy


#Algoritmo multiobjetivo utilizando problema binario

class BinProblem(BinaryProblem):
    def __init__(self, X, Y, kernel = 'sigmoid', gamma = 0.1, degree = 1, 
                 C = 1, coef0 = 0):
        
        super(BinProblem, self).__init__()

        self.instances = np.shape(X)[0]
        self.attributes = np.shape(X)[1]
        self.Xtrain = X
        self.Ytrain = Y

        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.C = C
        self.coef0 = coef0
        
    def evaluate(self, solution: BinarySolution):
        instances = solution.variables[0]
        attributes = solution.variables[1]

        X = self.Xtrain[instances, :]
        X = X[:, attributes]
        Y = self.Ytrain[instances]

        noInst = np.shape(X)[0]
        noAttr = np.shape(X)[1]
        
        if (noAttr <= 1): return solution
        
        model = SVC(gamma = self.gamma, C = self.C, degree=self.degree, kernel = self.kernel)
        model.fit(X = X, y = Y)

        
        solution.objectives[0] = (1 - model.score(X, Y))
        solution.objectives[1] = len(model.support_)
        solution.objectives[2] = noInst
        solution.objectives[3] = noAttr

        return solution

    def create_solution(self):
        sol = BinarySolution(number_of_variables=2, number_of_objectives=4)
        sol.variables[0] = [True if random.randint(0, 1) == 0 else False for _ in range(self.instances)]
        sol.variables[1] = [True if random.randint(0, 1) == 0 else False for _ in range(self.attributes)]
        return sol
    
    def get_name(self):
        return 'BinProblem'


class MultiObjectiveTestBinary(object):
    
    def __init__(self, Xtrain, Ytrain, kernel = 'sigmoid', gamma = 0.1, degree = 1, 
                 C = 1, coef0 = 0, maxEvaluations = 10000, populationSize = 100, 
                 seed = None,):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.C = C
        self.coef0 = coef0
        self.seed = check_random_state(seed)
        self.maxEvaluations = maxEvaluations
        self.popsize = populationSize

        self.model = None
    
    def train(self):
        problem = BinProblem(X=self.Xtrain, Y=self.Ytrain, kernel=self.kernel, gamma=self.gamma, degree=self.degree, C=self.C, coef0=self.coef0)

        max_evaluations = self.maxEvaluations
        algorithm = NSGAII(
            problem=problem,
            population_size=self.popsize,
            offspring_population_size=self.popsize,
            mutation=BitFlipMutation(probability=1.0 / np.shape(self.Xtrain)[0]),
            crossover=SPXCrossover(probability=1.0),
            termination_criterion=StoppingByEvaluations(max=max_evaluations)
        )

        algorithm.run()
        front = algorithm.get_result()

        normed_matrix = normalize(list(map(lambda result : result.objectives, front)))
        
        scores = list(map(lambda item : sum(item), normed_matrix ))
        solution = front[ scores.index( min(scores))]
        
        self.instances =  solution.variables[0]
        self.attributes = solution.variables[1]

        X = self.Xtrain[self.instances, :]
        X = X[:, self.attributes]
        Y = self.Ytrain[self.instances]

        self.model = SVC(gamma = self.gamma, C = self.C, degree=self.degree, kernel = self.kernel)
        self.model.fit(X = X, y = Y)
        
        return self.model
        
    def predict(self, Xtest):
        self.y_pred = self.model.predict(Xtest[:, self.attributes])
        return self.y_pred

    def accuracy(self, Ytest):
        self.accuracy = accuracy_score(Ytest, self.y_pred)
        return self.accuracy


# Get data from file
X = np.loadtxt(fname='./MusicHackathon/X copy.csv', delimiter=',', dtype=None)
y = np.loadtxt(fname='./MusicHackathon/Y copy.csv', delimiter=',', dtype=None)


#Discreet data to 1-4
for y1 in range(len(y)):
    if (y[y1] < 25):
        y[y1] = 1
    elif (y[y1] >= 25 and y[y1] < 50):
        y[y1] = 2
    elif (y[y1] >= 50 and y[y1] < 75):
        y[y1] = 3
    else:
        y[y1] = 4


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)



print("-------------SVM---------------")

testSVM = MultiObjectiveTestSVM(Xtrain=X_train, Ytrain=y_train, maxEvaluations=1000)
model = testSVM.train()
y_pred = testSVM.predict(X_test)
accuracy = testSVM.accuracy(y_test)
print(accuracy)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))



print("-------------Binary Problem---------------")

testB = MultiObjectiveTestBinary(Xtrain=X_train, Ytrain=y_train)
testB.train()
y_pred = testB.predict(X_test)
accuracy = testB.accuracy(y_test)

print("Accuracy: ", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
print("Clasification Report") 
print(classification_report(y_test,y_pred))




print("-------------KNN---------------")

knn_scores = []
for k in range(1,y_train.shape[0]):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    knn_classifier.fit(X_train, y_train)
    knn_scores.append(knn_classifier.score(X_test, y_test))

print("Max Accuracy: ", max(knn_scores))




print("-------------Random Forest---------------")


rf_scores = []
estimators = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
#estimators = [10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
for i in estimators:
    rf_classifier = RandomForestClassifier(n_estimators = i, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    rf_scores.append(rf_classifier.score(X_test, y_test))


print("Max Accuracy: ", max(rf_scores))
