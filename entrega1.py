#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 20:24:18 2019

Entrenamiento de una SVM Usando Algoritmos Evolutivos con Manejador de Restricciones
Inteligencia Computacional

Layla Tame
A01192934
"""

from scipy._lib._util import check_random_state
import numpy as np
import random
from numpy import genfromtxt


print("Layla Tame | A01192934")
print("Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey, me comprometo a que mi actuacion en este examen este regida por la honestidad academica.")

class EvolutionaryConstrainedSVM(object):
    
    def __init__(self, Xtrain, Ytrain, kernel = 'rbf', gamma = 0.1, degree = 1, 
                 C = 1, coef0 = 0, generations = 1000, populationSize = 100, 
                 constraint = 'feasibilityRules', CR = 0.9, F = 0.1, seed = None, 
                 threshold = 0.0001, feasibilityProbability = 0.5):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.seed = check_random_state(seed)
        self.generations = generations
        self.popsize = populationSize
        self.constraint = constraint
        self.CR = CR
        self.F = F
        self.threshold = 0.0001
        self.fp = feasibilityProbability
        self.C = C
        self.coef0 = coef0
        self.respFinal = None
        
    
    def _linear_kernel(self, X, Y):
        kernel = np.ones((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(Y)):
                kernel[i][j] = np.dot(X[i], np.transpose(Y[i]))
        return kernel
    
    def _polynomial_kernel(self, X, Y):
        kernel = np.ones((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(Y)):
                kernel[i][j] = ((np.dot(X[i], np.transpose(Y[j]))) + self.CR) ** self.degree
        return kernel
    
    def _rbf_kernel(self, X, Y):
        kernel = np.ones((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(Y)):
                kernel[i][j] = np.exp(-self.gamma * (np.linalg.norm(np.subtract(X[i], Y[j]))) ** 2)
        return kernel
    
    
    
    def _feasibility_rules(self, parents, fparents, cparents, offspring, 
                             foffspring, coffspring):
        # Revisar si los constrains son factibles
        # para evaluar si parents o offspring tiene mejor fitness
            #1. offspring mejor que parent
            #0. parent mejor que offspring
            
        if cparents != 0 and coffspring != 0:
            if cparents >= coffspring:
                return 1
            else:
                return 0 
        if cparents != 0 and coffspring == 0:
            return 1
        if cparents == 0 and coffspring == 0:
            if fparents <= foffspring:
                return 1 
            else:
                return 0
        if cparents == 0 and coffspring != 0:
            return 0 
    

    
    def _epsilon_constraint(self, parents, fparents, cparents, offspring, 
                             foffspring, coffspring):
        # Write your code here
        return 0
    
    def _stochastic_ranking(self, parents, fparents, cparents, offspring, 
                             foffspring, coffspring):
        # Write your code here
        return 0
    
    def _coevolution(self, parents, fparents, cparents, offspring, 
                             foffspring, coffspring):
        # Write your code here
        return 0
    
    
    

    
 
    def __compute_kernel(self, X, Y):
        if self.kernel == 'linear':
            return self._linear_kernel(X, Y)
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel(X, Y)
        elif self.kernel == 'rbf':
            return self._rbf_kernel(X, Y)
        else:
            raise ValueError("Please select a valid kernel function")
            

            
            
            

    def __constraint_handler(self, parents, fparents, cparents, offspring, 
                             foffspring, coffspring):
        if self.constraint == 'feasibilityRules':
            return self._feasibility_rules(parents, fparents, cparents, offspring, 
                             foffspring, coffspring)
        elif self.constraint == 'epsilonConstraint':
            return self._epsilon_constraint(parents, fparents, cparents, offspring, 
                             foffspring, coffspring)
        elif self.constraint == 'stochasticRanking':
            return self._stochastic_ranking(parents, fparents, cparents, offspring, 
                             foffspring, coffspring)
        elif self.constraint == 'coevolution':
            return self._coevolution(parents, fparents, cparents, offspring, 
                             foffspring, coffspring)
        else:
            raise ValueError("Please select a valid constraint-handling technique")
    
  
    
    def calcFitness(self, alpha):
        aY = np.multiply(alpha, np.transpose(self.Ytrain))
        p1 = np.dot((np.dot(aY, self.k)), np.transpose(aY))
        p2 = np.dot(alpha, np.ones(np.shape(Ytrain)[0]))
        r = (1/2) * p1 * (-p2)
        return r
    
    
    

    def train(self):
        bounds = np.array(np.multiply([0, self.C], np.ones((np.shape(self.Xtrain)[0],2))))
        kernel = self.__compute_kernel(self.Xtrain, self.Xtrain)

        #Crear valores aleatorios para alpha0
        parents = np.random.uniform(low= 0, high=self.C, size=(self.popsize, np.shape(bounds)[0]))
        
        #Obtener constraint y fitness inicial para parents
        cparents = np.dot(parents, self.Ytrain)
        fparents = self.calcFitness(parents)
        

        #Iterar para n generaciones
        iGen = 0
        while iGen < self.generations+1:
            
            #Iterar para el tamaño de la poblacion
            for i in range(self.popsize):
                
                #Elegir dos vectores de alpha para mutar
                vec1, vec2 = parents[np.random.choice((m for m in range(self.popsize)),2)]
            
                #Elegir offspring
                offspring = np.where((np.random.rand(np.shape(bounds)[0]) < self.CR))
            
                #Calcular constraint y fitness de los hijos
                coffspring = np.dot(offspring, self.Ytrain)
                foffspring = self.calcFitness(offspring)
            
                #Aplicar Constraint Handler para elegir entre parents y offspring para continuar con el algoritmo
                #if 1 -> offspring, 0 -> parents
                if self.__constraint_handler(parents[i], fparents[i], cparents[i], offspring, foffspring, coffspring) == 1:
                    fparents[i] = foffspring
                    parents[i] = offspring
            
            iGen += 1
            
        
        return parents[np.argmin(fparents)]
    
    
    
        
        
        
        
    def predict(self, Xtest):
        
        parentsY = np.multiply(np.transpose(self.respFinal), self.Ytrain)
        #Volver a calcular kernel con la matriz de entrada y la de prueba
        kernel = self.__compute_kernel(Xtest, self.Xtrain)
        yFinal = np.dot(kernel, parentsY)
        
        b = (1/2) * (np.max([yFinal[x] for x in range(np.shape(yFinal)[0]) if self.Ytrain[x] == -1 and self.respFinal[x] > 0])) + (np.min([yFinal[x] for x in range(np.shape(yFinal)[0]) if self.Ytrain[x] ==  1 and self.respFinal[x] > 0]))
        
        
        return np.sign(yFinal + b)
    




#Ejecturar programa
    
    #Escoger archivo de datos y cargar a matrix y vector
    file_load = str(input("Que archivo desea abrir? "))

    if file_load == 'echocardiogram':
        Xtrain = np.loadtxt(fname='./datasetCI2019/echocardiogram/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/echocardiogram/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/echocardiogram/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'fertility':
        Xtrain = np.loadtxt(fname='./datasetCI2019/fertility/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/fertility/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/fertility/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'flags':
        Xtrain = np.loadtxt(fname='./datasetCI2019/flags/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/flags/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/flags/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'heart-switzerland':
        Xtrain = np.loadtxt(fname='./datasetCI2019/heart-switzerland/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/heart-switzerland/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/heart-switzerland/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'low-res-spect':
        Xtrain = np.loadtxt(fname='./datasetCI2019/low-res-spect/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/low-res-spect/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/low-res-spect/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'molec-biol-splice':
        Xtrain = np.loadtxt(fname='./datasetCI2019/molec-biol-splice/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/molec-biol-splice/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/molec-biol-splice/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'monks-2':
        Xtrain = np.loadtxt(fname='./datasetCI2019/monks-2/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/monks-2/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/monks-2/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'pittsburg-bridges-SPAN':
        Xtrain = np.loadtxt(fname='./datasetCI2019/pittsburg-bridges-SPAN/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/pittsburg-bridges-SPAN/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/pittsburg-bridges-SPAN/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'spambase':
        Xtrain = np.loadtxt(fname='./datasetCI2019/spambase/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/spambase/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/spambase/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'steel-plates':
        Xtrain = np.loadtxt(fname='./datasetCI2019/steel-plates/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/steel-plates/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/steel-plates/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'teaching':
        Xtrain = np.loadtxt(fname='./datasetCI2019/teaching/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/teaching/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/teaching/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'titanic':
        Xtrain = np.loadtxt(fname='./datasetCI2019/titanic/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/titanic/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/titanic/Xtest.csv', delimiter=',', dtype=None)
    if file_load == 'yeast':
        Xtrain = np.loadtxt(fname='./datasetCI2019/yeast/Xtrain.csv', delimiter=',', dtype=None)
        Ytrain = np.loadtxt(fname='./datasetCI2019/yeast/Ytrain.csv', delimiter=',', dtype=None)
        Xtest = np.loadtxt(fname='./datasetCI2019/yeast/Xtest.csv', delimiter=',', dtype=None)
    else:
        print("No hay archivo llamado así")
    

    predFinal = EvolutionaryConstrainedSVM(Xtrain, Ytrain)
    #prefFinal = EvolutionaryConstrainedSVM()
    predFinal.train()
    predFinal.predict(Xtest)
    
    
    
    
    
    
    
    
    
