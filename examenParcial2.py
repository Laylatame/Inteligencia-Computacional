"""
Created on Oct 16 20:24:18 2019

Entrega Parcial 2
Inteligencia Computacional

Layla Tame
A01192934
"""


from scipy._lib._util import check_random_state

import numpy as np
from random import randint

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.problem import IntegerProblem

from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score


print("Layla Tame | A01192934")
print("Apegandome al Codigo de Etica de los Estudiantes del Tecnologico de Monterrey, me comprometo a que mi actuacion en este examen este regida por la honestidad academica.")


class Ejemplo(IntegerProblem):
    
    def __init__(self, X, Y):
        
        super(Ejemplo, self).__init__()
        
        self.number_of_variables = 5
        self.number_of_objectives = 4
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE, self.MINIMIZE, self.MINIMIZE]
        self.lower_bound = [2**(-10), 1, 2**(-3), 0, 0]
        self.upper_bound = [2**(3), 2**(10), 2**(10), 10, 3]
        self.obj_labels = ['error', 'NoSV', 'inst', 'atr']
        self.Xtrain = X
        self.Ytrain = Y

    
    def evaluate(self, solution: IntegerProblem):
        
        solution.masks = [[True if randint(0, 1) == 0 else False for _ in range(np.shape(Xtrain)[0])],
                            [True if randint(0, 1) == 0 else False for _ in range(np.shape(Xtrain)[1])]]
        
        gamma = solution.variables[0]
        C = solution.variables[1]
        coef0 = solution.variables[2]
        degree = solution.variables[3]
        kernel = solution.variables[4]
         
        # Entrenar
        model = SVM(Xtrain = Xtrain, Ytrain = Ytrain, gamma = gamma, C = C, kernel = kernel, coef0 = coef0, degree = degree).train()
        model.fit(X = self.Xtrain, y = self.Ytrain)
        
        error = 1 - model.score(self.Xtrain, self.Ytrain)
        noSV = len(model.support_)
        
        solution.objectives[0] = error
        solution.objectives[1] = noSV

        return solution
    
    
    def get_name(self):
        return 'Ejemplo'


class MultiObjectiveTest(object):
    
    def __init__(self, Xtrain, Ytrain,  kernel = 'rbf', gamma = 0.1, degree = 1,
                 C = 1, coef0 = 0, maxEvaluations = 100000, populationSize = 100, seed = None):
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
        
        
    def train(self):
        svmT = Ejemplo(X=self.Xtrain, Y=self.Ytrain)
        
        alg = NSGAII(
                           problem = svmT,
                           population_size = self.popsize,
                           offspring_population_size = self.popsize,
                           mutation = PolynomialMutation(probability = 1.0 / svmT.number_of_variables, distribution_index = 20),
                           crossover = SBXCrossover(probability = 1.0, distribution_index = 20),
                           termination_criterion = StoppingByEvaluations(max = self.maxEvaluations)
                           )
                                       
        alg.run()
        front = alg.get_result()
                          
                           
        # Normalizar y obtener minimo
        normM = normalize(list(map(lambda result : result.objectives, front)))
                           
        weights = list(map(lambda item : sum(item), normM))
        solution = front[ weights.index(min(weights))]
                           
        self.gamma = solution.variables[0]
        self.C = solution.variables[1]
        self.coef0 = solution.variables[2]
        self.degree = solution.variables[3]
        self.kernel = solution.variables[4]
                           
        self.instances = solution.masks[0]
        self.attributes = solution.masks[1]
 
        Xtrain = self.Xtrain[self.instances, :]

        self.model = SVM(Xtrain = Xtrain[:, self.attributes], Ytrain = self.Ytrain[self.instances],  kernel = self.kernel, C = self.C, degree = self.degree, coef0 = self.coef0,
                         gamma = self.gamma, seed = self.seed).train()

        return self.model
    
    
    def predict(self, Xtest):
        X = Xtest[:, self.attributes]
        self.y_pred = self.model.predict(X)
        return self.y_pred
    
    
    def accuracy(self, Ytest):
        print("Accuracy:", accuracy_score(Ytest, self.y_pred))


#Load Data Sample
Xtrain = np.loadtxt(fname='./datasetCI2019/echocardiogram/Xtrain.csv', delimiter=',', dtype=None)
Ytrain = np.loadtxt(fname='./datasetCI2019/echocardiogram/Ytrain.csv', delimiter=',', dtype=None)
Xtest = np.loadtxt(fname='./datasetCI2019/echocardiogram/Xtest.csv', delimiter=',', dtype=None)
Ytest = np.loadtxt(fname='./datasetCI2019/echocardiogram/Ytest.csv', delimiter=',', dtype=None)


#Escoger archivo de datos y cargar a matrix y vector
#file_load = str(input("Que archivo desea abrir? "))

"""
if file_load == 'echocardiogram':
    Xtrain = np.loadtxt(fname='./datasetCI2019/echocardiogram/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/echocardiogram/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/echocardiogram/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/echocardiogram/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'fertility':
    Xtrain = np.loadtxt(fname='./datasetCI2019/fertility/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/fertility/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/fertility/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/fertility/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'flags':
    Xtrain = np.loadtxt(fname='./datasetCI2019/flags/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/flags/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/flags/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/flags/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'heart-switzerland':
    Xtrain = np.loadtxt(fname='./datasetCI2019/heart-switzerland/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/heart-switzerland/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/heart-switzerland/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/heart-switzerland/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'low-res-spect':
    Xtrain = np.loadtxt(fname='./datasetCI2019/low-res-spect/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/low-res-spect/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/low-res-spect/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/low-res-spect/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'molec-biol-splice':
    Xtrain = np.loadtxt(fname='./datasetCI2019/molec-biol-splice/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/molec-biol-splice/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/molec-biol-splice/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/molec-biol-splice/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'monks-2':
    Xtrain = np.loadtxt(fname='./datasetCI2019/monks-2/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/monks-2/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/monks-2/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/monks-2/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'pittsburg-bridges-SPAN':
    Xtrain = np.loadtxt(fname='./datasetCI2019/pittsburg-bridges-SPAN/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/pittsburg-bridges-SPAN/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/pittsburg-bridges-SPAN/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/pittsburg-bridges-SPAN/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'spambase':
    Xtrain = np.loadtxt(fname='./datasetCI2019/spambase/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/spambase/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/spambase/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/spambase/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'steel-plates':
    Xtrain = np.loadtxt(fname='./datasetCI2019/steel-plates/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/steel-plates/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/steel-plates/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/steel-plates/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'teaching':
    Xtrain = np.loadtxt(fname='./datasetCI2019/teaching/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/teaching/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/teaching/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/teaching/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'titanic':
    Xtrain = np.loadtxt(fname='./datasetCI2019/titanic/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/titanic/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/titanic/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/titanic/Ytest.csv', delimiter=',', dtype=None)
if file_load == 'yeast':
    Xtrain = np.loadtxt(fname='./datasetCI2019/yeast/Xtrain.csv', delimiter=',', dtype=None)
    Ytrain = np.loadtxt(fname='./datasetCI2019/yeast/Ytrain.csv', delimiter=',', dtype=None)
    Xtest = np.loadtxt(fname='./datasetCI2019/yeast/Xtest.csv', delimiter=',', dtype=None)
    Ytest = np.loadtxt(fname='./datasetCI2019/yeast/Ytest.csv', delimiter=',', dtype=None)
else:
    print("No hay archivo llamado así")
"""



predFinal = MultiObjectiveTest(Xtrain=Xtrain, Ytrain=Ytrain, maxEvaluations=1000)
model = predFinal.train()
print("Prediction: ", predFinal.predict(Xtest))
print("Answer: ", accuracy_score(Ytest, predFinal.predict(Xtest)))
