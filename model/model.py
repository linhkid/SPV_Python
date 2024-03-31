import os
import numpy as np
from sklearn.base import BaseEstimator

class Model(BaseEstimator):
    def buildClassifier(self):
        raise NotImplementedError
    
    def evaluateFunction(self, sourceFile):
        raise NotImplementedError
    
    def predict(self, instance):
        raise NotImplementedError
    
    def distributionForInstance(self, instance):
        raise NotImplementedError
    
    def computeGrad(self, inst, probs, x_C):
        raise NotImplementedError
    
    def computeGradAndUpdateParameters(self, instance, probs, x_C):
        raise NotImplementedError
    
    def evaluateFunction(self, cvInstances):
        raise NotImplementedError
    
    def update(self, row):
        raise NotImplementedError