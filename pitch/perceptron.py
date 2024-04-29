import numpy as np
import math

class p():
    def __init__(self, dataSize:int, func_act:str=None):
        self.func_act = func_act
        self.b = 0 if self.func_act == 'sigmoid' else 0.01
        self.ws = self.__weight_init(dataSize)
        self.d = 0

    def __str__(self):
        return f"Weights: {self.ws}({self.ws.shape})   biais: {self.b}   derivée: {self.d}   activation: {self.func_act}"

    def __fagr(self, xi:np.array) -> float: # Agragate Function
            return np.dot(xi, self.ws) + self.b

    def __fact(self, x:float) -> int: # Activation Function
        match self.func_act:
            case 'heaviside':   return 1 if x > 0 else 0
            case 'relu':        return max(0, x)
            case 'leaky_relu':  return max(x * 0.01, x)
            case 'sigmoid' :    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))
            # Softmax is implemented in the model not on Neurones !
            case _ :            return x

    def __weight_init(self, datasize):
        match self.func_act:
            case 'relu' | 'leaky_relu': # methode He
                limit = np.sqrt(2 / datasize) # Uniforme pour simplification
                return np.random.uniform(-limit, limit, datasize)
            case 'tanh' | 'sigmoid': # methode Xavier (ou Glorot)
                limit = np.sqrt(6 / datasize) # Uniforme pour simplification
                return np.random.uniform(-limit, limit, datasize)

        return np.zeros(datasize)

    def deltaW(self, delta:np.array, type:str):
        match type:
            case 'output' : return delta * self.d
            case 'dense' :  return np.sum(delta.dot(self.ws.reshape(-1, 1).T)) * self.d
            case _ :        raise ValueError(f"in deltaW type({type}) error")

    def deriv(self, y_pred:float):
        match self.func_act:
            case 'heaviside':   self.d = y_pred
            case 'relu':        self.d = np.where(y_pred > 0, 1, 0)
            case 'leaky_relu':  self.d = np.where(y_pred > 0, 1, 0.01)
            case 'sigmoid' :    self.d = y_pred * (1 - y_pred)
            case 'softmax' :    self.d = 0 ### TO DO !!!!!
            case _ :            self.d = y_pred

    def predict(self, xi:np.array):
        return self.__fact(self.__fagr(xi))
