import numpy as np
import pandas as pd
from .perceptron import p

class Layer():
    def __init__(self, type:str, units:int, prev_units:int, func_act:str=None):
            self.n = [p(prev_units, func_act) for _ in range(units)]        # Neurones
            self.output_buffer = []                                         # Y output
            self.func_act = func_act                                        # Activation function of all neurone in the layer
            self.d = []                                                     # Delta vector
            self.type = type                                                # Type of the layer (input, dense, output)

    def getOut(self, inputs:np.array):
        self.output_buffer = np.array([neuron.predict(inputs) for neuron in self.n])

    def delta(self, dnl:np.array):
        self.d = []
        for i in range(len(self.n)):
            self.d.append(self.n[i].deltaW(dnl, self.type))
        self.d = np.array(self.d)
