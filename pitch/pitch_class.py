import numpy as np
import pandas as pd
import sys, os
import time
import json
import matplotlib.pyplot as plt

from .layer_class import Layer
from .perceptron import p
from pitch.vprint import Vprint, vprint
from tqdm import tqdm

class Pitch():
    def __init__(self, train_data, train_labels, test_data, test_labels):
        self.x_train = np.array(train_data)
        self.labels = np.array(train_labels)
        self.x_test = np.array(test_data)
        self.test_labels = np.array(test_labels)
        self.layers = []
        self.error_mean = []
        self.error_std = []

    # def __str__(self):
    #     return f"Model : weights {self.n.ws} biais {self.n.b}"

### --------------------------------------------------------------------------------------------
### Class Methodes
### --------------------------------------------------------------------------------------------
### load : load existing model parameters
### pitch_data : prepare the data from the dataset (DataFrame)
### --------------------------------------------------------------------------------------------

    ### ----------------------------------------------------------------------------------------
    ### NAME : load
    ### ----------------------------------------------------------------------------------------
    ### Description : Load model parameters (weights and bias)
    ### ----------------------------------------------------------------------------------------
    ### RETURN : None
    ### ----------------------------------------------------------------------------------------
    @classmethod
    def load(cls, file="model.p5"):
        with open(file, 'r') as file:
            model_data = json.load(file)

        model = Pitch((0,0),(0,0),(0,0),(0,0))

        model.n.ws = np.array(model_data['weights'])
        model.n.b = float(model_data['bias'])

        return model

    ### ----------------------------------------------------------------------------------------
    ### NAME : pitch_data
    ### ----------------------------------------------------------------------------------------
    ### Description : prepare the data from dataset (DataFrame) to classification training
    ###               in train and test data. The data are NOT stratified and NOT normalized
    ###               by default.
    ### ----------------------------------------------------------------------------------------
    ### data (type : pandas.DataFrame) : data from dataset with x and y.
    ### y_index (type : int,) : determine the index position of the y column.
    ### y_name (type : str) : determine the name of the y column.
    ### --- The parmeters y_index and y_name are applie with XOR logic. So use only 1 the 2
    ### --- paramters.
    ### strat (type : bool) : Set to True if you want stratified label data
    ### normed (type : bool) : Set to True if you want x_train and x_test normalized data output.
    ### seed (type : int) : put a seed if you want the same rand data sample.
    ### one_hot (type : bool) : set to True to transform y data (label) into one-hot vector.
    ### ----------------------------------------------------------------------------------------
    ### RETURN : x_train, y_train, x_test, y_test (numpy array)
    ### ----------------------------------------------------------------------------------------
    @classmethod
    def pitch_data(cls, data:pd.DataFrame, y_name:str=None, strat:bool=False, normed:bool=False, seed:int=None, one_hot:bool=False) -> pd.DataFrame:

        # Generate the RandomState for the sample
        rs = np.random.RandomState(seed=seed)

        if strat:      # Stratification of data
            sorted_data = data.sort_values(by=y_name)
            class_counts = sorted_data[y_name].value_counts()
            train_counts = (class_counts * 0.8).astype(int)
            train_data = pd.DataFrame()

            for label, count in train_counts.items():
                label_data = sorted_data[sorted_data[y_name] == label]
                sample_data = label_data.sample(n=count, random_state=rs)
                train_data = pd.concat([train_data, sample_data])

            test_data = data.drop(train_data.index)
        else :          # No stratified data
            # create a 80/20 ratio sample for random train and test data
            train_data = data.sample(frac=0.8, random_state=rs)
            test_data = data.drop(train_data.index)

        x_train = np.array(train_data.drop(y_name, axis=1))
        y_train = np.array(train_data[y_name])
        x_test = np.array(test_data.drop(y_name, axis=1))
        y_test = np.array(test_data[y_name])

        collaps_data = pd.DataFrame({'x1': x_test[:,0], 'x2': x_test[:,1]})

        # Data normalization
        if normed:
            mean = x_train.mean(axis=0)
            std = x_train.std(axis=0)
            x_train = (x_train - mean) / std
            x_test = (x_test - mean) / std

        if one_hot:
            y_train = pd.get_dummies(y_train)
            y_test = pd.get_dummies(y_test)

        return x_train, y_train, x_test, y_test, collaps_data


### --------------------------------------------------------------------------------------------
### Public Methodes
### --------------------------------------------------------------------------------------------
### predict : predict Y based on x(i) input
### save : save the weight and bias into file
### train : train the model
### verbose : set the model to verbose mode
### --------------------------------------------------------------------------------------------

    ### ----------------------------------------------------------------------------------------
    ### NAME : add_layer
    ### ----------------------------------------------------------------------------------------
    ### Description : add a layer to the model (input, ouput and dense)
    ### ----------------------------------------------------------------------------------------
    ### todo
    ### ----------------------------------------------------------------------------------------
    ### RETURN : None
    ### ----------------------------------------------------------------------------------------

    def add_layer(self, type:str, units:int=None, func_act:str=None):
        match type:
            case 'dense':
                if len(self.layers) == 0:
                    pu = self.x_train.shape[1]         # Number of neurones in previous layer
                else:
                    pu = len(self.layers[-1].n)        # Number of neurones in previous layer
                self.layers.append(Layer(type, units, pu, func_act))
            case 'output':
                if len(self.layers) == 0:
                    pu = self.x_train.shape[1]         # Number of neurones in previous layer
                else:
                    pu = len(self.layers[-1].n)        # Number of neurones in previous layer
                self.layers.append(Layer(type, units, pu, func_act))
            case _ :
                raise ValueError(f"add_layer : type{type} doesn't existe.")



    ### ----------------------------------------------------------------------------------------
    ### NAME : predict
    ### ----------------------------------------------------------------------------------------
    ### Description : Save the parameters (weights and biais) into a file to be used in
    ###               futher project.
    ### ----------------------------------------------------------------------------------------
    ### file (type : str) : name of the output file
    ### ----------------------------------------------------------------------------------------
    ### RETURN : np.array, list of input + prediction
    ### ----------------------------------------------------------------------------------------
    def predict(self, input:pd.DataFrame, trigger:float=0.5) -> pd.DataFrame:
        pred = np.empty((input.shape[0], input.shape[1]+1), dtype=object)
        for i, s_input in enumerate(np.array(input)):
            pred[i] = [*s_input, self.__predict(s_input)]

        return pd.DataFrame(pred, columns=[f'x{i+1}' for i in range(input.shape[1])] + ['pred'])

    ### ----------------------------------------------------------------------------------------
    ### NAME : save
    ### ----------------------------------------------------------------------------------------
    ### Description : Save the parameters (weights and biais) into a file to be used in
    ###               futher project.
    ### ----------------------------------------------------------------------------------------
    ### file (type : str) : name of the output file
    ### ----------------------------------------------------------------------------------------
    ### RETURN : None
    ### ----------------------------------------------------------------------------------------
    def save(self, file="model.p5"):
        model_data = {'weights': self.n.ws.tolist(), 'bias': self.n.b}
        with open(file, 'w') as file:
            json.dump(model_data, file)

    def show_model(self):
        for l_index, l in enumerate(self.layers):
            print(f"Layer ({l_index+1}) -> type: {l.type}  neurones: {len(l.n)}")
            for n_index, n in enumerate(l.n):
                print(f"\tn ({n_index+1}) -->", n)
        #print("Output layer error (mean):", self.error)

    def show_loss(self, file:str=None):
        plt.plot(self.error_mean)
        if file:
            plt.savefig(file)
            plt.close()
        else:
            plt.show()

    ### ----------------------------------------------------------------------------------------
    ### NAME : regLine
    ### ----------------------------------------------------------------------------------------
    ### Description : Find the 2 points of the regretion line
    ### --- Use only for 2 dimensions data
    ### ----------------------------------------------------------------------------------------
    ### todo
    ### ----------------------------------------------------------------------------------------
    ### RETURN : ((x1, x2), (y1, y2)) tuple of 2 tuples
    ### ----------------------------------------------------------------------------------------
    def regLine(self, data=None):
        try:
            w1, w2 = self.n.ws
            m = -w1 / w2
            b = -self.n.b / w2
        except:
            raise ValueError("Model have more then 2 x in input. Impossible to calculate regression line.")

        if data is None:
            try:
                x1 = np.min(self.x_train[:, 0])
                x2 = np.max(self.x_train[:, 0])
            except:
                raise ValueError("Model contain no data, maybe because the model was loaded from file with model.load(). For loaded model, please use model.regLine(data=[Your DataFrame])")
        else:
            if isinstance(data, pd.DataFrame):
                if data.shape[1] == 2:
                    x1 = np.min(data.iloc[:, 0])
                    x2 = np.max(data.iloc[:, 0])
                else:
                    raise ValueError(f"data shape in regLine(date:DataFrame) must be (:, 2) execept of {data.shape}")
            else:
                raise ValueError(f"data in regLine(date:DataFrame) must be DataFrame except of {type(data)}")

        # calculer les points y correspondants
        y1 = m * x1 + b
        y2 = m * x2 + b

        return ((x1, x2), (y1, y2))


    ### ----------------------------------------------------------------------------------------
    ### NAME : train
    ### ----------------------------------------------------------------------------------------
    ### Description : train the model based on x_train and y_train and eval the model with
    ###               x_test and y_test.
    ### ----------------------------------------------------------------------------------------
    ### todo
    ### ----------------------------------------------------------------------------------------
    ### RETURN : None
    ### ----------------------------------------------------------------------------------------
    def train(self, learning_rate:float, epochs:int, loss:str, metric:bool=False, trigger:float=0.5):
        vprint("*** Start Model Training ***")
        dataLen = len(self.x_train)
        start_time = time.time()
        # BOUCLE EPOCHE
        for epoch in range(epochs):
            i = vp = fp = vn = fn = 0
            error = []
            with tqdm(total=dataLen, desc=f"Epoch {epoch+1}",leave=True, ncols=100) as pbar:
                # BOUCLE INPUTE
                for x, label in zip(self.x_train, self.labels):
                    # PROPAGATION (FEEDFORWARD)
                    self.layers[0].getOut(x)                                            # Calcule the output of each neurones for the first layer based on input (x_train)
                    for n in range(len(self.layers[0].n)):
                        self.layers[0].n[n].deriv(self.layers[0].output_buffer[n])      # Calculate the derivate of each neurones of the first layer
                    for l in range(1,len(self.layers)):
                        self.layers[l].getOut(self.layers[l-1].output_buffer)           # Calcule the output of each neurones based on output of the previous layer
                        for n in range(len(self.layers[l].n)):
                            # NEURONE DERIVATION CALCULATION
                            self.layers[l].n[n].deriv(self.layers[l].output_buffer[n])  # Calcule the derivation of each neurones
                    # LOST FUNCTION RESULT and DERIVATION
                    error.append(self.__loss_func(self.layers[-1].output_buffer, label, func_loss=loss))
                    loss_deriv = np.array(self.__loss_derivation(self.layers[-1].output_buffer, label, func_loss=loss))

                    # BACK PROPAGATION
                    self.layers[-1].delta(loss_deriv)
                    for l in range(len(self.layers)-2, -1, -1):
                        self.layers[l].delta(self.layers[l+1].d)
                        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        ### Probleme dans la fonction delta !
                        ### Il y a une couille dans le pâté
                        ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                    # WEIGHTS and BIAS update by GRADIAN DESCENT
                    for l in range(len(self.layers)-1, 0, -1):
                        for n_index, n in enumerate(self.layers[l].n):
                            for w in range(len(n.ws)):
                                n.ws[w] -= learning_rate * n.d * self.layers[l-1].output_buffer[w]
                            n.b -= learning_rate * n.d
                    for n_index, n in enumerate(self.layers[0].n):
                        for w in range(len(n.ws)):
                            n.ws[w] -= learning_rate * n.d * x[w]
                        n.b -= learning_rate * n.d

                    if metric:
                        pass
                        ### TODO
                        # if label:
                        #     vp += 1 if self.n.y == label else 0
                        #     fn += 1 if self.n.y != label else 0
                        # else:
                        #     vn += 1 if self.n.y == label else 0
                        #     fp += 1 if self.n.y != label else 0
                    i += 1
                    pbar.update(1)

                ### END FOR EACH LAYER

                self.error_std.append(np.std(error))
                self.error_mean.append(np.mean(error))

            ### END WITH TDQM

            if metric:
                ### TODO
                # loss = self.error[epoch,:].std()
                # error = int(np.sum(np.abs(self.error[epoch,:])))/dataLen
                # accuracy = vp / (vp + fp)
                # sensitivity = vp / (vp + fn)
                # score_f1 = (accuracy * sensitivity) / (accuracy + sensitivity)
                # fpr = fp / (fp + vn)
                vprint(f"Loss mean: {self.error_mean[epoch]:.4f}  ", end="")
                vprint(f"Loss std: {self.error_std[epoch]:.4f}  ", end="\n")
                # vprint(f"Error: {error:.4f}  ", end="")
                # vprint(f"Accuracy: {accuracy:.4f}  ", end="")
                # vprint(f"Sensitivity: {sensitivity:.4f}  ", end="")
                # vprint(f"Score_f1: {score_f1:.4f}  ", end="")
                # vprint(f"FPR: {fpr:.4f}")

        ### END EPOCH LOOP

        end_time = time.time()
        vprint(f"*** Model Trained *** (in time : {end_time - start_time:.2f}s)")
        start_time = time.time()

        self.__test(self.x_test, self.test_labels, trigger)
        end_time = time.time()
        vprint(f"*** Tested *** (in time : {end_time - start_time:.2f}s)")

    ### ----------------------------------------------------------------------------------------
    ### NAME : verbose
    ### ----------------------------------------------------------------------------------------
    ### val (type : bool) : set the model to versbose (True) or not verbose (False)
    ### --- Not verbose by default
    ### ----------------------------------------------------------------------------------------
    ### RETURN : None
    ### ----------------------------------------------------------------------------------------
    def verbose(self, val:bool=False):
        Vprint.verbose=val

### --------------------------------------------------------------------------------------------
### Prived Methodes
### --------------------------------------------------------------------------------------------
### loss_func : Lost Function based on Binary test (1 or 0)
### predict : predict the value based on trained model weights and biais
### softmax : softmax function used for Multy Category Classification
### test : test the accuracy of the model
### --------------------------------------------------------------------------------------------
    ### ----------------------------------------------------------------------------------------
    ### NAME : loss_func
    ### ----------------------------------------------------------------------------------------
    ### Description : calcul the loss of the model
    ### ----------------------------------------------------------------------------------------
    ### y_pred (type : float or numpy array) : predicted value from model
    ### y_labbel (type : float or numpy array) : label from train data
    ### func_loss (type : str) : select the loss function to be used
    ### --- 'binary' : binary function 0 = no error, 1 = positiv error, -1 = negativ error
    ### --- 'bce' : Binary Cross Entropy
    ### --- 'cce' : Category Cross Entropy
    ### --- 'mse' : Mean Squared Error
    ### ----------------------------------------------------------------------------------------
    ### RETURN : float or numpy array
    ### ----------------------------------------------------------------------------------------
    def __loss_func(self, y_pred:np.array, label:np.array, func_loss:str='binary'):
        match func_loss:
            case 'binary':
                return label - y_pred
            case 'bce':
                epsilon = 1e-10     # Protec from log(0) or log(1)
                y_pred_c = np.clip(y_pred, epsilon, 1 - epsilon)
                return -1 * np.sum(label * np.log(y_pred_c) + (1 - label) * np.log(1 - y_pred_c))
            case 'cce':
                epsilon = 1e-10     # Protec from log(0) or log(1)
                y_pred_c = np.clip(y_pred, epsilon, 1 - epsilon)
                return -np.sum(label* np.log(y_pred_c))
            case 'mse':
                return np.mean((label - y_pred) ** 2)


    def __loss_derivation(self, y_pred, labels, func_loss:str):
        match func_loss:
            case 'binary':  return y_pred - labels
            case 'bce':
                e_labels = labels + 1e-10
                e_y_pred = y_pred + 1e-10
                return -e_labels / e_y_pred+ (1 - e_labels) / 1 - e_y_pred
            case 'cce':
                epsilon = 1e-15
                return -labels / y_pred
            case 'mse':
                return 2 * (y_pred - labels)

    def __predict(self, test_data:np.array) -> float:
        input_data = test_data
        for layer in self.layers:
            layer.getOut(input_data)
            input_data = layer.output_buffer
        return input_data

    def __softmax(self, x:np.array):
        e_x = np.exp(x - np.max(x))  # substract np.max(x) for numeric stability
        return e_x / e_x.sum(axis=0)

    def __test(self, test_data:np.array, test_labels:np.array, trigger:float) -> int:
        nb_test = 0
        nb_true = 0

        for test, label in zip(np.array(test_data), np.array(test_labels)):
            nb_test += 1
            test_pred = 1 if self.__predict(test) >= trigger else 0

            if test_pred == label:
                nb_true += 1

        # vprint("Test done : ", nb_test)
        # vprint(f"Test Accuracy : {round(nb_true/nb_test*100, 2)}%")
