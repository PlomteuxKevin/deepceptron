import pitch as pt
import aikiplot as ak
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras import models, layers, optimizers, losses, metrics

### Data preparation
data = pt.pd.read_csv("datas/exam.csv")
x_train, y_train, x_test, y_test, collaps_data = pt.Pitch.pitch_data(data, y_name='Result', strat=True ,seed=42)
print("Prepared Data (shape): ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(x_train.shape[1], )))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=512)

result = model.predict(x_test)
result = (result > 0.5).astype(int)

collaps_data['y'] = result.flatten()

print(collaps_data.describe())

ak.scat2cat(collaps_data, title="(Keras) Taux de réussite d'un exam", lim0=True, sp=1, xlabel="Work hours", ylabel="Sleep Hours", autoName=True, file="graph/keras.png")
