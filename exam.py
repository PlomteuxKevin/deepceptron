import pitch as pt
import aikiplot as ak
import math
import random
import matplotlib.pyplot as plt

### --------------------------------------------------------------------------------------------
### DATASET : Catégorisation des résultats d'examens par rapport au temps de sommeil
###           et au travail fournis
### --------------------------------------------------------------------------------------------
data = pt.pd.read_csv("datas/exam.csv")
data = data.head(1000)

# print("Dataset:\n",data, "\n")

x_train, y_train, x_test, y_test, collaps_data = pt.Pitch.pitch_data(data, y_name='Result', strat=True ,seed=42)

print("Prepared Data (shape): ", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# print(f"x_train: {x_train}   y_train: {y_train}")

### --------------------------------------------------------------------------------------------

# ak.scat2cat(data, title="Taux de réussite d'un exam", lim0=True, sp=1, autoName=True, file="graph/input_data.png")

model = pt.Pitch(x_train, y_train, x_test, y_test)
model.verbose(True)

# model.add_layer(type='dense', units=6, func_act='relu')
model.add_layer(type='dense', units=16, func_act='relu')
model.add_layer(type='output', units=1, func_act='sigmoid')
model.train(learning_rate=0.00001, epochs=100, loss='bce', metric=True)

model.show_loss()

pred_data = model.predict(data[['Work_Hours', 'Sleep_Hours']])
pred_data['pred'] = pred_data['pred'].apply(lambda x: 1 if x[0] >= 0.5 else 0)

print(pred_data['pred'].describe())

ak.scat2cat(pred_data, title="Taux de réussite d'un exam", lim0=True, sp=1, xlabel="Work hours", ylabel="Sleep Hours", autoName=True, file="graph/predict_data.png")
