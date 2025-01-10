import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import keras.api.models as mod
import keras.api.layers as lay

import numpy as np 
import matplotlib.pyplot as plt


pitch = 20
step = 5
N = 100
n_train = int(N*0.3)

def gen_data(x):
    return (x%pitch)/pitch

t = np.arange(1, N+1)
#y = [gen_data(i) for i in t]
y = np.sin(0.05*t*10) + 0.8 * np.random.rand(N)
y = np.array(y)


# plt.figure()
# plt.plot(y)
# plt.show()




def convertToMatrix(data, step=1):
    X, Y =[],[]
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d,])
        Y.append(data[d,])
        
    return np.array(X), np.array(Y)

train, test =  y[0:n_train],y[n_train:N]

x_train, y_train = convertToMatrix(train,step)
x_test, y_test = convertToMatrix(test,step)



print("Dimension (Before) : ", train.shape, test.shape)
print("Dimension (After)  : ", x_train.shape, x_test.shape)



x_train = x_train.reshape(-1, step, 1)
x_test = x_test.reshape(-1, step, 1)

model = mod.Sequential()
model.add(lay.SimpleRNN(units=256, input_shape=(step, 1), activation="relu"))
model.add(lay.Dense(units=1))


model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=500, batch_size=1, verbose=1)

y_pred = model.predict(x_test)


plt.figure()
plt.plot(np.arange(len(y_test)), y_test, label="Original Data", color="blue")
plt.plot(np.arange(len(y_pred)), y_pred, label="Predicted Data", color="red", linestyle="dashed")
plt.show()


model = mod.Sequential()
model.add(lay.SimpleRNN(units = 64,
                        input_shape=(step,1),
                        activation="relu"))
model.add(lay.Dense(units = 1))

model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=30, batch_size=1, verbose=1)

plt.plot(hist.history['loss'])
plt.show()