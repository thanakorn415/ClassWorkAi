import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# สร้างข้อมูล Blob 2 กลุ่ม
X1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=[[2.0, 2.0]],
                    cluster_std=0.75,
                    random_state=42)

X2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=[[3.0, 3.0]],
                    cluster_std=0.75,
                    random_state=42)

# รวมข้อมูลทั้งสองกลุ่ม
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(len(X1)), np.ones(len(X2))))

# สร้างโมเดล Neural Network
model = Sequential([
    Dense(8, input_dim=2, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

# ฝึก Neural Network
model.fit(X, y, epochs=100, batch_size=10, verbose=0)

# สร้าง decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightblue', 'lightcoral'], alpha=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='bwr')
plt.title("Simple Neural Network - Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
