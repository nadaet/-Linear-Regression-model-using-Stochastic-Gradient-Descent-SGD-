import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt  

FILE_NAME = 'MultipleLR.csv - MultipleLR.csv (1)'
LEARNING_RATE = 0.00005

EPOCHS = 500

def load_data(file_name):
    try:
        df = pd.read_csv(file_name, header=None)
    except FileNotFoundError:
        print(f"File '{file_name}' not found, using default data.")
        data_list = [
            (73, 80, 75, 152), (93, 88, 93, 185), (89, 91, 90, 180), (96, 98, 100, 196),
            (73, 66, 70, 142), (53, 46, 55, 101), (69, 74, 77, 149), (47, 56, 60, 115),
            (87, 79, 90, 175), (79, 70, 88, 164), (69, 70, 73, 141), (70, 65, 74, 141),
            (93, 95, 91, 184), (79, 80, 73, 152), (70, 73, 78, 148), (93, 89, 96, 192),
            (78, 75, 68, 147), (81, 90, 93, 183), (88, 92, 86, 177), (78, 83, 77, 159),
            (82, 86, 90, 177), (86, 82, 89, 175), (78, 83, 85, 175), (76, 83, 71, 149),
            (96, 93, 95, 192)
        ]
        df = pd.DataFrame(data_list)
        print("Using default dataset.")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_aug = np.insert(X, 0, 1, axis=1) 
    return X_aug, y

def predict(X, weights):
    return np.dot(X, weights)

def sgd_update(X, y, weights, learning_rate):
    N = X.shape[0]
    indices = list(range(N))
    random.shuffle(indices)
    for i in indices:
        y_hat_i = predict(X[i], weights)
        error = y[i] - y_hat_i
        gradient_step = 2 * learning_rate * error * X[i]
        weights = weights + gradient_step
    return weights

def mean_squared_error(X, y, weights):
    predictions = predict(X, weights)
    mse = np.mean((y - predictions) ** 2)
    return mse


X_aug, y = load_data(FILE_NAME)
num_features_with_bias = X_aug.shape[1]
weights = np.zeros(num_features_with_bias)
losses = [] 
print(f"Hyperparameters: Learning Rate={LEARNING_RATE}, Epochs={EPOCHS}\n")

for epoch in range(EPOCHS):
    weights = sgd_update(X_aug, y, weights, LEARNING_RATE)
    mse = mean_squared_error(X_aug, y, weights)
    losses.append(mse)

    print(f"Epoch {epoch + 1}/{EPOCHS}: MSE (Loss) = {mse:.4f}")

final_mse = mean_squared_error(X_aug, y, weights)
print("\nTraining complete.")
print("-" * 30)
print(f"Final Weights: {weights}")
print(f"Final MSE (Loss): {final_mse:.4f}")
print("-" * 30)

plt.plot(range(1, EPOCHS + 1), losses, color='blue')
plt.title("Loss (MSE) over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE (Loss)")
plt.grid(True)
plt.show()
