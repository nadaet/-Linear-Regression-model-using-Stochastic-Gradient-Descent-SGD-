# -Linear-Regression-model-using-Stochastic-Gradient-Descent-SGD-
Overview:
The goal of this project is to train a linear model that predicts the relationship between X and y.
The model learns to fit a straight line y = m * x + b that minimizes the mean squared error (MSE)
-------------------------------------------
How it works:
Initialize the slope (m) and intercept (b) to zero.

For each data point:

Compute the prediction y_pred = m * x + b

Calculate the error (y - y_pred)

Update m and b using gradient descent:

dm = -2 * x * (y - y_pred)
db = -2 * (y - y_pred)
Repeat for a number of epochs to minimize the error.
---------------------------------------------------
Technologies Used:
Python 3
NumPy