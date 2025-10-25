# -Linear-Regression-model-using-Stochastic-Gradient-Descent-SGD-
This project implements a Multiple Linear Regression model trained using Stochastic Gradient Descent (SGD) — completely from scratch, without using any machine learning libraries such as scikit-learn or TensorFlow.
----------------------------------
 Model Details

Algorithm: Multiple Linear Regression

Optimization: Stochastic Gradient Descent (SGD)

Loss Function: Mean Squared Error (MSE)
----------------------------------
How the Code Works

Data Loading:
Reads data from MultipleLR.csv. If the file doesn’t exist, uses predefined data.

Model Initialization:
Initializes all weights (including bias) to zero.

Training (SGD):

Iterates through the data for a specified number of epochs.

Updates weights after each training sample 

Evaluation:
Calculates Mean Squared Error (MSE) every 100 epochs to track progress.
--------------------------------
 Notes

If you get nan values in the output, try decreasing the learning rate.

You can also normalize the data for better convergence.
-------------------------
Author
Nada Etman
computer engineering student