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
------------------------------------
10 Types of Optimizers in Machine Learning (Evolution and Problem Solving)
1.	Stochastic Gradient Descent (SGD): Updates parameters based on the gradient calculated from a single training sample, but suffers from high variance and a fixed learning rate.
2.	Mini-batch Gradient Descent: Uses the gradient calculated from a small batch of data, which reduces variance and provides more stable updates than SGD.
3.	Momentum: Increases SGD speed by incorporating an exponentially decaying moving average of past gradients, which helps overcome local minima and navigate flat regions.
4.	Nesterov Accelerated Gradient (NAG): A modification of Momentum that "anticipates" the next step before calculating the gradient, leading to more accurate and effective corrections.
5.	Adagrad (Adaptive Gradient): The first adaptive optimizer: adjusts the learning rate per-parameter, but is prone to rapid learning rate decay over time, causing premature stopping.
6.	RMSprop (Root Mean Square Propagation): Addresses Adagrad's learning rate decay problem by using an Exponential Moving Average of squared gradients instead of accumulating all past squares.
7.	Adadelta: Improves upon Adagrad and RMSprop by eliminating the need to define a global learning rate and uses a history log of updates for normalization.
8.	Adam (Adaptive Moment Estimation): Combines the best of Momentum (first moment) and RMSprop (second moment), making it highly effective and the standard default choice for many tasks.
9.	Adamax: A stable variation of Adam that relies on the Infinity Norm to compute the adaptive learning rate component, which is more robust against extremely large gradients.
10.	Nadam (Nesterov-accelerated Adaptive Moment Estimation): Adam integrated with Nesterov acceleration (NAG), which often provides slightly better convergence and stability than the standard Adam optimizer.
