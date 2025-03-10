# Gradient Descent Algorithm

## Overview
Gradient Descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient. It is widely used in machine learning and deep learning for optimizing cost functions.

### Formula
For a linear regression problem, where the hypothesis function is represented as:

![Gradient Descent Formula](https://latex.codecogs.com/png.latex?%5Ctheta_j%20%3A%3D%20%5Ctheta_j%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Ctheta_j%7D%20J(%5Ctheta_0%2C%5Ctheta_1))

where:
- α is the learning rate (step size).
- J(θ) is the cost function to be minimized.
- (∂/∂θ_j) J(θ) is the gradient of the cost function.


## Steps of Gradient Descent
1. **Initialize parameters** \( \theta \) randomly or with zeros.
2. **Compute the gradient** of the cost function with respect to each parameter.
3. **Update the parameters** using the update rule.
4. **Repeat** steps 2 and 3 until convergence (i.e., the changes become very small or a maximum number of iterations is reached).

## Types of Gradient Descent
- **Batch Gradient Descent:** Computes the gradient using the entire dataset.
- **Stochastic Gradient Descent (SGD):** Computes the gradient using a single training example at a time.
- **Mini-Batch Gradient Descent:** Uses a small batch of training examples to compute the gradient.

## Implementation in Python
```python
import numpy as np

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        gradient = (1/m) * X.T @ (X @ theta - y)
        theta -= alpha * gradient
    return theta
```

## Applications
- Machine Learning (Linear Regression, Logistic Regression, Neural Networks)
- Deep Learning (Training Neural Networks)
- Optimization Problems

## Choosing the Learning Rate (\( \alpha \))
- Too small: Convergence is slow.
- Too large: May overshoot the minimum or not converge at all.
- Use techniques like learning rate decay or adaptive learning rates (e.g., Adam optimizer).

## Conclusion
Gradient Descent is a fundamental algorithm for optimization in machine learning. Choosing an appropriate learning rate and stopping criteria is crucial for effective training and model performance.

