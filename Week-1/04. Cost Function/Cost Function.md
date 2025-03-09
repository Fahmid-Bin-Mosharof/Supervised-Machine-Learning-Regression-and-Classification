# Cost Function

## Overview
A **Cost Function** is a mathematical function used to measure the performance of a machine learning model. It quantifies the difference between predicted values and actual values, helping to optimize the model.

## Formula
For a linear regression problem, where the hypothesis function is represented as:

\[
h_\theta(x) = mx + c
\]

The Mean Squared Error (MSE) cost function is:

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2
\]

Where:
- \(J(\theta)\) is the cost function.
- \(m\) is the number of training examples.
- \(h_\theta(x) = mx + c\) is the hypothesis function (model prediction).
- \(y\) is the actual output.

## Types of Cost Functions
- **Mean Squared Error (MSE)** – Common for regression problems.
- **Cross-Entropy Loss** – Used for classification problems.
- **Hinge Loss** – Used for SVM classification.

## Optimization
To minimize the cost function, algorithms such as **Gradient Descent** or **Stochastic Gradient Descent (SGD)** are used.

## Usage
Cost functions are used in various machine learning algorithms, including:
- Linear Regression
- Logistic Regression
- Neural Networks
- Support Vector Machines (SVM)

## Conclusion
The cost function is a crucial component in machine learning, helping to fine-tune models and improve accuracy. Choosing the right cost function depends on the problem being solved.

