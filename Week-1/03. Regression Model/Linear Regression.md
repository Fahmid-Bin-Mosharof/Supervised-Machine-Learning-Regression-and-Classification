# Linear Regression

## Introduction

Linear Regression is a fundamental machine learning algorithm used for predictive modeling. It establishes a relationship between independent variables (features) and a dependent variable (target) by fitting a linear equation to the data.

## Formula

The mathematical representation of simple linear regression is:

$$
Y = mX + c
$$

Where:

- \(Y\) is the dependent variable (target)
- \(X\) is the independent variable (feature)
- \(m\) is the slope (coefficient)
- \(c\) is the intercept

For multiple linear regression:

$$
Y = m_1 X_1 + m_2 X_2 + ... + m_n X_n + c
$$

## Types of Linear Regression

1. **Simple Linear Regression**: One independent variable.
2. **Multiple Linear Regression**: Multiple independent variables.

## Implementation

### Using Python (scikit-learn)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.show()
```

## Applications

- Predicting sales based on advertising expenditure
- Estimating house prices using various features
- Forecasting stock market trends
- Medical data analysis (e.g., predicting patient outcomes)

## Advantages

- Simple and easy to implement
- Interpretable results
- Works well with linearly correlated data

## Limitations

- Assumes linearity between variables
- Sensitive to outliers
- Cannot handle complex relationships effectively

## Conclusion

Linear Regression is a powerful and widely used algorithm in predictive modeling. However, for complex datasets, other techniques like polynomial regression or deep learning may be more suitable.

