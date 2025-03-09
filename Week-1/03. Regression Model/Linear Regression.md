# Linear Regression

## Introduction

Linear Regression is one of the most popular supervised machine learning. It predicts values within a continuous range, (e.g. sale prices, life expectancy, temperature, etc) instead of trying to classify them into categories (e.g. car, bus, bike and others). The main goal of the linear regression is to find the best fit line which describes the relationship between the data.

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

