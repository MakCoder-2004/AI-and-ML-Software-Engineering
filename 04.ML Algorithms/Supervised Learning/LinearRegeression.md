# Linear Regression

Linear regression is a supervised learning algorithm used to model the relationship between a dependent variable $y$ and one or more independent variables $x_1, x_2, \dots, x_n$. The goal is to find the best-fitting linear equation:

$$
y = m x + b
$$

for a single variable (simple linear regression) or

$$
y = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

for multiple variables (multiple linear regression).

### Why Use Linear Regression?

* To understand the strength of the relationship between variables.
* To predict future values based on existing data.
* It is one of the simplest and most interpretable models.

### When to Use Linear Regression

* When the relationship between variables is approximately linear.
* When the residuals (errors) are normally distributed and independent.
* When the dataset does not contain extreme multicollinearity.

### Loss Function

To measure how well a line fits the data, we use the **Mean Squared Error (MSE)**:

$$
MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

### Gradient Descent

Gradient descent is an optimization algorithm used to minimize the MSE by iteratively updating the parameters:

$$
m_{new} = m_{old} - \alpha \frac{\partial MSE}{\partial m}
$$

$$
b_{new} = b_{old} - \alpha \frac{\partial MSE}{\partial b}
$$

---

## Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

## Loading Data

```python
path = "data.csv"
df = pd.read_csv(path)
print(df)
```

## Visualize Relation Between Data

```python
plt.scatter(df.x, df.y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x vs y')
plt.show()
```

## Algorithm Implementation From Scratch

```python
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m*x + b)) ** 2
    return total_error / float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y
        m_gradient += (-2/n) * x * (y - (m_now * x + b_now))
        b_gradient += (-2/n) * (y - (m_now * x + b_now))
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

# Initialize parameters
m = 0
b = 0
L = 0.01
epochs = 1000

# Gradient Descent Loop
for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}, Loss: {loss_function(m, b, df):.4f}")
    m, b = gradient_descent(m, b, df, L)

print(f"Optimized parameters => m: {m}, b: {b}")

# Plotting the regression line
plt.scatter(df.x, df.y)
plt.plot(list(range(20, 80)), [m * x + b for x in range(20, 80)])
plt.show()
```

---

## How to Implement the Algorithm (Using sklearn)

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
import matplotlib.pyplot as plt
```

### Load Dataset

```python
df = pd.read_csv('insurance.csv')
print(df)
print('\nNumber of rows and columns in the data set: ', df.shape)
```

### Show First 5 Rows

```python
df.head()
```

### Dataset Description

```python
df.describe()
```

### Check for Missing Values

```python
missing_values = df.isnull().sum()
print(missing_values)
```

### Features and Output (Label)

```python
input_df = df.drop(columns='x')  # drop the label column
target_df = df['x']
```

### Encoding Categorical Columns

```python
# Replace with your own column names
columns_to_encode = ["X1", "X2", "X3"]
le = LabelEncoder()
for col in columns_to_encode:
    input_df["encoded_" + col] = le.fit_transform(input_df[col])

# Drop the original categorical columns
drop_columns = ["X1", "X2", "X3"]
input_df = input_df.drop(drop_columns, axis=1)
```

### Scaling Features

```python
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(input_df)
input_df = pd.DataFrame(scaled_features, columns=input_df.columns)
```

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(input_df, target_df, test_size=0.2, random_state=42)
```

### Apply Linear Regression

```python
reg = LinearRegression()
reg.fit(X_train, y_train)
```

### Testing / Prediction

```python
prediction_test = reg.predict(X_test)
print(y_test.values, prediction_test)
```

### Calculate Errors

```python
MAEValue = mean_absolute_error(y_test, prediction_test)
MSEValue = mean_squared_error(y_test, prediction_test)
MdSEValue = median_absolute_error(y_test, prediction_test)

print('Mean Absolute Error Value is : ', MAEValue)
print('Mean Squared Error Value is : ', MSEValue)
print('Median Absolute Error Value is : ', MdSEValue)
```

### Visualization
```python
plt.figure(figsize=(10,5))
sns.scatterplot(x=y_test, y=prediction_test, color="blue", alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Actual vs Predicted Values")
plt.show()
```

---
