# Logistic Regression

Logistic Regression is a supervised learning algorithm used for **classification problems**. Unlike linear regression, which predicts continuous values, logistic regression predicts the probability that a given input belongs to a particular class.

The hypothesis function is the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

The output of the sigmoid function is always between 0 and 1, which makes it suitable for classification.

---

### Why Use Logistic Regression?

* To solve **binary classification** problems (e.g., spam vs. not spam, disease vs. no disease).
* Interpretable model: coefficients represent the log-odds.
* Provides probability estimates, not just hard classifications.

---

### When to Use Logistic Regression

* When the target variable is **binary** (0 or 1).
* When the relationship between features and the log-odds of the target is approximately linear.
* When the dataset is not extremely large.

---

### Loss Function (Log Loss / Cross Entropy)

To measure how well the model fits the data, we use **Log Loss**:

$$
J(\theta) = - \frac{1}{n} \sum_{i=1}^n \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]
$$

where

* \$y\_i \in {0,1}\$ is the true label.
* \$\hat{y}\_i = \sigma(z\_i)\$ is the predicted probability.

---

### Gradient Descent

To minimize the log loss, we update weights using gradient descent:

$$
w_j = w_j - \alpha \frac{\partial J}{\partial w_j}
$$

The derivative:

$$
\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^n ( \hat{y}_i - y_i ) x_{ij}
$$

and

$$
\frac{\partial J}{\partial b} = \frac{1}{n} \sum_{i=1}^n ( \hat{y}_i - y_i )
$$

---

## Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

---

## Loading Data

```python
path = "data.csv"
df = pd.read_csv(path)
print(df)
```

---

## Visualize Data Distribution

```python
plt.scatter(df.x, df.y, c=df.y, cmap='bwr')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of x vs y')
plt.show()
```

---

## Algorithm Implementation From Scratch

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss_function(y, y_pred):
    return -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

def gradient_descent(X, y, w, b, L, epochs):
    n = len(y)
    for i in range(epochs):
        z = np.dot(X, w) + b
        y_pred = sigmoid(z)
        
        dw = (1/n) * np.dot(X.T, (y_pred - y))
        db = (1/n) * np.sum(y_pred - y)
        
        w -= L * dw
        b -= L * db
        
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss_function(y, y_pred):.4f}")
    return w, b

# Example dataset (binary classification)
X = df[['x']].values
y = df['y'].values

# Initialize parameters
w = np.zeros(X.shape[1])
b = 0
L = 0.01
epochs = 1000

w, b = gradient_descent(X, y, w, b, L, epochs)
print("Optimized Parameters => w:", w, " b:", b)

# Plot decision boundary
x_vals = np.linspace(X.min(), X.max(), 100)
y_vals = sigmoid(w[0]*x_vals + b)
plt.plot(x_vals, y_vals, color='red')
plt.scatter(X, y, c=y, cmap='bwr')
plt.show()
```

---

## How to Implement the Algorithm (Using sklearn)

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### Load Dataset

```python
df = pd.read_csv('data.csv')
print(df)
print('\nNumber of rows and columns in the data set: ', df.shape)
```

---

### Show First 5 Rows

```python
df.head()
```

---

### Dataset Description

```python
df.describe()
```

---

### Check for Missing Values

```python
missing_values = df.isnull().sum()
print(missing_values)
```

---

### Features and Output (Label)

```python
input_df = df.drop(columns='y')  # drop the label column
target_df = df['y']
```

---

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

---

### Scaling Features

```python
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(input_df)
input_df = pd.DataFrame(scaled_features, columns=input_df.columns)
```

---

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(input_df, target_df, test_size=0.3, random_state=1)
```

---

### Apply Logistic Regression

```python
clf = LogisticRegression()
clf.fit(X_train, y_train)
```

---

### Testing / Prediction

```python
prediction_test = clf.predict(X_test)
print(y_test.values, prediction_test)
```

---

### Model Evaluation

```python
accuracy = accuracy_score(y_test, prediction_test)
print("Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, prediction_test))

conf_matrix = confusion_matrix(y_test, prediction_test)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```
