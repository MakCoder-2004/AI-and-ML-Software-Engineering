# K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a **supervised learning algorithm** used for **classification** and **regression**.
It is based on the idea that a data point’s label is determined by the **labels (classification)** or **values (regression)** of its nearest neighbors in the feature space.

---

### How KNN Works

1. Choose the number of neighbors **k**.
2. Calculate the **distance** (e.g., Euclidean) between the new data point and all training points.
3. Select the **k-nearest neighbors**.
4. For classification: assign the class with the **majority vote** among neighbors.
   For regression: take the **average value** of the neighbors.

---

### Why Use KNN?

* Simple and intuitive to understand.
* Works for both **classification** and **regression**.
* No assumptions about data distribution (non-parametric).
* Effective when the decision boundary is irregular.

---

### When to Use KNN

* When you have a **small to medium-sized dataset** (since large datasets make it slow).
* When data is **not too high-dimensional** (curse of dimensionality).
* When you want an **easy-to-implement** algorithm for quick prediction.

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

## Visualize Class Distribution (for classification problems)

```python
plt.scatter(df.x1, df.x2, c=df.y, cmap='bwr')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of x1 vs x2')
plt.show()
```

---

## Implementation Using sklearn

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### Load Dataset

```python
df = pd.read_csv('data.csv')
print(df)
print('\nNumber of rows and columns in the dataset: ', df.shape)
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
# For classification (assuming 'y' is categorical)
X_class = df.drop(columns='y')
y_class = df['y']

# For regression (assuming 'target' is continuous)
X_reg = df.drop(columns='target')
y_reg = df['target']
```

---

### Encoding Categorical Columns (if any)

```python
# Replace with your own column names
columns_to_encode = ["X1", "X2", "X3"]
le = LabelEncoder()
for col in columns_to_encode:
    if col in X_class.columns:
        X_class["encoded_" + col] = le.fit_transform(X_class[col])
        X_class = X_class.drop(col, axis=1)
```

---

### Scaling Features

```python
scaler = MinMaxScaler()
X_class = pd.DataFrame(scaler.fit_transform(X_class), columns=X_class.columns)
X_reg   = pd.DataFrame(scaler.fit_transform(X_reg), columns=X_reg.columns)
```

---

## KNN for Classification

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.3, random_state=1
)
```

### Apply KNN Classifier

```python
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
```

### Testing / Prediction

```python
prediction_test = knn_clf.predict(X_test)
print(y_test.values, prediction_test)
```

### Evaluate Performance

```python
accuracy = accuracy_score(y_test, prediction_test)
print("Classification Accuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, prediction_test))

conf_matrix = confusion_matrix(y_test, prediction_test)
print("\nConfusion Matrix:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---

## KNN for Regression

### Split Data

```python
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=1
)
```

### Apply KNN Regressor

```python
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_reg, y_train_reg)
```

### Testing / Prediction

```python
prediction_test_reg = knn_reg.predict(X_test_reg)
```

### Evaluate Performance

```python
mse = mean_squared_error(y_test_reg, prediction_test_reg)
r2 = r2_score(y_test_reg, prediction_test_reg)

print("Regression Mean Squared Error:", mse)
print("Regression R2 Score:", r2)
```

### Visualization

```python
plt.scatter(y_test_reg, prediction_test_reg, color='blue')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('KNN Regression: Actual vs Predicted')
plt.show()
```
