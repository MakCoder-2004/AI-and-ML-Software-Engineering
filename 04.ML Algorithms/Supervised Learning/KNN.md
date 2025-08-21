# K-Nearest Neighbors (KNN)

K-Nearest Neighbors is a **supervised learning algorithm** used for **classification** and **regression**.
It is based on the idea that a data point’s label is determined by the **labels of its nearest neighbors** in the feature space.

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
* When you want an **easy-to-implement** algorithm for quick classification.

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

## Visualize Class Distribution

```python
plt.scatter(df.x1, df.x2, c=df.y, cmap='bwr')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of x1 vs x2')
plt.show()
```

---

## Algorithm Implementation From Scratch

```python
# Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Predict single sample
def predict_single(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_neighbor_labels = [y_train[i] for i in k_indices]
    # Majority vote
    values, counts = np.unique(k_neighbor_labels, return_counts=True)
    return values[np.argmax(counts)]

# Predict multiple samples
def predict(X_train, y_train, X_test, k):
    return np.array([predict_single(X_train, y_train, x, k) for x in X_test])

# Example dataset
X = df[['x1','x2']].values
y = df['y'].values

# Predict using k=3
y_pred = predict(X, y, X, k=3)

print("Predictions:", y_pred)
```

---

## How to Implement the Algorithm (Using sklearn)

### Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
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
input_df  = df.drop(columns='y')
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

# Drop original categorical columns
drop_columns = ["X1", "X2", "X3"]
input_df = input_df.drop(drop_columns, axis=1)
```

---

### Scaling Features

```python
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(input_df)
input_df      = pd.DataFrame(scaled_features, columns=input_df.columns)
```

---

### Split Data

```python
X_train, X_test, y_train, y_test = train_test_split(
    input_df, target_df, test_size=0.3, random_state=1
)
```

---

### Apply KNN

```python
knn = KNeighborsClassifier(n_neighbors=3)  # choose k=3
knn.fit(X_train, y_train)
```

---

### Testing / Prediction

```python
prediction_test = knn.predict(X_test)
print(y_test.values, prediction_test)
```

---

### Calculate Accuracy

```python
accuracy = accuracy_score(y_test, prediction_test)
print("Accuracy:", accuracy)
```

---

### Classification Report

```python
print("\nClassification Report:\n", classification_report(y_test, prediction_test))
```

---

### Confusion Matrix

```python
conf_matrix = confusion_matrix(y_test, prediction_test)
print("\nConfusion Matrix:\n", conf_matrix)
```

---

### Visualization

```python
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---
