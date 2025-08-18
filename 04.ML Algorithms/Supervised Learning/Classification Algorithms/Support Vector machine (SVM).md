# Support Vector Machine (SVM)

Support Vector Machine is a **supervised learning algorithm** used for both **classification** and **regression**, but it is mostly used for classification problems. SVM aims to find the best decision boundary (called the **hyperplane**) that separates different classes with the maximum margin.

---

### Why Use Support Vector Machine?

* Effective in **high-dimensional spaces**.
* Works well with clear margin of separation.
* Robust to overfitting, especially in high-dimensional data.
* Can be used for linear and non-linear classification (with kernels).

---

### When to Use SVM

* When the dataset has **clear boundaries** between classes.
* When you want a **maximum-margin classifier**.
* When the number of features is large compared to the number of samples.
* When non-linear relationships exist (using kernel tricks).

---

### Decision Boundary in SVM

For a linear SVM, the decision boundary is defined as:

$w^T x + b = 0$

where:

* \$w\$ is the weight vector.
* \$b\$ is the bias.

The goal is to maximize the margin between the two classes.

---

### Loss Function (Hinge Loss)

The hinge loss is used in SVM:

$J(w, b) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \max(0, 1 - y_i (w^T x_i + b))$

where:

* \$C\$ is the regularization parameter.
* \$y\_i \in {-1, +1}\$ are the true labels.

---

## Required Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

## Algorithm Implementation (Linear SVM from Scratch - Simplified)

```python
def svm_train(X, y, lr=0.001, lambda_param=0.01, epochs=1000):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0

    for epoch in range(epochs):
        for i, x_i in enumerate(X):
            condition = y[i] * (np.dot(x_i, w) - b) >= 1
            if condition:
                w -= lr * (2 * lambda_param * w)
            else:
                w -= lr * (2 * lambda_param * w - np.dot(x_i, y[i]))
                b -= lr * y[i]
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Weights: {w}, Bias: {b}")

    return w, b

# Example dataset (binary classification)
X = df[['x']].values
y = df['y'].values

# Convert labels to -1 and +1
y = np.where(y == 0, -1, 1)

w, b = svm_train(X, y)

# Plot decision boundary
x_vals = np.linspace(X.min(), X.max(), 100)
y_vals = np.zeros_like(x_vals)  # since we have 1D feature

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
from sklearn import datasets, svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
```

### Load Dataset

```python
df = pd.read_csv('data.csv')
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
input_df = df.drop(columns='y')  # drop the label column
target_df = df['y']
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
X_train, X_test, y_train, y_test = train_test_split(input_df, target_df, test_size=0.3, random_state=1)
```

### Apply Support Vector Machine

```python
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
```

### Testing / Prediction

```python
prediction_test = clf.predict(X_test)
print("True labels:     ", y_test.values)
print("Predicted labels:", prediction_test)
```

### Calculate Accuracy

```python
accuracy = accuracy_score(y_test, prediction_test)
print("Accuracy:", accuracy)
```

### Calculate Classification Report

```python
print("\nClassification Report:\n", classification_report(y_test, prediction_test))
```

### Calculate the Confusion Matrix

```python
conf_matrix = confusion_matrix(y_test, prediction_test)
print("\nConfusion Matrix:\n", conf_matrix)
```

### Calculate Precision/ Recall/ F1-score

```python
precision = precision_score(y_test, prediction_test, average='weighted') * 100
recall = recall_score(y_test, prediction_test, average='weighted') * 100
f1 = f1_score(y_test, prediction_test, average='weighted') * 100

# Print results
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")
```

### Visualization

```python
# Title for the plot
title = "SVC with linear kernel"

x0 = X_train['X'] 
x1 = X_train['Y'] 

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X_train,
    response_method="predict",
    cmap=plt.cm.coolwarm,
    alpha=0.75,
    ax=ax,
    xlabel="Feature 1",
    ylabel="Feature 2",
)

# Scatter plot of training points
ax.scatter(x0, x1, c=y_train, edgecolors="k")

ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)

plt.show()
```

---

## K-Fold Cross Validation with SVM

```python
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
```

### Ensure that x and y are numpy arrays if they are pandas DataFrames
```pyhton
x = np.array(input_df)  # Features
y = np.array(target_df)  # Labels
```

### Initialize 5-fold cross-validation
```pyhton
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

### Initialize lists to store metrics for each fold
```python
accuracies = []
precisions = []
recalls = []
f1_scores = []
```

### Loop through each fold
```python
for fold, (train_index, test_index) in enumerate(kf.split(x)):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Initialize and train SVC
    model = SVC(kernel='linear', random_state=42)
    model.fit(x_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(x_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store metrics
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    
    print(f"Fold {fold + 1}:")
    print(f"  Accuracy: {accuracy:.2f}")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")
    print()
```

### Print average metrics across all folds
```python
print("Average Metrics Across All Folds:")
print(f"  Accuracy: {np.mean(accuracies):.2f}")
print(f"  Precision: {np.mean(precisions):.2f}")
print(f"  Recall: {np.mean(recalls):.2f}")
print(f"  F1 Score: {np.mean(f1_scores):.2f}")
```
---