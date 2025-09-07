# Python Basics Cheat Sheet

## Table of Contents

* Data Types
* Collections (Arrays)
* Lists
* Tuples
* Sets
* Dictionaries
* Conditional Statements
* Loops
* Range Constructor
* Random Numbers

---

## Data Types

Python has various built-in data types:

| Example                 | Data Type  | Description                   |
| ----------------------- | ---------- | ----------------------------- |
| x = "Hello World"       | str        | Text/string type              |
| x = 20                  | int        | Integer number                |
| x = 20.5                | float      | Floating point number         |
| x = 1j                  | complex    | Complex number                |
| x = ["apple", "banana"] | list       | Ordered, mutable collection   |
| x = ("apple", "banana") | tuple      | Ordered, immutable collection |
| x = range(6)            | range      | Sequence of numbers           |
| x = {"name": "John"}    | dict       | Key-value pairs               |
| x = {"apple", "banana"} | set        | Unordered, unique elements    |
| x = frozenset(...)      | frozenset  | Immutable set                 |
| x = True                | bool       | Boolean (True/False)          |
| x = b"Hello"            | bytes      | Immutable bytes sequence      |
| x = bytearray(5)        | bytearray  | Mutable bytes sequence        |
| x = memoryview(bytes(5)) | memoryview | Memory view object            |
| x = None                | NoneType   | Represents absence of value   |

**Explicit type setting:**

```python
x = str("Hello World")         # str
x = int(20)                    # int
x = float(20.5)                 # float
x = complex(1j)                 # complex
x = list(("apple", "banana"))   # list
```

**Get Type:**

```python
x = 5
print(type(x))  # <class 'int'>
```

**Change Type (Type Casting):**

```python
x = int(3.5)        # Convert float to int → 3
y = float("4.2")    # Convert string to float → 4.2
z = str(25)         # Convert int to string → "25"
```

---

## Collections (Arrays)

### Lists

**Characteristics:** Ordered, Changeable, Allows duplicates.

**Common Methods:**

| Method    | Example                  | Description                  |
| --------- | ------------------------ | ---------------------------- |
| append()  | fruits.append("orange")  | Adds element at the end      |
| insert()  | fruits.insert(1, "mango") | Inserts at specific position |
| remove()  | fruits.remove("banana")  | Removes first matching item  |
| pop()     | fruits.pop(1)            | Removes item at index        |
| sort()    | fruits.sort()            | Sorts the list               |
| reverse() | fruits.reverse()         | Reverses the list            |
| copy()    | new_list = fruits.copy() | Creates a shallow copy       |

**Examples:**

```python
fruits = ["apple", "banana", "cherry"]
fruits.append("orange")
print(fruits[1])  # Output: banana
```

```python
numbers = [1,8,2,10,3,5,7,4,9]
numbers.sort() # Sorts the list ascending
numbers.reverse() # Sorts the list descending
numbers.sort(reverse=True) # Sorts the list descending
```

---

### Tuples

**Characteristics:** Ordered, Unchangeable, Allows duplicates.

**Methods:**

| Method  | Example                   | Description                  |
| ------- | ------------------------- | ---------------------------- |
| count() | thistuple.count("apple")  | Counts occurrences of value  |
| index() | thistuple.index("banana") | Returns first index of value |

**Examples:**

```python
thistuple = ("apple", "banana", "cherry")
print(len(thistuple))  # Output: 3

# Single item tuple
singletuple = ("apple",)
print(type(singletuple))  # <class 'tuple'>

# Deleting a tuple
del thistuple
```

---

### Sets

**Characteristics:** Unordered, Unchangeable (but can add/remove), No duplicates.

**Common Methods:**

| Method                 | Example                                  | Description                                |
| ---------------------- | ---------------------------------------- | ------------------------------------------ |
| add()                  | fruits.add("orange")                     | Adds an element to the set                 |
| remove()               | fruits.remove("banana")                  | Removes an element (error if not found)    |
| discard()              | fruits.discard("banana")                 | Removes an element (no error if not found) |
| pop()                  | fruits.pop()                             | Removes and returns a random element       |
| clear()                | fruits.clear()                           | Removes all elements                       |
| union()                | fruits.union({"kiwi", "mango"})          | Returns a new set with all unique elements |
| update()               | fruits.update({"kiwi", "mango"})         | Adds elements from another set/list        |
| intersection()         | fruits.intersection({"banana", "cherry"}) | Returns common elements between sets       |
| difference()           | fruits.difference({"banana", "cherry"})  | Returns elements only in the first set     |
| symmetric_difference() | fruits.symmetric_difference({"banana"})  | Returns elements not in both sets          |

**Example:**

```python
fruits = {"apple", "banana", "cherry"}
fruits.add("orange")
fruits.remove("banana")
```

---

### Dictionaries

**Characteristics:** Ordered (Python 3.7+), Changeable, No duplicates.

**Methods:**

| Method     | Example                    | Description                 |
|------------|----------------------------|-----------------------------|
| keys()     | x = car.keys()             | Returns all keys            |
| values()   | x = car.values()           | Returns all values          |
| items()    | x = car.items()            | Returns key-value pairs     |
| get()      | x = car.get("model")       | Gets value of key           |
| update()   | car.update({"year": 2020}) | Updates dictionary          |
| copy()     | new_car = car.copy()       | Creates a shallow copy      |
| pop()      | car.pop("model")           | Removes key-value pair      |
| popitem()  | car.popitem()              | Removes last key-value pair |
| del        | del car["model"]           | Removes key-value pair      |
| clear()    | car.clear()                | Removes all key-value pairs |
| del        | del car                    | Removes dictionary          |

**Example:**

```python
car = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(car["model"])  # Output: Mustang
```

### What is the difference between using the copy() method and the = operator to copy a dictionary?

- The `copy()` method creates a new dictionary that is a shallow copy of the original dictionary. This means that the new dictionary has its own copy of the key-value pairs, but the values themselves are still references to the same objects as in the original dictionary.
<br><br>
- On the other hand, the = operator creates a new reference to the same dictionary object. This means that both the original and new dictionary variables point to the same object in memory, and any changes made to the dictionary through one variable will be reflected in the other variable.

---

## Conditional Statements

**Syntax:**
```python
if <condition>:
    <code>
elif <condition>:
    <code>
else:
    <code>
```

```python
a = 200
b = 33

if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
else:
  print("a is greater than b")
```

**Logical Operators:**

* and → True if both statements are true
* or → True if one statement is true
* not → Reverse the result

**Examples:**

```python
x = 5
print(x > 3 and x < 10)  # True
print(x > 3 or x < 4)    # True
print(not(x > 3 and x < 10))  # False
```

**Short Hand If-Else:**

```python
a = 2
b = 330
print("A") if a > b else print("B")  # Output: B
```

---

## Loops

## While Loop:

```python
i = 1
while i < 6:
  print(i)
  i += 1
```

## For Loop:

**Syntax:**
```python
for <variable> in <list>:
    print(<variable>)
```

**Example:**
```python
fruits = ["apple", "banana", "cherry"]
for x in fruits:
  print(x)
```

**Enumerate Function:**
```python
for i, fruit in enumerate(fruits):
    print(i, fruit)
```

**Looping in dictionaries:**
```python
car = {"name": "Ford", "model": "Mustang", "year": 1964}
for key, value in car.items():
    print(key, value)
```

## Break and Continue:

**Break**
- we use break to exit the loop when a certain condition is met
```python
i = 1
while i < 6:
  print(i)
  if i == 3:
    break
  i += 1
```

**Continue**
- we use continue to skip the current iteration and move to the next iteration
```python
for x in range(6):
  if x == 3: continue
  print(x)
```

---

## Functions

**Syntax:**
```python
def <function_name>(<parameters>):
    <code>
    return <value>
```

**Example:**
```python
def add(a, b):
    return a + b

print(add(1, 2))
```

---

## Range Constructor

```python
# Prints 0 to 5
for x in range(6):
  print(x)

# Prints 2 to 5
for x in range(2,6):
  print(x)

# Prints 2, 5, 8
for x in range(2, 10, 3):
  print(x)
```

---

## Random Numbers

```python
import random

# Random number between 1 and 9
print(random.randrange(1, 10))

# Random float between 0 and 1
print(random.random())

# Random choice from a list
fruits = ["apple", "banana", "cherry"]
print(random.choice(fruits))
```
---

## String Functions

### Convert string to uppercase
```python
x = "Hello World"
print(x.upper())
```

### Convert string to lowercase
```python
x = "Hello World"
print(x.lower())
```

### Convert string to title (the first letter of each word is uppercase)
```python
x = "hello world"
print(x.title())
```

### Get the number of characters in a string
```python
x = "hello world"
print(len(x))
```

### Get how many times a character appears in a string
```python
x = "hello world"
print(x.count("l"))
```

### Replace a character in a string
```python
x = "hello world"
print(x.replace("l", "x"))
```

### Check if a string contains a specific character
```python
x = "hello world"
print("world" in x)
```

### Check if a string does not contain a specific character
```python
x = "hello world"
print("world" not in x)
```

### Slicing a string
```python
x = "hello world"
print(x[1:5])
```

### Remove whitespace from the beginning and end of a string
```python
x = " hello world "
print(x.strip())
```
---

## Math Functions

**Get the maximum value in a list**
```python
num = [1, 2, 3, 4, 5]
print(max(num))
```

**Get the minimum value in a list**
```python
num = [1, 2, 3, 4, 5]
print(min(num))
```

**Get the absolute value of a number**
```python
num = -5
print(abs(num))
```

**Round a number**
```python
num = 3.14159
print(round(num))
```

**Get the square root of a number**
```python
num = 4
print(sqrt(num))
```

**Get the power of a number**
```python
num = 2
print(pow(num, 3))
```

---

## Modules

### OS Module

**import**
```python
import os
```

**Get the current working directory**
```python
print(os.getcwd())
```

**List all files and directories in the current directory**
```python
print(os.listdir())
```

**Create a Folder**
```python
os.mkdir("new_folder")
```

---