# K-Nearest Neighbour (KNN) from Scratch

A Python implementation of the K-Nearest Neighbour (KNN) algorithm built from scratch without relying on machine learning libraries for the core classification logic.

## Overview

This project demonstrates how the K-Nearest Neighbour algorithm works by implementing:

- Distance calculations (Euclidean and/or Manhattan)
- Neighbor selection
- Majority voting classification
- Model evaluation
- Prediction on unseen data

The goal is to understand the inner workings of KNN rather than using pre-built implementations.

## How KNN Works

K-Nearest Neighbour is a supervised machine learning algorithm used primarily for classification and regression tasks.

The algorithm:

1. Calculates the distance between a test point and all training points.
2. Selects the **K** closest neighbors.
3. Determines the most common class among those neighbors.
4. Assigns that class to the test point.

## Features

✅ KNN implementation from scratch

✅ Custom distance metrics

✅ Euclidean distance support

✅ Manhattan distance support

✅ Classification using majority vote

✅ Simple and easy-to-understand code structure

## Project Structure

```text
K-Nearest-Neighbour/
│
├── dataset/
│   └── data.csv
│
├── knn.py
├── main.py
├── utils.py
└── README.md
```

> Note: The structure may vary depending on your implementation.

## Installation

Clone the repository:

```bash
git clone https://github.com/Ighayin1/K-Nearest-Neighbour.git
```

Navigate to the project directory:

```bash
cd K-Nearest-Neighbour
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```

Example:

```python
knn = KNearestNeighbour(k=3, distance_metric="Euclidean")

knn.fit(X_train, y_train)

prediction = knn.predict(X_test)
```

## Distance Metrics

### Euclidean Distance

```math
d = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}
```

### Manhattan Distance

```math
d = \sum_{i=1}^{n}|x_i - y_i|
```

## Example Workflow

1. Load dataset
2. Split data into training and testing sets
3. Select value of K
4. Train the model
5. Predict test instances
6. Evaluate accuracy

## Sample Output

```text
Distance Metric: Euclidean
K = 5

Accuracy: 94.2%
```

## Learning Objectives

This project was created to:

- Understand the fundamentals of KNN
- Learn distance-based classification
- Practice building machine learning algorithms from scratch
- Improve Python programming skills

## Future Improvements

- Weighted KNN
- KNN Regression
- Cross-validation
- Feature normalization
- Performance optimization using NumPy
- Visualization of decision boundaries

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib

## Author

**Ighayin**

GitHub: https://github.com/Ighayin1


---
If you found this project helpful, consider giving it a ⭐ on GitHub!

