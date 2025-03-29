

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    return np.sum(np.abs(a - b))

def minkowski_distance(a, b, p=3):
    return np.power(np.sum(np.abs(a - b) ** p), 1 / p)


def knn_predict(X_train, y_train, X_test, k=3, distance_metric='euclidean', p=3):

    y_pred = []

    for test_point in X_test:
        distances = []

        for idx, train_point in enumerate(X_train):
            if distance_metric == 'euclidean':
                dist = euclidean_distance(test_point, train_point)
            elif distance_metric == 'manhattan':
                dist = manhattan_distance(test_point, train_point)
            elif distance_metric == 'minkowski':
                dist = minkowski_distance(test_point, train_point, p)
            else:
                raise ValueError("Unsupported distance metric.")

            distances.append((dist, y_train.iloc[idx]))

        distances.sort(key=lambda x: x[0])
        neighbors = [label for (_, label) in distances[:k]]

        most_common = Counter(neighbors).most_common(1)[0][0]
        y_pred.append(most_common)

    return np.array(y_pred)




if __name__ == "__main__":

    data = load_wine()
    X = data.data
    y = data.target

    import pandas as pd
    X = pd.DataFrame(X, columns=data.feature_names)
    y = pd.Series(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    k = 5
    metric = 'manhattan'

    y_pred = knn_predict(X_train_scaled, y_train, X_test_scaled, k=k, distance_metric=metric)

 

