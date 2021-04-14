'''
MIT License

Copyright (c) 2021 Rizwan Hasan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import collections
import numpy as np
import pandas as pd

class KNN:
    def __init__(self, n_neighbors: int, minkowski_p: int = 2) -> None:
        self.__p: int = minkowski_p
        self.__n_neighbors: int = n_neighbors
        self.__X: pd.core.frame.DataFrame
        self.__y: pd.core.frame.DataFrame

    def fit(self, X: pd.core.frame.DataFrame, y: pd.core.frame.DataFrame):
        self.__X = X.copy()
        self.__y = y.copy()

    def __minkowski(self, X_test: pd.core.frame.DataFrame) -> dict:
        distances_list: list = list()
        y: list = X_test[1].copy().to_list()

        for row in self.__X.iterrows():
            x: list = row[1].copy().to_list()
            summation: list = list()
            for i in zip(x, y):
                tmp: int = abs(i[0] - i[1])
                tmp **= self.__p
                summation.append(tmp)
            summation: int = sum(summation)

            distance: int = summation ** (1 / self.__p)
            distances_list.append(distance)

        distances_dict: dict = dict()
        classifications_list: list = [
            row[1].copy().to_list().pop() for row in self.__y.iterrows()
        ]
        for i in range(len(distances_list)):
            distances_dict[distances_list[i]] = classifications_list[i]

        return distances_dict

    def predict(self, X: pd.core.frame.DataFrame):
        predictions_list: list = []
        
        for row in X.iterrows():
            distances: dict = self.__minkowski(X_test=row)
            sorted_distances: tuple = tuple(
                distances[key] for key in sorted(distances.keys())
            )
            n_neighbors: tuple = sorted_distances[: self.__n_neighbors]
            class_counts: dict = collections.Counter(n_neighbors)
            max_class_key = max(class_counts, key=class_counts.get)
            predictions_list.append(max_class_key)

        return np.array(predictions_list)

if __name__ == "__main__":
    main()
