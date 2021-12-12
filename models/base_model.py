from abc import ABC
from abc import abstractmethod

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class ModelInterface(ABC):
    def __init__(self, data: pd.DataFrame, scalar: MinMaxScaler = None, model = None, filename = None) -> None:
        self.data = data
        self.scalar = None
        self.model = None
        self.filename = None

    @abstractmethod
    def preprocessing(self):
        """preprocessing data"""
        pass

    @abstractmethod
    def train(self):
        """training data"""
        pass

    @abstractmethod
    def predict(self):
        """predicting results"""
        pass