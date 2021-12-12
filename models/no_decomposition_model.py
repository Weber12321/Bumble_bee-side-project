import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from definition import SAVE_FOLDER
from models.base_model import ModelInterface
from utils.helper import get_logger

_logger = get_logger('model')

class ModelNoDecomposition(ModelInterface):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data=data, scalar=MinMaxScaler(), model=LinearRegression(),
                         filename=Path(SAVE_FOLDER / f"no_decomposition_model_{datetime.now()}.joblib"))
        self.data = data
        # self.test_size = test_size
        self.scalar = MinMaxScaler()
        self.model = LinearRegression()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.preprocessing()

    def train(self):
        # X, y = self.scale()
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = 42)

        scoring = ['r2', 'neg_mean_squared_error']
        _logger.info(f'{type(self.X_train)}, {type(self.y_train)}')
        scores = cross_validate(self.model, self.X_train, self.y_train, scoring=scoring)

        # X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)
        # self.model.fit(self.X_train, self.y_train)
        # y_pred = self.model.predict(X_test)
        # return self.model.score(X_train, y_train)


        _logger.info(f'{scores}')
        test_scores = scores['test_r2']
        test_scores_max_idx = np.nanargmax(test_scores)
        estimator = scores['estimator'][test_scores_max_idx]
        dump(estimator, self.filename)

        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f'file {self.filename} is not found')

        return scores

    def predict(self):
        model = load(self.filename)
        predicted = model.predict(self.X_test)
        r2 = r2_score(self.y_test, predicted)
        mse = mean_squared_error(self.y_test, predicted)

        return r2, mse

    def preprocessing(self):
        data = self.data
        arr = data.iloc[:,1:-1].values
        y = data.iloc[:,-1].values
        X = self.scalar.fit_transform(arr)

        self.X_train = X
        self.y_train = y


class FileNotFoundError(Exception):
    """file not found"""
    pass



