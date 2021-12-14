import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from definition import SAVE_FOLDER
from models.base_model import ModelInterface
from utils.helper import get_logger

_logger = get_logger('model')

class ModelNoDecomposition(ModelInterface):
    def __init__(self, data: pd.DataFrame):
        super().__init__(data=data, scalar=MinMaxScaler(), model=LinearRegression(),
                         filename=Path(SAVE_FOLDER / f"no_decomposition_model.joblib"))
        self.data = data
        # self.test_size = test_size
        self.scalar = MinMaxScaler()
        self.model = LinearRegression()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.filename = Path(SAVE_FOLDER / f"no_decomposition_model.joblib")
        self.preprocessing()

    def train(self):
        # X, y = self.scale()
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = 42)

        scoring = ['r2', 'neg_mean_squared_error']
        _logger.info(f'{type(self.X_train)}, {type(self.y_train)}')

        result_dict = []
        count = 1
        _r2 = 0
        kf = KFold()
        for train_index, test_index in tqdm(kf.split(self.X_train)):
            # _logger.info(("TRAIN:", train_index, "TEST:", test_index))
            X_train, X_test = self.X_train[train_index], self.X_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]

            self.model.fit(X_train, y_train)
            r2 = self.model.score(X_train, y_train)
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            _logger.info(f'R squared: {r2}; MSE: {mse}')
            temp = {'r2': r2, 'mse': mse}
            result_dict.append(temp)


            if r2 > _r2:
                _logger.info(f'fold_{count}: {r2} is bigger than {_r2}, save model...')
                dump(self.model, self.filename)
                _r2 = r2

            count += 1

        return result_dict

    def predict(self):
        if not os.path.isfile(self.filename):
            raise FileNotFoundError(f'file {self.filename} is not found')
        model = load(self.filename)
        predicted = model.predict(self.X_test)
        r2 = model.score(self.X_test, self.y_test)
        mse = mean_squared_error(self.y_test, predicted)

        return r2, mse

    def preprocessing(self):

        data = self.data[~self.data.isna().any(axis=1)]
        # data.dropna(inplace=True)

        # data.replace([np.inf, -np.inf], " ", inplace=True)
        # _logger.info(f'Nan number : {len(data[data.isna().any(axis=1)])}')
        # data = data.dropna()
        arr = data.iloc[:,1:-1].values
        y = data.iloc[:,-1].values

        X = self.scalar.fit_transform(arr)

        self.X_train = X
        self.y_train = y.reshape(-1,1)

        _logger.info(f'{self.X_train.shape}')
        _logger.info(f'{self.y_train.shape}')


class FileNotFoundError(Exception):
    """file not found"""
    pass



