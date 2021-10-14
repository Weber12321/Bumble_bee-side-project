import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class RunModel(object):
    def __init__(self, data: pd.DataFrame, test_size: float) -> None:
        self.data = data
        self.test_size = test_size
        self.scalar = MinMaxScaler()
        self.model = LinearRegression()

    def run(self):
        X, y = self.scale()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size, random_state = 42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return self.model.score(X_train, y_train), mean_squared_error(y_test, y_pred)
    def scale(self):
        data = self.data
        arr = data.iloc[:,1:].values
        y = data.iloc[:,0].values
        return self.scalar.fit_transform(arr), y



