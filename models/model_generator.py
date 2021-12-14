from models.decomposition_model import ModelDecomposition
from models.no_decomposition_model import ModelNoDecomposition

class generator:

    @staticmethod
    def create_model(some_attribute, dataframe):
        if some_attribute == "decomposition":
            return ModelDecomposition(dataframe)
        if some_attribute == "no_decomposition":
            return ModelNoDecomposition(dataframe)