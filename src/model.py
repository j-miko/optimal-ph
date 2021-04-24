from sklearn import tree
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

class ModelRNN:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        model_data = pickle.load(open(self.model_file_path, "rb"))
        self.model = keras.Sequential.from_config(model_data['config'])
        self.model.set_weights(model_data['weights'])
        
    def predict(self, input_df):
        return self.model.predict(input_df)

class ModelADA:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path
        self.model = pickle.load(open(self.model_file_path, "rb"))
    
    def predict(self, input_df):
        return self.model.predict(input_df)
        
class BaselineModel:
    def __init__(self, model_file_path):
        self.model_file_path = model_file_path

    def vectorize_sequences(self, sequence_array):
        vectorize_on_length = np.vectorize(len)
        return np.reshape(vectorize_on_length(sequence_array), (-1, 1))

    def train(self, df_train):
        X = self.vectorize_sequences(df_train['sequence'].to_numpy())
        y = df_train['mean_growth_PH'].to_numpy()

        model = tree.DecisionTreeRegressor()
        model.fit(X, y)

        with open(self.model_file_path, 'wb') as model_file:
            pickle.dump(model, model_file)

    def predict(self, df_test):
        with open(self.model_file_path, 'rb') as model_file:
            model: tree.DecisionTreeRegressor = pickle.load(model_file)

        X = df_test['sequence'].to_numpy()
        X_vectorized = self.vectorize_sequences(X)
        return model.predict(X_vectorized)
