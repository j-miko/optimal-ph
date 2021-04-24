import argparse
import pandas as pd
import numpy as np
from model import ModelADA
from tensorflow.keras.preprocessing import sequence


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', default='input.csv')
args = parser.parse_args()

# Config
output_file_path = 'predictions.csv'

# Load input.csv
with open(args.input_csv) as input_csv:
    df = pd.read_csv(input_csv)

#preprocess data
max_len = 1000

chars = 'A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'.split(',')
chars_encoder = {char: ii for ii, char in enumerate(chars)}
sequence_encoded = [np.array([chars_encoder[char] for char in row]) for row in df.to_numpy().flatten()]
sequence_encoded = sequence.pad_sequences(sequence_encoded, maxlen=max_len)


df = pd.DataFrame(sequence_encoded)

# Run predictions
# y_predictions = BaselineModel(model_file_path='src/model.pickle').predict(df)
# y_predictions = ModelRNN(model_file_path='src/model_3.pkl').predict(df)
y_predictions = ModelADA(model_file_path='src/model_ada.pkl').predict(df)

# Save predictions to file
df_predictions = pd.DataFrame({'prediction': y_predictions.flatten()}, index=df.index)
df_predictions.to_csv(output_file_path, index=False)

print(f'{len(y_predictions)} predictions saved to a csv file')
