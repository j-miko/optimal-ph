import argparse
import pandas as pd
import numpy as np
from model import ModelRNN
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
sequence_ascii = [np.array([ord(cha) for cha in row]) for row in df]
df = pd.DataFrame(sequence.pad_sequences(sequence_ascii, maxlen=max_len))

# Run predictions
# y_predictions = BaselineModel(model_file_path='src/model.pickle').predict(df)
y_predictions = ModelRNN(model_file_path='src/model_2.pkl').predict(df)

# Save predictions to file
df_predictions = pd.DataFrame({'prediction': y_predictions.flatten()}, index=df.index)
df_predictions.to_csv(output_file_path, index=False)

print(f'{len(y_predictions)} predictions saved to a csv file')
