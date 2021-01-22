# Data is from https://www.kaggle.com/c/asap-aes/data
# Lines 16-19 from https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/

import numpy as np
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot
import pandas as pd
import pickle

data = pd.read_excel("data/training_set_rel3.xls")

data = data[["essay", "rater1_domain1", "rater2_domain1"]]

complete_data = []

for index, row in data.iterrows():
	essay = row["essay"]
	words = set(text_to_word_sequence(essay))
	vocab_size = len(words)
	encoded_essay = one_hot(essay, round(vocab_size*1.3))

	# Everything above is for one hot encoding

	complete_data.append([encoded_essay, row["rater1_domain1"]])

complete_data = np.array(complete_data)

x_train = []
y_train = []

for feature, label in complete_data:
	x_train.append(feature)
	y_train.append(label)

with open("X.pickle", "wb") as f:
	pickle.dump(x_train, f)

with open("y.pickle", "wb") as f:
	pickle.dump(y_train, f)
