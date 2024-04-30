import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Import IMDB dataset keeping only 10,000 most common unique words
training_set, testing_set = imdb.load_data(num_words=10000)

# Zero-Padding
training_padded = sequence.pad_sequences(training_set[0], maxlen=100)
testing_padded = sequence.pad_sequences(testing_set[0], maxlen=100)

# Model Building
def train_model(Optimizer, X_train, y_train, X_val, y_val):
    model = Sequential()
    # **ADD YOUR CODE HERE**
    return 

# Train Model


# Plot accuracy per epoch


# Plot confusion matrix