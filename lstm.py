import matplotlib
import sklearn.metrics
matplotlib.use("TkAgg")
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



# Import IMDB dataset keeping only 10,000 most common unique words
training_set, testing_set = imdb.load_data(num_words=10000)

# Zero-Padding
training_padded = sequence.pad_sequences(training_set[0], maxlen=100)
testing_padded = sequence.pad_sequences(testing_set[0], maxlen=100)

hyperparameters = {'output_dim': 32, 'batch_size': 128, 'epochs': 5}

def train_model(Optimizer, X_train, y_train, X_val, y_val):
    # Model Building
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=hyperparameters['output_dim'], input_length=100))
    model.add(LSTM(units=64))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Train Model
    model.compile(optimizer=Optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train,y_train,batch_size=hyperparameters['batch_size'],epochs=hyperparameters['epochs'],validation_data=(X_val,y_val))
    return model

def model_prediction(X_val,y_val):
    model=train_model(RMSprop(lr=0.1), training_padded, training_set[1], testing_padded, testing_set[1])
    print("EVALUATING MODEL ACCURACY ON TESTING DATA")
    testing_predictions=model.predict(X_val)
    predictions=[1 if pred > 0.5 else 0 for pred in testing_predictions]
    correct=(predictions==y_val).sum()
    accuracy="{:.2f}".format((correct/len(y_val))*100)
    print(f"accuracy: {accuracy}%")



model_prediction(testing_padded, testing_set[1])
# Plot accuracy per epoch
    # accuracy_history=model.history['acc']
    # epochs=list(range(1,hyperparameters['epochs']+1))
    # epoch_accuracy_graph=sns.stripplot(epochs,accuracy_history)
    # epoch_accuracy_graph.set(xlabel='Epoch',ylabel='Accuracy')
    # plt.title('Epoch Accuracy')
    # plt.show()

# Plot confusion matrix


# Embedding layer https://keras.io/2.16/api/layers/core_layers/embedding/ https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding
# LSTM layer https://keras.io/2.16/api/layers/recurrent_layers/lstm/ https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
# Dense layer https://keras.io/2.16/api/layers/core_layers/dense/ https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
