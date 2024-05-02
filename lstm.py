import tensorflow as tf
tf.get_logger().setLevel('ERROR') #added to suppress warnings for more comprehendable printing
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

hyperparameters = {'output_dim': 64, 'batch_size': 128, 'epochs': 5}    #defining hyperparameters

def train_model(Optimizer, X_train, y_train, X_val, y_val):
    # Model Building
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=hyperparameters['output_dim'], input_length=100))
    model.add(LSTM(units=64))
    model.add(Dense(units=1, activation='sigmoid'))
    
    # Train Model
    model.compile(optimizer=Optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    trained_model=model.fit(X_train,y_train,batch_size=hyperparameters['batch_size'],epochs=hyperparameters['epochs'],validation_data=(X_val,y_val))
    return model, trained_model

def model_evaluation(model, X_val, y_val):  #makes evaluation on testing data based off trained model
    print("EVALUATING MODEL ON TESTING DATA")
    loss, accuracy = model.evaluate(testing_padded, testing_set[1])
    accuracy="{:.2f}".format(accuracy*100) #turns accuracy into percentage
    loss="{:.2f}".format(loss*100) #turns accuracy into percentage
    print(f"Loss: {loss}%")
    print(f"Accuracy: {accuracy}%")
    print("CREATING GRAPHS")


#gets base model and trained model and prints out model evaluation
model, trained_model=train_model(RMSprop(lr=0.1), training_padded, training_set[1], testing_padded, testing_set[1])
model_evaluation(model, testing_padded, testing_set[1])

#gets list of predictions from testing data
testing_predictions=[1 if pred > 0.5 else 0 for pred in model.predict(testing_padded)]

#allows us to display 2 graphs
fig, axs = plt.subplots(2,1,figsize=(8,8))

# Plot accuracy per epoch
history=[acc*100 for acc in trained_model.history['acc']]   #turns accuracy into percentage
epochs=list(range(1,hyperparameters['epochs']+1))
epoch_accuracy_graph=sns.stripplot(epochs,history,ax=axs[0])
epoch_accuracy_graph.set(xlabel='Epoch',ylabel='Accuracy(%)')
axs[0].set_title('Epoch Accuracy Progression for Training Data')

# Plot confusion matrix
cm=confusion_matrix(testing_set[1],testing_predictions)
for i in range(len(cm)):    #turns total for each square into percentage
    for j in range(len(cm[0])):
        cm[i][j]=(cm[i][j]/25000) * 100
sns.heatmap(cm, annot=True, cbar=False, fmt=".2f", ax=axs[1])
axs[1].set_title('Percentage Confusion Matrix for Testing Data Predictions')
plt.xlabel('Predicted Labels')
plt.ylabel('Real Labels')
plt.tight_layout()
plt.show()