import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_predict, learning_curve
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Build data sets from mailbox .csv files
phish_data = pd.read_csv("/Users/jamesknepper/Documents/MACHLEARN/group_project/phishing3.mbox-export.csv")
good_data = pd.read_csv("/Users/jamesknepper/Documents/MACHLEARN/group_project/features-enron.csv")

# Concatenate phish_data and good_data
raw_data = pd.concat([phish_data,good_data],ignore_index=True)

# Rearrange columns - Put "Phishy" as last col
temp1 = raw_data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]]
temp2 = raw_data.iloc[:, 12]
raw_data = pd.concat([temp1, temp2],axis=1)

# Transform strings in col 3 to ints
le = LabelEncoder()
le.fit(raw_data.iloc[:,3])
raw_data.iloc[:,3] = le.transform(raw_data.iloc[:,3])

# Convert all values to ints for model
raw_data = raw_data.astype(int)

# Split data int train/test X/Y
trainX, testX, trainY, testY = train_test_split(raw_data.iloc[:, :12] , raw_data.iloc[:, 12], test_size=.4, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
trainX = sc.fit_transform(trainX)
testX = sc.fit_transform(testX)

import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the neural network
classifier = Sequential()
# adding input layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 12))
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = classifier.fit(trainX, trainY, batch_size = 10, nb_epoch = 100)

y_pred = classifier.predict(testX)
y_pred = (y_pred > 0.5).astype(int)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testY, y_pred)

loss = history.history['loss']
acc = history.history['acc']

# Create count of the number of epochs
epoch_count = range(1, len(loss) + 1)

# Visualize loss history
plt.plot(epoch_count, loss, 'r--')
plt.plot(epoch_count, acc, 'b-')
plt.legend(['Loss', 'Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();