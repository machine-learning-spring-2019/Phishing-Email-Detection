import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, learning_curve
from sklearn.metrics import r2_score, accuracy_score



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

# Convert all values to ints
raw_data = raw_data.astype(int)

# Split data int train/test X/Y
trainX, testX, trainY, testY = train_test_split(raw_data.iloc[:, :12] , raw_data.iloc[:, 12], test_size=.4, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
trainX = sc.fit_transform(trainX)
testX = sc.fit_transform(testX)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)

classifier.fit(trainX, trainY)

predicted = cross_val_predict(classifier, trainX, trainY, cv = 10)

print(accuracy_score(trainY, predicted))

y_pred = classifier.predict(testX)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testY, y_pred)

