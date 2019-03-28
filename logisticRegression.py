# Created with Python version 2.7.15
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_predict, learning_curve
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Build data sets from mailbox .csv files
phish_data = pd.read_csv("./datasets/phishing3.mbox-export.csv")
good_data = pd.read_csv("./datasets/features-enron.csv")

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

# Logistic regression on Y1
logi = LogisticRegression(random_state=0, solver="lbfgs", multi_class='multinomial',max_iter=300)
logi.fit(trainX, trainY)

# Variance of algorithm
yPred = logi.predict(testX)
print('Variance score for Y1: %.2f' % r2_score(testY, yPred))

# Cross validation
predicted = cross_val_predict(logi, trainX, trainY, cv=5)
print "CV accuracy: ", accuracy_score(trainY, predicted)

# Graph training score and cross-validation score

#train_sizes, train_scores, test_scores = learning_curve(logi, trainX, trainY, n_jobs=-1, cv=5, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)
#
#train_scores_mean = np.mean(train_scores, axis=1)
#train_scores_std = np.std(train_scores, axis=1)
#test_scores_mean = np.mean(test_scores, axis=1)
#test_scores_std = np.std(test_scores, axis=1)
#
#plt.figure()
#plt.title("Logistic Regression")
#plt.legend(loc="best")
#plt.xlabel("Training examples")
#plt.ylabel("Score")
#plt.gca().invert_yaxis()
#
## box-like grid
#plt.grid()
#
## plot the std deviation as a transparent range at each training set size
#plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
#plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
#
## plot the average training and test score lines at each training set size
#plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
#plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
#
## sizes the window for readability and displays the plot
## shows error from 0 to 1.1
#plt.ylim(-.1,1.1)
#plt.show()