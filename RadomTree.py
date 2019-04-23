import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, learning_curve # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

col_names = ["@ in URLs", "Attachments", "Css", "Encoding", "External Resources", "Flash content", "HTML content",
             "Html Form", "Html iFrame", "IPs in URLs", "Javascript", "Phishy", "URLs"]
# Build data sets from mailbox .csv files
phish_data = pd.read_csv("./datasets/phishing3.mbox-export.csv")
good_data = pd.read_csv("./datasets/features-enron.csv")

# Concatenate phish_data and good_data
raw_data = pd.concat([phish_data,good_data],ignore_index=True)

# Transform strings in col 3 to ints
le = LabelEncoder()
le.fit(raw_data.Encoding)
raw_data.Encoding = le.transform(raw_data.Encoding)

#raw_data = raw_data.astype(int)

raw_data.head()

feature_cols = ["@ in URLs", "Attachments", "Css", "Encoding", "External Resources", "Flash content", "HTML content",
                "Html Form", "Html iFrame", "IPs in URLs", "Javascript", "URLs"]
X = raw_data[feature_cols]  # Features
y = raw_data.Phishy  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = RandomForestClassifier(n_estimators=1000)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print('Variance score for Y1: %.2f' % r2_score(y_test, y_pred))

train_sizes, train_scores, test_scores = learning_curve(clf, X_train, y_train, n_jobs=-1, cv=5, train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure()
plt.title("Classification Random Tree")
plt.legend(loc="best")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.gca().invert_yaxis()

## box-like grid
plt.grid()

## plot the std deviation as a transparent range at each training set size
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")

## plot the average training and test score lines at each training set size
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
#
## sizes the window for readability and displays the plot
## shows error from 0 to 1.1
plt.ylim(-.1,1.1)
plt.show()
