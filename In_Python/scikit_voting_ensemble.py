#!/usr/bin/python

#
#   * * * Completed by Vincent A. Saulys
#   * * * B.Eng Student at McGill University
#   Email me at valexandersaulys@gmail.com
#   This particularly method makes use of ensemble learning
#	A simple voting method is used to look at the linear SVM
#	classifications to determine 
#	

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from progressbar import ProgressBar
import time


print 'starting...'

df = pd.read_csv('train.csv')
print 'read data !'

print 'modifying data'
# >>> df.shape
#  (891, 12) 
# So we have 891 rows with 12 columns
# >>> df.columns
#  Index([u'PassengerId', u'Survived', u'Pclass', u'Name', u'Sex', u'Age', u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked'], dtype='object')  # All the column names

df = df.drop('Name', 1)
df = df.drop('Cabin', 1)
df = df.drop('Embarked', 1)
df = df.drop('Ticket', 1)

for i in range(len(df.index)):
    if df.iloc[i,1] == 0:
        df.iloc[i,1] = -1

# Create FamilySize variable and scrap what it replaces       
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = df.drop('SibSp', 1)
df = df.drop('Parch', 1)

# Insert average age into missing age values in dataset
avgnumbah = df['Age'].mean()
for i in range(len(df.index)):
    if pd.isnull(df.iloc[i,4]):
        df.iloc[i,4] = round(avgnumbah)

df['Age'] = df['Age'].astype(int)

for i in range(len(df.index)):
    if df.iloc[i,3] == 'male':
        df.iloc[i,3] = 1
    elif df.iloc[i,3] == 'female':
        df.iloc[i,3] = 0
    else:
        df.iloc[i,3] = 1 

training, validation = df[:668], df[668:]
training = training.as_matrix().astype(int)
validation = validation.as_matrix().astype(int)
#training = training.astype('float64')
#validation = training.astype('float64')
print 'partitioned data...'

# Splitting Training Up
y_train = training[:,1]    # labels
x_train = training[:,2:]   # everything else
# Splitting validation up
y_valid = validation[:,1]  # read y values
x_valid = validation[:,2:]


# Build classifiers
# Tests were conducted to find the parameters
clf = []
clf.append(LinearSVC(C=0.1, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='l2', multi_class='ovr', penalty='l2',
     random_state=None, tol=1e-07, verbose=0
     ))	

clf.append(SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0001, degree=3,
  gamma=0.0, kernel='rbf', max_iter=-1, probability=True,
  random_state=None, shrinking=True, tol=0.0001, verbose=False
  ))

clf.append(RandomForestClassifier(bootstrap=True, compute_importances=None,
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_density=None, min_samples_leaf=1,
            min_samples_split=2, n_estimators=250, n_jobs=1,
            oob_score=False, random_state=None, verbose=0
            )) 

print "Now onto building our predictions"
# The idea is to loop through the array and build a single linear 
# regression against each single feature. Then take down the 
# probabilities (R^2 scores) of it being correct as a weight
# This weight then figures into a simple voting procedure down below
classes = []; 
probs = []; 
preds = []; 
pbar = ProgressBar(maxval=len(clf))


for j in range(len(clf)):
	fitted = clf[j].fit(X=x_train,y=y_train)
	v = fitted.predict(x_valid)
	#print v
	preds.append( v )
	# We'll subsitute scores for probabilities here, in this case the R^2
	if j == 2:
		probs.append( clf[j].predict_proba(X=x_valid) )
	else:
		probs.append( clf[j].score(X=x_train,y=y_train) )
	pbar.update(j)

pbar.finish()


# # # # # # # # # NOW WE APPEND THE PREDICTORS

# Declares lists to hold final predictions
print "Now Going through the Voting"
holder_predictions = []
vote_preds = []

for i in range(len(preds[2])):
	h = 0
	for j in range(3):
		if j==0 or j==1:
			h += probs[j] * preds[j][i]
		else: # When j==2
			if preds[j][i] == 0:
				h += probs[j][i][0] * preds[j][i]
			else:
				h += probs[j][i][1] * preds[j][i]
	holder_predictions.append(h)

for i in range(len(holder_predictions)):
	if holder_predictions[i] <= int(0):
		vote_preds.append(0)
	else:
		vote_preds.append(1)

# Convert y_valid back
for i in range(len(y_valid)):
	if y_valid[i] == int(-1):
		y_valid[i] = int(0)
	else:
		y_valid[i] = int(1)

count = 0
for i in range( len(vote_preds) ):
	if vote_preds[i]==1:
		count += 1
print "Number of 1s => %d, Number of 0s => %d" % (count, len(vote_preds) - count)

# # # Now build and display accuracy and confusion matrix
cm = confusion_matrix(y_valid,vote_preds)
asm = accuracy_score(y_valid,vote_preds)
print ""
print cm
print asm


"""
# Submitting...
pID = np.asarray(tt)[:,0]
w = {'PassengerId' : map(int,pID),
     'Survived' : vote_preds }
     
wf = pd.DataFrame(w)

wf.to_csv("Submittion_bagging_linear_regression.csv",index=False)
"""
