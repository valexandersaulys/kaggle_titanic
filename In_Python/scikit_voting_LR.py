#!/usr/bin/python

#
#   * * * Completed by Vincent A. Saulys
#   * * * B.Eng Student at McGill University
#   Email me at valexandersaulys@gmail.com
#   This particularly method makes use of ensemble learning
#	A simple voting method is used to determine which is most 
#	correct using linear regressions against single features
#	

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd


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
training = np.asarray(training)
validation = np.asarray(validation)
#training = training.astype('float64')
#validation = training.astype('float64')
print 'partitioned data...'

# Splitting Training Up
y_train = training[:,1]    # labels
x_train = training[:,2:]   # everything else
# Splitting validation up
y_valid = validation[:,1]  # read y values
x_valid = validation[:,2:]


# Build classifier
clf = LinearRegression()

print "Now onto building our predictions"
# The idea is to loop through the array and build a single linear regression against each single feature
# Then take down the probabilities (R^2 scores) of it being correct as a weight
# This weight then figures into a simple voting procedure down below
classes = []; probs = []; preds = []
for j in range(x_train.shape[1]):
	fitted = clf.fit(X=x_train[:,j],y=y_train)
	v = fitted.predict(x_valid[:,j].tolist())
	print v
	preds.append( v )
	# We'll subsitute scores for probabilities here, in this case the R^2
	probs.append( clf.score(X=x_train[:,j], y=y_train) )

# Declares lists to hold final predictions
print "Now Going through the Voting"
holder_predictions = []
vote_preds = []

for i in len(number_of_classifiers):
	holder_predictions[i] += probs[i] * preds[i]

for i in len(holder_predictions):
	if holder_predictions[i] <= 0:
		vote_preds.append(0)
	else:
		vote_preds.append(1)

pID = np.asarray(tt)[:,0]
w = {'PassengerId' : map(int,pID),
     'Survived' : vote_preds }
     
wf = pd.DataFrame(w)

wf.to_csv("Submittion_bagging_linear_regression.csv",index=False)

