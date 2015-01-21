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
from sklearn import cross_validation as cv
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

avg_fare = sum(df.iloc[:,5])/len(df.index)

print 'partitioned data...'

# Splitting Training Up
y_train = df.iloc[:,1]
x_train = df.iloc[:,2:]
y_train = y_train.as_matrix().astype(int)    # labels
x_train = x_train.as_matrix().astype(int)   # everything else


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

##########################################################################
##########################################################################
##########################################################################
# # # # # # Now we look to create the submittion file
#
#
print 'Now we begin the testing data!'
tt = pd.read_csv('test.csv')
tt = tt.drop('Name', 1)
tt = tt.drop('Cabin', 1)
tt = tt.drop('Embarked', 1)
tt = tt.drop('Ticket', 1)

for i in range(len(tt.index)):
    if tt.iloc[i,2] == 'male':
        tt.iloc[i,2] = 1
    elif tt.iloc[i,2] == 'female':
        tt.iloc[i,2] = 0
    else:
        tt.iloc[i,2] = 1 

tt['FamilySize'] = tt['SibSp'] + tt['Parch'] + 1
tt = tt.drop('SibSp', 1)
tt = tt.drop('Parch', 1)


avgnumbah = tt['Age'].mean()
for i in range(len(tt.index)):
    if pd.isnull(tt.iloc[i,3]):
        tt.iloc[i,3] = round(avgnumbah)

tt['Age'] = tt['Age'].astype(int)


for i in range(len(tt.index)):
    if tt.iloc[i,3] == 'male':
        tt.iloc[i,3] = 1
    elif tt.iloc[i,3] == 'female':
        tt.iloc[i,3] = 0
    else:
        tt.iloc[i,3] = 1 

for i in range( len(tt.index) ):
	if pd.isnull(tt.iloc[i,5]):
		tt.iloc[i,5] = avg_fare

for i in range( len(tt.index) ):
	for j in range( len(tt.columns) ):
		if pd.isnull(tt.iloc[i,j]):
			tt.iloc[i,j] = 0

#tt = tt.astype('float64')
x_test = tt.iloc[:,1:]
x_test = x_test.as_matrix().astype(int)
pID = tt.iloc[:,0] # Passenger ID's
pID = pID.as_matrix().astype(int)


pbar = ProgressBar(maxval=len(clf))
probs = []; 
preds = []; 
scores = [0,0,0];

for j in range(len(clf)):
	fitted = clf[j].fit(X=x_train,y=y_train)
	v = fitted.predict(x_test)
	scores[j] = cv.cross_val_score(clf[j],x_train,y_train,cv=10).mean()
	preds.append( v )
	if j==0:
		# We'll subsitute scores for probabilities here, in this case the R^2
		probs.append( clf[j].score(X=x_train,y=y_train))
	else:
		probs.append( clf[j].predict_proba(X=x_test) )


holder_predictions = []; vote_preds = []
for i in range(len(preds[2])):
	h = 0
	for j in range(3):
		if j==0:
			h += scores[j] * probs[j] * preds[j][i]
		else: # When j==1 or j==2
			if preds[j][i] == 0:
				h += scores[j] * probs[j][i][0] * preds[j][i]
			else:
				h += scores[j] * probs[j][i][1] * preds[j][i]
	holder_predictions.append(h)

for i in range(len(holder_predictions)):
    if holder_predictions[i]:
		vote_preds.append(0)
    else:
        vote_preds.append(1)

count = 0
for i in range( len(vote_preds) ):
	if vote_preds[i]==1:
		count += 1
print "Number of 1s => %d, Number of 0s => %d" % (count, len(vote_preds) - count)


w = {'PassengerId' : map(int,pID),
     'Survived' : vote_preds }
     
wf = pd.DataFrame(w)

wf.to_csv("Submission_with_voting.csv",index=False)