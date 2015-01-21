#
# * * * Completed by Vincent A. Saulys
# * * * These are test to decide parameters for
#		a randomforest classifier
#

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC, SVC
from sklearn.grid_search import GridSearchCV
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

tune_grid = [{'kernel':['rbf','sigmoid'],
            'C':[0.001,0.01,0.1,1.0],
			'tol':[0.0001,0.00001,0.000001,0.0000001],
            #'degree':[3,4,5],
            'coef0':[0.0001,0.001,0.01,0.1,0.0,1.0,2.0]
			}]

best_model = GridSearchCV( SVC(), tune_grid, cv=10, verbose=2, n_jobs=5).fit(x_train,y_train)

y_pred = best_model.predict(x_valid)

p = []; v = [];
for i in range(len(y_pred)):
	p.append(int(y_pred[i]))

for j in range(len(y_valid)):
	v.append(int(y_valid[j]))

cm = confusion_matrix(v,p)
asm = accuracy_score(v,p)
print cm
print "Accuracy: %f" % (asm)
print best_model.best_estimator_
