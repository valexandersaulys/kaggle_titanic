#!/usr/bin/python

#
#   * * * Completed by Vincent A. Saulys
#   * * * B.Eng Student at McGill University
#   Email me at valexandersaulys@gmail.com
#   Completed with ample help from the internets
#
#

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import numpy
import pandas as pd

print 'starting...'

df = pd.read_csv('train.csv')
print 'read data !'
# 
# >>> df.shape
#  (891, 12) 
# So we have 891 rows with 12 columns
# >>> df.columns
#  Index([u'PassengerId', u'Survived', u'Pclass', u'Name', u'Sex', u'Age', u'SibSp', u'Parch', u'Ticket', u'Fare', u'Cabin', u'Embarked'], dtype='object')  # All the column names
#  


#  We'll begin by eliminating useless variables from the data set
#  Namely, the non-numeric ones
df = df.drop('Name', 1)
df = df.drop('Cabin', 1)
df = df.drop('Embarked', 1)
df = df.drop('Ticket', 1)



# >>> df.shape
#   (891,8)

# Now we'll calculate the gender as a binary
# 1 for male, 0 for female
for i in range(len(df.index)):
    if df.iloc[i,3] == 'male':
        df.iloc[i,3] = 1
    elif df.iloc[i,3] == 'female':
        df.iloc[i,3] = 0
    else:
        df.iloc[i,3] = 1 
#If they didn't survive, chances are they were male

# Create FamilySize variable and scrap what it replaces       
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df = df.drop('SibSp', 1)
df = df.drop('Parch', 1)

# Insert average age into missing age values in dataset
avgnumbah = df['Age'].mean()
for i in range(len(df.index)):
    if pd.isnull(df.iloc[i,4]):
        df.iloc[i,4] = round(avgnumbah)
        
df['Sex'] = df['Sex'].astype(int)
df['Age'] = df['Age'].astype(int)

blankers = df['Fare'].mean() # for later
df['Fare'] = df['Fare'].astype(int)

print 'created little variables...'


training, validation = df[:668], df[668:]
training = numpy.asarray(training)
validation = numpy.asarray(validation)
#training = training.astype('float64')
#validation = training.astype('float64')
print 'partitioned data...'


dt = DecisionTreeClassifier(max_depth=2
                            #,
                            #,
                            )
bdt = AdaBoostClassifier(dt
                         ,algorithm="SAMME"
                         ,n_estimators=600
                         ,learning_rate=1
                         )

# Splitting Training Up
y_train = training[:,1]    # labels
x_train = training[:,2:]   # everything else
# Splitting validation up
y_valid = validation[:,1]  # read y values
x_valid = validation[:,2:]
print 'did necesary prep...'                         
                        
scores = cross_val_score(bdt, x_train, y_train,cv=10).fit(x_valid,y_valid)
scores.mean() 
print scores
print scores.mean()
