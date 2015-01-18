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
#If they didn't survive, chanes are they were male

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
                         ,learning_Rate=1
                         )

# Splitting Training Up
y_train = training[:,1]    # labels
x_train = training[:,2:]   # everything else
# Splitting validation up
y_valid = validation[:,1]  # read y values
x_valid = validation[:,2:]
print 'did necesary prep...'                         
                        
model = bdt.fit(x_train,y_train)
y_pred = model.predict(x_valid)

# Turn into integer arrays from object arrays
y_valid = map(int,y_valid)
y_pred = map(int,y_pred)

#print y_pred

cm = confusion_matrix(y_valid, y_pred)
asm = accuracy_score(y_valid,y_pred)
print (cm)
print "Accuracy -> %f" % (asm)


# Then the testing data...
#
#
print 'now we begin the testing data!'
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


# Using the previous average b/c its whats trained
for j in range(len(tt.index)):
    if pd.isnull(tt.iloc[j,3]):
        tt.iloc[j,3] = round(avgnumbah)        

tt['Sex'] = tt['Sex'].astype(int)
tt['Age'] = tt['Age'].astype(int)

for j in range(len(tt.index)):
    if pd.isnull(tt.iloc[j,4]):
        tt.iloc[j,4] = round(blankers)    
tt['Fare'] = tt['Fare'].astype(int)

#tt = tt.astype('float64')
x_test = numpy.asarray(tt)[:,1:]
pID = numpy.asarray(tt)[:,0]
preds = model.predict(x_test)
w = {'PassengerId' : map(int,pID),
     'Survived' : preds }
     
wf = pd.DataFrame(w)

wf.to_csv("Submittion_112214_Uno.csv",index=False)
