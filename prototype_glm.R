rm(list = ls())

library(caret)
library(h2o)
set.seed(6645)
set <- read.csv('train.csv')
backup <- set

# Create a Family Size Variable
set$FamilySize <- set$SibSp + set$Parch + 1

# Create Find age median and replace it in the missing values
male_median <- median(set[set$Sex=='male',]$Age)
female_median <- median(set[set$Sex=='female',]$Age)
set[is.na(set$Age) & set$Sex=='male',]$Age <- male_median 
set[is.na(set$Age) & set$Sex=='female',]$Age <- female_median

# Create Age^Fare Variable
#set$AgeFare <- set$Age^set$Fare

# Cabin or not?
set$Cabin <- as.character(set$Cabin)
set[set$Cabin=="",]$Cabin <- 0
set[set$Cabin != "0",]$Cabin <- 1
set$Cabin <- as.factor(set$Cabin)

# Gender to binary
set$Sex <- as.character(set$Sex)
set[set$Sex=='male',]$Sex <- '1'
set[set$Sex=='female',]$Sex <- '0'
set$Sex <- as.factor(set$Sex)

# Clear Useless Variables
set$Ticket <- NULL
#set$Cabin <- NULL #thinking the incomplete natre of cabin can be handled by h2o
set$Name <- NULL
set$SibSp <- NULL
set$Parch <- NULL
set$Embarked <- NULL

dataPart <- createDataPartition(set$Survived,p=0.7, list=FALSE)
training <- set[dataPart,]
validation <- set[-dataPart,]

#
#  Begin NEural Network Learning Sequence
#

localH2O  <- h2o.init(Xmx = '6g' )

h2o.clusterInfo(localH2O)

train.h2o <- as.h2o(localH2O, training, key='train.hex')
valid.h2o <- as.h2o(localH2O, validation, key='valid.hex')

independent <- names(train.h2o)[!names(train.h2o) %in% c('Survived','PassengerId')]
dependent <- 'Survived'

beat_boots <- boot(data=training,)


model.h2o <- h2o.glm(x=independent, y=dependent, data=train.h2o
                    ,nfolds=10
                    ,family='gaussian'
                    #,link='identity'
                        )
model.h2o

predictions <- as.data.frame(h2o.predict(model.h2o, valid.h2o))
validation$pred <- round(as.numeric(predictions$predict),digits=0)
mats <- confusionMatrix(validation$pred, validation$Survived)
mats

#=============================================
#Writing Submition FIle
#
#
#
testing <- read.csv('test.csv', stringsAsFactors=FALSE)

# First separates titles from names and then removes the name variables
testing$Name <- as.character(testing$Name)
testing$Title <- sapply(testing$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
testing$Title <- sub(' ', '', testing$Title)
#testing$Title <- factor(testing$Title)

testing$FamilySize <- testing$SibSp + testing$Parch + 1

# Create Find age median and replace it in the missing values
male_median <- median(testing[testing$Sex=='male',]$Age)
female_median <- median(testing[testing$Sex=='female',]$Age)
testing[is.na(testing$Age) & testing$Sex=='male',]$Age <- male_median 
testing[is.na(testing$Age) & testing$Sex=='female',]$Age <- female_median

# Cabin or not?
testing$Cabin <- as.character(set$Cabin)
testing[testing$Cabin=="",]$Cabin <- 0
testing[testing$Cabin != "0",]$Cabin <- 1
testing$Cabin <- as.numeric(testing$Cabin)

# Gender to binary
testing$Sex <- as.character(testing$Sex)
testing[testing$Sex=='male',]$Sex <- '1'
testing[testing$Sex=='female',]$Sex <- '0'
testing$Sex <- as.numeric(testing$Sex)

# Clear Useless Variables
testing$Ticket <- NULL
#testing$Cabin <- NULL #thinking the incomplete natre of cabin can be handled by h2o
testing$Name <- NULL
testing$SibSp <- NULL
testing$Parch <- NULL
testing$Embarked <- NULL
testing$Title <- NULL

test.h2o <- as.h2o(localH2O, testing, key='test.hex')
predictions <- as.data.frame(h2o.predict(model.h2o,test.h2o))
submit <- data.frame(PassengerId=testing$PassengerId, 
                     Survived=predictions$predict)
write.csv(submit,file='submittion_111814_dos',row.names=FALSE)
