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

# Cabin or not?
set$Cabin <- as.character(set$Cabin)
set[set$Cabin=="",]$Cabin <- 0
set[set$Cabin != "0",]$Cabin <- 1
set$Cabin <- as.numeric(set$Cabin)

# Gender to binary
set$Sex <- as.character(set$Sex)
set[set$Sex=='male',]$Sex <- '1'
set[set$Sex=='female',]$Sex <- '0'
set$Sex <- as.numeric(set$Sex)

# Clear Useless Variables
set$Ticket <- NULL
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

independent <- names(train.h2o)[!names(train.h2o) %in% c('Survived')]
dependent <- 'Survived'

model_rf <- h2o.randomForest(x=independent,y=dependent,data=train.h2o
                               ,key='model.hex'
                             ,classification=TRUE
                             ,ntree=1000
                             ,depth=30
                             ,nfolds=10
                             ,nbins=20
                             ,stat.type='GINI'
                             ,balance.classes=TRUE
                             ,type='BigData'
                             )

predictions <- as.data.frame(h2o.predict(model_rf, valid.h2o))
validation$pred <- as.numeric(predictions$predict)
mats <- confusionMatrix(validation$pred, validation$Survived)
mats