rm(list = ls())
library(caret)
library(gbm)
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
set$Cabin <- as.numeric(set$Cabin)

# Gender to binary
set$Sex <- as.character(set$Sex)
set[set$Sex=='male',]$Sex <- '1'
set[set$Sex=='female',]$Sex <- '0'
set$Sex <- as.numeric(set$Sex)

# Clear Useless Variables
set$Ticket <- NULL
set$Cabin <- NULL #thinking the incomplete natre of cabin can be handled by h2o
set$Name <- NULL
set$SibSp <- NULL
set$Parch <- NULL
set$Embarked <- NULL

dataPart <- createDataPartition(set$Survived,p=0.7, list=FALSE)
training <- set[dataPart,]
validation <- set[-dataPart,]

# Classification Tree with rpart
library(rpart)

# grow tree 
fit <- rpart(Survived ~ .,
             method="class", 
             data=training)

# prune the tree 
model_rpart <- prune(fit, 
                     cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])

model_rpart

predictions <- as.data.frame(predict(model_rpart,validation))
colnames(predictions) <- c("zero", "one")
predictions$final <- 0
predictions[ predictions$zero >= predictions$one,]$final <- 0
predictions[predictions$one >= predictions$zero,]$final <- 1
confusionMatrix(predictions$final, validation$Survived)

# ========================Submitting Stuff


