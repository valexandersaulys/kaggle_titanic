rm(list = ls())
library(caret)
library(gbm)
set.seed(6645)
set <- read.csv('train.csv')
backup <- set

# Prepping Ze Data
# First separates titles from names and then removes the name variables
set$Name <- as.character(set$Name)
set$Title <- sapply(set$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
set$Title <- sub(' ', '', set$Title)
set$Title <- factor(set$Title)

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
#set$Cabin <- NULL #thinking the incomplete natre of cabin can be handled by h2o
set$Name <- NULL
set$SibSp <- NULL
set$Parch <- NULL
set$Embarked <- NULL

dataPart <- createDataPartition(set$Survived,p=0.7, list=FALSE)
training <- set[dataPart,]
validation <- set[-dataPart,]

ctrl <- trainControl(
  method = "repeatedcv",
  number=10,
  repeats=1,
  returnResamp = "all",
  classProbs = TRUE  ) 

TG <- data.frame(
  n.trees=100,  # default of 100
  interaction.depth=4,  # default of 3
  shrinkage=0.01)  #default of 0.1

indy <- training[!names(training) %in% c('Survived')]
depy <- as.factor(training$Survived)

model_gbm <- train(x=indy,
                     y=depy, 
                     method='ada',
                     preProcess='pca',
                     #type='Classification',
                     trControl=ctrl,
                     tuneGrid=TG,
                     tuneLength=3)
model_gbm

predictions <- predict(model_gbm,validation)
confusionMatrix(predictions, validation$Survived)

# ========================Submitting Stuff


