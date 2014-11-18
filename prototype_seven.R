rm(list = ls())
library(caret)
library(ada)
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

# Clear Useless Variables
set$Ticket <- NULL
set$Cabin <- NULL #thinking the incomplete natre of cabin can be handled by h2o
set$Name <- NULL
set$SibSp <- NULL
set$Parch <- NULL
set$Embarked <- NULL

# randomize the set
nr<-dim(set)[1]
set[sample.int(nr),]

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
  n.trees=50,  # default of 100
  interaction.depth=2,  # default of 3
  shrinkage=0.1)  #default of 0.1

nukem <- 'BoxCox'

indy <- training[!names(training) %in% c('Survived')]
depy <- as.factor(training$Survived)

model_rpart <- train(x=indy,
                   y=depy, 
                   method='rpart',
                   #preProcess=nukem,
                   #type='Classification',
                   trControl=ctrl,
                   #tuneGrid=TG,
                   cp=0.000001,
                   tuneLength=3)
model_rpart

predictions <- predict(model_rpart,validation)
confusionMatrix(predictions, validation$Survived)

# ========================Submitting Stuff


