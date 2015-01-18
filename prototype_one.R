rm(list = ls())
library(caret)
library(rpart)
set.seed(2565186)
set <- read.csv('train.csv')
backup <- set


# Prepping Ze Data
# First separates titles from names and then removes the name variables
set$Name <- as.character(set$Name)
set$Title <- sapply(set$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
set$Title <- sub(' ', '', set$Title)
set$Title <- factor(set$Title)

# Clear Useless Variables 
#set$Ticket <- NULL
#set$Cabin <- NULL
set$Name <- NULL


dataPart <- createDataPartition(set$Survived,p=0.7, list=FALSE)
training <- set[dataPart,]
validation <- set[-dataPart,]

formula <- as.formula(Survived ~ .)

MyTrainControl=trainControl(
  method = "repeatedcv",
  number=10,
  repeats=5,
  returnResamp = "all",
  classProbs = TRUE
) 

magic <- preProcess(training, 
            method = 'pca', 
            thresh = 0.95,
            pcaComp = NULL,
            na.remove = TRUE,
            k = 5,
            knnSummary = mean,
            outcome = NULL,
            fudge = .2,
            numUnique = 3,
            verbose = FALSE
)

modelFit_rpart <- train(form=as.formula(Survived ~ .),
                        data=training,
                        method='rpart',
                        trControl=MyTrainControl,
                        na.action=na.omit)

predictions <- predict(modelFit_rpart,validation,type='raw')
validation$pred1 <- round(predictions, digits = 0)
confusionMatrix(validation$pred1,validation$Survived)


