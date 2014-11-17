rm(list = ls())
library(caret)
library(h2o)
set.seed(45684812)
testing <- read.csv('test.csv', stringsAsFactors=FALSE)

# Setup h2o
#localH2O  <- h2o.init(max_mem_size='6g')

# Prepping Ze Data
# First separates titles from names and then removes the name variables
testing$Name <- as.character(testing$Name)
testing$Title <- sapply(testing$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
testing$Title <- sub(' ', '', testing$Title)
#testing$Title <- factor(testing$Title)

testing$FamilySize <- testing$SibSp + testing$Parch + 1

# Clear Useless Variables
testing$Ticket <- NULL
testing$Cabin <- NULL #thinking the incomplete natre of cabin can be handled by h2o
testing$Name <- NULL
testing$SibSp <- NULL
testing$Parch <- NULL
testing$Embarked <- NULL

testing[is.na(testing$Age),]$Age <- 0
testing[is.na(testing$Fare),]$Fare <- 0



#test.h2o <- as.h2o(localH2O, testing, key='test.hex')

# ========================Submitting Stuff
#predictions <- as.data.frame(h2o.predict(modelFinal,newdata=test.h2o))
#submit <- data.frame(PassengerId=as.data.frame(test.h2o)$PassengerId,
 #                    Survived = predictions$predict)
testing[testing$Title=='Dona',]$Title <- 'Don'
testing$Title <- as.factor(testing$Title)
predictions <- predict(modelFinal, testing)
testing$Survived <- predictions
#testing$Survived <- round(predictions,digit=0)

# Notable Exceptions:======
testing[testing$Age==0 & testing$Sex=='female',]$Survived <- 0  # 100% in training
testing[testing$Title=='Mr', ]$Survived <- 0 # only 15% survived
testing[testing$Embarked=='']$Survived <- 0

submit <- data.frame(PassengerId=testing$PassengerId, Survived=testing$Survived)

write.csv(submit, file = "submittion_111714_tres.csv", row.names = FALSE)

#=========================Close up H2o
h2o.shutdown(localH2O)
localH2O = h2o.init(nthreads = -1)
