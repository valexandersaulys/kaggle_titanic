rm(list = ls())
library(caret)
library(h2o)
set.seed(45684812)
testing <- read.csv('test.csv')

# Setup h2o
#localH2O  <- h2o.init(max_mem_size='6g')

# Prepping Ze Data
# First separates titles from names and then removes the name variables
testing$Name <- as.character(testing$Name)
testing$Title <- sapply(testing$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
testing$Title <- sub(' ', '', testing$Title)
testing$Title <- factor(testing$Title)

testing$FamilySize <- testing$SibSp + testing$Parch + 1

# Clear Useless Variables
testing$Ticket <- NULL
testing$Cabin <- NULL #thinking the incomplete natre of cabin can be handled by h2o
testing$Name <- NULL
testing$SibSp <- NULL
testing$Parch <- NULL
testing$Embarked <- NULL

#test.h2o <- as.h2o(localH2O, testing, key='test.hex')

# ========================Submitting Stuff
#predictions <- as.data.frame(h2o.predict(modelFinal,newdata=test.h2o))
#submit <- data.frame(PassengerId=as.data.frame(test.h2o)$PassengerId,
 #                    Survived = predictions$predict)

predictions <- predict(modelFinal, testing)
submit <- data.frame(PassengerId=testing$PassengerId, Survived=predictions)

write.csv(submit, file = "submittion_111614_uno.csv", row.names = FALSE)

#=========================Close up H2o
h2o.shutdown(localH2O)
localH2O = h2o.init(nthreads = -1)
