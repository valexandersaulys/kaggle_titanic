rm(list = ls())
library(caret)
library(h2o)
set.seed(9484)
set <- read.csv('train.csv')
backup <- set

# Setup h2o
localH2O  <- h2o.init(max_mem_size='6g')

# Prepping Ze Data
# First separates titles from names and then removes the name variables
set$Name <- as.character(set$Name)
set$Title <- sapply(set$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
set$Title <- sub(' ', '', set$Title)
set$Title <- factor(set$Title)

# Clear Useless Variables
#set$Ticket <- NULL
#set$Cabin <- NULL #thinking the incomplete natre of cabin can be handled by h2o
#set$Name <- NULL

dataPart <- createDataPartition(set$Survived,p=0.7, list=FALSE)
training <- set[dataPart,]
validation <- set[-dataPart,]

train.h2o <- as.h2o(localH2O, training, key='train.hex')
valid.h2o <- as.h2o(localH2O, validation, key='valid.hex')
set.h2o <- as.h2o(localH2O, set, key='set.hex')

indy <- names(train.h2o)[!names(train.h2o) %in% c('Survived')]
depy <- 'Survived'

model.glm <- h2o.glm(x=indy,y=depy,
                               data=train.h2o,
                             nfolds=10, family='binomial',
                             #hidden=c(100,100,100),
                             #activation='Rectifier',
                             #l1=0.00000001,l2=0.00000001,
                     )
predictions <- as.data.frame(h2o.predict(model.nn,newdata=valid.h2o))
validation$pred1 <- predictions$predict
confusionMatrix(validation$pred1,validation$Survived)

# ========================Submitting Stuff


#=========================Close up H2o
h2o.shutdown(localH2O)
