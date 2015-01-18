rm(list = ls())
library(caret)
library(h2o)
set.seed(9841651)
set <- read.csv('train.csv')
backup <- set

# Setup h2o
#localH2O  <- h2o.init(max_mem_size='6g')

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

train.h2o <- as.h2o(localH2O, training, key='train.hex')
valid.h2o <- as.h2o(localH2O, validation, key='valid.hex')
set.h2o <- as.h2o(localH2O, set, key='set.hex')

indy <- names(train.h2o)[!names(train.h2o) %in% c('Survived')]
depy <- 'Survived'

model.nn <- h2o.deeplearning(x=indy,y=depy,
                               data=train.h2o,
                             classification=TRUE,
                             nfolds=20, epochs=10, 
                             hidden=c(15,15,15,15,15),
                             activation='Tanh',
                             balance_classes=TRUE,
                             single_node_mode=TRUE,
                             l1=0.00000001,l2=0.00000001, 
                             seed=999845)
predictions <- h2o.predict(model.nn,newdata=valid.h2o)
h2o.confusionMatrix(predictions$predict,valid.h2o$Survived)

# ========================Submitting Stuff


#=========================Close up H2o
h2o.shutdown(localH2O)
localH2O = h2o.init(nthreads = -1)
