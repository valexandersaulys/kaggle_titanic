rm(list = ls())
library(caret)
library(h2o)
set.seed(2565186)
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

# Second make a cutesy variable
set$FamilySize <- set$SibSp + set$Parch + 1

# Clear Useless Variables
#set$Ticket <- NULL
set$Cabin <- NULL # thinking the incomplete nature of cabin can be handled by h2o
set$Name <- NULL

dataPart <- createDataPartition(set$Survived,p=0.7, list=FALSE)
training <- set[dataPart,]
validation <- set[-dataPart,]

# randomize the training
nr<-dim(training)[1]
training[sample.int(nr),]

train.h2o <- as.h2o(localH2O, training, key='train.hex')
valid.h2o <- as.h2o(localH2O, validation, key='valid.hex')

indy <- names(train.h2o)[!names(train.h2o) %in% c('Survived')]
depy <- 'Survived'

model.rf <- h2o.randomForest(x=indy,y=depy,
                              data=train.h2o,
                              nfolds=10, mtries=10,
                              classification=TRUE,
                              depth=10000, ntree=500,
			                        nodesize=10, seed=654197,
			                        balance.classes=TRUE)
predcictions <- h2o.predict(model.rf,valid.h2o)
h2o.confusionMatrix(predcictions$predict,valid.h2o$Survived)

# ========================Submitting Stuff


#=========================Close up H2o
h2o.shutdown(localH2O)
localH2O = h2o.init(nthreads = -1)
