#
# Neural Net
#

source('AMS.R')

require(ROCR)
require(plyr)
require(h2o)

# Read Data and prep
# --------------------------------------------------------------
train = read.csv(file.path("data","training.csv"))

train$PRI_jet_num <- factor(train$PRI_jet_num)
train[train==-999] <- NA

head(train)

# Save response vectors to use later
train_WL = train[, c('EventId', 'Weight', 'Label')]

head(train_WL)

# set the Label, b=0, s=1
#train$Label <- ifelse(train_WL$Label == 'b', 0, 1)

table(train_WL$Label)
table(train_WL$Label)/length(train_WL$Label)

#
# Neural Net for h2o
#

localH2O  <- h2o.init(Xmx = '6g' )

h2o.clusterInfo(localH2O)

# upload to h2o
train.h2o <- as.h2o(localH2O, train, key='train.hex')

summary(train.h2o)
class(train.h2o)

# model
independent <- names(train.h2o)[!names(train.h2o) %in% c('EventId', 'Weight', 'Label')]
dependent <- 'Label'

model.h2o <- h2o.deeplearning(x=independent, y=dependent, data=train.h2o
                              ,classification=TRUE
                              ,activation = 'Rectifier'
                              ,hidden = c(30,30)
                              ,epochs = 1000.0
                              ,train_samples_per_iteration = -1
                              ,adaptive_rate = TRUE
                              ,rho = 0.999
                              ,epsilon = 1e-6
                              ,input_dropout_ratio = 0.0
                              ,score_interval = 5.0
                              ,variable_importances = TRUE
                              ,diagnostics = TRUE
                              ,single_node_mode = TRUE
                              )

model.h2o

predict.h2o <- h2o.predict(object = model.h2o, newdata = train.h2o)

preds <- as.data.frame(predict.h2o)

head(preds)

table(train_WL$Label)/length(train_WL$Label)
table(preds$predict)/length(preds$predict)

nrow(preds)

model.h2o
AMS(pred=preds$predict,real=train_WL$Label,weight=train$Weight)

# Make predictions on test set and create submission file
test = read.csv(file.path("data","test.csv"))

test$PRI_jet_num <- factor(as.character(test$PRI_jet_num))

# upload to h2o
test.h2o <- as.h2o(localH2O, test, key='test.hex')

nnTestPrediction.h2o = h2o.predict(object=model.h2o, newdata=test.h2o)

nnTestPrediction <- as.data.frame(nnTestPrediction.h2o)
head(nnTestPrediction)

weightRank = rank(nnTestPrediction$s, ties.method= "random")

# table
table(nnTestPrediction$predict)
table(nnTestPrediction$predict)/length(nnTestPrediction$predict)

submission = data.frame(EventId = test$EventId, RankOrder = weightRank, Class = nnTestPrediction$predict)
write.csv(submission, "nnet_submission_3.csv", row.names=FALSE)





# cleanup
h2o.shutdown(localH2O, prompt = FALSE)
