fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

attach(training)

rf.model <- train(Survived ~ . ,
                    training,
                    #distribution = "gaussian",
                    method = "rf",
                    trControl = fitControl )
                    #verbose = FALSE

