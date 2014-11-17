#plotting bits
library(caret)

training[is.na(training$Age),]$Age <- 0

ggplot(data=training, aes(training$Sex,training$Fare)) +
  geom_point(
    aes(colour=factor(training$Survived))
  )

focue <- formula(
  Survived ~ Fare + Age + Sex + Pclass + FamilySize #+ Title
  )

fit3 <- lm(
  formula=focue,
  data=training
  )

#validation[is.na(validation$Age),]$Age <- 0

#predictions <- predict(fit, validation)
predictions3 <- predict(fit3, validation)
validation$pred3 <- round(predictions3,digit=0)
confusionMatrix(validation$pred3, validation$Survived)
