#plotting bits
library(caret)

training[is.na(training$Age),]$Age <- 0

ggplot(data=set, 
       aes(Fare, Survived)
       ) + 
  geom_point(
    aes(colour=factor(set$Survived))
  )




focue <- formula(
  Survived ~ Fare + Age + Sex + FamilySize #+ Title
  )
training[is.na(training$Age),]$Age <- 0
training[is.na(training$Fare),]$Fare <- 0
training[is.na(training$FamilySize),]$FamilySize <- 0

fit3 <- lm(
  formula=focue,
  data=training
  )

validation[is.na(validation$Age),]$Age <- 0

#predictions <- predict(fit, validation)
predictions3 <- predict(fit3, validation)
validation$pred3 <- round(predictions3,digit=0)
validation[validation$pred3>1,]$pred3 <- 1
confusionMatrix(validation$pred3, validation$Survived)





dta <- set[!names(set)=='Sex' || !names(set)=='Title']
dta.r <- abs(cor(dta)) # get correlations
dta.col <- dmat.color(dta.r) # get colors
# reorder variables so those with highest correlation
# are closest to the diagonal
dta.o <- order.single(dta.r) 
cpairs(dta, dta.o, panel.colors=dta.col, gap=.5,
       main="Variables Ordered and Colored by Correlation" )
