library(caret)
library(h2o)
set.seed(6645)
set <- read.csv('train.csv')
backup <- set

# Create a Family Size Variable
set$FamilySize <- set$SibSp + set$Parch + 1

# Create Find age median and replace it in the missing values
male_median <- median(set[set$Sex=='male',]$Age,na.rm=TRUE)
female_median <- median(set[set$Sex=='female',]$Age,na.rm=TRUE)
set[is.na(set$Age) & set$Sex=='male',]$Age <- male_median 
set[is.na(set$Age) & set$Sex=='female',]$Age <- female_median

# Create Age^Fare Variable
#set$AgeFare <- set$Age^set$Fare

# Cabin or not?
set$Cabin <- as.character(set$Cabin)
set[set$Cabin=="",]$Cabin <- 0
set[set$Cabin != "0",]$Cabin <- 1
set$Cabin <- as.factor(set$Cabin)

# Gender to binary
set$Sex <- as.character(set$Sex)
set[set$Sex=='male',]$Sex <- '1'
set[set$Sex=='female',]$Sex <- '0'
set$Sex <- as.factor(set$Sex)

# Clear Useless Variables
set$Ticket <- NULL
#set$Cabin <- NULL #thinking the incomplete natre of cabin can be handled by h2o
set$Name <- NULL
set$SibSp <- NULL
set$Parch <- NULL
set$Embarked <- NULL

dataPart <- createDataPartition(set$Survived,p=0.7, list=FALSE)
training <- set[dataPart,]
validation <- set[-dataPart,]

#================lm values
fm <- formula(Survived ~ Pclass + Sex + Age + Fare + Cabin + FamilySize)
library(boot) #load the package
# Now we need the function we would like to estimate
# In our case the beta:
betfun = function(data,b,formula){  
  # b is the random indexes for the bootstrap sample
  d = data[b,] 
  return(lm(d[,1]~d[,2], data = d)$coef)  
  # thats for the beta coefficient
}
# now you can bootstrap:
bootbet = boot(data=training, statistic=betfun, formula=fm,R=5000) 
# R is how many bootstrap samples
names(bootbet)
plot(bootbet)
hist(bootbet$t, breaks = 100)

fit <- lm(formula=fm,data=training)





