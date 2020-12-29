download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", dest="plm-training.csv", method="curl")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", dest="plm-testing.csv", method="curl")
rawdata <- read.csv("plm-training.csv", na.strings = c("#DIV/0!","NA",""))
dim(rawdata)

## 160 variables, need to reduce # of variables to fit model
## First remove columns for meta dat information
othervars <- c(1:7)
rawdata <- rawdata[, -othervars]

## Survey NA ratio per column
naratios <- NULL
for (i in (1:ncol(rawdata))){
  naratio <-  sum(is.na(rawdata[,i]))/nrow(rawdata)
  naratios <- c(naratios, naratio)
} 
summary(naratios)
quantile(naratios, c(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1))
## we have decided to exclude all columns with NA ratio > 0.3
navars <- NULL
for(i in (1: length(naratios))){
  if(naratios[i] >= 0.3)
    navars <- c(navars, names(rawdata)[i])
}
rawdata <- rawdata[,!(names(rawdata) %in% navars)] 
dim(rawdata)
rawdata$classe <- as.factor(rawdata$classe)

## We can further eliminate columns with zero variance
nearZeroVar(rawdata[, -53], names=TRUE)
## Turns out there is no variable with zero variance
## Now we have 52 variables plus one outcome

## Data partition
library(caret)
set.seed(799)
inTrain <- createDataPartition(y=rawdata$classe, p=0.7, list=FALSE)
train <- rawdata[inTrain, ]
test <- rawdata[-inTrain, ]

## Use PCA model to reduce # of variables further
pca_model <- prcomp(train[ , -53])
summary(pca_model)
## PC1 to PC9 explains 95.14% of total variance, we can use these 9
pca.df <- as.data.frame(pca_model$x[, 1:9])
pca.df <- cbind(pca.df, train$classe)
names(pca.df)[10] <- "classe"

## ntree = 10
rf.10 <- randomForest(x=pca.df[, 1:9], y=pca.df[, 10], ntree=10, mtry=3, proximity=TRUE)
## Accuracy : 0.83

## ntree = 100
rf.100 <- randomForest(x=pca.df[, 1:9], y=pca.df[, 10], ntree=100, mtry=3, proximity=TRUE)
## Accuracy : 0.9379          

## ntree = 500
rf.500 <- randomForest(x=pca.df[, 1:9], y=pca.df[, 10], ntree=500, mtry=3, proximity=TRUE)
confusionMatrix(rf.500$predicted, train$classe)
## Accuracy : 0.9458   

## ntree = 1000
rf.1000 <- randomForest(x=pca.df[, 1:9], y=pca.df[, 10], ntree=1000, mtry=3, proximity=TRUE)
confusionMatrix(rf.1000$predicted, pca.df$classe)
## Accuracy : 0.949

## run model once on test  data set
test_pca <- predict(pca_model, test[, -53])
test.pca.df <- as.data.frame(test_pca[, 1:9])
test.pca.df <- cbind(test.pca.df, "classe"=test$classe)
test.rf.500 <- randomForest(x=test.pca.df[, 1:9], y=test.pca.df[,10], ntree=500, mtry=3, proximity=TRUE)
confusionMatrix(test.rf.500$predicted, test.pca.df$classe)
## Accuracy : 0.8904 worse than train as expected



library(rpart)
library(rpart.plot)
rpart <- rpart(classe~., data=pca.df, control = rpart.control(cp = 0.0001))
bestcp <- rpart$cptable[which.min(rpart$cptable[,"xerror"]),"CP"]
rpart.pruned <- prune(rpart, cp = bestcp)
conf.matrix <- table(pca.df$classe, predict(rpart.pruned,type="class"))
## result is pretty bad, not going to use this model
