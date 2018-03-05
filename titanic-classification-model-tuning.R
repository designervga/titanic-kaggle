setwd("~/R Projects/lectures-ml/competition/titanic-classification/titanic-kaggle")

library(doParallel) 
cl <- makeCluster(detectCores(), type='PSOCK')
registerDoParallel(cl)

library(ggplot2)
library(plyr)
library(dplyr)
library(pROC)
library(zoo)
library(caret)


training <- read.csv("./data/train.csv", stringsAsFactors = FALSE, na.strings=c(""," ","NA"))
testing <- read.csv("./data/test.csv", stringsAsFactors = FALSE, na.strings=c(""," ","NA"))
testing$Survived <- NA
data <- rbind(training, testing)
PassengerId <- testing$PassengerId



# FEATURE ENGINEERING

# get social status title of person from their name
data$Title <- ifelse(grepl("Mr", data$Name), "Mr", 
              ifelse(grepl("Mrs", data$Name), "Mrs", 
              ifelse(grepl("Miss", data$Name), "Miss", "nothing")))

# fill NAs of Age with decision tree
library(rpart)
rpartFit_age <- rpart(Age ~ Survived + Sex + Pclass + Title + Fare, data = data[!is.na(data$Age), ], 
                      method = "anova", control = rpart.control(cp = 0.001))
data$Age[is.na(data$Age)] <- predict(rpartFit_age, data[is.na(data$Age), ])

# fill NAs of Embarked with decision tree
rpartFit_Embarked <- rpart(Embarked ~ Survived + Sex + Pclass + Title + Fare, data = data[!is.na(data$Embarked), ],
                         control = rpart.control(cp = 0.001))
data$Embarked[is.na(data$Embarked)] <- as.character(predict(rpartFit_Embarked, data[is.na(data$Embarked), ], type = "class"))

# fill NAs of Fare with median age
data$Fare[is.na(data$Fare)] <- median(data$Fare, na.rm = TRUE)


# convert variables to correct class
data$Pclass <- as.ordered(data$Pclass) # will make hot encoding work


# combine ex and class
data$PclassSex[data$Pclass == 1 & data$Sex == "female"] <- "P1Female"
data$PclassSex[data$Pclass == 2 & data$Sex == "female"] <- "P2Female"
data$PclassSex[data$Pclass == 3 & data$Sex == "female"] <- "P3Female"
data$PclassSex[data$Pclass == 1 & data$Sex == "male"] <- "P1Male"
data$PclassSex[data$Pclass == 2 & data$Sex == "male"] <- "P2Male"
data$PclassSex[data$Pclass == 3 & data$Sex == "male"] <- "P3Male"



# categorical age
data$Age_group[data$Age <= 10] <- "child"
data$Age_group[data$Age > 10 & data$Age <= 50] <- "adult"
data$Age_group[data$Age > 50] <- "elder"


# categorical age and sex
data$Age_sex[data$Age_group == "child" & data$Sex == "male"]    <- "child_male"
data$Age_sex[data$Age_group == "child" & data$Sex == "female"] <- "child_female"
data$Age_sex[data$Age_group == "adult" & data$Sex == "male"]    <- "adult_male"
data$Age_sex[data$Age_group == "adult" & data$Sex == "female"] <- "adult_male"
data$Age_sex[data$Age_group == "elder" & data$Sex == "male"]    <- "elder_male"
data$Age_sex[data$Age_group == "elder" & data$Sex == "female"]  <- "elder_female"


# embarked and sex
data$Sex_embarked[data$Sex == "male" & data$Embarked == "Q"]   <- "male_Q"
data$Sex_embarked[data$Sex == "female" & data$Embarked == "Q"] <- "female_Q"
data$Sex_embarked[data$Sex == "male" & data$Embarked == "S"]   <- "male_S"
data$Sex_embarked[data$Sex == "female" & data$Embarked == "S"] <- "female_S"
data$Sex_embarked[data$Sex == "male" & data$Embarked == "C"]   <- "male_C"
data$Sex_embarked[data$Sex == "female" & data$Embarked == "C"] <- "female_C"



# fare cat
data$Fare_cat[data$Fare == 0]  <- "free"
data$Fare_cat[data$Fare > 0 & data$Fare <= 100]  <- "normal"
data$Fare_cat[data$Fare > 100]  <- "expensive"


# log of numeric
data$Age <- log(data$Age +1)
data$Fare <- log(data$Fare +1)




# select data
data <- data %>% select(Pclass, Age, Sex, Title, Survived, SibSp, Parch, Fare, Embarked, PclassSex, Age_group, Age_sex,
                        Fare_cat, Sex_embarked)
# data$Pclass <- as.factor(data$Pclass) # 1st is upper and 3rd is lower class
# data$Title <- as.factor(data$Title)
# data$Sex <- as.factor(data$Sex)

# near zero variance here? select by hand with variable importance analysis?
# nzv <- nearZeroVar(data.clean, saveMetrics = TRUE)
# nzvNames <- row.names(nzv[nzv$nzv == TRUE, ])
# data.clean <- data.clean[, !nzv$nzv]

# create dummy variables from levels of factors
Pclass <- data$Pclass
data.dummy <- dummyVars(~ ., data = data[, -1], fullRank = FALSE)
data <- as.data.frame(predict(data.dummy, data)) # no more levels as text
data$Pclass <- Pclass


# convert response to factor class
data$Survived <- as.factor(ifelse(data$Survived == 1, "survived", "died"))
prop.table(table(data$Survived)) # 61.8% died



# unbind testing and training data (Now none have NAs)
testing <- data[is.na(data$Survived), ]
training <- data[!is.na(data$Survived), ]



# split training into train and test (to get quality metric estimation before sending to Kaggle)
set.seed(42)
inTrain <- createDataPartition(training$Survived, p = 0.6, list = FALSE)
training.train <- training[inTrain, ]
training.test <- training[-inTrain, ]





# train control with tuned parameters
folds <- 5
trControl_tuned <- trainControl(
  method = "repeatedcv", number = 5, repeats = 1, search = "grid",
  index = cv_folds,
  # summaryFunction = twoClassSummary, # add for ROC
  classProbs = TRUE, # Important for classification
  verboseIter = TRUE
)
cv_folds <- createMultiFolds(training.train$Survived, k = folds, times = 1)

# train control for searching parameter
trControl_search <- trainControl(
  method = "repeatedcv", number = folds, repeats = 2, search = "random",
  index = cv_folds,
  classProbs = TRUE, # Important for classification
  verboseIter = TRUE
)






# TRAINING-TRAIN

# LASSO (FINAL) 
myGrid_lasso <- expand.grid(
  alpha = 1,
  lambda = 0.012
)

set.seed(42)
lassoFit <- train(Survived ~ ., data = training.train,
                   method = "glmnet",
                   tuneGrid = myGrid_lasso,
                   # tuneLength = 10,
                   metric = "Accuracy",
                   trControl = trControl_tuned)
plot(lassoFit$finalModel, label = TRUE)
plot(varImp(lassoFit))
# training.train Accuracy 0.83027

train_pred <- predict(lassoFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 0.8282    

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.8078






# bag of lasso
predictorNames <- names(predictors)
length_divisor <- 1
predictions <- 0
predictions <- foreach(i=1:10,.combine=cbind) %dopar% { 
  set.seed(i)
  sampleRows <- sample(nrow(training.train), size = floor((nrow(training.train)/length_divisor)), replace = TRUE)
  fit <- train(Survived ~ ., data = training.train[sampleRows, ],
               method = "glmnet",
               tuneGrid = myGrid_lasso,
               # tuneLength = 10,
               metric = "Accuracy",
               trControl = trControl_tuned)
  predictions[i] <- data.frame(predict(fit, newdata = training.test, type = "prob")[1]) # pred > .5 died
}
auc <- roc(training.test$Survived, rowMeans(predictions), plot = TRUE)
print(auc)

bag_mean_pred <- rowMeans(predictions) # prob of dying
bag_mean_pred <- ifelse(bag_mean_pred < .5, "survived", "died")
confusionMatrix(training.test$Survived, bag_mean_pred)
# accuracy 0.8028 









# Elastic Net 
myGrid_glmnet <- expand.grid(
  alpha = 0,
  lambda = seq(0.001, 0.2, 0.001)
)

set.seed(42)
glmnetFit <- train(Survived ~ ., data = training.train,
                method = "glmnet",
                tuneGrid = myGrid_glmnet,
                # tuneLength = 1000,
                metric = "Accuracy",
                trControl = trControl_search)
plot(glmnetFit$finalModel, label = TRUE)
plot(varImp(glmnetFit))
# training.train Accuracy 0.8241789

train_pred <- predict(glmnetFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 0.8394   

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.8183






# Support Vector Machines with Radial Basis Function Kernel 
myGrid_svmRadial <- expand.grid(
  sigma = 0.062,
  C = 2.7
)
set.seed(42)
svmRadialFit <- train(Survived ~ ., data = training.train,
                   method = "svmRadial",
                   tuneGrid = myGrid_svmRadial,
                   # tuneLength = 100,
                   metric = "Accuracy",
                   trControl = trControl_tuned)
plot(svmRadialFit, label = TRUE)
plot(varImp(svmRadialFit))
# training.train Accuracy 

train_pred <- predict(svmRadialFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 0.8056

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.7853





# Decision tree with ctree
myGrid_rpart <- expand.grid(
  cp = seq(0.001, 0.01, 0.001)
)

set.seed(42)
rpartFit <- train(Survived ~ ., data = training.train,
                   method = "rpart",
                   tuneGrid = myGrid_rpart,
                   # tuneLength = 100,
                   metric = "Accuracy",
                   trControl = trControl_tuned)
plot(rpartFit) # while tuning
library(rattle)
library(rpart.plot)
rpart.plot(rpartFit$finalModel)
# training.train Accuracy 0.8041191

train_pred <- predict(rpartFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 0.8141

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.7946






# kernel k-nearest neighboors
myGrid_kknn <- expand.grid(
  kmax = 11,
  distance = 2,
  kernel = "optimal")

set.seed(42)
kknnFit <- train(Survived ~ ., data = training.train,
                  method = "kknn",
                  tuneGrid = myGrid_kknn,
                  # tuneLength = 100,
                  metric = "Accuracy",
                  trControl = trControl_tuned)
plot(kknnFit$finalModel)
# training.train Accuracy 

train_pred <- predict(kknnFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 0.8

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.7863






# random forest with ranger
myGrid_ranger <- expand.grid(
  mtry = 8,
  splitrule = "extratrees",
  min.node.size = 8
)

set.seed(42)
rangerFit <- train(Survived ~ ., data = training.train,
                  method = "ranger",
                  tuneGrid = myGrid_ranger,
                  # tuneLength = 10,
                  metric = "Accuracy",
                  trControl = trControl_tuned)
plot(rangerFit$finalModel)
# training.train Accuracy 0.8059536

train_pred <- predict(rangerFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 0.8394 

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.8225







# random forest with rf
myGrid_rf <- expand.grid(
  mtry = c(2:20)
)

set.seed(42)
rfFit <- train(Survived ~ ., data = training.train,
                   method = "rf",
                   tuneGrid = myGrid_rf,
                   # tuneLength = 10,
                   metric = "Accuracy",
                   trControl = trControl_tuned)
plot(rfFit$finalModel)
# training.train Accuracy 0.8059536

train_pred <- predict(rfFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 0.8423 

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.8178






# stocastic gradient boosting with gbm
myGrid_gbm <- expand.grid(
  n.trees = 500,
  interaction.depth = 7,
  shrinkage = 0.01,
  n.minobsinnode = 10
)

set.seed(42)
gbmFit <- train(Survived ~ ., data = training.train,
                   method = "gbm",
                   tuneGrid = myGrid_gbm,
                   # tuneLength = 100,
                   metric = "Accuracy",
                   trControl = trControl_tuned)
plot(gbmFit$finalModel)
# training.train Accuracy 0.7965758

train_pred <- predict(gbmFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 0.8423

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.8192






# Xtreme Gradient Boosting with xgbLinear
myGrid_xgbLinear <- expand.grid(
  nrounds = 100,
  lambda = 0.1,
  alpha = 1,
  eta = 0.5
)

set.seed(42)
xgbLinearFit <- train(Survived ~ ., data = training.train,
                method = "xgbLinear",
                tuneGrid = myGrid_xgbLinear,
                # tuneLength = 20,
                metric = "Accuracy",
                trControl = trControl_tuned)
plot(xgbLinearFit$finalModel)
# training.train Accuracy 0.8197 

train_pred <- predict(xgbLinearFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.7954
# validation is not correlated to test set. How to split/validate this dataset?

# extreme regularized gradient boosting with xgbTree (xboost)
myGrid_xgbTree <- expand.grid(
  nrounds = 600,
  eta = 0.15,
  gamma = 0.3,
  colsample_bytree = 0.04,
  min_child_weight = 3,
  subsample = 0.9,
  max_depth = 6
)

set.seed(42)
xgbTreeFit <- train(Survived ~ ., data = training.train,
                method = "xgbTree",
                tuneGrid = myGrid_xgbTree,
                # tuneLength = 20,
                metric = "Accuracy",
                trControl = trControl_tuned)
plot(xgbTreeFit$finalModel)
# training.train Accuracy 0.8307943

train_pred <- predict(xgbTreeFit, newdata = training.test)
confusionMatrix(training.test$Survived, train_pred)
# 0.8254 | Tune 2 0.8423, overfitted on test

train_pred <- ifelse(train_pred == "survived", 1, 0)
auc <- roc(training.test$Survived, train_pred, plot = TRUE) 
print(auc)
# training.test AUC 0.7999 | Tune 2 0.8192




# MODEL COMPARISONS
resamps <- resamples(list(XBOOST = xgbTreeFit,
                          GBM = gbmFit))
summary(resamps)
trellis.par.set(caretTheme())
dotplot(resamps, metric = "Accuracy")




# FINAL TRAINING
set.seed(42)
finalFit <- train(Survived ~ ., data = training,
                   method = "glmnet",
                   tuneGrid = myGrid_lasso,
                   # tuneLength = 100,
                   metric = "Accuracy",
                   trControl = trControl_tuned)

pred <- predict(finalFit, newdata = testing)
pred <- ifelse(pred == "survived", 1, 0)
# ranger    0.8290637
# xgbTree   0.8342908
# lasso     0.8327961 - best so far Kaggle
# ridge     0.8294124
# svmRadial 0.8175871
# gbm       0.831067
# rf        0.8303788
# xgbLinear 0.8260479    


submit <- data.frame(PassengerId = PassengerId, Survived = pred)


# submission glmnet
write.csv(submit, file = "submission.glmnet.05.oneHotEncoding.36var.alpha1.lambda0012.csv", row.names = FALSE)

# submission xgbTree
write.csv(submit, file = "submission.03.oneHotEncoding.xgbLinear.nrounds100.lambda01.alpha1.eta05.csv", row.names = FALSE)

