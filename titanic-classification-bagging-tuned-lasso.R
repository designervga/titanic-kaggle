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
cv_folds <- createMultiFolds(training.train$Survived, k = folds, times = 1)
trControl_tuned <- trainControl(
  method = "boot", number = 5, repeats = 1, search = "grid",
  index = cv_folds,
  # summaryFunction = twoClassSummary, # add for ROC
  classProbs = TRUE, # Important for classification
  verboseIter = TRUE
)

# train control for searching parameter
trControl_search <- trainControl(
  method = "repeatedcv", number = folds, repeats = 2, search = "random",
  index = cv_folds,
  classProbs = TRUE, # Important for classification
  verboseIter = TRUE
)






# TRAINING-TRAIN

# LASSO Tuning
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
cutoff <- coords(auc, "best", ret = c("threshold"))
plot.roc(auc, print.thres = cutoff)
# training.test AUC 0.8078






# bag of lasso
predictorNames <- names(predictors)
length_divisor <- 1
predictions <- 0
predictions <- foreach(i=1:100,.combine=cbind) %dopar% { 
  set.seed(i)
  sampleRows <- sample(nrow(training.train), size = floor((nrow(training.train)/length_divisor)), replace = TRUE)
  for(m in i) {
    lambda_test = 0.012 + log(sqrt(m))/1000
    myGrid_lasso <- expand.grid(
      alpha = 1,
      lambda = lambda_test
    )
  }
  fit <- train(Survived ~ ., data = training.train[sampleRows, ],
               method = "glmnet",
               tuneGrid = myGrid_lasso,
               metric = "Accuracy",
               trControl = trControl_tuned)
  predictions[i] <- data.frame(predict(fit, newdata = training.test, type = "prob")[1]) # pred > .5 died
}
auc <- roc(training.test$Survived, rowMeans(predictions), plot = TRUE)
cutoff <- coords(auc, "best", ret = c("threshold"))
print(auc)

# predicting with average of probabilities
bag_mean_pred <- rowMeans(predictions) # prob of dying
bag_mean_pred <- ifelse(bag_mean_pred < cutoff, "survived", "died")
confusionMatrix(training.test$Survived, bag_mean_pred)
# accuracy 0.831 

# predicting with votes
mostvoted <- function(x) {
  names(sort(table(x), decreasing = TRUE))[1]
}
predictions2 <- ifelse(predictions < 0.5, "survived", "died")
bag_mostvoted_pred <- apply(predictions2, 1, mostvoted)
confusionMatrix(training.test$Survived, bag_mostvoted_pred)
# 0.8254






# statistical analysis of difference in bagged predictions with voting vs simple average of probabilities
# test of equal given proportions
vote_vs_mean <- as.data.frame(cbind(bag_mostvoted_pred, bag_mean_pred))
colnames(vote_vs_mean) = c("vote", "mean")
barplot(table(vote_vs_mean), beside = TRUE, legend = names(vote_vs_mean))
# 'suvived' predictions always have 100% of agreement (cuttoff ~ 0.6)
barplot(table(vote_vs_mean$vote), legend = names(vote_vs_mean), ylim = c(0, 250))
barplot(table(vote_vs_mean$mean), legend = names(vote_vs_mean), ylim = c(0, 250))
# is there a difference in proportion of 'suvived' between each method?


# two sample test of equal proportion
p1 <- xtabs(~ vote, data = vote_vs_mean)
p2 <- xtabs(~ mean, data = vote_vs_mean)
p1p2 <- rbind(p1, p2)

# H0: p1 = p2 | Ha: p1 ≠ P2
prop.test(x = c(p1p2[1], p1p2[2]), n = c(p1p2[1] + p1p2[3], p1p2[2] + p1p2[4]))
# we are 95% confident that proportion of 'suvived' is different between average and voting (when cuttoff is little above 0.5)
# with cutoff 0.5 we have almost the same proportions. There os almost no difference between bagging with average vs voting.
# altough proportions are different, we saw the same result in the test dataset (same position in the LB)







# statistical analysis of difference in bagged lasso vs no bagging
# two sample test of equal proportion
train_pred <- ifelse(train_pred == 1, "survived", "died") 
bag_vs_single <- as.data.frame(cbind(bag_mostvoted_pred, train_pred))
u1 <- xtabs(~ bag_mostvoted_pred, data = bag_vs_single)
u2 <- xtabs(~ train_pred, data = bag_vs_single)
u1u2 <- rbind(u1, u2)

# H0: u1 = u2 | Ha: u1 ≠ u2
prop.test(x = c(u1u2[1], u1u2[2]), n = c(u1u2[1] + u1u2[3], u1u2[2] + u1u2[4]))
# we are not 95% confident that there is a difference between prediction from bagged and single lasso model
# obs: dataset is to small to detect a small difference in proportion






# FINAL TRAINING - Bagged Simple Average
finalPredictions <- 0
length_divisor <- 1
finalPredictions <- foreach(i=1:100,.combine=cbind) %dopar% { 
  set.seed(i)
  sampleRows <- sample(nrow(training), size = floor((nrow(training)/length_divisor)), replace = TRUE)
  for(m in i) {
    lambda_test = 0.012 + log(sqrt(m))/1000
    myGrid_lasso <- expand.grid(
      alpha = 1,
      lambda = lambda_test
    )
  }
  fit <- train(Survived ~ ., data = training[sampleRows, ],
               method = "glmnet",
               tuneGrid = myGrid_lasso,
               # tuneLength = 10,
               metric = "Accuracy",
               trControl = trControl_tuned)
  finalPredictions[i] <- data.frame(predict(fit, newdata = testing, type = "prob")[1]) # pred > .5 died
}

# predicting with average of probabilities
bag_mean_finalPred <- rowMeans(finalPredictions) # prob of dying
bag_mean_finalPred <- ifelse(bag_mean_finalPred < 0.5, "survived", "died")
bag_mean_finalPred <- ifelse(bag_mean_finalPred == "survived", 1, 0)
submit <- data.frame(PassengerId = PassengerId, Survived = bag_mean_finalPred)


# predicting with votes
finalPredictions2 <- ifelse(finalPredictions < 0.5, "survived", "died")
bag_mostvoted_finalPred <- apply(finalPredictions2, 1, mostvoted)
bag_mostvoted_finalPred <- ifelse(bag_mostvoted_finalPred == "survived", 1, 0)
submit <- data.frame(PassengerId = PassengerId, Survived = bag_mostvoted_finalPred) # votes
# bagged results for both probability average and votes showed no improvements in the test dataset


# submission bagging.lasso
write.csv(submit, file = "submission06.bagging.lasso.100.oneHotEncoding.36var.alpha1.lambdaNEAR0012.votes.csv", row.names = FALSE)
