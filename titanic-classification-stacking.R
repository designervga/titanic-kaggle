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
trControl_tuned <- trainControl(
  method = "cv", number = 5, search = "grid",
  index = cv_folds,
  savePredictions = TRUE,
  classProbs = TRUE,
  verboseIter = TRUE
)


# TRAINING-TRAIN

# LASSO (best at test set) 
lassoGrid <- expand.grid(
  alpha = 1,
  lambda = 0.012)


# Elastic Net 
glmnetGrid <- expand.grid(
  alpha = 0,
  lambda = seq(0.001, 0.2, 0.001))


# Support Vector Machines with Radial Basis Function Kernel 
svmRadialGrid <- expand.grid(
  sigma = 0.062,
  C = 2.7)


# Decision tree with rpart
rpartGrid <- expand.grid(
  cp = seq(0.001, 0.01, 0.001))


# kernel k-nearest neighboors
kknnGrid <- expand.grid(
  kmax = 11,
  distance = 2,
  kernel = "optimal")


# random forest with ranger
rangerGrid <- expand.grid(
  mtry = 8,
  splitrule = "extratrees",
  min.node.size = 8)


# random forest with rf
rfGrid <- expand.grid(
  mtry = c(2:20))


# stocastic gradient boosting with gbm
gbmGrid <- expand.grid(
  n.trees = 500,
  interaction.depth = 7,
  shrinkage = 0.01,
  n.minobsinnode = 10)


# Xtreme Gradient Boosting with xgbLinear
xgbLinearGrid <- expand.grid(
  nrounds = 100,
  lambda = 0.1,
  alpha = 1,
  eta = 0.5)


# extreme regularized gradient boosting with xgbTree (xboost)
xgbTreeGrid <- expand.grid(
  nrounds = 600,
  eta = 0.15,
  gamma = 0.3,
  colsample_bytree = 0.04,
  min_child_weight = 3,
  subsample = 0.9,
  max_depth = 6)






# STACKING TUNED MODELS
library(caretEnsemble)
set.seed(1234)
model_list <- caretList(
  Survived ~.,
  data = training,
  trControl = trControl_tuned,
  tuneList=list(
    gbm        = caretModelSpec(method = "gbm",        tuneGrid = gbmGrid),
    xgbTree    = caretModelSpec(method = "xgbTree",    tuneGrid = xgbTreeGrid),
    xgbLinear  = caretModelSpec(method = "xgbLinear",  tuneGrid = xgbLinearGrid),
    kknn       = caretModelSpec(method = "kknn",       tuneGrid = kknnGrid),
    rpart      = caretModelSpec(method = "rpart",      tuneGrid = rpartGrid),
    svmRadial  = caretModelSpec(method = "svmRadial",  tuneGrid = svmRadialGrid),
    glmnet     = caretModelSpec(method = "glmnet",     tuneGrid = lassoGrid),
    glmnet     = caretModelSpec(method = "glmnet",     tuneGrid = glmnetGrid),
    rf         = caretModelSpec(method = "rf",         tuneGrid = rfGrid),
    ranger     = caretModelSpec(method = "ranger",     tuneGrid = rangerGrid)))


p <- as.data.frame(predict(model_list, newdata=head(testing)))
print(p)
xyplot(resamples(model_list))

modelCor(resamples(model_list))
# their predicitons are fairly un-correlated, but their overall accuaracy is similar.





#  linear greedy optimization on RMSE
my_control <- trainControl(
  method = "boot",
  number = 5,
  verboseIter = TRUE,
  savePredictions = "final"
)

#Make a linear regression ensemble
glm_ensemble <- caretStack(
  model_list,
  method = "glm",
  trControl = my_control)

summary(glm_ensemble)
plot(glm_ensemble$ens_model$finalModel)


# caretStacking prediction
pred_stack <- predict(glm_ensemble, testing)
pred_stack <- ifelse(pred_stack == "survived", 1, 0)
submit <- data.frame(PassengerId = PassengerId, Survived = pred_stack)


# submission stacking
write.csv(submit, file = "submission02.glm.staking.gbm.xgbTree.xgbLinear.kknn.ranger.lasso.elasticnet.rf.svmRadial.csv", row.names = FALSE)
