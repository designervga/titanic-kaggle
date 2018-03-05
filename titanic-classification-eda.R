setwd("~/R Projects/lectures-ml/competition/titanic-classification")

library(doParallel) 
cl <- makeCluster(detectCores(), type='PSOCK')
registerDoParallel(cl)

library(caret)
library(ggplot2)
library(plyr)
library(dplyr)


training <- read.csv("./data/train.csv", stringsAsFactors = FALSE, na.strings=c(""," ","NA"))
testing <- read.csv("./data/test.csv", stringsAsFactors = FALSE, na.strings=c(""," ","NA"))
testing$Survived <- NA
data <- rbind(training, testing)





# EXPLATORY DATA ANALYSIS
# look for sequence pattern with index
plot(index(training), training$Fare)

# age histogram by sex
g1 <- ggplot(na.omit(training), aes(Age)) + geom_histogram(aes(fill = Sex), bins = 20)
g1 +  facet_grid(. ~ Survived) +
     labs(x = "How many people died and survived on the Titanic?") +
     geom_label(stat="count", aes(label=..count..))
table(training$Sex, training$Survived)

# survived class vs sex
g2 <- ggplot(training, aes(x = factor(Survived)))
g2  + geom_bar(aes(fill = factor(Sex)), position = "dodge") + 
     labs(x = "How many people died and survived on the Titanic?") +
     geom_label(stat="count", aes(label=..count..))
# male died more


# survived by age and sex
g3 <- ggplot(training, aes(x = Age, y = Survived))
g3  + geom_bar(stat = "identity") + facet_grid(. ~ Sex)

# survived pclass vs sex
g4 <- ggplot(training, aes(x = factor(Survived)))
g4  + geom_bar(aes(fill = Sex), position = "dodge") + facet_grid(. ~ Pclass)
# rich and mid class female almost survived
# poor and mid class male always died

# age by sex
g5 <- ggplot(na.omit(training), aes(x = Age))
g5  + geom_density(aes(fill = Sex), alpha = 0.5)
# females are younger

# survived by age and sex
g6 <- ggplot(na.omit(training), aes(x = Age, fill = factor(Survived)))
g6  + geom_bar(stat = "count", width = 1) + facet_grid(Survived ~ .)
# infants tend to survive

# proportion of survivers by sex and class
g7 <- ggplot(na.omit(training), aes(x = Pclass, fill = factor(Survived)))
g7 +  geom_bar(stat="count", position="fill") +
      labs(x = 'Training data only', y= "Percent") + facet_grid(. ~ Sex) +
      theme(legend.position="none")
# Pclass 1 and 2 are almost guaranteed survival for women, and Pclass 2 is almost as bad as Pclass 3 for men

# age vs fare
g8 <- ggplot(na.omit(training), aes(x = Age, y = Fare, colour = factor(Pclass))) +
      geom_point() + geom_smooth()
# fare is balanced across all ages
# TESTING - age vs fare
g8_testing <- ggplot(testing, aes(x = Age, y = Fare, colour = factor(Pclass))) +
              geom_point() + geom_smooth()
library(gridExtra)
grid.arrange(g8, g8_testing, ncol = 2)
# training and testing seems balanced for this features. No need to super handle validation.

# embarked vs fare
g9 <- ggplot(na.omit(training), aes(x = Embarked, fill = factor(Pclass)))
g9  + geom_bar(stat = "count", position = "dodge") + facet_grid(. ~ Sex)
# rich tends (p1) to embark at C. No separation between gender.

# embarked vs survived
g10 <- ggplot(na.omit(training), aes(x = Embarked, fill = factor(Survived)))
g10  + geom_bar(stat = "count", position = "dodge") + facet_grid(. ~ Sex)
# almost all females that embarked at C survived
# all male that embarket at Q died


# count of NA per column / feature
sapply(data, function(x) sum(is.na(x))) # fix age and fare







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



# group of people by ticket
ticket_group <- ddply(data, ~ Ticket, function(x) c(Ticket_group_size = length(x$Ticket)))
# merge
data <- left_join(data, ticket_group, by = "Ticket")

data$Ticket_group[data$Ticket_group_size == 1] <- "Alone"
data$Ticket_group[data$Ticket_group_size == 2] <- "Couple"
data$Ticket_group[data$Ticket_group_size >= 3 & data$Ticket_group_size <= 5] <- "Group"
data$Ticket_group[data$Ticket_group_size >5] <- "LargeGroup"


# select data
data <- data %>% select(Pclass, Age, Sex, Title, Survived, SibSp, Parch, Fare, Embarked, PclassSex, Age_group, Age_sex,
                        Fare_cat, Sex_embarked, Ticket_group_size, Ticket_group)
# data$Pclass <- as.factor(data$Pclass) # 1st is upper and 3rd is lower class
# data$Title <- as.factor(data$Title)
# data$Sex <- as.factor(data$Sex)



# create dummy variables from levels of factors
Pclass <- data$Pclass
data.dummy <- dummyVars(~ ., data = data[, -1], fullRank = TRUE)
data <- as.data.frame(predict(data.dummy, data)) # no more levels as text
data$Pclass <- Pclass


# convert response to factor class
data$Survived <- as.factor(ifelse(data$Survived == 1, "survived", "died"))
prop.table(table(data$Survived)) # 61.8% died



# unbind testing and training data (Now none have NAs)
testing <- data[is.na(data$Survived), ]
training <- data[!is.na(data$Survived), ]





# EXPLATORY DATA ANALYSIS - AFTER FEATURE ENGINEERING
sapply(data, function(x) sum(is.na(x)))

# survived vs age
g8 <- ggplot(training, aes(x = Age, fill = factor(Survived)))
g8  + geom_bar(stat = "count", width = 1) + facet_grid(Pclass ~ .)
# older were more rich (1st class)
# younger males tend to be poor
# the peak is the inputed median age for NA.

# remove hotencoding the see below plot
g9 <- ggplot(training, aes(x = Age, fill = factor(Survived)))
g9 +  geom_bar(stat = "count", width = 1) + facet_grid(Age_group ~ .)

# survived vs age group
g10 <- ggplot(training, aes(x = factor(Survived)))
g10  + geom_bar(aes(fill = Age_group), position = "dodge") + facet_grid(. ~ Pclass)
# no mid class child died


# age vs fare
g11 <- ggplot(na.omit(training), aes(x = Age, y = Fare, colour = factor(Pclass)))
g11  + geom_point() + geom_smooth()
# fare is balanced across all ages

