'''
title:                      "Team_10_Project_ML2"
author:       "Lindabeth Doby,Zach Hosseinian,& Braxton Justice"
'''
 

# ############################## Special Function ##############################

# Select Clusters Function (Optional)
# select_clusters <- function(data, column, method, count) {
#                column_levels <- levels(data[[column]])
#                return_list <- list()
#                method_name <- list('both', 'random')
#                method_select <- method_name[method]
#                if (method_select == 'both'){
#                               total <- as.integer(length(column_levels))
#                               half <- as.integer(count/2)
#                               lower_limit <- total-half+1
#                               forward <- column_levels[1:half]
#                               backward <- column_levels[lower_limit:total]
#                               #print(backward)
#                               return_list <- c(forward,backward)
#                               #print(return_list)
#                } else {
#                               return_list <- sample(column_levels, count, replace = FALSE)
#                               print(return_list)
#                }
#                return(return_list)
# }

 

######################### Loading in and Basic Cleaning} ######################

rm(list=ls())
#setwd("C:\\Users\\ladob\\Documents\\MSBA\\MSBA Courses\\Semester 2\\MachineLearning2 - 5082\\MLII_Team_Prez")

# Libraries
library(ISLR)
library(caret)
library(e1071)
library(ggplot2)
library(MLmetrics)
library(randomForest)
library(caret)
library(lubridate)
library(readr)
library(dplyr)
library(caTools)
library(class)
library(psych)

# set the seed
set.seed(1)

#Loading in the data
train <- read.csv("train.csv", nrows = 10000)
# destinations <- read_csv("destinations.csv") # We can use this if we want

# Preliminary Cleaning of Data
train <- na.omit(train)  # omit the NAs

# Check Column Names
print(names(train))

# Check Data Structure
print(str(train))


 

##############################  Heavy Cleaning  ###############################
# Here we are changing these columns to Factors
factor_columns <- list(2, 3, 4, 5, 6, 8, 9, 10, 11, 17, 18, 19, 21, 22, 23, 24)
# Here we are changing these columns to Int
integer_columns <- list(7, 14, 14, 16, 20)
# # Here we are Dropping these columns
drop_columns <- list("date_time", "user_id", "user_location_city", "hotel_market", "srch_destination_id")



# THis is a loop that Lindabeth made to make this more efficent 
for (i in factor_columns){
               train[,i] <- as.factor(train[,i])
}

for (i in integer_columns) {
               train[,i] <- as.integer(train[,i])
}

for (i in drop_columns) {
               train[,i] <- NULL
}

# Lets look at the srtucter and see how the variables have changed
str(train)
 

##########################  Limiting the Content  #############################


# Here we are creating a table for the 3 X-Features that are to large to use
# As well as our target Variable
cluster_column <- table(train$hotel_cluster)
region_column <- table(train$user_location_region)
country_column <- table(train$hotel_country)

# Here we are selecting the top N of the tables, we are using 10 for speed
top_num <- 10
top_50_cluster <- head(sort(cluster_column,decreasing=TRUE),top_num)
top_50_U_R<- head(sort(region_column,decreasing=TRUE),top_num)
top_50_Hotel_country <- head(sort(country_column,decreasing=TRUE),top_num)

# Convert those new items to a Factor
top_50_cluster <- as.factor(top_50_cluster)
top_50_U_R<- as.factor(top_50_U_R)
top_50_Hotel_country <- as.factor(top_50_Hotel_country)

# Here we are now creating a new DF with only the rows that meet the criteria
# That we specified above.
holding <- train[train$hotel_cluster %in% names(top_50_cluster),]
holding$hotel_cluster <- as.factor(as.integer(holding$hotel_cluster))

new_df <- holding[holding$user_location_region %in% names(top_50_U_R), ]
new_df$user_location_region <- as.factor(as.integer(new_df$user_location_region))

newer_df <- new_df[new_df$hotel_country %in% names(top_50_Hotel_country),]
newer_df$hotel_country <- as.factor(as.integer(newer_df$hotel_country))

# Readjust Factors, 
newer_df$user_location_region<- as.factor(as.integer(newer_df$user_location_region))
newer_df$hotel_country<- as.factor(as.integer(newer_df$hotel_country))
 

############  Special Formatting, Checking Factors Count #######################

newer_df$srch_ci <- as.POSIXct(newer_df$srch_ci, format = "%Y-%m-%d")
newer_df$srch_co <- as.POSIXct(newer_df$srch_co, format = "%Y-%m-%d")
newer_df$srch_ci <- month(newer_df$srch_ci)
newer_df$srch_co <- month(newer_df$srch_co)

######################### Check Factor Length #################################

over_factors <- list()

for (name in names(newer_df)){
               if (length(levels(newer_df[[name]])) > 50){
                              over_factors <- append(over_factors, name)
               }
}

# Call Cluster Selection Function (If Needed)
#----------------------------------------------------------------------------

#clusters_random <- select_clusters(newer_df, 'hotel_cluster', 2, 50)
#newest_df <- newer_df
#newest_df$hotel_cluster <- as.factor(as.integer(newest_df$hotel_cluster))

# Check that Levels Have Changed
#print(names(newest_df$hotel_cluster))
#-----------------------------------------------------------------------------

# Final DF Set Up if Cluster Selection Not Needed
#-----------------------------------------------------------------------------
newest_df <- newer_df
newest_df$hotel_cluster <- as.factor(as.integer(newest_df$hotel_cluster))
newest_df <- na.omit(newest_df)
#-----------------------------------------------------------------------------
 

########################## Removing for Memory Issues #########################
#If you are running this on massive amounts of rows (+1,000,000) run this chunk
# This will free up a good amount of memory
remove(holding)
remove(factor_columns)
remove(integer_columns)
remove(drop_columns)
remove(new_df)
remove(newer_df)
remove(over_factors)
remove(cluster_column)
remove(country_column)
remove(region_column)
remove(top_50_cluster)
remove(top_50_Hotel_country)
remove(top_50_U_R)
#remove(train)
 

############################# Random Forest Model #############################
# Create X DF
X <- newest_df[, -19]

# Create Y Vector
Y <- newest_df[, 19]

# ntree
n <- 30

rf_model <- randomForest(X, Y, ntree = n, importance = TRUE, do.trace=TRUE)

# Confusion Matrix
confusion <- (rf_model$confusion)

print(confusion)
# Check Model
print(rf_model)

#print(confusion)
print(rf_model$err.rate)

#Importance of Variables
varImpPlot(rf_model)


############################### KNN Model Prep ################################

# Here we are showing the 2 most important features in the RF model
# Plotted witht the 10 predictions colored, this is to show you the complexity
# Before the KNN Model

ggplot(newest_df, aes(x = user_location_region, y = orig_destination_distance, color = hotel_cluster)) + 
               geom_point()

# Pre-Scale Numeric Columns
numeric_columns <- c("orig_destination_distance", "srch_adults_cnt", "srch_children_cnt", "srch_rm_cnt", "cnt")

#Here we are Scaling the values
newest_df_scaled_values <- scale(newest_df[numeric_columns])

# Here we are reassigning the scaled values to the scaled DF
scaled_df <- newest_df
for (i in numeric_columns) {
               scaled_df[,i] <- newest_df_scaled_values[,i]
}

#Here we can see the Structure of the scaled_DF
str(scaled_df)

#Creating a function that adds the name to the X-Feature of the scaled value 
#SO there are no repeats
factor_columns <- c("site_name", "posa_continent", "user_location_country", "user_location_region", "channel", "srch_destination_type_id", "hotel_continent", "hotel_country","is_mobile")

for (i in factor_columns) {
               dummy_column <- as.data.frame(dummy.code(scaled_df[,i]))
               
               for (j in 1:length(dummy_column)) {
                              colnames(dummy_column)[j] = paste(i,"_", as.character(j))
               }
               
               scaled_df[,i] <- NULL
               scaled_df <- cbind(scaled_df, dummy_column)
}

#Omiting the NA's
scaled_df <- na.omit(scaled_df) 


 

################################## KNN MODEL ##################################
# Spliting the Data for the KNN Model

set.seed(1)

split <- sample.split(scaled_df, SplitRatio = 0.8)
knn_train <- subset(scaled_df, split == "TRUE")
knn_test <- subset(scaled_df, split == "FALSE")

knn_test_Y<-knn_test$hotel_cluster
knn_test$hotel_cluster <- NULL

knn_train_Y<-knn_train$hotel_cluster
knn_train$hotel_cluster <- NULL

# Best Model

best_knn_model <- knn(
               train = knn_train,
               test = knn_test,
               cl = knn_train_Y,
               k = 1
)
# where one is a pre-calculated best K

#taking the summary of our KNN Model
summary(best_knn_model)

#PLotting the histogram of the KNN Model
plot(best_knn_model)

# Accuracy
confusion <- table(knn_test$hotel_cluster, best_knn_model)
print(confusion)

accuracy <- mean(best_knn_model == knn_test_Y)
print(accuracy)

# METHOD 2
set.seed(1)

divideData <- createDataPartition(scaled_df$hotel_cluster, p = 0.8, list = FALSE)

knn_train <- scaled_df[divideData,]
knn_test <- scaled_df[-divideData,]

#trctrl <- trainControl(method = 'boot', number = 10, repeats = 3)
#knnfit <- train(hotel_cluster~., data = knn_train, method = "knn", trControl = trctrl, tuneLength = 10)
knnfit <- train(hotel_cluster~., data = knn_train, method = "knn", tuneLength = 10)
knnfit

plot(knnfit)

# Best K
knnfit$bestTune

# Predict
knnclass <- predict(knnfit, newdata = knn_test)

# Confusion Matrix
confusionMatrix(knnclass, knn_test$hotel_cluster)

# Accuracy
accuracy <- mean(knnclass == knn_test$hotel_cluster)
accuracy

 

################################ SVM Model #####################################
set.seed(1)

divideData <- createDataPartition(newest_df$hotel_cluster, p = 0.8, list = FALSE)

svm_train <- newest_df[divideData,]
svm_test <- newest_df[-divideData,]
svm_test_x <- subset(svm_test, select = -c(hotel_cluster))
svm_test_y <- svm_test$hotel_cluster

# Train SVM model with Radial Kernel

svm_model <- svm(hotel_cluster~., 
                 data = svm_train, 
                 kernel = "radial",  
                 gamma = 1, 
                 cost = 1)

summary(svm_model)

#plot(svm_model, newest_df)

# Tune Model
set.seed(1)

tune.out <- tune(svm, 
                 hotel_cluster~., 
                 data = svm_train, 
                 kernel = "radial", 
                 ranges = list(cost = c(0.1, 1, 10, 100),
                               gamma = c(0.5, 1, 2, 3)))

grid <- expand.grid(C = c(0.1, 1, 10, 100))

summary(tune.out)
names(tune.out)

summary(tune.out$best.model)

# Predict
yhat <- predict(tune.out$best.model,newdata=svm_test)

confusionMatrix(yhat, svm_test$hotel_cluster)

accuracy <- mean(yhat == svm_test$hotel_cluster)
accuracy
