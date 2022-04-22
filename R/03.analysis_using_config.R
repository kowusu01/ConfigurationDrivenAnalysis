########################################################################
##
## in this implementation, all  configurable items have been moved to 
## the configuration file (config.yml)
##
########################################################################
set.seed(2022)

if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")
if (!require(config)) install.packages("config")

library(dplyr)
library(readr)
library(caret)
library(config)

# function to download a file 
fnDownloadData <- function(remote_base_url, 
                           remote_path,  
                           file_name, 
                           local_folder) {

  # construct the full path 
  remote_full_path <-  paste0(remote_base_url, remote_path, file_name)
  
  # check if local directory exists
  if(!dir.exists(local_folder))
    dir.create(local_folder)
  
  # construct the path for the local folder where the downloaded will be saved
  local_path <- file.path(local_folder, file_name)
  
  # download the file, only if it does not exist locally
  if(!file.exists(local_path))
    download.file(remote_full_path, local_path,  method="curl", mode = "w")
}


#######################################################################
# 1. load config into memory and assign to an object
# 2. use the object to access config elements
#
# by default, R will load the configuration file
# if located at the home directory and named config.yml
########################################################################

# load config and assign to an object
config <- config::get() 

# 1.1 pass the config elements to a function to download the training data from remote
fnDownloadData(remote_base_url = config$remote.base_url, 
               remote_path = config$remote.path, 
               file_name = config$training_data.file_name, 
               local_folder = config$local_folder)

# 1.2 pass the config elements to a function to download the test data from remote
fnDownloadData(remote_base_url = config$remote.base_url, 
               remote_path = config$remote.path, 
               file_name = config$test_data.file_name, 
               local_folder = config$local_folder)


# 1.3 load the downloaded file
train_set <- readRDS(file.path(config$local_folder, config$training_data.file_name))

# 1.4 load test data
test_set <- readRDS(file.path(config$local_folder, config$test_data.file_name))

# 2.1 set up cross validation training scheme
n <- sample(nrow(train_set), config$sample_size)
sample_data <- train_set[n, ]


# 2.2 select a small sample to train a model
cv_control <- trainControl(method = "cv", 
                           number = config$cross_validation_folds, 
                           summaryFunction=twoClassSummary, 
                           classProbs=TRUE, savePredictions = T)

# print model training stats
print (paste0(Sys.time(), " - training with ", config$cross_validation_folds,
						 "-fold cross validation, tuneLength for k hyperparameter: ", 
						 config$k_hyperparameter.tuneLength, ", sample size: ", config$sample_size) )


# 3.1 train the model using the sample data
print(paste0(Sys.time(), " - begin model training..."))

knnModel <- train(
  over50K ~ .,
  method = "knn", 
  metric="ROC",
  data = sample_data,
  trControl = cv_control,
  tuneLength=config$k_hyperparameter.tuneLength
)
print(paste0(Sys.time(), " - print cross validation results..."))
knnModel
print(paste0(Sys.time(), " - done training model, best k = ", knnModel$bestTune$k)) 

# 3.2 fit a model using the entire training set
print(paste0(Sys.time(), " - fitting a final model on the entire training set...")) 
knn_fit <- knn3(over50K ~ ., data=train_set, k=knnModel$bestTune$k)
print(paste0(Sys.time(), " - done fitting a model.")) 


# 4.0 making predictions 
print(paste0(Sys.time(), " - making predictions using the final model...")) 
knn_predictions <- predict(knn_fit, test_set, type="class")

print(paste0(Sys.time(), " - print first 20 items from the predictions..."))
knn_predictions %>% head(n=20)

print(paste0(Sys.time(), " - print first 20 actual values from the test_set..."))
test_set$over50K %>% head(n=20)
print(paste0(Sys.time(), " - done with predictions.")) 


# show confusion matrix
cm <- confusionMatrix(test_set$over50K, knn_predictions)
print(paste0(Sys.time(), " - print performance details from the confusionMatrix..."))
cm$overall["Accuracy"]
cm$byClass["Sensitivity"]
cm$byClass["Specificity"]
print(paste0(Sys.time(), " - Done."))
