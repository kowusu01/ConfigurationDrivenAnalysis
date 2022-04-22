########################################################################
##
## this is the standard R script with all
## parameters hard coded
##
########################################################################

#set seed to allow reproducibility
set.seed(2022)

# load required libraries
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(caret)) install.packages("caret")

library(dplyr)
library(readr)
library(caret)

# define a function to download a file, we will reuse this later 
fnDownloadData <- function(remote_base_url, 
                           remote_path, file_name, 
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


# 1.1 call the function fnDownloadData() to attempt to download the training data
#     from remote
fnDownloadData(remote_base_url = "https://raw.githubusercontent.com/kowusu01", 
               remote_path = "/TransformedDatasets/main/adult_income/rda/", 
               file_name = "training_set_final.rda", 
               local_folder = "rda")

# 1.2 call the function fnDownloadData() to download the test data
fnDownloadData(remote_base_url = "https://raw.githubusercontent.com/kowusu01", 
               remote_path = "/TransformedDatasets/main/adult_income/rda/", 
               file_name = "test_set_final.rda", 
               local_folder = "rda")

# 1.3 load the downloaded file
train_set <- readRDS(file.path("rda", "training_set_final.rda"))

# 1.4 load test data
test_set <- readRDS(file.path("rda", "test_set_final.rda"))


# 2.1 select a small sample of 1000 rows to train a model
n <- sample(nrow(train_set), 1000)
sample_data <- train_set[n, ]

# 2.2 set up cross validation training scheme
cv_control <- trainControl(method = "cv", 
                           number = 5, 
                           summaryFunction=twoClassSummary, 
                           classProbs=TRUE, 
                           savePredictions = T)



# print model training stats
print (paste0(Sys.time(), " - training with 5-fold cross validation, ",
						  "tuneLength for k hyperparameter: 10, sample size: 1000.") )

print(paste0(Sys.time(), " - begin model training..."))

# 3.1 train the model using the sample data
knnModel <- train(
  over50K ~ .,
  method = "knn", 
  metric="ROC",
  data = sample_data,
  trControl = cv_control,
  tuneLength=10
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
