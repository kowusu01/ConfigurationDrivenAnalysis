########################################################################
##
## this is the standard R script with all
## parameters hard coded
## it has been refactored to pull all configurable items in one place
##
########################################################################

set.seed(2022)

if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

library(dplyr)
library(readr)
library(caret)

# define a function to download a file 
fnDownloadData <- function(remote_base_url, remote_path,  file_name, local_folder) {
  
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


########################################################################
# The first attempt towards configuration files
# is to move all the configuration items to one place
########################################################################

# url to my github location
remote.base_url <- "https://raw.githubusercontent.com/kowusu01"
remote.path <- "/TransformedDatasets/main/adult_income/rda/"

training_data.file_name <- "training_set_final.rda"
test_data.file_name <- "test_set_final.rda"

# local folder where copies are saved
local.folder_name <- "rda"

# sample size for training
sample_size <- 1000

# number of cross validation folds
cross_validation_folds <- 5

# tuning length allows you to tell caret how many values
# it should pick for the hyperparameter, in this case k (knn hyper parameter)
k_hyperparameter.tuneLength <- 10

# 1.1 call the function fnDownloadData() to attempt to download the training data
#     from remote
fnDownloadData(remote_base_url =  remote.base_url, 
               remote_path = remote.path, 
               file_name = training_data.file_name, 
               local_folder = local.folder_name)

# 1.2 call the function fnDownloadData() to download the test data
fnDownloadData(remote_base_url =  remote.base_url, 
               remote_path = remote.path, 
               file_name = test_data.file_name, 
               local_folder = local.folder_name)

# 1.3 load the downloaded training data
train_set <- readRDS(file.path(local.folder_name, training_data.file_name))

# 1.4 load the downloaded test data
test_set <- readRDS(file.path(local.folder_name, test_data.file_name))


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
print (paste0(Sys.time(), " - training with ", cross_validation_folds,
              "-fold cross validation, tuneLength for k hyperparameter: ", 
              k_hyperparameter.tuneLength, ", sample size: ", sample_size) )


# 3.1 train the model using the sample data
print(paste0(Sys.time(), " - begin model training..."))

knnModel <- train(
  over50K ~ .,
  method = "knn", 
  metric="ROC",
  data = sample_data,
  trControl = cv_control,
  tuneLength=k_hyperparameter.tuneLength
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
