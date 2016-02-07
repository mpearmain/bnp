## packages loading ####
library(data.table)
library(caret)
library(stringr)
library(readr)
library(lubridate)
library(Rtsne)
require(lme4)
require(chron)

set.seed(260681)

## functions ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

replaceNA = function(DT) {
  # by name :
  for (j in names(DT))
    set(DT,which(is.na(DT[[j]])),j,-1)
}

## build the datasets ####
# for convenience, each dataset construction is wrapped in a function

# basically a clone of MP1 from homesite
buildMP1 <- function()
{
  train <- fread('input/train.csv')
  test <- fread('input/test.csv')
  
  # Lets first align the datasets for equal vars to work with.
  y <- train[, target]
  train[, c('target') := NULL]
  train[, dset := 0]
  test[, dset := 1]
  
  # Join the datasets for simple manipulations.
  bigD <- rbind(train, test)
  rm(list = c('train', 'test'))
  
  ## Data Manipulations 
  # Count -1's across the data set
  bigD[, V_NAs := rowSums(is.na(.SD)), .SDcols = grep("v", names(bigD))]
  
  # simple hack for NA's to -1
  replaceNA(bigD)
  
  # Catch factor columns
  fact_cols <- which(lapply(bigD, class) == "character")
  # Map all categoricals into numeric.
  cat("Assuming text variables are categorical & replacing them with numeric ids\n")
  for (f in colnames(bigD)) {
    if (class(bigD[[f]])=="character") {
      print(f)
      levels <- unique(bigD[[f]])
      bigD[[f]] <- as.integer(factor(bigD[[f]], levels=levels))
    }
  }
  
  ## Split files & Export 
  xtrain <- bigD[dset == 0, ]
  xtest <- bigD[dset == 1, ]
  rm(bigD)
  
  xtrain[, dset := NULL]
  xtest[, dset := NULL]
  
  xtrain[, target := y]
  
  write.csv(xtrain, 'input/xtrain_mp1.csv', row.names = F)
  write.csv(xtest, 'input/xtest_mp1.csv', row.names = F)
  
  rm(xtrain)
  rm(xtest)
  return(cat("MP1 dataset built"))
}

# MP1 with some extras:
# -
buildKB1 <- function()
{
  train <- read_csv('input/train.csv')
  test <- read_csv('input/test.csv')
  
  # Lets first align the datasets for equal vars to work with.
  y <- train$target; train$target <- NULL
  train$dset <- 0; test$dset <- 1
  
  # Join the datasets for simple manipulations.
  bigD <- rbind(train, test)
  rm(list = c('train', 'test'))
  ID <- bigD$ID; bigD$ID <- NULL
  dset <- bigD$dset; bigD$dset <- NULL
    
  # column types
  column_types <- sapply(bigD, class)
  numeric_columns <- which(column_types == "numeric")
  character_columns <- which(column_types == "character")
  character_names <- colnames(bigD)[character_columns]
  # Count NAs across the data set
  count_nas <- rowSums(is.na(bigD))
 
  # attach to the dataset
  bigD$count_nas <- count_nas; rm(count_nas)
  
  # replace NA with -1
  bigD[is.na(bigD)] <- -1
  
  # create bivariate combos of factors
  xcomb <- combn(character_columns,2)
  for (ii in 1:ncol(xcomb))
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], sep = "")
    bigD[,xname] <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], sep = "")
      }
    
  # Map all categoricals into numeric.
  cat("Assuming text variables are categorical & replacing them with numeric ids\n")
  for (f in colnames(bigD)) {
    if (class(bigD[[f]])=="character") {
      print(f)
      levels <- unique(bigD[[f]])
      bigD[[f]] <- as.integer(factor(bigD[[f]], levels=levels))
    }
  }
  
  bigD$dset <- dset
  bigD$ID <- ID
  ## Split files & Export 
  xtrain <- bigD[dset == 0, ]
  xtest <- bigD[dset == 1, ]
  rm(bigD)
  
  xtrain$dset <- NULL
  xtest$dset <- NULL
  
  xtrain$target <- y
  
  write.csv(xtrain, 'input/xtrain_kb1.csv', row.names = F)
  write.csv(xtest, 'input/xtest_kb1.csv', row.names = F)
  
  rm(xtrain)
  rm(xtest)
  return(cat("KB1 dataset built"))
}
