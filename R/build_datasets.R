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

## building the datasets ####
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
# - add quadratic factors (pairwise combos)
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

# KB1 as basis:
# - add differences of correlated ones
# - cutoff parameter
buildKB2 <- function(cut_level = 0.99)
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
  
  ## analysis of correlations
  xnum <- bigD[,numeric_columns]; bigD <- bigD[,-numeric_columns]
  xcor <- cor(xnum)
  flc <- findCorrelation(xcor, cut_level)
  corr_pairs <- which(xcor > cut_level, arr.ind = T)
  corr_pairs <- corr_pairs[corr_pairs[,1] > corr_pairs[,2],]
  # create new features
  xnum1 <- array(0, c(nrow(xnum), nrow(corr_pairs)))
  for (ii in 1:nrow(corr_pairs))
  {
    xnum1[,ii] <- apply(xnum[,corr_pairs[ii,]],1,diff)
    msg(ii)
  }
  colnames(xnum1) <- paste("diff", 1:ncol(xnum1), sep = "")
  xnum <- xnum[,-flc]; xnum <- data.frame(xnum, xnum1)
  bigD <- data.frame(bigD, xnum)
  
    
  # Map all categoricals into numeric.
  cat("Assuming text variables are categorical & replacing them with numeric ids\n")
  for (f in colnames(bigD)) {
    if (class(bigD[[f]])=="character") {
      print(f)
      levels <- unique(bigD[[f]])
      bigD[[f]] <- as.integer(factor(bigD[[f]], levels=levels))
    }
    print(f)
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
  
  write.csv(xtrain, paste('input/xtrain_kb2',str_replace(cut_level, "[.]",""),'.csv', sep = ""), row.names = F)
  write.csv(xtest, paste('input/xtest_kb2',str_replace(cut_level, "[.]",""),'.csv', sep = ""), row.names = F)
  
  rm(xtrain)
  rm(xtest)
  return(cat("KB2 dataset built"))
}

## actual construction ####
buildMP1()
buildKB1()
buildKB2(cut_level = 0.99)
buildKB2(cut_level = 0.95)

