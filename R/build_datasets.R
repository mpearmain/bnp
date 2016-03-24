## packages loading ####
library(data.table)
library(caret)
library(stringr)
library(readr)
library(lubridate)
require(lme4)
require(chron)
require(Metrics)
require(kohonen)
require(h2o)
require(Rtsne)

set.seed(260681)

## functions: helper ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

# wrapper around logloss preventing Inf/-Inf for 1/0 values
log_loss <- function(predicted, actual, cutoff = 1e-15)
{
  predicted <- pmax(predicted, cutoff)
  predicted <- pmin(predicted, 1- cutoff)
  return(logLoss(actual,predicted))
}

replaceNA = function(DT) {
  # by name :
  for (j in names(DT))
    set(DT,which(is.na(DT[[j]])),j,-1)
}

## functions: building the datasets ####
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
  bigD[, VNAs := rowSums(is.na(.SD)), .SDcols = grep("v", names(bigD))]
  
  # simple hack for NA's to -1
  replaceNA(bigD)
  
  # Catch factor columns
  fact_cols <- which(lapply(bigD, class) == "character")
  # Map all categoricals into numeric.
  cat("Assuming text variables are categorical & replacing them with numeric ids\n")
  for (f in colnames(bigD)) {
    if (class(bigD[[f]])=="character") {
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
  msg("MP1 dataset built")
}

# MP1 with extras:
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
  countnas <- rowSums(is.na(bigD))
 
  # attach to the dataset
  bigD$countnas <- countnas; rm(countnas)
  
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
      levels <- unique(bigD[[f]])
      bigD[[f]] <- as.integer(factor(bigD[[f]], levels=levels))
    }
  }
  
  bigD$ID <- ID
  ## Split files & Export 
  xtrain <- bigD[which(dset == 0), ]
  xtest <- bigD[which(dset == 1), ]
  rm(bigD)
  
  xtrain$target <- y
  
  write.csv(xtrain, 'input/xtrain_kb1.csv', row.names = F)
  write.csv(xtest, 'input/xtest_kb1.csv', row.names = F)
  
  rm(xtrain)
  rm(xtest)
  msg("KB1 dataset built")
}

# MP1 with extras:
# - add quadratic factors (pairwise combos)
# - add cubic factors (three-way interactions)
buildKB2 <- function()
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
  countnas <- rowSums(is.na(bigD))
  
  # attach to the dataset
  bigD$countnas <- countnas; rm(countnas)
  
  # replace NA with -1
  bigD[is.na(bigD)] <- -1
  
  # create bivariate combos of factors
  xcomb <- combn(character_columns,2)
  for (ii in 1:ncol(xcomb))
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], sep = "")
    bigD[,xname] <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], sep = "")
  }

  # create trivariate combos of factors
  xcomb <- combn(character_columns,3)
  for (ii in 1:ncol(xcomb))
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], colnames(bigD)[xcomb[3,ii]], sep = "")
    bigD[,xname] <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], bigD[,xcomb[3,ii]], sep = "")
  }
  
  # Map all categoricals into numeric.
  cat("Assuming text variables are categorical & replacing them with numeric ids\n")
  for (f in colnames(bigD)) {
    if (class(bigD[[f]])=="character") {
      levels <- unique(bigD[[f]])
      bigD[[f]] <- as.integer(factor(bigD[[f]], levels=levels))
    }
  }
  
  bigD$ID <- ID
  ## Split files & Export 
  xtrain <- bigD[which(dset == 0), ]
  xtest <- bigD[which(dset == 1), ]
  rm(bigD)
  
  xtrain$target <- y
  
  write.csv(xtrain, 'input/xtrain_kb2.csv', row.names = F)
  write.csv(xtest, 'input/xtest_kb2.csv', row.names = F)
  
  rm(xtrain)
  rm(xtest)
  msg("KB2 dataset built")
}

# KB1 with extras:
# - all factors mapped to response rates
buildKB3 <- function()
{
  set.seed(260681)
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
  countnas <- rowSums(is.na(bigD))
  
  # attach to the dataset
  bigD$countnas <- countnas; rm(countnas)
  
  # replace NA with -1
  bigD[is.na(bigD)] <- -1
  
  # create bivariate combos of factors
  xcomb <- combn(character_columns,2)
  for (ii in 1:ncol(xcomb))
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], sep = "")
    bigD[,xname] <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], sep = "")
  }
  
  bigD$ID <- ID
  ## Split files & Export 
  xtrain <- bigD[which(dset == 0), ]
  xtest <- bigD[which(dset == 1), ]
  rm(bigD)
  
  # replace categorical ones with response rates
  xfold <- read_csv(file = "./input/xfolds.csv")
  idFix <- list()
  for (ii in 1:5)
  {
    idFix[[ii]] <- which(xfold$fold5 == ii)
  }
  rm(xfold,ii)  
  
  col_types <- sapply(xtrain, class)
  factor_vars <- colnames(xtrain)[which(col_types == "character")]
  for (varname in factor_vars)
  {
    # placeholder for the new variable values
    x <- rep(NA, nrow(xtrain))
    for (ii in seq(idFix))
    {
      # separate ~ fold
      idx <- idFix[[ii]]
      # x0 <- xtrain[-idx, factor_vars]; x1 <- xtrain[idx, factor_vars]
      x0 <- xtrain[-idx, varname, drop = F]; x1 <- xtrain[idx, varname, drop = F]
      y0 <- y[-idx]; y1 <- y[idx]
      # take care of factor lvl mismatches
      x0[,varname] <- factor(as.character(x0[,varname]))
      # fit LMM model
      myForm <- as.formula (paste ("y0 ~ (1|", varname, ")"))
      myLME <- lmer (myForm, x0, REML=FALSE, verbose=F)
      myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
      # table to match to the original
      myLMERDF <- data.frame (levelName = as.character(levels(x0[,varname])), myDampVal = myRanEf+myFixEf)
      rownames(myLMERDF) <- NULL
      x[idx] <- myLMERDF[,2][match(xtrain[idx, varname], myLMERDF[,1])]
      x[idx][is.na(x[idx])] <- mean(y0)
    }
    rm(x0,x1,y0,y1, myLME, myLMERDF, myFixEf, myRanEf)
    # add the new variable
    xtrain[,paste(varname, "dmp", sep = "")] <- x
    
    # create the same on test set
    xtrain[,varname] <- factor(as.character(xtrain[,varname]))
    x <- rep(NA, nrow(xtest))
    # fit LMM model
    myForm <- as.formula (paste ("y ~ (1|", varname, ")"))
    myLME <- lmer (myForm, xtrain[,factor_vars], REML=FALSE, verbose=F)
    myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
    # table to match to the original
    myLMERDF <- data.frame (levelName = as.character(levels(xtrain[,varname])), myDampVal = myRanEf+myFixEf)
    rownames(myLMERDF) <- NULL
    x <- myLMERDF[,2][match(xtest[, varname], myLMERDF[,1])]
    x[is.na(x)] <- mean(y)
    xtest[,paste(varname, "dmp", sep = "")] <- x
    # msg(varname)
  }
  
  # drop the factors
  ix <- which(colnames(xtrain) %in% factor_vars)
  xtrain <- xtrain[,-ix]
  ix <- which(colnames(xtest) %in% factor_vars)
  xtest <- xtest[,-ix]
  
  xtrain$target <- y
  
  write.csv(xtrain, 'input/xtrain_kb3.csv', row.names = F)
  write.csv(xtest, 'input/xtest_kb3.csv', row.names = F)
  
  rm(xtrain)
  rm(xtest)
  return(cat("KB3 dataset built"))
  
}

# KB2 with extras:
# - bi- and tri- element factor combos
# - all factors mapped to response rates
buildKB4 <- function()
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
  countnas <- rowSums(is.na(bigD))
  
  # attach to the dataset
  bigD$countnas <- countnas; rm(countnas)
  
  # replace NA with -1
  bigD[is.na(bigD)] <- -1
  
  # create bivariate combos of factors
  xcomb <- combn(character_columns,2)
  for (ii in 1:ncol(xcomb))
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], sep = "")
    bigD[,xname] <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], sep = "")
  }
  
  # create trivariate combos of factors
  xcomb <- combn(character_columns,3)
  for (ii in 1:ncol(xcomb))
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], colnames(bigD)[xcomb[3,ii]], sep = "")
    bigD[,xname] <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], bigD[,xcomb[3,ii]], sep = "")
  }
  
  
  bigD$ID <- ID
  xtrain <- bigD[which(dset == 0), ]
  xtest <- bigD[which(dset == 1), ]
  rm(bigD)
  
  # replace categorical ones with response rates
  xfold <- read_csv(file = "./input/xfolds.csv")
  idFix <- list()
  for (ii in 1:5)
  {
    idFix[[ii]] <- which(xfold$fold5 == ii)
  }
  rm(xfold,ii)  
  
  col_types <- sapply(xtrain, class)
  factor_vars <- colnames(xtrain)[which(col_types == "character")]
  for (varname in factor_vars)
  {
    # placeholder for the new variable values
    x <- rep(NA, nrow(xtrain))
    for (ii in seq(idFix))
    {
      # separate ~ fold
      idx <- idFix[[ii]]
      # x0 <- xtrain[-idx, factor_vars]; x1 <- xtrain[idx, factor_vars]
      x0 <- xtrain[-idx, varname, drop = F]; x1 <- xtrain[idx, varname, drop = F]
      y0 <- y[-idx]; y1 <- y[idx]
      # take care of factor lvl mismatches
      x0[,varname] <- factor(as.character(x0[,varname]))
      # fit LMM model
      myForm <- as.formula (paste ("y0 ~ (1|", varname, ")"))
      myLME <- lmer (myForm, x0, REML=FALSE, verbose=F)
      myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
      # table to match to the original
      myLMERDF <- data.frame (levelName = as.character(levels(x0[,varname])), 
                              myDampVal = myRanEf+myFixEf)
      rownames(myLMERDF) <- NULL
      x[idx] <- myLMERDF[,2][match(xtrain[idx, varname], myLMERDF[,1])]
      x[idx][is.na(x[idx])] <- mean(y0)
    }
    rm(x0,x1,y0,y1, myLME, myLMERDF, myFixEf, myRanEf)
    # add the new variable
    xtrain[,paste(varname, "dmp", sep = "")] <- x
    
    # create the same on test set
    xtrain[,varname] <- factor(as.character(xtrain[,varname]))
    x <- rep(NA, nrow(xtest))
    # fit LMM model
    myForm <- as.formula (paste ("y ~ (1|", varname, ")"))
    myLME <- lmer (myForm, xtrain[,factor_vars], REML=FALSE, verbose=F)
    myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
    # table to match to the original
    myLMERDF <- data.frame (levelName = as.character(levels(xtrain[,varname])), 
                            myDampVal = myRanEf+myFixEf)
    rownames(myLMERDF) <- NULL
    x <- myLMERDF[,2][match(xtest[, varname], myLMERDF[,1])]
    x[is.na(x)] <- mean(y)
    xtest[,paste(varname, "dmp", sep = "")] <- x
    # msg(varname)
  }
  
  # drop the factors
  ix <- which(colnames(xtrain) %in% factor_vars)
  xtrain <- xtrain[,-ix]
  ix <- which(colnames(xtest) %in% factor_vars)
  xtest <- xtest[,-ix]
  
  xtrain$target <- y
  write.csv(xtrain, 'input/xtrain_kb4.csv', row.names = F)
  write.csv(xtest, 'input/xtest_kb4.csv', row.names = F)
  
  rm(xtrain)
  rm(xtest)
  return(cat("KB4 dataset built"))
  
}

# KB3 with extras:
# - add pairwise differences of correlated numerical features
# - cutoff parameter for correlated pairs as argument
buildKB5 <- function(cut_level = 0.99)
{
  xtrain <- read_csv('input/xtrain_kb3.csv')
  xtest <- read_csv('input/xtest_kb3.csv')

  # Lets first align the datasets for equal vars to work with.
  y <- xtrain$target; xtrain$target <- NULL
  xtrain$dset <- 0; xtest$dset <- 1
  
  # Join the datasets for simple manipulations.
  bigD <- rbind(xtrain, xtest)
  rm(list = c('xtrain', 'xtest'))
  ID <- bigD$ID; bigD$ID <- NULL
  dset <- bigD$dset; bigD$dset <- NULL
  
  # column types
  column_types <- sapply(bigD, class)
  numeric_columns <- which(column_types == "numeric")
  character_columns <- which(column_types == "character")
  character_names <- colnames(bigD)[character_columns]
  
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
    # msg(ii)
  }
  colnames(xnum1) <- paste("diff", 1:ncol(xnum1), sep = "")
  xnum <- xnum[,-flc]; xnum <- data.frame(xnum, xnum1)
  bigD <- data.frame(bigD, xnum)
  
  
  bigD$ID <- ID
  ## Split files & Export 
  xtrain <- bigD[which(dset == 0), ]
  xtest <- bigD[which(dset == 1), ]
  rm(bigD)
  
  xtrain$target <- y
  
  write.csv(xtrain, paste('input/xtrain_kb5',str_replace(cut_level, "[.]",""),'.csv', sep = ""), row.names = F)
  write.csv(xtest, paste('input/xtest_kb5',str_replace(cut_level, "[.]",""),'.csv', sep = ""), row.names = F)
  
  rm(xtrain)
  rm(xtest)
  return(cat("KB5 dataset built"))
}

# KB4 with extras:
# - add pairwise differences of correlated numerical features
# - cutoff parameter for correlated pairs as argument
buildKB6 <- function(cut_level = 0.99)
{
  xtrain <- read_csv('input/xtrain_kb4.csv')
  xtest <- read_csv('input/xtest_kb4.csv')
  
  # Lets first align the datasets for equal vars to work with.
  y <- xtrain$target; xtrain$target <- NULL
  xtrain$dset <- 0; xtest$dset <- 1
  
  # Join the datasets for simple manipulations.
  bigD <- rbind(xtrain, xtest)
  rm(list = c('xtrain', 'xtest'))
  ID <- bigD$ID; bigD$ID <- NULL
  dset <- bigD$dset; bigD$dset <- NULL
  
  # column types
  column_types <- sapply(bigD, class)
  numeric_columns <- which(column_types != "character")
  character_columns <- which(column_types == "character")
  character_names <- colnames(bigD)[character_columns]
  
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
  
  
  bigD$ID <- ID
  ## Split files & Export 
  xtrain <- bigD[which(dset == 0), ]
  xtest <- bigD[which(dset == 1), ]
  rm(bigD)
  
  xtrain$target <- y
  
  write.csv(xtrain, paste('input/xtrain_kb6',str_replace(cut_level, "[.]",""),'.csv', sep = ""), row.names = F)
  write.csv(xtest, paste('input/xtest_kb6',str_replace(cut_level, "[.]",""),'.csv', sep = ""), row.names = F)
  
  rm(xtrain)
  rm(xtest)
  return(cat("KB6 dataset built"))
}

# kmeans on pure numeric datasets
buildKB7 <- function(ref_data = 'kb4', nof_clusters = 50)
{
  xtrain <- read_csv(paste('input/xtrain_',ref_data,'.csv', sep = ""))
  xtest <- read_csv(paste('input/xtest_',ref_data,'.csv', sep = ""))
  
  y <- xtrain$target; xtrain$target <- NULL
  id_train <- xtrain$ID; id_test <- xtest$ID
  xtrain$ID <- xtest$ID <- NULL
  
  # standardize and build kmeans
  prep0 <- preProcess(x = xtrain, method = c("range"))
  xtrain <- predict(prep0, xtrain)
  xtest <- predict(prep0, xtest)
  
  # map to distances from kmeans clusters
  km0 <- kmeans(xtrain, centers = nof_clusters)
  dist1 <- array(0, c(nrow(xtrain), nof_clusters))
  for (ii in 1:nof_clusters)
  {
    dist1[,ii] <- apply(xtrain,1,function(s) sd(s - km0$centers[ii,]))
    msg(ii)
  }
  dist2 <- array(0, c(nrow(xtest), nof_clusters))
  for (ii in 1:nof_clusters)
  {
    dist2[,ii] <- apply(xtest,1,function(s) sd(s - km0$centers[ii,]))
    msg(ii)
  }
  
  # storage
  dist1 <- data.frame(dist1)
  dist2 <- data.frame(dist2)
  dist1$target <- y
  dist1$ID <- id_train
  dist2$ID <- id_test
  
  xtrain$target <- y
  
  write.csv(dist1, paste('input/xtrain_kb7c',nof_clusters,'d',ref_data,'.csv', sep = ""), row.names = F)
  write.csv(dist2, paste('input/xtest_kb7c',nof_clusters,'d',ref_data,'.csv', sep = ""), row.names = F)
  
  return(cat("KB7 dataset built"))
}

# PCA version
buildKB8 <- function(ref_data = 'kb4', cut_level = 0.999)
{
  xtrain <- read_csv(paste('../input/xtrain_',ref_data,'.csv', sep = ""))
  xtest <- read_csv(paste('../input/xtest_',ref_data,'.csv', sep = ""))
  
  y <- xtrain$target; xtrain$target <- NULL
  id_train <- xtrain$ID; id_test <- xtest$ID
  xtrain$ID <- xtest$ID <- NULL

  # drop linear dependencies
  xcor <- cor(xtrain)
  flc <- findLinearCombos(xcor)
  xtrain <- xtrain[,-flc$remove]
  xtest <- xtest[,-flc$remove]
  
  # transform to PCA-representation
  prep0 <- preProcess(x = xtrain, method = c("pca"), thresh = cut_level)
  xtrain <- predict(prep0, xtrain)
  xtest <- predict(prep0, xtest)
  
  xtrain$ID <- id_train
  xtest$ID <- id_test
  xtrain$target <- y
  
  write.csv(xtrain, paste('../input/xtrain_',ref_data,'c',str_replace(cut_level, "[.]",""),   '.csv', sep = ""), row.names = F)
  write.csv(xtest, paste('../input/xtest_',ref_data,'c',str_replace(cut_level, "[.]",""),   '.csv', sep = ""), row.names = F)
  
  return(cat("KB8 dataset built"))
}

# varia: tsne
buildKB9 <- function(ref_data = 'kb1')
{
  # prep data
  xtrain <- read_csv(paste('../input/xtrain_',ref_data,'.csv', sep = ""))
  xtest <- read_csv(paste('../input/xtest_',ref_data,'.csv', sep = ""))
  isTrain <- 1:nrow(xtrain)
  
  y <- xtrain$target; xtrain$target <- NULL
  id_train <- xtrain$ID; id_test <- xtest$ID
  xtrain$ID <- xtest$ID <- NULL
  
  # tsne
  xdat <- rbind(xtrain, xtest)
  tsne <- Rtsne(as.matrix(xdat), check_duplicates = FALSE, pca = FALSE, 
                perplexity=30, theta=0.5, dims=2, verbose = T)
  
  xtrain <- data.frame(tsne$Y[isTrain,])
  xtest <- data.frame(tsne$Y[-isTrain,])
  xtrain$target <- y; xtrain$ID <- id_train
  xtest$ID <- id_test
  
  write.csv(xtrain, paste('../input/xtrain_',ref_data,'tsne.csv', sep = ""), row.names = F)
  write.csv(xtest, paste('../input/xtest_',ref_data,'tsne.csv', sep = ""), row.names = F)
  
  return(paste(ref_data, " + tsne: built"))
  

}

# reduced version of lvl2 features
buildKB10 <- function(ref_data = 'lvl220160317')
{
  train <- read_csv(paste('../input/xtrain_',ref_data,'.csv', sep = "" ))
  test <- read_csv(paste('../input/xtest_',ref_data,'.csv', sep = "" ))
  
  # Lets first align the datasets for equal vars to work with.
  y <- train$target; train$target <- NULL
  id_train <- train$ID; train$ID <- NULL
  id_test <- test$ID; test$ID <- NULL
  
  ix <- grep("xgb|mars", colnames(train))
  train <- train[,ix]
  test <- test[,ix]
  
  train$ID <- id_train; test$ID <- id_test
  train$target <- y
  
  write.csv(train, paste('../input/xtrain_',ref_data,'red.csv', sep = ""), row.names = F)
  write.csv(test, paste('../input/xtest_',ref_data,'red.csv', sep = ""), row.names = F)
  
  return("finished")
}

# combo version of lvl2 features - differences of heavily correlated ones
buildKB11 <- function(ref_data = 'lvl220160317', cut_level = 0.99)
{
  train <- read_csv(paste('../input/xtrain_',ref_data,'.csv', sep = "" ))
  test <- read_csv(paste('../input/xtest_',ref_data,'.csv', sep = "" ))
  
  # Lets first align the datasets for equal vars to work with.
  y <- train$target; train$target <- NULL
  id_train <- train$ID; train$ID <- NULL
  id_test <- test$ID; test$ID <- NULL
  
  xdat <- rbind(train, test); isTrain <- 1:nrow(train)
  # correlated pairs
  ## analysis of correlations
  xcor <- cor(xdat)
  flc <- findCorrelation(xcor, cut_level)
  corr_pairs <- which(xcor > cut_level, arr.ind = T)
  corr_pairs <- corr_pairs[corr_pairs[,1] > corr_pairs[,2],]
  # create new features
  xnum1 <- array(0, c(nrow(xdat), nrow(corr_pairs)))
  for (ii in 1:nrow(corr_pairs))
  {
    xnum1[,ii] <- apply(xdat[,corr_pairs[ii,]],1,diff)
    print(colnames(xdat)[corr_pairs[ii,]])
  }
  colnames(xnum1) <- paste("diff", 1:ncol(xnum1), sep = "")
  xdat <- xdat[,-flc]; xdat <- data.frame(xdat, xnum1)
  xtrain <- xdat[isTrain,]
  xtest <- xdat[-isTrain,]
  
  
  xtrain$ID <- id_train; xtest$ID <- id_test
  xtrain$target <- y
  
  write.csv(xtrain, paste('../input/xtrain_',ref_data,'diff.csv', sep = ""), row.names = F)
  write.csv(xtest, paste('../input/xtest_',ref_data,'diff.csv', sep = ""), row.names = F)
  
  return("finished")
}

# aggregate multiple datasets
buildKB12 <- function(data_list = c('kb1', 'kb1tsne', 'kb2tsne'))
{
  dset <- data_list[1]
  train <- read_csv(paste('../input/xtrain_',dset,'.csv', sep = "" ))
  test <- read_csv(paste('../input/xtest_',dset,'.csv', sep = "" ))
  
  # correction of column names in case there are duplicates
  y <- train$target; train$target <- NULL;  id_train <- train$ID; train$ID <- NULL
  id_test <- test$ID; test$ID <- NULL
  
  
  for (ii in 2:length(data_list))
  {
    xtr <- read_csv(paste('../input/xtrain_',data_list[ii],'.csv', sep = "" ))
    xte <- read_csv(paste('../input/xtest_',data_list[ii],'.csv', sep = "" ))
    
    train <- merge(x = train, y = xtr, by = c("ID", "target"))
    test <- merge(x = test, y = xte, by = c("ID"))
  }

  write.csv(xtrain, paste('../input/xtrain_',ref_data,'diff.csv', sep = ""), row.names = F)
  write.csv(xtest, paste('../input/xtest_',ref_data,'diff.csv', sep = ""), row.names = F)
  
  
}

# check 2,3 and 4-way combinations
buildKB17 <- function(cutoff = 25, cutoff2 = 50)
{
  train <- read_csv('../input/train.csv')
  test <- read_csv('../input/test.csv')
  
  # separation
  y <- train$target; train$target <- NULL
  train$dset <- 0; test$dset <- 1
  
  # Join the datasets for simple manipulations.
  bigD <- rbind(train, test); isTrain <- 1:nrow(train)
  rm(list = c('train', 'test'))
  ID <- bigD$ID; bigD$ID <- NULL
  dset <- bigD$dset; bigD$dset <- NULL
  
  # column types
  column_types <- sapply(bigD, class)
  numeric_columns <- which(column_types == "numeric")
  character_columns <- which(column_types == "character")
  character_names <- colnames(bigD)[character_columns]
  # Count NAs across the data set
  countnas <- rowSums(is.na(bigD))
  
  # attach to the dataset
  bigD$countnas <- countnas; rm(countnas)
  
  # replace NA with -1
  bigD[is.na(bigD)] <- -1
  
  xfolds <- read_csv("../input/xfolds.csv")
  xfolds$fold_index <- xfolds$fold5
  xfolds <- xfolds[,c("ID", "fold_index")]
  nfolds <- length(unique(xfolds$fold_index))
  idFix <- list()
  for (ii in 1:5)
  {
    idFix[[ii]] <- which(xfolds$fold_index == ii)
  }
  rm(xfolds,ii)  
  
  
  ## create bivariate combos of factors and keep the relevant ones
  xcomb <- combn(character_columns,2)
  # storage matrix for summary stats
  relev_values <- rep(0, ncol(xcomb))
  for (ii in 1:ncol(xcomb))
  {
    # construct candidate feature
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], sep = "")
    xfeat <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], sep = "")
    # lump infrequent values (= anything below top "cutoff" ones)
    xtab <- table(xfeat)
    freqnames <- names(tail(sort(xtab), cutoff ))
    xfeat[!(xfeat %in% freqnames)] <- "rare"
    xfeat <- factor(xfeat)
    # cross validated variation in response rates
    cv_vec <- rep(0, nfolds)
    for (jj in 1:nfolds)
    {
      idx <- idFix[[jj]]
      y0 <- y[isTrain][idx]; feat0 <- xfeat[isTrain][idx]
      mx <- aggregate(y0, by = list(feat0), mean)
      cv_vec[jj] <- sd(mx[,2])
    }
    relev_values[ii] <- mean(cv_vec)
    msg(ii)
  }
  # pick the relevant ones and add them to the matrix
  big_ones <- which(rank(-relev_values) < cutoff2)
  for (ii in big_ones)
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], sep = "")
    xfeat <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], sep = "")
    bigD[,xname] <- xfeat
  }

  # create trivariate combos of factors
  xcomb <- combn(character_columns,3)
  for (ii in 1:ncol(xcomb))
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], colnames(bigD)[xcomb[3,ii]], sep = "")
    xfeat <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], bigD[,xcomb[3,ii]],sep = "")

    # lump infrequent values (= anything below top "cutoff" ones)
    xtab <- table(xfeat)
    freqnames <- names(tail(sort(xtab), cutoff ))
    xfeat[!(xfeat %in% freqnames)] <- "rare"
    xfeat <- factor(xfeat)
    # cross validated variation in response rates
    cv_vec <- rep(0, nfolds)
    for (jj in 1:nfolds)
    {
      idx <- which(xfolds$fold_index  == jj)
      y0 <- y[isTrain][idx]; feat0 <- xfeat[isTrain][idx]
      mx <- aggregate(y0, by = list(feat0), mean)
      cv_vec[jj] <- sd(mx[,2])
    }
    relev_values[ii] <- mean(cv_vec)
    msg(ii)
    
  }
  # pick the relevant ones and add them to the matrix
  big_ones <- which(rank(-relev_values) < cutoff2)
  for (ii in big_ones)
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], colnames(bigD)[xcomb[3,ii]], sep = "")
    xfeat <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], bigD[,xcomb[3,ii]],sep = "")
    bigD[,xname] <- xfeat
  }
  
  # create 4-element combos of factors
  xcomb <- combn(character_columns,4)
  for (ii in 1:ncol(xcomb))
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], 
                   colnames(bigD)[xcomb[3,ii]],colnames(bigD)[xcomb[3,ii]], sep = "")
    xfeat <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], 
                   bigD[,xcomb[3,ii]], bigD[,xcomb[4,ii]] ,sep = "")
    # lump infrequent values (= anything below top "cutoff" ones)
    xtab <- table(xfeat)
    freqnames <- names(tail(sort(xtab), cutoff ))
    xfeat[!(xfeat %in% freqnames)] <- "rare"
    xfeat <- factor(xfeat)
    # cross validated variation in response rates
    cv_vec <- rep(0, nfolds)
    for (jj in 1:nfolds)
    {
      idx <- which(xfolds$fold_index  == jj)
      y0 <- y[isTrain][idx]; feat0 <- xfeat[isTrain][idx]
      mx <- aggregate(y0, by = list(feat0), mean)
      cv_vec[jj] <- sd(mx[,2])
    }
    relev_values[ii] <- mean(cv_vec)
    msg(ii)
  }
  # pick the relevant ones and add them to the matrix
  big_ones <- which(rank(-relev_values) < cutoff2)
  for (ii in big_ones)
  {
    xname <- paste(colnames(bigD)[xcomb[1,ii]],colnames(bigD)[xcomb[2,ii]], 
                   colnames(bigD)[xcomb[3,ii]],colnames(bigD)[xcomb[3,ii]], sep = "")
    xfeat <- paste(bigD[,xcomb[1,ii]], bigD[,xcomb[2,ii]], 
                   bigD[,xcomb[3,ii]], bigD[,xcomb[4,ii]] ,sep = "")
    bigD[,xname] <- xfeat
  }
  
  # separate into train and test 
  bigD$ID <- ID
  ## Split files & Export 
  xtrain <- bigD[which(dset == 0), ]
  xtest <- bigD[which(dset == 1), ]
  rm(bigD)
  
  
  # map categoricals to response rates
  col_types <- sapply(xtrain, class)
  factor_vars <- colnames(xtrain)[which(col_types == "character")]
  for (varname in factor_vars)
  {
    # placeholder for the new variable values
    x <- rep(NA, nrow(xtrain))
    for (ii in seq(idFix))
    {
      # separate ~ fold
      idx <- idFix[[ii]]
      x0 <- xtrain[-idx, varname, drop = F]; x1 <- xtrain[idx, varname, drop = F]
      y0 <- y[-idx]; y1 <- y[idx]
      # take care of factor lvl mismatches
      x0[,varname] <- factor(as.character(x0[,varname]))
      # fit LMM model
      myForm <- as.formula (paste ("y0 ~ (1|", varname, ")"))
      myLME <- lmer (myForm, x0, REML=FALSE, verbose=F)
      myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
      # table to match to the original
      myLMERDF <- data.frame (levelName = as.character(levels(x0[,varname])), 
                              myDampVal = myRanEf+myFixEf)
      rownames(myLMERDF) <- NULL
      x[idx] <- myLMERDF[,2][match(xtrain[idx, varname], myLMERDF[,1])]
      x[idx][is.na(x[idx])] <- mean(y0)
    }
    rm(x0,x1,y0,y1, myLME, myLMERDF, myFixEf, myRanEf)
    # add the new variable
    xtrain[,paste(varname, "dmp", sep = "")] <- x
    
    # create the same on test set
    xtrain[,varname] <- factor(as.character(xtrain[,varname]))
    x <- rep(NA, nrow(xtest))
    # fit LMM model
    myForm <- as.formula (paste ("y ~ (1|", varname, ")"))
    myLME <- lmer (myForm, xtrain[,factor_vars], REML=FALSE, verbose=F)
    myFixEf <- fixef (myLME); myRanEf <- unlist (ranef (myLME))
    # table to match to the original
    myLMERDF <- data.frame (levelName = as.character(levels(xtrain[,varname])), 
                            myDampVal = myRanEf+myFixEf)
    rownames(myLMERDF) <- NULL
    x <- myLMERDF[,2][match(xtest[, varname], myLMERDF[,1])]
    x[is.na(x)] <- mean(y)
    xtest[,paste(varname, "dmp", sep = "")] <- x
    # msg(varname)
  }
  
  # drop the factors
  ix <- which(colnames(xtrain) %in% factor_vars)
  xtrain <- xtrain[,-ix]
  ix <- which(colnames(xtest) %in% factor_vars)
  xtest <- xtest[,-ix]
  
  xtrain$target <- y
  
  write.csv(xtrain, paste('../input/xtrain_kb17c',cutoff, 'c', cutoff2, '.csv', sep = ""), row.names = F)
  write.csv(xtest, paste('../input/xtest_kb17c',cutoff, 'c', cutoff2, '.csv', sep = ""), row.names = F)
  
  msg("KB17 dataset built")
}



# TODO
# nbayes -> Python -> 2- and 3-way interactions, along with feature selection
# tsne
# proper bagging

## actual construction ####
# buildMP1()
# buildKB1()
# buildKB2()
# buildKB3()
# buildKB4()
# buildKB5(cut_level = 0.99)
# buildKB5(cut_level = 0.95)
# buildKB6(cut_level = 0.99)
# drop KB6 with cut_level = 0.95 since its HUGE - over 7000 columns
# buildKB6(cut_level = 0.95)
# buildKB7(ref_data = 'kb4', nof_clusters = 50)
# buildKB7(ref_data = 'kb4', nof_clusters = 250)
# buildKB7(ref_data = 'kb6099', nof_clusters = 50)
# buildKB7(ref_data = 'kb6099', nof_clusters = 250)
# buildKB8(ref_data = 'kb3', cut_level = 0.999)
# buildKB8(ref_data = 'kb4', cut_level = 0.999)
buildKB9(ref_data = 'kb2')
buildKB9(ref_data = 'kb3')
buildKB9(ref_data = 'kb4')
buildKB17(cutoff = 50, cutoff2 = 100)


