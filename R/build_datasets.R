## packages loading ####
library(data.table)
library(caret)
library(stringr)
library(readr)
library(lubridate)
require(lme4)
require(chron)

set.seed(260681)

## functions: helper ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
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

## actual construction ####
buildMP1()
buildKB1()
buildKB2()
buildKB3()
buildKB4()
buildKB5(cut_level = 0.99)
buildKB5(cut_level = 0.95)
# buildKB6(cut_level = 0.99)
# KB6 with cut_level = 0.95 since its HUGE - over 7000 columns
buildKB6(cut_level = 0.95)

