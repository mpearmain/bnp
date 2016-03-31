## wd etc ####
require(readr)
require(glmnet)
require(caret)
require(stringr)

dataset_version <- "c20160330"
seed_value <- 43
model_type <- "glmnet"
todate <- str_replace_all(Sys.Date(), "-","")

## functions ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}


auc<-function (actual, predicted) {
  
  r <- as.numeric(rank(predicted))
  
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
  
}

## data ####
# read actual data
xtrain <- read_csv(paste("../input2/xtrain_",dataset_version,".csv", sep = ""))
xtest <- read_csv(paste("../input2/xtest_",dataset_version,".csv", sep = ""))
y <- xtrain$target; xtrain$target <- NULL
id_train <- xtrain$ID
id_test <- xtest$ID
xtrain$ID <- xtest$ID <- NULL

# division into folds: 5-fold
xfolds <- read_csv("../input/xfolds.csv"); xfolds$fold_index <- xfolds$fold5
xfolds <- xfolds[,c("ID", "fold_index")]
nfolds <- length(unique(xfolds$fold_index))

## fit models ####
# parameter grid
param_grid <- expand.grid(alpha = seq(0,10)/10,
                          stand = c(TRUE, FALSE))

# storage structures 
mtrain <- array(0, c(nrow(xtrain), nrow(param_grid)))
mtest <- array(0, c(nrow(xtest), nrow(param_grid)))

# loop over parameters
for (ii in 1:nrow(param_grid))
{
  
  # loop over folds 
  for (jj in 1:nfolds)
  {
    isTrain <- which(xfolds$fold_index != jj)
    isValid <- which(xfolds$fold_index == jj)
    x0 <- as.matrix(xtrain[isTrain,]); x1 <- as.matrix(xtrain[isValid,])
    y0 <- y[isTrain]; y1 <- y[isValid]
    
    mod0 <- glmnet(x = x0, y = y0, family = "binomial", alpha = param_grid$alpha[ii],
                   standardize = param_grid$stand[ii])
    pred0 <- predict(mod0, x1,type = "response")
    pred0 <- pred0[,ncol(pred0)]
    mtrain[isValid,ii] <- pred0
  }
  
  # full version 
  mod0 <- glmnet(x = as.matrix(xtrain), y = y, family = "binomial", alpha = param_grid$alpha[ii],
         standardize = param_grid$stand[ii])
  pred_full <- predict(mod0, as.matrix(xtest), type = "response")
  pred_full <- pred_full[,ncol(pred_full)]
  mtest[,ii] <- pred_full
  msg(ii)
}

## store complete versions ####
mtrain <- data.frame(mtrain)
mtest <- data.frame(mtest)
colnames(mtrain) <- colnames(mtest) <- paste(model_type, 1:ncol(mtrain), sep = "")
mtrain$ID <- id_train
mtest$ID <- id_test
mtrain$target <- y


# Remove any linear combos.
# trim linearly dependent ones 
print(paste("Pre linear combo trim size ", dim(mtrain)[2]))
flc <- findLinearCombos(mtrain)
if (length(flc$remove))
{
  mtrain <- mtrain[,-flc$remove]
  mtest <- mtest[,-flc$remove]
}

print(paste(" Number of cols after linear combo extraction:", dim(mtrain)[2]))
# store the metas
write_csv(mtrain, path = paste("../metafeatures2/prval_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = "" ))
write_csv(mtest, path = paste("../metafeatures2/prfull_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = "" ))

# store the parameters
write_csv(param_grid, path = paste("../meta_parameters2/params_",model_type,"_", todate, "_data", dataset_version, "_seed", seed_value, ".csv",sep = "" ))

