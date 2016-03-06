## wd etc ####
require(readr)
require(stringr)
require(glmnet)
require(caret)
require(xgboost)
require(nnet)
require(ranger)
require(Metrics)

seed_value <- 450
todate <- str_replace_all(Sys.Date(), "-","")
nbag <- 5

## functions ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}
# build an ensemble, input = parameters(initSize,howMany,blendIt, blendProp),
# input x, input y (x0 / y0 in c-v)
# output = list(weight)
buildEnsemble <- function(parVec, xset, yvec)
{
  set.seed(20130912)
  # ensemble settings
  initSize <- parVec[1]; howMany <- parVec[2];
  blendIt <- parVec[3]; blendProp <- parVec[4]
  
  # storage matrix for blending coefficients
  arMat <- array(0, c(blendIt, ncol(xset)))
  colnames(arMat) <- colnames(xset)
  
  # loop over blending iterations
  dataPart <- createDataPartition(1:ncol(arMat), times = blendIt, p  = blendProp)
  for (bb in 1:blendIt)
  {
    idx <- dataPart[[bb]];    xx <- xset[,idx]
    
    # track individual scores
    trackScore <- apply(xx, 2, function(x) logLoss(yvec,x))
    
    # select the individual best performer - store the performance
    # and create the first column -> this way we have a non-empty ensemble
    bestOne <- which.max(trackScore)
    mastaz <- (rank(-trackScore) <= initSize)
    best.track <- trackScore[mastaz];
    hillNames <- names(best.track)
    hill.df <- xx[,mastaz, drop = FALSE]
    
    # loop over adding consecutive predictors to the ensemble
    for(ee in 1 : howMany)
    {
      # add a second component
      trackScoreHill <- apply(xx, 2,
                              function(x) logLoss(yvec,rowMeans(cbind(x , hill.df))))
      
      best <- which.max(trackScoreHill)
      best.track <- c(best.track, max(trackScoreHill))
      hillNames <- c(hillNames,names(best))
      hill.df <- data.frame(hill.df, xx[,best])
      msg(ee)
    }
    
    ww <- summary(factor(hillNames))
    arMat[bb, names(ww)] <- ww
    msg(paste("blend: ",bb, sep = ""))
  }
  
  wgt <- colSums(arMat)/sum(arMat)
  
  return(wgt)
}

# wrapper around logloss preventing Inf/-Inf for 1/0 values
log_loss <- function(actual, predicted, cutoff = 1e-15)
{
  predicted <- pmax(predicted, cutoff)
  predicted <- pmin(predicted, 1- cutoff)
  return(logLoss(actual,predicted))
}

## data ####
# list the groups 
xlist_val <- dir("./metafeatures/", pattern =  "prval", full.names = T)
xlist_full <- dir("./metafeatures/", pattern = "prfull", full.names = T)

# aggregate validation set
ii <- 1
mod_class <- str_split(xlist_val[[ii]], "_")[[1]][[2]]
xvalid <- read_csv(xlist_val[[ii]])
mod_cols <- grep(mod_class, colnames(xvalid))
colnames(xvalid)[mod_cols] <- paste(mod_class, ii, 1:length(mod_cols), sep = "")

for (ii in 2:length(xlist_val))
{
  mod_class <- str_split(xlist_val[[ii]], "_")[[1]][[2]]
  xval <- read_csv(xlist_val[[ii]])
  mod_cols <- grep(mod_class, colnames(xval))
  colnames(xval)[mod_cols] <- paste(mod_class, ii, 1:length(mod_cols), sep = "")
  xvalid <- merge(xvalid, xval)
  msg(ii)
  print(dim(xvalid))
}

y <- xvalid$target; xvalid$target <- NULL
id_valid <- xvalid$ID; xvalid$ID <- NULL

# aggregate test set
ii <- 1
mod_class <- str_split(xlist_full[[ii]], "_")[[1]][[2]]
xfull <- read_csv(xlist_full[[ii]])
mod_cols <- grep(mod_class, colnames(xfull))
colnames(xfull)[mod_cols] <- paste(mod_class, ii, 1:length(mod_cols), sep = "")

for (ii in 2:length(xlist_full))
{
  mod_class <- str_split(xlist_full[[ii]], "_")[[1]][[2]]
  xval <- read_csv(xlist_full[[ii]])
  mod_cols <- grep(mod_class, colnames(xval))
  colnames(xval)[mod_cols] <- paste(mod_class, ii, 1:length(mod_cols), sep = "")
  xfull <- merge(xfull, xval)
  msg(ii)
  print(dim(xfull))
}

rm(xval)
id_full <- xfull$ID; xfull$ID <- NULL

## building ####

# folds for cv evaluation
xfolds <- read_csv("./input/xfolds.csv")
xfolds$fold_index <- xfolds$fold5
xfolds <- xfolds[,c("ID", "fold_index")]
nfolds <- length(unique(xfolds$fold_index))

# storage for results
storage_matrix <- array(0, c(nfolds, 5))

# storage for level 2 forecasts 
xvalid2 <- array(0, c(nrow(xvalid),5))
xfull2 <- array(0, c(nrow(xfull),5))


for (ii in 1:nfolds)
{
  # mix with glmnet: average over multiple alpha parameters 
  isTrain <- which(xfolds$fold_index != ii)
  isValid <- which(xfolds$fold_index == ii)
  x0 <- xvalid[isTrain,];   
  x1 <- xvalid[isValid,]
  y0 <- y[isTrain];  
  y1 <- y[isValid]
  prx1 <- y1 * 0
  for (jj in 1:11)
  {
    mod0 <- glmnet(x = as.matrix(x0), y = y0, alpha = (jj-1) * 0.1)
    prx <- predict(mod0,as.matrix(x1))  
    prx <- prx[,ncol(prx)]
    prx1 <- prx1 + prx
  }
  storage_matrix[ii,1] <- log_loss(y1,prx1/11)
  xvalid2[isValid,1] <- prx1
  
  # mix with xgboost: bag over multiple seeds
  x0d <- xgb.DMatrix(as.matrix(x0), label = y0)
  x1d <- xgb.DMatrix(as.matrix(x1), label = y1)
  watch <- list(valid = x1d)
  prx2 <- y1 * 0
  for (jj in 1:nbag)
  {
    set.seed(seed_value + 1000*jj + 2^jj + 3 * jj^2)
    clf <- xgb.train(booster = "gbtree", 
                     maximize = TRUE, 
                     print.every.n = 50, 
                     nrounds = 668,
                     eta = 0.0083448206764984816, 
                     max.depth = 9,
                     colsample_bytree = 0.85469693813676129, 
                     subsample = 0.84436176592642664,
                     min_child_weight = 9.3990898152906937,
                     gamma=0.00058227567837996846,
                     data = x0d, 
                     objective = "binary:logistic",
                     eval_metric = "logloss")
    prx <- predict(clf, x1d)
    prx2 <- prx2 + prx
  }
  prx2 <- prx2 / nbag
  storage_matrix[ii,2] <- logLoss(y1,prx2)
  xvalid2[isValid,2] <- prx2
  
  # mix with nnet:  
  prx3 <- y1 * 0
  for (jj in 1:nbag)
  {
    set.seed(seed_value + 1000*jj + 2^jj + 3 * jj^2)
    net0 <- nnet(factor(y0) ~ ., data = x0, size = 40, MaxNWts = 20000, decay = 0.02)
    prx3 <- prx3 + predict(net0, x1)
  }
  prx3 <- prx3 /nbag
  storage_matrix[ii,3] <- logLoss(y1,prx3)
  xvalid2[isValid,3] <- prx3

  # mix with hillclimbing
  par0 <- buildEnsemble(c(1,15,5,0.6), x0,y0)
  prx4 <- as.matrix(x1) %*% as.matrix(par0)
  storage_matrix[ii,4] <- logLoss(y1,prx4)
  xvalid2[isValid,4] <- prx4
  
  # mix with random forest
  rf0 <- ranger(factor(y0) ~ ., 
                data = x0, 
                mtry = 25, 
                num.trees = 350,
                write.forest = T, 
                probability = T,
                min.node.size = 10, 
                seed = seed_value,
                num.threads = 8)
  prx5 <- predict(rf0, x1)$predictions[,2]
  storage_matrix[ii,5] <- logLoss(y1,prx5)
  xvalid2[isValid,5] <- prx5
  
  msg(paste("fold ",ii,": finished", sep = ""))
}

## build prediction on full set
# glmnet
prx1 <- rep( 0, nrow(xfull))
for (jj in 1:11)
{
  mod0 <- glmnet(x = as.matrix(xvalid), y = y, alpha = (jj-1) * 0.1)
  prx <- predict(mod0,as.matrix(xfull))  
  prx <- prx[,ncol(prx)]
  # storage_matrix[ii,jj] <- logLoss(y1,prx1)
  prx1 <- prx1 + prx
}
prx1 <- rank(prx1)/length(prx1)
xfull2[,1] <- prx1

# xgboost
x0d <- xgb.DMatrix(as.matrix(xvalid), label = y)
x1d <- xgb.DMatrix(as.matrix(xfull))
prx2 <- rep(0, nrow(xfull))
for (jj in 1:nbag)
{
  set.seed(seed_value + 1000*jj + 2^jj + 3 * jj^2)
  clf <- xgb.train(booster = "gbtree", 
                   maximize = TRUE, 
                   print.every.n = 50, 
                   nrounds = 668,
                   eta = 0.0083448206764984816, 
                   max.depth = 9,
                   colsample_bytree = 0.85469693813676129, 
                   subsample = 0.84436176592642664,
                   min_child_weight = 9.3990898152906937,
                   gamma=0.00058227567837996846,
                   data = x0d, 
                   objective = "binary:logistic",
                   eval_metric = "logloss")
  prx <- predict(clf, x1d)
  prx2 <- prx2 + prx
}
prx2 <- prx2 / nbag
xfull2[,2] <- prx2

# mix with nnet 
prx3 <- rep(0, nrow(xfull))
for (jj in 1:nbag)
{
  set.seed(seed_value + 1000*jj + 2^jj + 3 * jj^2)
  net0 <- nnet(factor(y) ~ ., data = xvalid, size = 40, MaxNWts = 20000, decay = 0.02)
  prx3 <- prx3 + predict(net0, xfull)
}
prx3 <- prx3 /nbag
xfull2[,3] <- prx3

# mix with hillclimbing
par0 <- buildEnsemble(c(1,15,5,0.6), xvalid,y)
prx4 <- as.matrix(xfull) %*% as.matrix(par0)
xfull2[,4] <- prx4

# mix with ranger
rf0 <- ranger(factor(y) ~ ., 
              data = xvalid, 
              mtry = 25, 
              num.trees = 350,
              write.forest = T, 
              probability = T,
              min.node.size = 10, 
              seed = seed_value,
              num.threads = 8)
prx5 <- predict(rf0, xfull)$predictions[,2]
xfull2[,5] <- prx5

rm(y0,y1, x0d, x1d, rf0, prx1,prx2,prx3,prx4,prx5)
rm(par0, net0, mod0,mod_class, clf,x0, x1)


## final ensemble forecasts ####
# evaluate performance across folds
storage2 <- array(0, c(nfolds,3))
param_mat <- array(0, c(nfolds, 5))
for (ii in 1:nfolds)
{
  isTrain <- which(xfolds$fold_index != ii)
  isValid <- which(xfolds$fold_index == ii)
  x0 <- apply(xvalid2[isTrain,],2,rank)/length(isTrain)
  x1 <- apply(xvalid2[isValid,],2,rank)/length(isValid)
  x0 <- data.frame(x0)
  x1 <- data.frame(x1)
  y0 <- y[isTrain]
  y1 <- y[isValid]
  
  par0 <- buildEnsemble(c(1,15, 5,0.6), x0,y0)
  pr1 <- as.matrix(x1) %*% as.matrix(par0)
  storage2[ii,1] <- logLoss(y1, pr1)
  param_mat[ii,] <- par0
  
}

# find the best combination of mixers
xvalid2 <- apply(xvalid2,2,rank)/nrow(xvalid2)
xfull2 <- apply(xfull2,2,rank)/nrow(xfull2)
xvalid2 <- data.frame(xvalid2)
xfull2 <- data.frame(xfull2)

# construct forecast
par0 <- buildEnsemble(c(1,15, 5,0.6), xvalid2,y)
prx <- as.matrix(xfull2) %*% as.matrix(par0)
xfor <- data.frame(ID = id_full, PredictedProb = prx)

print(paste("mean: ", mean(storage2[,1])))
print(paste("sd: ", sd(storage2[,1])))

# store
todate <- str_replace_all(Sys.Date(), "-","")
write_csv(xfor, path = paste("./submissions/ens_bag",nbag,"_",todate,"_seed",seed_value,".csv", sep = ""))