## wd etc ####
require(readr)
require(stringr)
require(caret)
require(ranger)
require(Metrics)
require(Boruta)

seed_value <- 181
todate <- str_replace_all(Sys.Date(), "-","")

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
    trackScore <- apply(xx, 2, function(x) -log_loss(yvec,x))
    
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
                              function(x) -log_loss(yvec,rowMeans(cbind(x , hill.df))))
      
      best <- which.max(trackScoreHill)
      best.track <- c(best.track, max(trackScoreHill))
      hillNames <- c(hillNames,names(best))
      hill.df <- data.frame(hill.df, xx[,best])
      #msg(ee)
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
xvalid <- read_csv("../input2/xtrain_lvl220160329.csv")
y <- xvalid$target; xvalid$target <- NULL
id_valid <- xvalid$ID; xvalid$ID <- NULL

xfull <- read_csv("../input2/xtest_lvl220160329.csv")
id_full <- xfull$ID; xfull$ID <- NULL

## pick top lvl1 model ####

# # division into folds: 5-fold
# xfolds <- read_csv("../input/xfolds.csv"); xfolds$fold_index <- xfolds$fold5
# xfolds <- xfolds[,c("ID", "fold_index")]
# nfolds <- length(unique(xfolds$fold_index))
# 
# storage_mat <- array(0, c(nfolds, ncol(xvalid)))
# for (ii in 1:nfolds)
# {
#   isTrain <- which(xfolds$fold_index != ii)
#   isValid <- which(xfolds$fold_index == ii)
#   x0 <- xvalid[isTrain,]; x1 <- xvalid[isValid,]
#   y0 <- y[isTrain]; y1 <- y[isValid]
#   
#   storage_mat[ii,] <- apply(x0,2,function(s) log_loss(y0,s))
# }

## compute reduced version ####

boruta_imp1 <- Boruta(factor(y)~.,data=xvalid,doTrace=2)
