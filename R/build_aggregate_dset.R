## wd etc ####
require(readr)
require(stringr)
require(caret)
require(Metrics)
require(gbm)

seed_value <- 450
todate <- str_replace_all(Sys.Date(), "-","")

metas_source <- "metafeatures"
target_data_folder <- "input2"

## functions ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
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
xlist_val <- dir(paste("../",metas_source, "/", sep = ""), pattern =  "prval", full.names = T)
xlist_full <- dir(paste("../",metas_source, "/", sep = ""), pattern = "prfull", full.names = T)

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
  print(dim(xvalid))
  msg(ii)
}

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
  print(dim(xfull))
  msg(ii)
}

rm(xval)

## process the sets ####

# separate stuff to ensure columns are identical
id_train <- xvalid$ID; xvalid$ID <- NULL
id_test <- xfull$ID; xfull$ID <- NULL
y <- xvalid$target; xvalid$target <- NULL

# trim linearly dependent ones 
print(paste("Pre linear combo trim size ", dim(xvalid)[2]))
flc <- findLinearCombos(xvalid)
if (length(flc$remove))
{
  xvalid <- xvalid[,-flc$remove]
  xfull <- xfull[,-flc$remove]
}
print(paste(" Number of cols after linear combo extraction:", dim(xvalid)[2]))

## save the datasets  ####
xvalid$target <- y 
xvalid$ID <- id_train
write.csv(xvalid, paste('../',target_data_folder,'/xtrain_',todate,'.csv', sep = ""), row.names = F)

xfull$ID <- id_test
write.csv(xfull, paste('../',target_data_folder,'/xtest_',todate,'.csv', sep = ""), row.names = F)

## reduced version via feature selection ####
y <- xvalid$target; xvalid$target <- NULL
id_train <- xvalid$ID; xvalid$ID <- NULL
id_test <- xfull$ID; xfull$ID <- NULL

idFix <- createDataPartition(y = y, times = 40, p = 0.15)
relev_mat <- array(0, c(ncol(xvalid), length(idFix)))
# loop over folds 
for (ii in 1:length(idFix))
{
  idx <- idFix[[ii]]
  x0 <- xvalid[idx,]; y0 <- y[idx];
  
  mod0 <- gbm.fit(x = x0, y = y0, distribution = "bernoulli", 
                    n.trees = 150, interaction.depth = 25, shrinkage = 0.001, verbose = F)
  
  relev_mat[,ii] <- summary(mod0, order = F, plot = F)[,2]
  msg(ii)
}

# pick features that are always relevant
idx <- which(apply(relev_mat,1,prod) != 0)
xv <- xvalid[,idx]; xf <- xfull[,idx]
xv$target <- y; xv$ID <- id_train; xf$ID <- id_test
write.csv(xv, paste('../',target_data_folder,'/xtrain_',todate,'r1.csv', sep = ""), row.names = F)
write.csv(xf, paste('../',target_data_folder,'/xtest_',todate,'r1.csv', sep = ""), row.names = F)

# pick features with average importance > 1pct
idx <- which(rowMeans(relev_mat) > 0.01)
xv <- xvalid[,idx]; xf <- xfull[,idx]
xv$target <- y; xv$ID <- id_train; xf$ID <- id_test
write.csv(xv, paste('../',target_data_folder,'/xtrain_',todate,'r3.csv', sep = ""), row.names = F)
write.csv(xf, paste('../',target_data_folder,'/xtest_',todate,'r3.csv', sep = ""), row.names = F)

# pick features with average importance > 5pct
idx <- which(rowMeans(relev_mat) > 0.05)
xv <- xvalid[,idx]; xf <- xfull[,idx]
xv$target <- y; xv$ID <- id_train; xf$ID <- id_test
write.csv(xv, paste('../',target_data_folder,'/xtrain_',todate,'r5.csv', sep = ""), row.names = F)
write.csv(xf, paste('../',target_data_folder,'/xtest_',todate,'r5.csv', sep = ""), row.names = F)

## VARIA ####
# evaluate by cross-validated performance
# idFix <- createDataPartition(y = y, times = 30, p = 2/3)
# xmat <- array(0, c(length(idFix), ncol(xvalid) * 2 + 2))
# # loop over folds 
# for (jj in 1:length(idFix))
# {
#   idx <- idFix[[jj]]
#   x0 <- xvalid[idx,]; x1 <- xvalid[-idx,]
#   y0 <- y[idx]; y1 <- y[-idx]
#   
#   xmat[jj,1:ncol(x1)] <- apply(x1,2,function(s) log_loss(y1,s))
#   xmat[jj,(1:ncol(x1)) + ncol(x1)] <- apply(x1,2,function(s) log_loss(y1,s + mean(y0) - mean(s)))
#   xmat[jj, 2 * ncol(x1) + 1] <- log_loss(y1, rowMeans(x1))
#   xmat[jj, 2 * ncol(x1) + 2] <- log_loss(y1, exp(rowMeans(log(x1))))
# }

# # combo
# idx <- grep("xgb|mars|nnet|srk|msk", colnames(xvalid))
# xv <- xvalid[,idx]; xf <- xfull[,idx]
# xv$ID <- id_train; xv$target <- y; xf$ID <- id_test
# write.csv(xv, paste('../input2/xtrain_lvl2',todate,'combo.csv', sep = ""), row.names = F)
# write.csv(xf, paste('../input2/xtest_lvl2',todate,'combo.csv', sep = ""), row.names = F)
# 
# # combo lvl3
# idx <- grep("xgb|nnet", colnames(xvalid))
# xv <- xvalid[,idx]; xf <- xfull[,idx]
# xv$ID <- id_train; xv$target <- y; xf$ID <- id_test
# write.csv(xv, paste('../input3/xtrain_lvl3',todate,'xgbnnet.csv', sep = ""), row.names = F)
# write.csv(xf, paste('../input3/xtest_lvl3',todate,'xgbnnet.csv', sep = ""), row.names = F)
