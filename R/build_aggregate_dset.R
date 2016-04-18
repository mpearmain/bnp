## wd etc ####
require(readr)
require(stringr)
require(caret)
require(Metrics)
require(gbm)

seed_value <- 450
todate <- str_replace_all(Sys.Date(), "-","")
todate <- paste(todate, "v2", sep = "")

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
  print(mod_class)
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

# add summary statistics
xmin <- apply(xvalid,1,min); xmax <- apply(xvalid,1,max); xmed <- apply(xvalid,1,median); 
xq1 <- apply(xvalid,1,function(s) quantile(s, 0.1))
xq2 <- apply(xvalid,1,function(s) quantile(s, 0.25))
xq3 <- apply(xvalid,1,function(s) quantile(s, 0.75))
xq4 <- apply(xvalid,1,function(s) quantile(s, 0.9))
xvalid$xmin <- xmin; xvalid$xmax <- xmax; xvalid$xmed <- xmed
xvalid$xq1 <- xq1; xvalid$xq2 <- xq2; xvalid$xq3 <- xq3; xvalid$xq4 <- xq4

xmin <- apply(xfull,1,min); xmax <- apply(xfull,1,max); xmed <- apply(xfull,1,median); 
xq1 <- apply(xfull,1,function(s) quantile(s, 0.1))
xq2 <- apply(xfull,1,function(s) quantile(s, 0.25))
xq3 <- apply(xfull,1,function(s) quantile(s, 0.75))
xq4 <- apply(xfull,1,function(s) quantile(s, 0.9))
xfull$xmin <- xmin; xfull$xmax <- xmax; xfull$xmed <- xmed
xfull$xq1 <- xq1; xfull$xq2 <- xq2; xfull$xq3 <- xq3; xfull$xq4 <- xq4

# compare the distributions with ks.test
ks_mat <- array(0,c(ncol(xvalid),2))
for (j in 1:ncol(xvalid))
{
  kst <- ks.test(xvalid[,j], xfull[,j])
  ks_mat[j,1] <- kst$statistic
  ks_mat[j,2] <- kst$p.value
}

## save the datasets  ####
xvalid$target <- y 
xvalid$ID <- id_train
write.csv(xvalid, paste('../',target_data_folder,'/xtrain_',todate,'.csv', sep = ""), row.names = F)

xfull$ID <- id_test
write.csv(xfull, paste('../',target_data_folder,'/xtest_',todate,'.csv', sep = ""), row.names = F)

## reduced version via feature selection: ks.test ####
id_train <- xvalid$ID; xvalid$ID <- NULL
id_test <- xfull$ID; xfull$ID <- NULL
y <- xvalid$target; xvalid$target <- NULL

# compare whether the prval and prfull parts come from same distribution
ix <- which(ks_mat[,1] < 0.01)
xv <- xvalid[,ix]; xf <- xfull[,ix]
xv$target <- y; xv$ID <- id_train; xf$ID <- id_test
write.csv(xv, paste('../',target_data_folder,'/xtrain_',todate,'ks1.csv', sep = ""), row.names = F)
write.csv(xf, paste('../',target_data_folder,'/xtest_',todate,'ks1.csv', sep = ""), row.names = F)

ix <- which(ks_mat[,1] < 0.05)
xv <- xvalid[,ix]; xf <- xfull[,ix]
xv$target <- y; xv$ID <- id_train; xf$ID <- id_test
write.csv(xv, paste('../',target_data_folder,'/xtrain_',todate,'ks2.csv', sep = ""), row.names = F)
write.csv(xf, paste('../',target_data_folder,'/xtest_',todate,'ks2.csv', sep = ""), row.names = F)


## reduced version via feature selection: gbm ####
# y <- xvalid$target; xvalid$target <- NULL

idFix <- createDataPartition(y = y, times = 40, p = 0.15)
relev_mat <- array(0, c(ncol(xvalid), length(idFix)))
# loop over folds 
for (ii in 1:length(idFix))
{
  idx <- idFix[[ii]]
  x0 <- xvalid[idx,]; y0 <- y[idx];
  
  mod0 <- gbm.fit(x = x0, y = y0, distribution = "bernoulli", 
                    n.trees = 200, interaction.depth = 7, shrinkage = 0.001, verbose = T)
  
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
xvalid <- read_csv("../metafeatures2/prval_xgbc_20160418_data20160418v2_seed12.csv")
xfull <- read_csv("../metafeatures2/prfull_xgbc_20160418_data20160418v2_seed12.csv")
y <- xvalid$target
set.seed(2388)
idFix <- createDataPartition(y = y, times = 100, p = 2/3)
xmat <- array(0, c(length(idFix), 4 ))
# loop over folds 
for (jj in 1:length(idFix))
{
  idx <- idFix[[jj]]
  x0 <- xvalid[idx,]; x1 <- xvalid[-idx,]
  y0 <- y[idx]; y1 <- y[-idx]
  
  xmat[jj,1] <- log_loss(y1,xvalid[-idx,1])
  xmat[jj,2] <- log_loss(y1,xvalid[-idx,1] + mean(y0) - mean(xvalid[-idx,1]))
}
xfull$xgb0 <- xfull$xgb0 + mean(xvalid$target) - mean(xfull$xgb0)
xfor <- data.frame(ID = xfull$ID, PredictedProb = xfull$xgb0)
write_csv(xfor, path = "../submissions/calibrated_xgb_on_lvl1.csv")