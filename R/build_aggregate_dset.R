## wd etc ####
require(readr)
require(stringr)
require(caret)
require(Metrics)

seed_value <- 450
todate <- str_replace_all(Sys.Date(), "-","")

## functions ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

## data ####
# list the groups 
xlist_val <- dir("../metafeatures/", pattern =  "prval", full.names = T)
xlist_full <- dir("../metafeatures/", pattern = "prfull", full.names = T)

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
write.csv(xvalid, paste('../input2/xtrain_lvl2',todate,'.csv', sep = ""), row.names = F)

xfull$ID <- id_test
write.csv(xfull, paste('../input2/xtest_lvl2',todate,'.csv', sep = ""), row.names = F)

## limited versions of the dataset (restricted to specific model class)
y <- xvalid$target; xvalid$target <- NULL
id_train <- xvalid$ID; xvalid$ID <- NULL
id_test <- xfull$ID; xfull$ID <- NULL

# xgb only
idx <- grep("xgb", colnames(xvalid))
xv <- xvalid[,idx]; xf <- xfull[,idx]
xv$ID <- id_train; xv$target <- y; xf$ID <- id_test
write.csv(xv, paste('../input2/xtrain_lvl2',todate,'xgb.csv', sep = ""), row.names = F)
write.csv(xf, paste('../input2/xtest_lvl2',todate,'xgb.csv', sep = ""), row.names = F)

# keras only
idx <- grep("keras", colnames(xvalid))
xv <- xvalid[,idx]; xf <- xfull[,idx]
xv$ID <- id_train; xv$target <- y; xf$ID <- id_test
write.csv(xv, paste('../input2/xtrain_lvl2',todate,'keras.csv', sep = ""), row.names = F)
write.csv(xf, paste('../input2/xtest_lvl2',todate,'keras.csv', sep = ""), row.names = F)

# combo
idx <- grep("xgb|mars|nnet", colnames(xvalid))
xv <- xvalid[,idx]; xf <- xfull[,idx]
xv$ID <- id_train; xv$target <- y; xf$ID <- id_test
write.csv(xv, paste('../input2/xtrain_lvl2',todate,'combo.csv', sep = ""), row.names = F)
write.csv(xf, paste('../input2/xtest_lvl2',todate,'combo.csv', sep = ""), row.names = F)
