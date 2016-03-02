## wd etc ####
require(readr)
require(stringr)
require(caret)
require(Metrics)

seed_value <- 450
todate <- str_replace_all(Sys.Date(), "-","")
nbag <- 5

## functions ####

msg <- function(mmm,...)
{
  cat(sprintf(paste0("[%s] ",mmm),Sys.time(),...)); cat("\n")
}

## data ####
# list the groups 
xlist_val <- dir("./metafeatures/", pattern =  "prval", full.names = T)
xlist_full <- dir("./metafeatures/", pattern = "prfull", full.names = T)

# aggregate validation set
ii <- 1
mod_class <- str_split(xlist_val[[ii]], "_")[[1]][[2]]
xvalid <- read_csv(xlist_val[[ii]])
xcols <- colnames(xvalid)[1:(ncol(xvalid)-2)]
xcols <- paste(xcols , ii, sep = "")
colnames(xvalid)[1:(ncol(xvalid)-2)] <- xcols

for (ii in 2:length(xlist_val))
{
  mod_class <- str_split(xlist_val[[ii]], "_")[[1]][[2]]
  xval <- read_csv(xlist_val[[ii]])
  xcols <- colnames(xval)[1:(ncol(xval)-2)]
  xcols <- paste(xcols , ii, sep = "")
  colnames(xval)[1:(ncol(xval)-2)] <- xcols
  xvalid <- merge(xvalid, xval)
  msg(ii)
}

# aggregate test set
ii <- 1
mod_class <- str_split(xlist_full[[ii]], "_")[[1]][[2]]
xfull <- read_csv(xlist_full[[ii]])
xcols <- colnames(xfull)[1:(ncol(xfull)-1)]
xcols <- paste(xcols , ii, sep = "")
colnames(xfull)[1:(ncol(xfull)-1)] <- xcols

for (ii in 2:length(xlist_val))
{
  xval <- read_csv(xlist_full[[ii]])
  xcols <- colnames(xval)[1:(ncol(xval)-1)]
  xcols <- paste(xcols , ii, sep = "")
  colnames(xval)[1:(ncol(xval)-1)] <- xcols
  xfull <- merge(xfull, xval)
  msg(ii)
}

rm(xval)

# prepare the data
y <- xvalid$target; xvalid$target <- NULL
id_valid <- xvalid$ID; xvalid$ID <- NULL
id_full <- xfull$ID; xfull$ID <- NULL

# trim linearly dependent ones 
print(paste("Pre linear combo trim size ", dim(xvalid)[2]))
flc <- findLinearCombos(xvalid)
if (length(flc$remove))
{
  xvalid <- xvalid[,-flc$remove]
  xfull <- xfull[,-flc$remove]
}
print(paste(" Number of cols after linear combo extraction:", dim(xvalid)[2]))

# amend the data
xMed <- apply(xvalid,1,median); xMin <- apply(xvalid,1,min)
xMax <- apply(xvalid,1,max); xMad <- apply(xvalid,1,mad)
xq1 <- apply(xvalid,1, function(s) quantile(s, 0.1))
xq2 <- apply(xvalid,1, function(s) quantile(s, 0.25))
xq3 <- apply(xvalid,1, function(s) quantile(s, 0.75))
xq4 <- apply(xvalid,1, function(s) quantile(s, 0.9))
xvalid$xmed <- xMed 
xvalid$xmax <- xMax 
xvalid$xmin <- xMin 
xvalid$xmad <- xMad
xvalid$xq1 <- xq1 
xvalid$xq2 <- xq2 
xvalid$xq3 <- xq3
xvalid$xq4 <- xq4

xq1 <- apply(xfull,1, function(s) quantile(s, 0.1))
xq2 <- apply(xfull,1, function(s) quantile(s, 0.25))
xq3 <- apply(xfull,1, function(s) quantile(s, 0.75))
xq4 <- apply(xfull,1, function(s) quantile(s, 0.9))
xMed <- apply(xfull,1,median); xMin <- apply(xfull,1,min)
xMax <- apply(xfull,1,max); xMad <- apply(xfull,1,mad)
xfull$xmed <- xMed 
xfull$xmax <- xMax 
xfull$xmin <- xMin 
xfull$xmad <- xMad
xfull$xq1 <- xq1 
xfull$xq2 <- xq2 
xfull$xq3 <- xq3 
xfull$xq4 <- xq4

rm(xq1, xq2, xq3, xq4, xMad, xMax, xMed, xMin)


# To save dataset for quick optimizations
xvalid$target <- y 
xvalid$ID <- id_valid
write.csv(xvalid, paste('./input/xvalid_ensemble_base.csv', sep = ""), row.names = F)

xfull$ID <- id_full
write.csv(xfull, paste('./input/xfull_ensemble_base.csv', sep = ""), row.names = F)

