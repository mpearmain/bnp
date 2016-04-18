require(data.table)
require(caret)
require(readr)
set.seed(92)

tmp1 = fread('../input/train.csv',data.table=F); trainY = as.matrix(tmp1$target); 
isTrain <- 1:nrow(tmp1); id_train <- tmp1$ID
tmp1$target=NULL; 
tmp2 = fread('../input/test.csv',data.table=F); id_test <- tmp2$ID
tmp1=rbind(tmp1,tmp2)
tmp1$ID=NULL
tmp2 = matrix(0,nrow(tmp1),ncol(tmp1))
tmp2[is.na(tmp1)]=1
tmp1[is.na(tmp1)]=-1

delCols = c(8,23,25,36,37,46,31,75,79,51,53,54,63,73,81,82,89,92,95,105,107,108,109,110,116,117,118,119,123,124,128)
tmp1=tmp1[,-delCols]

tmpX = NULL
tmpScores=NULL
tmpJ = 1:ncol(tmp1)
for (j in tmpJ) {
  #  print(j)
  if (typeof(tmp1[,j])!="character") {
    #tmp1[,j] = round(tmp1[,j],0)#1,2,3,4
  }
  else {
    
    tmpX = cbind(tmpX,tmp1[,j])  
  }
}
#tmpX = cbind(tmpX,tmp2)
tmpX = cbind(tmpX,tmp1$v129,tmp1$v72,tmp1$v62,tmp1$v38)

tmp = combn(1:ncol(tmpX),3) #if you change this to 4
tmpXX = matrix(0,nrow=nrow(tmpX),ncol=ncol(tmp))
tmpX2 = matrix(0,nrow=nrow(tmpX),ncol=ncol(tmp))
tmpJ = 1:ncol(tmp)
for (j in tmpJ) {
  print(j/816)
  x = tmp[1,j]; y = tmp[2,j]
  z = tmp[3,j]; #a = tmp[4,j]
  
  tmp0 = paste(tmpX[,x],tmpX[,y],tmpX[,z],sep='_') #want to add tmpX[,a] if you change to 4.
  tmp00 = tmp1$v50
  tmp2 = data.table(cbind(tmp0,tmp00))
  tmp3 = tmp2[ , `:=`( COUNT = .N , COUNT2=min(as.numeric(tmp00)) , IDX = 1:.N ) , by = tmp0 ]
  tmpXX[,j] = tmp3$COUNT
  tmpX2[,j] = tmp3$COUNT2
}

tmpXX = cbind(tmpXX,tmpX2)
tmpXX = data.frame(tmpXX)
head(tmpXX)
dim(tmpXX)

# write.csv(tmpXX, file='../input/ds_v12mean.csv', quote=FALSE,row.names=FALSE)

xtrain <- tmpXX[isTrain,]; xtrain$target <- trainY; xtrain$ID <- id_train
xtest <- tmpXX[-isTrain,]; xtest$ID <- id_test
write_csv(xtrain, path = "../input/xtrain_v50min.csv")
write_csv(xtest, path = "../input/xtest_v50min.csv")