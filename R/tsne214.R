require(data.table)
set.seed(92)

tmp1 = fread('../input/train.csv',data.table=F); trainY = as.matrix(tmp1$target); 
tmp1$target=NULL; 
tmp2 = fread('../input/test.csv',data.table=F)
tmp1=rbind(tmp1,tmp2)
tmp1$ID=NULL
tmp1[is.na(tmp1)]=-1

delCols = c(8,23,25,36,37,46,31,75,79,51,53,54,63,73,81,82,89,92,95,105,107,108,109,110,116,117,118,119,123,124,128)
tmp1=tmp1[,-delCols]

tmpX = NULL
tmpScores=NULL
tmpJ = 1:ncol(tmp1)
for (j in tmpJ) {
  #  print(j)
  if (typeof(tmp1[,j])!="character") {
    tmp1[,j] = round(tmp1[,j],0)#1,2,3,4
  }
  else {

  tmpX = cbind(tmpX,tmp1[,j])  
  }
}


tmp = combn(1:ncol(tmpX),4)
tmpXX = matrix(0,nrow=nrow(tmpX),ncol=ncol(tmp))
tmpJ = 1:ncol(tmp)
for (j in tmpJ) {
 print(j)
 x = tmp[1,j]; y = tmp[2,j]
 z = tmp[3,j]; a = tmp[4,j]
 
 tmp0 = paste(tmpX[,x],tmpX[,y],tmpX[,z],tmpX[,a],sep='_')
 tmp2 = data.table(tmp0)
 tmp3 = tmp2[ , `:=`( COUNT = .N , IDX = 1:.N ) , by = tmp0 ]
 tmpXX[,j] = tmp3$COUNT
}
  


tmpXX = data.frame(tmpXX)
head(tmpXX)

write.csv(tmpXX, file='/home/mikeskim/Desktop/kaggle/bnp/features/tsne214.csv', quote=FALSE,row.names=FALSE)