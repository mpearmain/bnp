require(data.table)
require(glmnet)
require(caret)
require(infotheo)
set.seed(92)

tmp1 = fread('/home/mikeskim/Desktop/kaggle/bnp/data/train.csv',data.table=F); trainY = as.matrix(tmp1$target); 
tmp1$target=NULL; 
tmp2 = fread('/home/mikeskim/Desktop/kaggle/bnp/data/test.csv',data.table=F)
tmp1=rbind(tmp1,tmp2)
tmp1$ID=NULL
#dummied already: v3, v24, v47  (v30,v31,v52 no good)
tmp1[is.na(tmp1)]=-1

delCols = c(8,23,25,36,37,46,31,75,79,51,53,54,63,73,81,82,89,92,95,105,107,108,109,110,116,117,118,119,123,124,128)
tmp1=tmp1[,-delCols]

tmpX = NULL
tmpScores=NULL
tmpJ = 1:ncol(tmp1)
for (j in tmpJ) {
  #v125,v50
  #  print(j)
  if (typeof(tmp1[,j])!="character") {
    tmp1[,j] = round(tmp1[,j],0)#1,2,3,4
  }
  else {

  tmpX = cbind(tmpX,tmp1[,j])  
  #tmp2 = data.table(tmp1)
  #tmp3 = tmp2[ , `:=`( COUNT = .N , IDX = 1:.N ) , by = eval(names(tmp1)[j]) ]
  #tmpX = cbind(tmpX,tmp3$COUNT)
  
  #tmpP = mutinformation(trainY, tmp3$COUNT[1:114321])
  #tmpScores = c(tmpScores,tmpP)
  
  }
}

tmpXX = NULL
for (j in 1:14) {
  print(j)
  if (j > 12) {
    break
  }
  for (k in (j+1):13) {
    for (m in (k+1):14) {
    
    tmp0 = paste(tmpX[,j],tmpX[,k],tmpX[,m],sep='_')
    tmp2 = data.table(tmp0)
    
    tmp3 = tmp2[ , `:=`( COUNT = .N , IDX = 1:.N ) , by = tmp0 ]
    tmpXX = cbind(tmpXX,tmp3$COUNT)
    }
  }
}

tmpXX = data.frame(tmpXX)
head(tmpXX)

write.csv(tmpXX, file='/home/mikeskim/Desktop/kaggle/bnp/features/tsne213.csv', quote=FALSE,row.names=FALSE)