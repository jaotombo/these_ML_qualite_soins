# Loading caret package
library("caret")
library("psych")
library("doSNOW")
library("hmeasure")
library("gbm")
library("ranger")

cl <- makeCluster(14, type = "SOCK")
registerDoSNOW(cl)


#===============================================================
#====   Loading data
#===============================================================

setwd("E:/Rehospitalisation/Medecine Baltimore/data")

rawData = read.csv("rawData.csv")

varNames = c("RHurg30","agecat","SEXE","AME","CMU","libelle_DA","sevav3cat",
             "ASO","MEdom","URGav","dureeAV4cat","MSdom", "yURG6mav",
             "CH_RENAL","CH_RD","CH_PVD","CH_PUD","CH_PLEGIA","CH_MSL",
             "CH_MILD_LIVER_D","CH_METS","CH_MALIGNANCY","CH_HIV","CH_DM_COMP",
             "CH_DM","CH_DEMENTIA","CH_CVD","CH_COPD","CH_CHF","CH_MI")

rehosData = rawData[, varNames]

rm(rawData)

#======================================
# Definition of outcome and predictors
#======================================

outcomeName = "RHurg30"
predNames = varNames[-1]

#=============================================================
#====   Pre processing
#===============================================================


# Formatting all predictors as categorical
rehosData[,predNames] = data.frame(apply(rehosData[,predNames],2,as.factor))

# Check structure of data
str(rehosData,list.len = 1000)
summary(rehosData)
sum(is.na(rehosData))

# set up training and testing data stratified on the outcome
table(rehosData[,outcomeName])
prop.table(table(rehosData[,outcomeName]))

# select 10 % stratified sample of the data
set.seed(654321)
itrain = createDataPartition(rehosData[,outcomeName], p=0.10, list=FALSE)
train = rehosData[itrain,]

#===============================================================
#===   Preparing data for analysis
#===============================================================

#Checking the structure of train file
train_stat = describe(train)

# Identifying constant variables
sd_0 = train_stat$vars[train_stat$sd==0]
sd_names = rownames(train_stat)[sd_0]


#Checking the structure of train
str(train)
prop.table(table(train[,outcomeName]))

#============================
# === START RUNNING HERE ===
#============================

# predictors --------------------------------------------------------------
predictors = names(train)[!names(train) %in% c(outcomeName, sd_names)]
#--------------------------------------------------------------------------

# converting the outcome to a factor
train[,outcomeName] <- as.factor(train[,outcomeName])



#===============================================================
# No Model Selection Analysis
#===============================================================

## This summary function is used to evaluate the models.
fiveStats <- function(...) c(twoClassSummary(...), defaultSummary(...))


# Tuning parameters
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10 ,
  repeats = 3 ,
  summaryFunction = fiveStats,
  classProbs = TRUE)

levels(train[,outcomeName]) <- make.names(levels(factor(train[,outcomeName])))


#-------------------------------------------------------------
# Cross Validation to select the best hyper parameters
#-------------------------------------------------------------

set.seed(654321)
index <- createDataPartition(train[,outcomeName], p=0.7, list=FALSE)
trainSet = train[index,]
testSet = train[-index,]



#==============================================================
# CARET machine learning using CART best split
#==============================================================

# Logistic regression
model_lr1 = train(trainSet[,predictors],trainSet[,outcomeName],method='glm', family='binomial',
                  trControl=fitControl, metric="ROC",  tuneLength=10)
param_lr1 = model_lr1$bestTune


# CART
model_cart1 = train(trainSet[,predictors],trainSet[,outcomeName],method='rpart',
                    trControl=fitControl, metric="ROC",  tuneLength=15)
param_cart1 = model_cart1$bestTune


# Gradient Boosting
model_gbm1 = train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',
                   trControl=fitControl, metric="ROC",  tuneLength=10)
param_gbm1 = model_gbm1$bestTune


# randomForest
model_rf1 = train(trainSet[,predictors],trainSet[,outcomeName],method='rf',
                  trControl=fitControl, metric="ROC", ntrees=500,  tuneLength=10)
param_rf1 = model_rf1$bestTune


# ranger
model_rf2 = train(trainSet[,predictors],trainSet[,outcomeName],method='ranger',
                  trControl=fitControl, metric="ROC", num.trees=500,  tuneLength=10,importance="impurity")
param_rf2 = model_rf2$bestTune


# Neural Network
model_nnet1 = train(trainSet[,predictors],trainSet[,outcomeName],method='nnet',
                    trControl=fitControl, metric="ROC",  tuneLength=10, maxit=1000)
param_nnet1 = model_nnet1$bestTune




#======================================================
# Running all the required functions for modeling
#======================================================

#==================================================================================
# Cette fonction calcule le taux de faux positifs (tfp) et de vrais positifs (tvp)
#==================================================================================
rocTFVP = function(score,class) {
  rocData = cbind.data.frame(score,class)
  rocData = rocData[order(score, decreasing = TRUE),]
  
  tfp = c()
  tvp = c()
  for (i in 1:nrow(rocData)) {
    tfp = append(tfp,sum(rocData$class[1:i]=="X0")/sum(rocData$class=="X0") )
    tvp = append(tvp,sum(rocData$class[1:i]=="X1")/sum(rocData$class=="X1") )
  }
  return(data.frame(tfp,tvp))
}
# Sortie : data frame


#=================================================================================
# Cette fonction calcule l'importance des variables pour la regression logistique
#=================================================================================
lr_Imp_fct = function (trainSet, outcomeName) {
  # base train
  # Calcul de la contribution de chaque modalite apres disjonction totale
  predictors<-names(train)[!names(train) %in% outcomeName]
  X=trainSet[,predictors]
  X_model = glm(paste(outcomeName," ~ .",sep=""), data=trainSet, family = binomial())
  
  
  varDev = data.frame(matrix(NA, ncol=2, nrow=length(predictors))) # initialize data frame
  rownames(varDev)=predictors
  colnames(varDev)=c("deviance","p-value")
  for (i in 1:length(predictors)) {
    model = glm(as.formula(paste(outcomeName, " ~ X[,",i,"]", sep="")), data=trainSet, family = "binomial")
    varDev[i,1] = X_model$null.deviance - model$deviance
    varDev[i,2] = 1-pchisq(varDev[i,1], 1)
  }
  varDev$importance = varDev$deviance/max(varDev$deviance)*100
  return(varDev)
}
# Sortie : vecteur avec le nom des predicteurs en ligne


#==========================================================================
# Cette fonction calcule l'importance des variables en fonction du modele
#==========================================================================
importance = function(model, trainSet, predNames, outcomeName, useModel) {
  if (model$method == "glm") {
    imp = lr_Imp_fct(trainSet,outcomeName)$importance
    res = data.frame(imp)
    rownames(res) = predNames
    colnames(res) = model$method
    return(res)
  }
  imp = varImp(model, useModel)$importance
  res = data.frame(imp[predNames,1])
  rownames(res) = predNames
  colnames(res) = model$method
  return(res)
}
# Sortie : vecteur avec nom des predicteurs


#=======================================================================
# Cette fonction calcule l'AUC sous le ROC avec la methode de trapezes
#=======================================================================
aucMoyen = function(tfp,tvp) {
  auc=0
  for (i in 2:length(tfp)){
    auc = auc + (tfp[i]-tfp[i-1])*(tvp[i]+tvp[i-1])/2
  }
  return(auc)
}


#####################################
# Apprentissage et Test + repetition
#####################################

lrtfp = carttfp = gbmtfp = rf1tfp = rf2tfp = nnettfp = c()
lrtvp = carttvp = gbmtvp = rf1tvp = rf2tvp = nnettvp = c()
lrmetric = cartmetric = gbmmetric = rf1metric = rf2metric = nnetmetric = c()
lrimp = cartimp = gbmimp = rf1imp = rf2imp = nnetimp = data.frame(matrix(NA, ncol=1, nrow=length(predictors)))

repetition = 2

for (i in 1:repetition) {
  
  fitControl <- trainControl(method = "none", classProbs = TRUE)
  
  print(c("nombre total de repetitions", repetition))
  
  indice <- createDataPartition(train[,outcomeName], p=0.7, list=FALSE)
  trainSet = train[indice,]
  testSet = train[-indice,]
  
  print(c("repetition #", i, "LR"))
  #=================================================================================
  # Cette partie calcule la performance moyen d'un modele LR avec 
  # les donnees pour calculer les tfp, tvp, ROC et importance sur plusieurs tirages
  #=================================================================================
  
  best_lr1 = train(trainSet[,predictors],trainSet[,outcomeName],method='glm', family='binomial',
                   trControl=fitControl, metric="ROC", tuneGrid=param_lr1)
  
  lrpred = predict(best_lr1, testSet, type="prob")
  lrrocData = rocTFVP(lrpred$"X1",testSet[,outcomeName])
  lrtfp = cbind(lrtfp, lrrocData$tfp)
  lrtvp = cbind(lrtvp, lrrocData$tvp)
  
  lrmetric = rbind(lrmetric, HMeasure(testSet[,outcomeName],lrpred$X1)$metrics)
  lrimp = cbind(lrimp,importance(best_lr1,trainSet,predictors,outcomeName)[[1]])
  
  print(c("repetition #", i, "CART"))
  #=================================================================================
  # Cette partie calcule la performance moyen d'un modele CART avec 
  # les donnees pour calculer les tfp, tvp, ROC et importance sur plusieurs tirages
  #=================================================================================  
  
  best_cart1 = train(trainSet[,predictors],trainSet[,outcomeName],method='rpart',
                     trControl=fitControl, metric="ROC", tuneGrid=param_cart1)
  
  cartpred = predict(best_cart1, testSet, type="prob")
  cartrocData = rocTFVP(cartpred$"X1",testSet[,outcomeName])
  carttfp = cbind(carttfp, cartrocData$tfp)
  carttvp = cbind(carttvp, cartrocData$tvp)
  
  cartmetric = rbind(cartmetric, HMeasure(testSet[,outcomeName],cartpred$X1)$metrics)
  cartimp = cbind(cartimp,importance(best_cart1,trainSet,predictors,outcomeName,useModel=TRUE)[[1]])
  
  print(c("repetition #", i, "GBM"))
  #=================================================================================
  # Cette partie calcule la performance moyen d'un modele GBM avec
  # les donnees pour calculer les tfp, tvp, ROC et importance sur plusieurs tirages
  #=================================================================================
  
  best_gbm1 = train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',
                    trControl=fitControl, metric="ROC", tuneGrid=param_gbm1)
  
  gbmpred = predict(best_gbm1, testSet, type="prob")
  gbmrocData = rocTFVP(gbmpred$"X1",testSet[,outcomeName])
  gbmtfp = cbind(gbmtfp, gbmrocData$tfp)
  gbmtvp = cbind(gbmtvp, gbmrocData$tvp)
  
  gbmmetric = rbind(gbmmetric, HMeasure(testSet[,outcomeName],gbmpred$X1)$metrics)
  gbmimp = cbind(gbmimp,importance(best_gbm1,trainSet,predictors,outcomeName,useModel=TRUE)[[1]])
  
  print(c("repetition #", i, "RF1"))
  #=================================================================================
  # Cette partie calcule la performance moyen d'un modele RF1 avec
  # les donnees pour calculer les tfp, tvp, ROC et importance sur plusieurs tirages
  #=================================================================================
  
  best_rf1 = train(trainSet[,predictors],trainSet[,outcomeName],method='rf',
                   trControl=fitControl, metric="ROC", tuneGrid=param_rf1)
  
  rf1pred = predict(best_rf1, testSet, type="prob")
  rf1rocData = rocTFVP(rf1pred$"X1",testSet[,outcomeName])
  rf1tfp = cbind(rf1tfp, rf1rocData$tfp)
  rf1tvp = cbind(rf1tvp, rf1rocData$tvp)
  
  rf1metric = rbind(rf1metric, HMeasure(testSet[,outcomeName],rf1pred$X1)$metrics)
  rf1imp = cbind(rf1imp,importance(best_rf1,trainSet,predictors,outcomeName,useModel=TRUE)[[1]])
  
  print(c("repetition #", i, "RF2"))
  #=================================================================================
  # Cette partie calcule la performance moyen d'un modele RF2 avec
  # les donnees pour calculer les tfp, tvp, ROC et importance sur plusieurs tirages
  #=================================================================================
  
  best_rf2 = train(trainSet[,predictors],trainSet[,outcomeName],method='ranger',
                   trControl=fitControl, metric="ROC", tuneGrid=param_rf2, importance='impurity')
  
  rf2pred = predict(best_rf2, testSet, type="prob")
  rf2rocData = rocTFVP(rf2pred$"X1",testSet[,outcomeName])
  rf2tfp = cbind(rf2tfp, rf2rocData$tfp)
  rf2tvp = cbind(rf2tvp, rf2rocData$tvp)
  
  rf2metric = rbind(rf2metric, HMeasure(testSet[,outcomeName],rf2pred$X1)$metrics)
  rf2imp = cbind(rf2imp,importance(best_rf2,trainSet,predictors,outcomeName,useModel=TRUE)[[1]])
  
  print(c("repetition #", i, "NNET"))
  #=================================================================================
  # Cette partie calcule la performance moyen d'un modele NNET avec
  # les donnees pour calculer les tfp, tvp, ROC et importance sur plusieurs tirages
  #=================================================================================
  
  best_nnet1 = train(trainSet[,predictors],trainSet[,outcomeName],method='nnet',
                     trControl=fitControl, metric="ROC", tuneGrid=param_nnet1)
  
  nnetpred = predict(best_nnet1, testSet, type="prob")
  nnetrocData = rocTFVP(nnetpred$"X1",testSet[,outcomeName])
  nnettfp = cbind(nnettfp, nnetrocData$tfp)
  nnettvp = cbind(nnettvp, nnetrocData$tvp)
  
  nnetmetric = rbind(nnetmetric, HMeasure(testSet[,outcomeName],nnetpred$X1)$metrics)
  nnetimp = cbind(nnetimp,importance(best_nnet1,trainSet,predictors,outcomeName,useModel=FALSE)[[1]])
  
}

#================================================================================
lrtfp_means = rowMeans(lrtfp)
lrtvp_means = rowMeans(lrtvp)
lrmetric_means = colMeans(lrmetric)
lrimp_means = rowMeans(lrimp, na.rm=TRUE)
lrres = list(data.frame(lrtfp_means,lrtvp_means), data.frame(lrmetric_means), 
             data.frame(lrimp_means), data.frame(lrmetric))
names(lrres) = c("roc","metric","importance","metric_data")
#================================================================================
carttfp_means = rowMeans(carttfp)
carttvp_means = rowMeans(carttvp)
cartmetric_means = colMeans(cartmetric)
cartimp_means = rowMeans(cartimp, na.rm=TRUE)
cartres = list(data.frame(carttfp_means,carttvp_means), data.frame(cartmetric_means), 
               data.frame(cartimp_means), data.frame(cartmetric))
names(cartres) = c("roc","metric","importance","metric_data")
#================================================================================
gbmtfp_means = rowMeans(gbmtfp)
gbmtvp_means = rowMeans(gbmtvp)
gbmmetric_means = colMeans(gbmmetric)
gbmimp_means = rowMeans(gbmimp, na.rm=TRUE)
gbmres = list(data.frame(gbmtfp_means,gbmtvp_means), data.frame(gbmmetric_means), 
              data.frame(gbmimp_means), data.frame(gbmmetric))
names(gbmres) = c("roc","metric","importance","metric_data")
#================================================================================
rf1tfp_means = rowMeans(rf1tfp)
rf1tvp_means = rowMeans(rf1tvp)
rf1metric_means = colMeans(rf1metric)
rf1imp_means = rowMeans(rf1imp, na.rm=TRUE)
rf1res = list(data.frame(rf1tfp_means,rf1tvp_means), data.frame(rf1metric_means), 
              data.frame(rf1imp_means), data.frame(rf1metric))
names(rf1res) = c("roc","metric","importance","metric_data")
#================================================================================
rf2tfp_means = rowMeans(rf2tfp)
rf2tvp_means = rowMeans(rf2tvp)
rf2metric_means = colMeans(rf2metric)
rf2imp_means = rowMeans(rf2imp, na.rm=TRUE)
rf2res = list(data.frame(rf2tfp_means,rf2tvp_means), data.frame(rf2metric_means), 
              data.frame(rf2imp_means), data.frame(rf2metric))
names(rf2res) = c("roc","metric","importance","metric_data")
#================================================================================
nnettfp_means = rowMeans(nnettfp)
nnettvp_means = rowMeans(nnettvp)
nnetmetric_means = colMeans(nnetmetric)
nnetimp_means = rowMeans(nnetimp, na.rm=TRUE)
nnetres = list(data.frame(nnettfp_means,nnettvp_means), data.frame(nnetmetric_means), 
               data.frame(nnetimp_means), data.frame(nnetmetric))
names(nnetres) = c("roc","metric","importance","metric_data")


#--------------
# Performances
#--------------

lrMetric = lrres$metric
lrAUC = aucMoyen(lrres$roc$lrtfp_means,lrres$roc$lrtvp_means)

cartMetric = cartres$metric
cartAUC = aucMoyen(cartres$roc$carttfp_means,cartres$roc$carttvp_means)

gbmMetric = gbmres$metric
gbmAUC = aucMoyen(gbmres$roc$gbmtfp_means,gbmres$roc$gbmtvp_means)

rf1Metric = rf1res$metric
rf1AUC = aucMoyen(rf1res$roc$rf1tfp_means,rf1res$roc$rf1tvp_means)

rf2Metric = rf2res$metric
rf2AUC = aucMoyen(rf2res$roc$rf2tfp_means,rf2res$roc$rf2tvp_means)

nnetMetric = nnetres$metric
nnetAUC = aucMoyen(nnetres$roc$nnettfp_means,nnetres$roc$nnettvp_means)


metricMat = cbind(lrMetric,cartMetric,gbmMetric,rf1Metric,rf2Metric,nnetMetric)
colnames(metricMat)=c("LR","CART","GBM","RF1","RF2","NN")

#-------------
# Importance
#-------------
impMat = cbind(lrimp_means,cartimp_means,gbmimp_means,rf1imp_means,rf2imp_means,nnetimp_means)
colnames(impMat) = c("LR","CART","GBM","RF1","RF2","NN")
rownames(impMat) = predictors


#--------------
# Courbes ROC
#--------------
plot(lrres$roc$lrtfp_means,lrres$roc$lrtvp_means, type="s", col="blue",lwd=2, 
     xlab="False Positive Rates", ylab="True Positive Rates",
     main = "Classifiers performance")
lines(cartres$roc$carttfp_means,cartres$roc$carttvp_means, col="cyan",lwd=2)
lines(gbmres$roc$gbmtfp_means,gbmres$roc$gbmtvp_means, col="green",lwd=2)
lines(rf1res$roc$rf1tfp_means,rf1res$roc$rf1tvp_means, col="red",lwd=2)
lines(rf2res$roc$rf2tfp_means,rf2res$roc$rf2tvp_means, col="black",lwd=2)
lines(nnetres$roc$nnettfp_means,nnetres$roc$nnettvp_means, col="purple",lwd=2)
abline(0,1)

legend("bottomright",c("LR","CART","GBM","RF1","RF2","NNET"),col=c("blue","cyan","green","red","black","purple"),lwd=2)
auc_values = c(lrMetric[3,],cartMetric[3,],gbmMetric[3,],rf1Metric[3,],rf2Metric[3,],nnetMetric[3,])
legend(0.5,0.5,c(sprintf(auc_values, fmt = '%#.3f')),col=c("blue","cyan","green","red","black","purple"),
       bty="n",lwd=2, title="AUC Values")

#---------------------------------------------------------------
# Fin du calcul parallele
stopCluster(cl)

