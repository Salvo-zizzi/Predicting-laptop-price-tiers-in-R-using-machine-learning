
# setWD -----
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

options(scipen=999)
# df ----
df<-read.csv("data/laptops.csv",
              stringsAsFactors=TRUE, na.strings=c("NA","NaN", "", "N/A", "?"))
head(df)
summary(df)

# Missing ----
library(VIM)
na_pattern = aggr(df, numbers=TRUE, sortVars=TRUE, labels=names(df), cex.axis=.5, gap=3, ylab=c("Proportion of missingness","Missingness Pattern"))

library(funModeling)
library(dplyr)

# control if there are strange things
status=df_status(df, print_results = F)
head(status%>% arrange(type))
head(status%>% arrange(unique))
head(status%>% arrange(-p_na))

sapply(df, function(x)(sum(is.na(x))))

# Imputation ----

library(mice)
names(df)
a<-df[, c(1:22)]
a
tempData <- mice(df[, c(1:22)], m=5, maxit=20, meth='pmm', seed=500)

ls(tempData)

tempData$method
tempData$pred
tempData$nmis

ls(tempData$imp)

dim(tempData$imp$OpSys)
dim(tempData$imp$Memory)

head(tempData$imp$OpSys)
head(tempData$imp$Memory)

df_imputed <- complete(tempData,1)
dim(df_imputed)
names(df_imputed)

status_imp=df_status(df_imputed, print_results = F)
status_imp%>%arrange(-q_na)

all=cbind(df$Price ,df_imputed)
names(all)[names(all)=="df$Price"] <- "Price"
names(all)

## Check missing ----
status=df_status(all, print_results = F)
status%>%arrange(-q_na)

sapply(all, function(x)(sum(is.na(x))))

# Optimal grouping ----


library(dplyr)
library(factorMerger)

reduce_levels <- mergeFactors(response = all$Price, factor = all$cpu_name)
reduce_levels2 <- mergeFactors(response = all$Price, factor = all$Memory)
reduce_levels3 <- mergeFactors(response = all$Price, factor = all$gpu_name)
reduce_levels4 <- mergeFactors(response = all$Price, factor = all$Company)
reduce_levels5 <- mergeFactors(response = all$Price, factor = all$TypeName)
reduce_levels6 <- mergeFactors(response = all$Price, factor = all$cpu_brand)


## save reduced levels ----

og=cutTree(reduce_levels)
og1=cutTree(reduce_levels2)
og2=cutTree(reduce_levels3)
og3=cutTree(reduce_levels4)
og4=cutTree(reduce_levels5)
og5=cutTree(reduce_levels6)


## add to dataset ----

all$optimal_group=as.numeric(og)
all$optimal_group1=as.numeric(og1)
all$optimal_group2=as.numeric(og2)
all$optimal_group3=as.numeric(og3)
all$optimal_group4=as.numeric(og4)
all$optimal_group5=as.numeric(og5)



all$cpu_nameOG=as.factor(all$optimal_group)
all$MemoryOG=as.factor(all$optimal_group1)
all$gpu_nameOG=as.factor(all$optimal_group2)
all$CompanyOG=as.factor(all$optimal_group3)
all$TypeNameOG=as.factor(all$optimal_group4)
all$cpu_brandOG=as.factor(all$optimal_group5)

all$cpu_name<-NULL
all$Memory<-NULL
all$gpu_name<-NULL
#all$Weight_kg<-NULL
all$Company<-NULL
all$TypeName<-NULL
all$cpu_brand<-NULL
#all$cpu_speed<-NULL
#all$ssd<-NULL
#all$Ram<-NULL
#all$resolution_width<-NULL


all$optimal_group<-NULL
all$optimal_group1<-NULL
all$optimal_group2<-NULL
all$optimal_group3<-NULL
all$optimal_group4<-NULL
all$optimal_group5<-NULL

#Save workspace ----
save.image('MBOG.RData')

#Load workspace ----
load('MBOG.RData')

# Correlazioni ----
library(caret)
numeric<-unlist(lapply(all, is.numeric)) #per vedere quali variabili sono numeriche
x=all[,numeric]
R=cor(x)

correlatedPredictors = findCorrelation(R, cutoff = 0.95, names = TRUE)
correlatedPredictors

# Dopo devo droppare quella. 

df2=subset(all, select = -c(resolution_height))
df2$Price=as.factor(ifelse(df$Price>60000, "High", "Low"))
#df2$Price=ifelse(df$Price>60000, 1, 0)
head(df2)

table(df2$Price)
prop.table(table(df2$Price))
# NZV ----
nzv = nearZeroVar(df2, saveMetrics = TRUE)
nzv 
# train and test ----
library(caret)
set.seed(1234)
split <- createDataPartition(y=df2$Price, p = 0.70, list = FALSE)
train <- df2[split,]
test <- df2[-split,]

# 1. model selection? best preprocess?? ----
# Ogni modello lavora sul suo dataset (all covariates o quelle selezionate se richiede model selection)

##  tree model selection ----
set.seed(1234)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
rpartTuneCvA <- train(Price ~ ., data = train, method = "rpart",
                      tuneLength = 5, preProcess=c("corr", "nzv"),
                      trControl = cvCtrl)


rpartTuneCvA
getTrainPerf(rpartTuneCvA)

# var imp of the tree
plot(varImp(object=rpartTuneCvA),main="train tuned - Variable Importance")

## select only important variables ----
vimp=varImp(rpartTuneCvA)
vimp=data.frame(vimp[1])
vimp$var=row.names(vimp)
head(vimp)

# select covariate with imp>30% than most important
vimp2=vimp[vimp$Overall>30,]
head(vimp2)

# new train and test with selected covariates+ target
train2=train[,c("Ram", "MemoryOG", "ssd", "cpu_speed")]
train2=cbind(train$Price, train2)
head(train2)
names(train2)[1] <- "Price"
head(train2)

## next we will use train2 of the tree for models requiring mod selection ----


# 2 quale metrica usare per  i models? ----
metric="ROC"

## lanciamo tre modelli ad esempio ----
## fate tenfold validation k=10, qui con pochi dati k=5
## glm  on selected covariates ----

## glm ---
set.seed(1234)
glm=train(Price~. ,data=train2, method = "glm", preProcess=c("corr", "nzv"), 
    trControl = cvCtrl, tuneLength=5, trace=TRUE, metric=metric)
## knn ----
knnFit <- train(Price ~., data=train2,
                 method = "knn", tuneLength = 5,
                 preProcess = c("center", "scale", "corr", "nzv"),
                  metric=metric,
                 trControl = cvCtrl)
print(knnFit)

## forest ----
set.seed(1234)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
rf <- train(Price ~ ., data = train2, method = "rf",
                tuneLength = 5,
                trControl = cvCtrl)
rf
print(rf)

## pls ----
set.seed(1234)
pls=train(Price~.,data=train2 , method = "pls", preProcess=c("corr", "nzv"), 
          trControl = cvCtrl, tuneLength=5)
print(pls)
## lasso  ----
set.seed(1234)
grid = expand.grid(.alpha=1,.lambda=seq(0, 1, by = 0.01))
lasso=train(Price~.,data=train2 , method = "glmnet", family ="binomial",
                      trControl = cvCtrl, tuneLength=5, tuneGrid=grid, preProcess=c("corr", "nzv"), 
                      metric=metric)
print(lasso)

## ADA ----
set.seed(1234)
ada=train(Price~.,data=train2 , method = "adaboost", preProcess=c("corr", "nzv"), 
          trControl = cvCtrl, tuneLength=5)

print(ada)

## C5.0 ----

set.seed(1234)
C=train(Price~.,data=train2 , method = "C5.0", preProcess=c("corr", "nzv"), 
          trControl = cvCtrl, tuneLength=5)

print(C)

## NodeHarvest ----

set.seed(1234)
NH=train(Price~.,data=train2 , method = "nodeHarvest", preProcess=c("corr", "nzv"), 
        trControl = cvCtrl, tuneLength=5)

print(NH)

## PcaNNet ----

set.seed(1234)
PCAnn=train(Price~.,data=train2 , method = "pcaNNet", preProcess=c("scale", "corr", "nzv"), 
        trControl = cvCtrl, tuneLength=5)

print(PCAnn)

## ----

set.seed(1234)
GB=train(Price~.,data=train2 , method = "gbm", preProcess=c("scale", "corr", "nzv"), 
        trControl = cvCtrl, tuneLength=5)

print(GB)

## R with plot ----
library(rpart)
library(rpart.plot)
cv.caret <- rpart( Price~ ., data = train2, method = "class",  minsplit = 5, xval = 5)

rpart.plot(cv.caret, type = 4, extra = 1)

## compare results ----
getTrainPerf(rf)
getTrainPerf(glm)
getTrainPerf(pls)
getTrainPerf(lasso)
getTrainPerf(rpartTuneCvA)
getTrainPerf(knnFit)
getTrainPerf(ada)
getTrainPerf(C)
getTrainPerf(NH)
getTrainPerf(PCAnn)
getTrainPerf(GB)

trellis.par.set(caretTheme())
plot(rf, metric='ROC')
#plot(glm, metric='ROC')
plot(pls, metric='ROC')
plot(lasso, metric='ROC')
plot(rpartTuneCvA, metric='ROC')
plot(knnFit, metric='ROC')
plot(ada, metric='ROC')
plot(C, metric='ROC')
plot(NH, metric='ROC')
plot(PCAnn, metric='ROC')
plot(GB, metric='ROC')

results <- resamples(list(glm=glm, pls=pls, lasso=lasso, rf=rf, rpartTuneCvA=rpartTuneCvA, knn=knnFit, adaBoost=ada, C5.0=C, NodeHarvest=NH, PCANN=PCAnn, GradienBoosting=GB))
results
bwplot(results)

## test difference of accuracy using bonferroni adjustement ----
Diffs <- diff(results)
summary(Diffs)

## estimate probs P(M) ----
test$p1 = predict(glm       , test, "prob")[,1]
test$p2 = predict(pls         , test, "prob")[,1]
test$p3 = predict(lasso    , test, "prob")[,1]
test$p4 = predict(rf      , test, "prob")[,1]
test$p5 = predict(rpartTuneCvA, test, "prob")[,1]
test$p6 = predict(knnFit, test, "prob")[,1]
test$p7 = predict(ada, test, "prob")[,1]
test$p8 = predict(C, test, "prob")[,1]
test$p9 = predict(NH, test, "prob")[,1]
test$p10 = predict(PCAnn, test, "prob")[,1]
test$p11 = predict(GB, test, "prob")[,1]

## roc values ----
library(pROC)

r1=roc(Price ~ p1, data = test)
r2=roc(Price ~ p2, data = test)
r3=roc(Price ~ p3, data = test)
r4=roc(Price ~ p4, data = test)
r5=roc(Price ~ p5, data = test)
r6=roc(Price ~ p6, data = test)
r7=roc(Price ~ p7, data = test)
r8=roc(Price ~ p8, data = test)
r9=roc(Price ~ p9, data = test)
r10=roc(Price ~ p10, data = test)
r11=roc(Price ~ p11, data = test)


plot(r1)
plot(r2,add=T,col="red")
plot(r3,add=T,col="blue")
plot(r4,add=T,col="yellow")
plot(r5,add=T,col="violet")
plot(r6,add=T,col="green")
plot(r7, add=T, col="orange")
plot(r8, add=T, col="brown")
plot(r9, add=T, col="black")
plot(r10, add=T, col="gold")
plot(r11, add=T, col="pink")

r1
r2
r3
r4
r5
r6
r7
r8
r9
r10
r11

## lift values ----
require(caret)
lift_chart <- caret::lift(Price~p1+p2+p3+p4+p5+p6+p7+p8+p9+p10+p11 , class='He', data=test,cuts = 200)
lift_chart_plot<-ggplot(lift_chart) + labs(title="Competing models Lift Charts")+geom_point(x=20,y=60, colour="red")
lift_chart_plot

# 3a step: confusion matrix del/dei best model) ----

r10

## DO AT HAND THE predicted M....for best model ----

pred_y=ifelse(test$p8>0.5, "High","Low")
pred_y=as.factor(pred_y)
head(pred_y)

confusionMatrix(as.factor(pred_y), as.factor(test$Price), positive="High")

# 3b step: studio la soglia ----

y=test$Price
y=ifelse(y=="High",1,0)
predProbT=test$p10

library(ROCR)
predRoc <- prediction(predProbT,y)

## this step do nothing, useful only to have an ROCR object..giving us all metric when varying threshold

class(predRoc)

## roc curve ----

roc.perf = performance(predRoc, measure = "tpr", x.measure = "fpr")
plot(roc.perf)
abline(a=0, b=1)

# Plot tutti insieme ----

acc.perf = performance(predRoc, measure = "acc")
spec.perf = performance(predRoc, measure = "spec")
sens.perf = performance(predRoc, measure = "sens")
prec.perf = performance(predRoc, measure = "prec")
par(mfrow=c(2,2))
plot(acc.perf, main='Accuracy best model')
plot(spec.perf, main='Specificity best model')
plot(sens.perf, main='Sensitivity best model')
plot(prec.perf, main='Precision best model')

## see lift for curiosity: funmodeling model the higher class of target thus R ----

test$pR = predict(C, test, "prob")[,1]

library(funModeling)
gain_lift(data = test, score = 'pR', target = 'Price')

# 4 step ----
## newdata=scoredata simulato ----
newdata1=df2[sample(nrow(df2), size=(nrow(df2)*0.10)), ]
newdata=newdata1[, -1]
head(newdata)
newdata$prob = predict(C, newdata, "prob")
head(newdata$prob)
probM=newdata$prob[,1]
newdata$pred_y=ifelse(probM>0.6, "High","Low") # interessa la specificity
head(newdata)

table(newdata1$Price, newdata$pred_y)
confusionMatrix(as.factor(newdata1$Price), as.factor(newdata$pred_y))

## 4.1 ----
newdata1=df2[sample(nrow(df2), size=(nrow(df2)*0.10)), ]
newdata=newdata1[, -1]
newdata$prob = predict(rf, newdata, "prob")
head(newdata$prob)
probHE=newdata$prob[,1]
newdata$pred_y=ifelse(probHE>0.65, "High","Low")
head(newdata)
## Reti neurali ----

#y=ifelse(train2$Price=="High",1,0)

# do a MPL nnet package (classical tuning parm) ########
# do a MPL with classical tuning parm
library(nnet)
set.seed(1234)
mynet <- nnet(train2[,-1], y ,  entropy=T, size=3, decay=0.1, maxit=2000, trace=T)
mynet
# see architecture: MLP
library(NeuralNetTools)
plotnet(mynet, alpha=0.6)

# confusion matrix of training set
mynet.pred <- as.numeric(predict(mynet, train2[,-1], type='class'))
table(mynet.pred, y)
prop.table(table(mynet.pred,y))

# use caret for more metrics of confusion matrix (also for models not fitted with caret)

library(caret)
confusionMatrix(as.factor(mynet.pred),as.factor(y))

set.seed(7)
metric <- "Accuracy"
ctrl = trainControl(method="cv", number=10, search = "grid")
nnet_fit <- train(train2[-1], train2$Price,
                     method = "nnet",
                     preProcess = c("scale", "corr", "nzv"), 
                     metric=metric, trControl=ctrl,
                     trace = TRUE, # use true to see convergence
                     maxit = 300)

print(nnet_fit)
plot(nnet_fit)

confusionMatrix(nnet_fit)

# 1-5 HL -----

set.seed(7)
metric <- "Accuracy"
ctrl = trainControl(method="cv", number=10, search="grid")
tunegrid <- expand.grid(size=c(1:5), decay = c(0.001, 0.01, 0.05 , .1, .3))
nnet_grid <- train(train2[-1], train2$Price,
                         method = "nnet",
                         preProcess = c("scale", "corr", "nzv"), 
                         tuneLength = 10, metric=metric, trControl=ctrl, tuneGrid=tunegrid,
                         trace = TRUE,
                         maxit = 300)

print(nnet_grid)
plot(nnet_grid)  
confusionMatrix(nnet_grid)

## different size ----
set.seed(7)
metric <- "Accuracy"
ctrl = trainControl(method="cv", number=10, search="grid")
tunegrid <- expand.grid(size=c(1:15), decay = c(0.001, 0.01, 0.05 , .1, .3))
nnet_size <- train(train2[-1], train2$Price,
                   method = "nnet",
                   preProcess = c("scale", "corr", "nzv"), 
                   tuneLength = 10, metric=metric, trControl=ctrl, tuneGrid=tunegrid,
                   trace = TRUE,
                   maxit = 300)

print(nnet_size)
confusionMatrix(nnet_size)
plot(nnet_size)  

# Architecture ----

library(NeuralNetTools)
plotnet(nnet_fit)
plotnet(nnet_fit2, alpha=0.6)
plotnet(nnet_size, alpha=0.6)

## Confronto nn -----
cvValues <- resamples(list(Length5= nnet_fit, Grid_nnet=nnet_grid, Big_nnet=nnet_size))#,   default_length7=nnetFit_deflength)) #
summary(cvValues)
dotplot(cvValues)
# this is better
bwplot(cvValues, layout = c(1, 3))

## -Confronto roc
library(pROC)
test$nn1 = predict(nnet_fit, test, "prob")[,1]
test$nn2 = predict(nnet_grid, test, "prob")[,1]
test$nn3 = predict(nnet_size, test, "prob")[,1]

rn1=roc(Price ~ nn1, data = test)
rn2=roc(Price ~ nn2, data = test)
rn3=roc(Price ~ nn3, data = test)

rn1
rn2
rn3
# XAI ----
library(DALEX)
explainer  <- explain(nnet_size, data = train2)
sv  <- single_variable(explainer, variable = c("Ram", "ssd", "cpu_speed"),  type = "partial", prob = TRUE)
plot(sv)

## Prediction reti neurali ----

test$pnn=predict(nnet_size, test, "prob")[,1]

y=test$Price
y=ifelse(y=="High",1,0)
predProbT=test$pnn

library(ROCR)
predRoc <- prediction(predProbT,y)

# Plot tutti insieme ----

acc.perf = performance(predRoc, measure = "acc")
spec.perf = performance(predRoc, measure = "spec")
sens.perf = performance(predRoc, measure = "sens")
prec.perf = performance(predRoc, measure = "prec")
par(mfrow=c(2,2))
plot(acc.perf, main='Accuracy best model')
plot(spec.perf, main='Specificity best model')
plot(sens.perf, main='Sensitivity best model')
plot(prec.perf, main='Precision best model')


## newdata=scoredata simulato ----
newdata1=df2[sample(nrow(df2), size=(nrow(df2)*0.10)), ]
newdata=newdata1[, -1]
newdata$prob = predict(nnet_size, newdata, "prob")
probM=newdata$prob[,1]
newdata$pred_y=ifelse(probM>0.5, "High","Low")

confusionMatrix(as.factor(newdata1$Price), as.factor(newdata$pred_y))