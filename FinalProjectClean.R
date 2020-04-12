##### satisfaction clean

satisfaction <- read.csv("train.csv") 
str(satisfaction)
summary(satisfaction)
# libraries

library(dplyr)  # Used for data manupulation
library(caret) # Useed to remove highly correlated varables
library(ElemStatLearn) # used for the feature selection stage
library(rJava) # used for the feature selection stage
library(FSelector) # used for the feature selection stage
library(ROCR)# ROC curve 
library(rpart)  #tree
library(rpart.plot) # tree plot
library(ipred)# bagging
library(randomForest) # random forest
########################### Data cleaning

# remove duplicates

reduced_satisfaction <- distinct(satisfaction)
dim(reduced_satisfaction) # no duplicates

# get rid of ID variable since it's not useful for modeling

reduced_satisfaction <- reduced_satisfaction[,-1]

# some variables only have only 0's. Therefore exlude zero variance variables
# exclude zero variance variabels as they don't give us any info...

variance1 <- sapply(reduced_satisfaction, var)
eliminate <- which(variance1 == 0)
reduced_satisfaction <- reduced_satisfaction[,-eliminate]
dim(reduced_satisfaction)  # we removed 34 variables

# remove duplicated columns (corr = 1)

reduced_satisfaction <- reduced_satisfaction[,colnames(unique(as.matrix(reduced_satisfaction),MARGIN=2))] #
dim(reduced_satisfaction)# we just removed 29 variables

##### eliminate higly correlated variables (corr > 0.7)
# Given that we have so many variables that are highly cor. we want to avoid multicolienarity in our modeling

df2 = cor(reduced_satisfaction)
hc = findCorrelation(df2, cutoff=0.7) # putt any value as a "cutoff" 
hc = sort(hc)
reduced_satisfaction = reduced_satisfaction[,-c(hc)]
dim(reduced_satisfaction)  # we just removed 203 variables
?findCorrelation
# checking for missing values

sum(colSums(is.na(reduced_satisfaction)))  # no missing values

########## Abnormal values detection

# Abnormal positive
max_outliers <- sapply(reduced_satisfaction, max)  ## na.rm=TRUE
max(max_outliers) # they seem to be anormal values

# what variables are the abnormal values located in 
which(max_outliers >= 999999999 )

x2 <- which(reduced_satisfaction$delta_imp_aport_var33_1y3 >= 999999999 )
x3 <- which(reduced_satisfaction$delta_imp_compra_var44_1y3 >= 999999999 )
x6 <- which(reduced_satisfaction$delta_imp_venta_var44_1y3 >= 999999999 )

reduced_satisfaction[x2,"delta_imp_aport_var33_1y3"]=NA
reduced_satisfaction[x3,"delta_imp_compra_var44_1y3"]=NA
reduced_satisfaction[x6,"delta_imp_venta_var44_1y3"]=NA

# as seen below the maximm values of these variables have gone from 1e10 to 6.2
summary(reduced_satisfaction$delta_imp_aport_var33_1y3)
summary(reduced_satisfaction$delta_imp_compra_var44_1y3)
summary(reduced_satisfaction$delta_imp_venta_var44_1y3)

# we don't have abnormal values anymore
max_outliers <- sapply(reduced_satisfaction, max, na.rm=TRUE)  ## na.rm=TRUE
max(max_outliers)


## Abnormal negative
min_outliers <-sapply(reduced_satisfaction, min , na.rm=TRUE)
minimum_value <- min(min_outliers)
which(min_outliers <= minimum_value ) # the var var3 has  candidate abnormal values
# based on the boxplot seems to be abnormal
boxplot(reduced_satisfaction$var3)

# impute the abnormal values for missing values
y1 <- which(reduced_satisfaction$var3 == min(reduced_satisfaction$var3))
reduced_satisfaction[y1,"var3"]=median(reduced_satisfaction$var3)
summary(reduced_satisfaction$var3)

# the new minimum seems to be normal now, so we don't have any other abnormal values
min_outliers <-sapply(reduced_satisfaction, min , na.rm=TRUE)
min(min_outliers)

#############################

######### feature selection: Done to to help cut down on runtime and eliminate unecessary features prior to building a prediction model.

# for this step I am going to use Fselector package to get the 30 most influential variables (from the 104 that we currently have) on the satisfaction level of the customers
# https://miningthedetails.com/blog/r/fselector/
### 3 means that code has been indented to make sure we don't lose computational time

###typeof(reduced_satisfaction[,104])  # the response variable is an integer and we change it to factor just to do the feature selection
###reduced_satisfaction$TARGET <- as.factor(reduced_satisfaction$TARGET)
###att.scores <- random.forest.importance(TARGET ~ ., reduced_satisfaction)

###x <- head(arrange(att.scores,desc(attr_importance)), n = 30)

# Approach selected: select the 30 most important variables forthe modeling stage
###selected_variables <- cutoff.k(att.scores, k = 30)

# final dataset
###satisfaction_final <- reduced_satisfaction[,selected_variables] # we need to add TARGET at the end


hardcoded_variables <- c("num_var45_hace3","saldo_var8","saldo_medio_var5_hace2", "num_var22_hace3", "num_var45_ult1", "num_meses_var39_vig_ult3", "num_op_var41_hace2","imp_op_var41_comer_ult1","num_var43_recib_ult1", "var36","imp_op_var41_efect_ult3","ind_var43_recib_ult1", "saldo_var37", "num_var22_hace2","var15" ,"num_var37_med_ult2","num_meses_var5_ult3", "saldo_medio_var8_hace2", "num_var24_0","saldo_medio_var12_hace2" ,"num_var42_0","saldo_var5",  "num_var5_0" , "saldo_medio_var5_hace3", "num_op_var41_efect_ult1", "saldo_medio_var13_corto_hace2" , "imp_aport_var13_hace3" ,"var3","saldo_var25" ,  "saldo_medio_var13_corto_hace3", "TARGET"  ) 
clean_data <- reduced_satisfaction[,hardcoded_variables]
View(clean_data)

##### from here, before modeling, we could try to do some EDA

pie(clean_data$TARGET)
barplot(clean_data$TARGET)
a <- c(73012,3008)
pie(a)

###### train test

set.seed(123)
index <- sample(nrow(clean_data),nrow(clean_data)*0.8)
satisfaction.train <- clean_data[index,]
satisfaction.test <- clean_data[-index,]

sum(colSums(is.na(clean_data)))
############################################### modeling


## logistic regression

satisfaction.glm<- glm(TARGET~., family=binomial, data=satisfaction.train)
summary(satisfaction.glm)

#doing the following takes almost 5 minutes, so I have hardcoded below the variabels that this model takes
##fullmodel <- satisfaction.glm
##nullmodel <- glm(TARGET~1, family=binomial, data=satisfaction.train)
##model.step.s<- step(nullmodel, scope=list(lower=nullmodel, upper=fullmodel), direction='both')
##summary(model.step.s)

# hardcoded model using automatic stepwise...
log_finalmodel <- glm(TARGET~num_meses_var5_ult3 + var15 + num_var24_0 + imp_aport_var13_hace3+var3 + saldo_medio_var5_hace2 + num_op_var41_efect_ult1+ saldo_medio_var8_hace2 + num_var22_hace2 + num_var22_hace3+ var36 + saldo_medio_var12_hace2 + num_var43_recib_ult1+num_var45_ult1 + saldo_medio_var13_corto_hace2 + saldo_var5+imp_op_var41_comer_ult1 + num_var42_0 + imp_op_var41_efect_ult3+saldo_var8 + saldo_medio_var13_corto_hace3 + saldo_var37+ num_var45_hace3, family = binomial, data = satisfaction.train)
summary(log_finalmodel)


#### in sample roc curve. 
#in sample prediction...less important
pred.glm<- predict(log_finalmodel, type="response")
library(ROCR)
pred <- prediction(pred.glm, satisfaction.train$TARGET) # this prediction function is different form the predict above. this function basically calculates many confusion matrices with different cut-off probability
perf <- performance(pred, "tpr", "fpr")  # calculates tpr and fpr 
plot(perf, colorize=TRUE)  # each point represents to a different p-cut
unlist(slot(performance(pred, "auc"), "y.values")) #AUC we get 0.78

#### out of sample ROC curve
pred.glm.test<- predict(log_finalmodel, newdata = satisfaction.test, type="response")
library(ROCR)
pred1 <- prediction(pred.glm.test, satisfaction.test$TARGET) # this prediction function is different form the predict above. this function basically calculates many confusion matrices with different cut-off probability
perf1 <- performance(pred1, "tpr", "fpr")  # calculates tpr and fpr 
plot(perf1, colorize=TRUE)  # each point represents to a different p-cut
unlist(slot(performance(pred1, "auc"), "y.values"))  #AUC of 0.78

# precision-recall

# in sample pr curve
pred.glm<- predict(log_finalmodel, type="response")


# in sample sample pr curve
library("PRROC")
score1 <- pred.glm[satisfaction.train$TARGET==1]
score0 <- pred.glm[satisfaction.train$TARGET==0]
pr <- pr.curve(score1, score0, curve = T)
pr
plot(pr)

# out of sample PR curve


pred.glm.test<- predict(log_finalmodel, newdata = satisfaction.test, type="response")

library("PRROC")
score1.test <- pred.glm.test[satisfaction.test$TARGET==1]
score0.test <- pred.glm.test[satisfaction.test$TARGET==0]
pr.test <- pr.curve(score1.test, score0.test, curve = T)
pr.test
plot(pr.test)




############# p cut
# define a cost function with input "obs" being observed response 
# and "pi" being predicted probability, and "pcut" being the threshold.
costfunc = function(obs, pred.p, pcut){   # if the cost is symmetric jsut assign 1/2 and 1/2 and then the cost should be equal to the MR
  weight1 = 10/11   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1/11    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function


# Next, define a sequence from 0.01 to 1 by 0.01
p.seq = seq(0.01, 1, 0.01) 

#Then, you need to calculate the cost (as you defined before) for each probability in the sequence p.seq.
# write a loop for all p-cut to see which one provides the smallest cost
# first, need to define a 0 vector in order to save the value of cost from all pcut
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = satisfaction.train$TARGET, pred.p = pred.glm, pcut = p.seq[i])  
} # end of the loop


#Last, draw a plot with cost against p.seq, and find the p-cut that gives you the minimum cost.
# draw a plot with X axis being all pcut and Y axis being associated cost
plot(p.seq, cost)



# find the optimal pcut
optimal.pcut.glm = p.seq[which(cost==min(cost))]

##### for training sample
# step 1. get binary classification
class.glm<- (pred.glm>optimal.pcut.glm)*1   # to see waht observation belong to default or non default
# step 2. get confusion matrix, MR, FPR, FNR
table(satisfaction.train$TARGET, class.glm, dnn = c("True", "Predicted"))

# (equal-weighted) misclassification rate
MR<- mean(satisfaction.train$TARGET!=class.glm)
FPR<- sum(satisfaction.train$TARGET==0 & class.glm==1)/sum(satisfaction.train$TARGET==0)
FNR<- sum(satisfaction.train$TARGET==1 & class.glm==0)/sum(satisfaction.train$TARGET==1)
cost<- costfunc(obs = satisfaction.train$TARGET, pred.p = pred.glm, pcut = optimal.pcut.glm) # this value can be seen in the plot

##### for testing sample
class.glm.test<- (pred.glm.test>optimal.pcut.glm)*1   # to see waht observation belong to default or non default
# step 2. get confusion matrix, MR, FPR, FNR
table(satisfaction.test$TARGET, class.glm.test, dnn = c("True", "Predicted"))

# (equal-weighted) misclassification rate
MR<- mean(satisfaction.test$TARGET!=class.glm.test)
FPR<- sum(satisfaction.test$TARGET==0 & class.glm.test==1)/sum(satisfaction.test$TARGET==0)
FNR<- sum(satisfaction.test$TARGET==1 & class.glm.test==0)/sum(satisfaction.test$TARGET==1)
cost<- costfunc(obs = satisfaction.test$TARGET, pred.p = pred.glm.test, pcut = optimal.pcut.glm)




################## Classification tree

library(rpart)  #tree
library(rpart.plot)

#start with initial big classification tree to prune it

## largetree with assymetric cost
largetree.satisfaction <- rpart(formula = TARGET ~ ., data = satisfaction.train, method="class",
                        parms= list(loss=matrix(c(0,10/11,1/11,0))), cp = 0.001) 
prp(largetree.satisfaction, digits = 4, extra = 1)

# prune the tree
#cp plot
plotcp(largetree.satisfaction)
#cp table
printcp(largetree.satisfaction)

#optimal cp
opt.cp <- largetree.satisfaction$cptable[which(largetree.satisfaction$cptable[,2]==7),1]

# pruned tree
opttree.satisfaction <- rpart(formula = TARGET ~ ., data = satisfaction.train, method="class",
                                parms= list(loss=matrix(c(0,10/11,1/11,0))), cp = opt.cp) 
prp(opttree.satisfaction, digits = 4, extra = 1)

# in sample prediction
satisfaction.class.tree <- predict(opttree.satisfaction, type="class")
table(satisfaction.train$TARGET, satisfaction.class.tree, dnn=c("True", "False"))
MR<- mean(satisfaction.train$TARGET!=satisfaction.class.tree)

# out of sample prediction
satisfaction.class.tree1 <- predict(opttree.satisfaction, newdata =satisfaction.test,  type="class")
table(satisfaction.test$TARGET, satisfaction.class.tree1, dnn=c("True", "False"))
MR<- mean(satisfaction.test$TARGET!=satisfaction.class.tree1)
# cost
cost <- function(r, pi){
  weight1 = 10/11
  weight0 = 1/11
  c1 = (r==1)&(pi==0) #logical vector - true if actual 1 but predict 0
  c0 = (r==0)&(pi==1) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0))
}

# in sample cost
cost(satisfaction.train$TARGET,satisfaction.class.tree)
# out of sample cost
cost(satisfaction.test$TARGET,satisfaction.class.tree1)

#####ROC curves

## in sample

satisfaction.train.prob.rpart<- predict(opttree.satisfaction,satisfaction.train, type="prob")

library(ROCR)
pred = prediction(satisfaction.train.prob.rpart[,2], satisfaction.train$TARGET) 
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]


# out of smaple
satisfaction.test.prob.rpart<- predict(opttree.satisfaction,satisfaction.test, type="prob")

library(ROCR)
pred = prediction(satisfaction.test.prob.rpart[,2], satisfaction.test$TARGET) 
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]




######################     BAgging
# doesn't take assymetric costs....
library(ipred)

# takes like 2 mintes to run this
satisfaction.train$TARGET<- as.factor(satisfaction.train$TARGET)
satisfaction.bag<- bagging(TARGET~., data = satisfaction.train, nbagg=100)

satisfaction.bag.pred.train<- predict(satisfaction.bag, newdata = satisfaction.train, type="prob")[,2]
satisfaction.bag.pred<- predict(satisfaction.bag, newdata = satisfaction.test, type="prob")[,2]

costfunc = function(obs, pred.p, pcut){
  weight1 = 10/11   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1/11    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} 
p.seq = seq(0.01, 0.5, 0.01)
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = satisfaction.test$TARGET, pred.p = satisfaction.bag.pred, pcut = p.seq[i])  
}
plot(p.seq, cost)

# optimal pcut
optimal.pcut = p.seq[which(cost==min(cost))]

satisfaction.bag.class.train<- (satisfaction.bag.pred.train>optimal.pcut)*1
satisfaction.bag.class<- (satisfaction.bag.pred>optimal.pcut)*1

# in sample
table(satisfaction.train$TARGET, satisfaction.bag.class.train, dnn = c("True", "Pred"))
MR<- mean(satisfaction.train$TARGET!=satisfaction.bag.class.train)
cost(satisfaction.train$TARGET,satisfaction.bag.class.train)

# out of sample
table(satisfaction.test$TARGET, satisfaction.bag.class, dnn = c("True", "Pred"))
MR<- mean(satisfaction.test$TARGET!=satisfaction.bag.class)
cost(satisfaction.test$TARGET,satisfaction.bag.class)

#### ROC curves

# in sample
# couple of minutes to run
satisfaction.bag.pred.train<- predict(satisfaction.bag, newdata = satisfaction.train, type="prob")

library(ROCR)
pred = prediction(satisfaction.bag.pred.train[,2], satisfaction.train$TARGET) 
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]

# out of sample
satisfaction.bag.pred<- predict(satisfaction.bag, newdata = satisfaction.test, type="prob")

library(ROCR)
pred = prediction(satisfaction.bag.pred[,2], satisfaction.test$TARGET) 
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]



########################### Random forest

library(randomForest)
#### takes some time to run like 2-3 minutes..rememebr that radnom forest doesnt' support asymetric loss
satisfaction.rf<- randomForest(TARGET~., data = satisfaction.train , ntree = 100)
satisfaction.rf
?randomForest

plot(satisfaction.rf)
legend("right", legend = c("OOB Error", "FPR", "FNR"), lty = c(1,2,3), col = c("black", "red", "green"))

satisfaction.rf.pred<- predict(satisfaction.rf, type = "prob")[,2]

costfunc = function(obs, pred.p, pcut){
  weight1 = 10/11   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1/11   # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} 
p.seq = seq(0.01, 0.5, 0.01)
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = satisfaction.train$TARGET, pred.p = satisfaction.rf.pred, pcut = p.seq[i])  
}
plot(p.seq, cost)

optimal.pcut = p.seq[which(cost==min(cost))]

satisfaction.rf.pred<- predict(satisfaction.rf, type = "prob")[,2]
satisfaction.rf.pred.test<- predict(satisfaction.rf,newdata = satisfaction.test, type = "prob")[,2]

# in sample
satisfaction.rf.class<- (satisfaction.rf.pred>optimal.pcut)*1
table(satisfaction.train$TARGET, satisfaction.rf.class, dnn = c("True", "Pred"))
MR<- mean(satisfaction.train$TARGET!=satisfaction.rf.class)
cost(satisfaction.train$TARGET,satisfaction.rf.class)

# out of sample
satisfaction.rf.class.test<- (satisfaction.rf.pred.test>optimal.pcut)*1
table(satisfaction.test$TARGET, satisfaction.rf.class.test, dnn = c("True", "Pred"))
MR<- mean(satisfaction.test$TARGET!=satisfaction.rf.class)
cost(satisfaction.test$TARGET,satisfaction.rf.class.test)

#### ROC curves

# in sample
# couple of minutes to run

satisfaction.rf.pred<- predict(satisfaction.rf, type = "prob")

library(ROCR)
pred = prediction(satisfaction.rf.pred[,2], satisfaction.train$TARGET) 
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]

# out of sample

satisfaction.rf.pred.test<- predict(satisfaction.rf,newdata = satisfaction.test, type = "prob")

library(ROCR)
pred = prediction(satisfaction.rf.pred.test[,2], satisfaction.test$TARGET) 
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

slot(performance(pred, "auc"), "y.values")[[1]]

############### boosting


install.packages("adabag")
library(adabag)

adaboost<-boosting(TARGET~., data=satisfaction.train, boos=TRUE, mfinal=20,coeflearn='Breiman')
summary(adaboost)
adaboost$trees
adaboost$weights
adaboost$importance
a <- errorevol(adaboost,satisfaction.train)
b <- predict(adaboost,satisfaction.train)
t1<-adaboost$trees[[1]]
library(tree)
plot(t1)
text(t1,pretty=0)











############################# GAM


str(clean_data)
summary(clean_data)
clean.data <- clean_data


clean.data$num_meses_var39_vig_ult3<- as.factor(clean.data$num_meses_var39_vig_ult3)
clean.data$ind_var43_recib_ult1<- as.factor(clean.data$ind_var43_recib_ult1)
clean.data$num_meses_var5_ult3<- as.factor(clean.data$num_meses_var5_ult3)



library(mgcv)
satisfaction.gam <- gam(formula =TARGET~num_var45_hace3+saldo_var8+saldo_medio_var5_hace2+num_var45_ult1+s(num_meses_var39_vig_ult3)+num_op_var41_hace2+imp_op_var41_comer_ult1+num_var43_recib_ult1+var36+imp_op_var41_efect_ult3+s(ind_var43_recib_ult1)+saldo_var37+ num_var22_hace2+var15+num_var37_med_ult2+s(num_meses_var5_ult3)+saldo_medio_var8_hace2+num_var24_0+saldo_medio_var12_hace2+num_var42_0+saldo_var5 +num_var5_0+saldo_medio_var5_hace3+num_op_var41_efect_ult1+saldo_medio_var13_corto_hace2+imp_aport_var13_hace3+var3+saldo_var25+saldo_medio_var13_corto_hace3, family = binomial, data = clean.data)
summary(satisfaction.gam)




## discriminant analysis
??lda
library(MASS)
satisfaction.train$TARGET = as.factor(satisfaction.train$TARGET)
satisfaction.lda <- lda(TARGET ~ ., data = satisfaction.train)
prob.lda.in <- predict(satisfaction.lda, data = satisfaction.train)

costfunc = function(obs, pred.p, pcut){   # if the cost is symmetric jsut assign 1/2 and 1/2 and then the cost should be equal to the MR
  weight1 = 10/11   # define the weight for "true=1 but pred=0" (FN)
  weight0 = 1/11    # define the weight for "true=0 but pred=1" (FP)
  c1 = (obs==1)&(pred.p<pcut)    # count for "true=1 but pred=0"   (FN)
  c0 = (obs==0)&(pred.p>=pcut)   # count for "true=0 but pred=1"   (FP)
  cost = mean(weight1*c1 + weight0*c0)  # misclassification with weight
  return(cost) # you have to return to a value when you write R functions
} # end of the function


# Next, define a sequence from 0.01 to 1 by 0.01
p.seq = seq(0.01, 1, 0.01) 

#Then, you need to calculate the cost (as you defined before) for each probability in the sequence p.seq.
# write a loop for all p-cut to see which one provides the smallest cost
# first, need to define a 0 vector in order to save the value of cost from all pcut
cost = rep(0, length(p.seq))  
for(i in 1:length(p.seq)){ 
  cost[i] = costfunc(obs = satisfaction.train$TARGET, pred.p = prob.lda.in$posterior[, 2], pcut = p.seq[i])  
} # end of the loop


#Last, draw a plot with cost against p.seq, and find the p-cut that gives you the minimum cost.
# draw a plot with X axis being all pcut and Y axis being associated cost
plot(p.seq, cost)

optimal.pcut.lda = p.seq[which(cost==min(cost))]




# in sample performance
pred.lda.in <- (prob.lda.in$posterior[, 2] >= optimal.pcut.lda) * 1
table(satisfaction.train$TARGET, pred.lda.in, dnn = c("Obs", "Pred"))
class.lda<- (pred.lda.in>optimal.pcut.lda)*1 

MR<- mean(satisfaction.train$TARGET!=class.lda)
cost<- costfunc(obs = satisfaction.train$TARGET, pred.p = pred.lda.in, pcut = optimal.pcut.lda)

## out of sample performance
prob.lda.in.test <- predict(satisfaction.lda, newdata = satisfaction.test)
pred.lda.in.test <- (prob.lda.in.test$posterior[, 2] >= optimal.pcut.lda) * 1
table(satisfaction.test$TARGET, pred.lda.in.test, dnn = c("Obs", "Pred"))
class.lda.test<- (pred.lda.in.test>optimal.pcut.lda)*1 

MR<- mean(satisfaction.test$TARGET!=class.lda.test)
cost<- costfunc(obs = satisfaction.test$TARGET, pred.p = pred.lda.in.test, pcut = optimal.pcut.lda)



## neural network




###################
# need to find a performance metric: By having MR in an ibalance data, we can just do all 0's so we have only 4% of MR.

# one big problem that we have with the supervised learning methods is that the are sensitive to the unbalanced data (since we only have a 3.95% of satisfied customers)


