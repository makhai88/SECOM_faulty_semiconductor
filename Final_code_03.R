#SECOM Case study

#Required packages

if(!require("pacman")) install.packages("pacman")
library("pacman")
p_load("foreign")
p_load("ggplot2")
p_load("dplyr")
p_load("caret") 
p_load("scales") 
p_load("RCurl") 
p_load("Matrix")  
p_load("stats") 
p_load("reshape2") 
p_load("raster")
p_load("timeSeries")
p_load("VIM")
p_load("laeken")
p_load("mlbench")
p_load("FSelector")
p_load("Boruta")
p_load("ROSE")
p_load("DMwR")
p_load("smotefamily")
p_load("viridis")
p_load("randomForest")
p_load("caret")
p_load("MLmetrics")
p_load("ROCR")
p_load("haven")
#----------------------------------------------------------------


## Import SECOM dataset and load to secom.data


secom <- as.data.frame(read_sav("secom_mod.SAV"))
# secom<-read.spss("C:\\Users\\Sushil PC\\Desktop\\Second Sem\\Data Mining\\Case Study\\A03_Case_Study_SEMICONDUCTOR\\secom_mod.SAV",to.data.frame=TRUE)
secom.data <- secom[,4:ncol(secom)]

#--------------Analysis of Data

# The SECOM dataset has 1567 rows as instances and 590 columns representing 590 features 
dim(secom.data)
View(secom.data)

#--------------Analysis of Pass and Fail Data case

# Bar chart of Pass and Fail cases
#1472 Pass cases and 95 Fail cases
secom.bar.PF<-barplot(table(secom$class),names.arg = c("Pass","Fail"), 
                      ylim = c(0,2000), col = c("darkgoldenrod1","red"),
                      main = "Frequency of Pass and Fail")
text(secom.bar.PF,y = table(secom$class),labels = table(secom$class), pos = 3)
secom.bar.PF

#----------------------------------------------------------------
## Spliting Training and Test data set
#----------------------------------------------------------------

## Splitting the dataset into Training and Test datset(80%/20%)
set.seed(55)
secom.part<-createDataPartition(secom$class, times = 1,p = 0.8, list = FALSE)

#-------- Training Dataset
secom.training<-secom[secom.part,]

#-------- Test Dataset
secom.test<-secom[-secom.part,]

#-------- Training Data with only features
secom.train.data<-secom.training[,-c(1,2,3)]

#-------- Loading data to secom.data.1 for further processing
secom.data.1 <- secom.train.data


#-------- Test Data with only features
secom.test.data<-secom.test[,-c(1,2,3)]

#----------------------------------------------------------------

#-------- Check the ratio of Fail in Training and Test Dataset
Ratio <- nrow(secom[secom$class==1,])/nrow(secom)*100
Training_Ratio <- nrow(secom.training[secom.training$class==1,])/nrow(secom.training)*100
Test_Ratio <- nrow(secom.test[secom.test$class==1,])/nrow(secom.test)*100
Ratio
Training_Ratio
Test_Ratio


#--------- Plot for Pass and Fail for SECOM Dataset, Training and Test Dataset

DS <- c("secom","secom","Training","Training","Test","Test")
Status <- c("Pass","Fail","Pass","Fail","Pass","Fail")
case <- c(nrow(secom[secom$class==0,]),nrow(secom[secom$class==1,]),nrow(secom.training[secom.training$class==0,]),nrow(secom.training[secom.training$class==1,]),nrow(secom.test[secom.test$class==0,]),nrow(secom.test[secom.test$class==1,]))
secom.pf.1 <- data.frame(Dataset = DS, Pass_Fail = Status, Cases = case)

overall.PF.plot <- ggplot(data = secom.pf.1, aes(x = Dataset, y = Cases, fill = Pass_Fail)) + geom_bar(stat="identity") + 
  geom_text(aes(label = Cases), position = position_stack(vjust = 0.5))+
  theme_bw() + ggtitle("No. of Pass and Fail Cases")+
  theme(plot.title = element_text(hjust = 0.5,face = "bold"))
overall.PF.plot 


#----------------------------------------------------------------
# Data preparation
#----------------------------------------------------------------


## Feature removeval base on number of missing values 

# Columns containing more than 55% missing values removed
# Input: secom.data.1
# Output: secom.feature.rem1


#-------- count and percentage of NAs in columns

secom.NA.count.col<-colSums(is.na(secom.data.1))
secom.NA.perc.col<-(colMeans(is.na(secom.data.1)))*100
secom.NA.col<-data.frame(Feature = 1:ncol(secom.data.1), Percentage = round(secom.NA.perc.col,3), Frequency = secom.NA.count.col)

View(secom.NA.col)

#-------- Percentage of NA Cutoff value to remove features
secom.NA.cutoff<-55


#-------- Histogram of Percentage of NAs per feature

secom.hist.NA<-ggplot(secom.NA.col) +
  geom_histogram(aes(x = Percentage),boundary =0, binwidth = 10, fill="darkorange4", col = "blue") + 
  theme_bw()+
  labs(title = "Histogram of Percentage of NAs of Feature", y = "Number of Features", x = "Percentage of NA")+
  theme(plot.title = element_text(hjust = 0.5,face = "bold"),axis.text = element_text(color = "black")) +
  geom_vline(aes(xintercept= secom.NA.cutoff),color="springgreen4", size =1.1, linetype = "longdash")
secom.hist.NA


#-------- Removal of features with NAs greater than 55% from dataset
secom.feature.rem1<-secom.data.1[, ! secom.NA.perc.col>secom.NA.cutoff]
View(secom.feature.rem1)
ncol(secom.feature.rem1)


#-------- count and percentage of NA per row

# Input: secom.feature.rem1
# Output:secom.feature.rem1
secom.NA.count.row <-rowSums(is.na(secom.feature.rem1))
secom.NA.perc.row<-rowMeans((is.na(secom.feature.rem1)))*100
secom.NA.row <-data.frame(Percentage = round(secom.NA.perc.row,3), Frequency = secom.NA.count.row)
View(secom.NA.row)


#-------- Since no rows have 100% or majority as NAs, no instances were removed

ncol(secom.feature.rem1)
#-----------------------------------------------------------------

## 4.Remove features with near zero variance 
# Input: secom.feature.rem1
# Output: secom.feature.rem2

remove_cols <- nearZeroVar(secom.feature.rem1)
secom.feature.rem2 <- secom.feature.rem1[, -remove_cols]
View(secom.feature.rem2)
ncol(secom.feature.rem2)
secom.data.5 <- secom.feature.rem2

#-----------------------------------------------------------------

# ## 5.Feature removal through volatility check
# # Input: secom.feature.rem2
# # Output: secom.feature.rem3
# 
# 
# #-------- Coefficient of Variance 
# secom.cv<-apply(secom.feature.rem2, MARGIN = 2,cv, na.rm=TRUE)
# secom.cv <- secom.cv/100
# secom.cv.table<-data.frame(Feature = 1:ncol(secom.feature.rem2), Coeff_Variation = round(secom.cv,3))
# View(secom.cv.table)
# 
# #-------- Coeffient of Variation of features varies from -200 to 85 nearly
# secom.cv.plot<-ggplot(secom.cv.table,aes(x= Feature,y = Coeff_Variation)) +
#   geom_point(size = 2, colour = "brown3", fill = "darkmagenta", shape = 23, na.rm = TRUE) +
#   theme_bw() +
#   ylim(c(-200,100))+
#   labs(title = "CV Distribution of Features", x = "Features", y = "Coeff of Variation")+
#   theme(plot.title = element_text(hjust = 0.5,face = "bold"))
# secom.cv.plot
# 
# #-------- CV higher than absolute value of 0.01 can be considered to be of high variance and hence can be threshold value.
# cv.threshold<- 0.01
# 
# #-------- Histogram of coeffecient of variation of features
# secom.cv.hist <- ggplot(secom.cv.table, aes(x=abs(Coeff_Variation))) + 
#   geom_histogram(binwidth = 1,color="darkblue", fill="coral")+
#   theme_bw()+
#   xlim(c(0,40))+
#   ylim(c(0,150))+
#   labs(title = "Histogram of Coeffecient of Variaton", y = "No. of Features", x = "Absolute Value(Coeff of Variation)")+
#   theme(plot.title = element_text(hjust = 0.5,face = "bold"))
# secom.cv.hist
# 
# 
# #-------- SECOM dataset after removal of features with low-volatility
# secom.feature.rem3<-secom.feature.rem2[,which(abs(secom.cv)>cv.threshold)]
# 
# View(secom.feature.rem3)
# ncol(secom.feature.rem3)
# 
# #-------- Storing the dataset into new variable for further processing
# secom.data.5 <- secom.feature.rem3 


#-----------------------------------------------------------------

# Code for feature removal by correlation (Will be used if rebuilding of model is required)

# ## Feature removal through Correlation
# # Input: secom.feature.rem3
# # Output: secom.feature.rem4
# 
# #-------- correlation matrix
# 
# secom_corrm <- cor(secom.feature.rem3, use="pairwise.complete.obs")
# 
# #-------- Cutoff value for correlation cofficient
# corr_cutoff <- .90
# 
# #-------- Plot of correlation matrix
# library(corrplot)
# corrplot(secom_corrm[0:40,0:40], method="color")
# 
# #----------------------------------------------------------------
# library(reshape2)
# secom_corr2 <- round(secom_corrm, 2)
# diag(secom_corr2)<-0
# corr.pairwise <- melt(secom_corr2)
# 
# #-------- remove feature paires with NA in correlation
# corr.pairwise <- corr.pairwise[!is.na(corr.pairwise$value),]
# 
# #-------- Histogram of correlation cofficient
# 
# h <- as.data.frame(corr.pairwise)
# ggplot(corr.pairwise, aes(x= value)) + 
#   geom_histogram(color="black", fill="chartreuse3")+
#   labs(title="Histogram of Feature Correlations",x=" Correlation Cofficient", y = "Frequency")+
#   theme_classic()+ 
#   geom_vline(aes(xintercept= 0.90), color="blue", linetype="dashed", size=1)
# 
# #-------- find and  remove highly correlated features from the data set
# highCorr <- sum(abs(secom_corrm[upper.tri(secom_corrm)]) > corr_cutoff)
# highlyCorDescr <- findCorrelation(secom_corrm, cutoff = corr_cutoff)
# 
# secom.feature.rem4 <- secom.feature.rem3[,-(highlyCorDescr)]
# 
# #Storing the dataset into new variable for further processing
# secom.data.2 <- secom.feature.rem4


#-----------------------------------------------------------------
# Functions
#-----------------------------------------------------------------


#Function :: Outlier Identification and replace with NAs

secom_outlier <- function(secom_data_outlier){
  for (i in 1:ncol(secom_data_outlier)){
    u.bound <- mean(secom_data_outlier[,i], na.rm = TRUE) + (3*sd(secom_data_outlier[,i], na.rm = TRUE))
    l.bound <- mean(secom_data_outlier[,i], na.rm = TRUE) - (3*sd(secom_data_outlier[,i], na.rm = TRUE))
    secom_data_outlier[,i] = ifelse(secom_data_outlier[,i] > u.bound , u.bound, secom_data_outlier[,i])
    secom_data_outlier[,i] = ifelse(secom_data_outlier[,i] < l.bound , l.bound, secom_data_outlier[,i])
  }
  return(secom_data_outlier)
}


##Function :: KNN Impuatation
# set KNN parameter

secom_impute.knn <- function(secom_data_impute){
  secom.knn_dataset.1<-VIM::kNN(secom_data_impute,k=10,numFun = weightedMean)
  secom.knn_dataset <- subset(secom.knn_dataset.1,select = c(1:ncol(secom_data_impute)))
  return(secom.knn_dataset)
}

# #Function :: Mice Imputation
# 
# secom_impute.mice <- function(secom_data_impute){
#   data_miceimputation <- mice(secom_data_impute, m=1, maxit = 1, method = 'pmm',seed=500)
#   return(data_miceimputation)
# }
# 

## Functions for case balancing

##Function :: ROSE 

secom_balance.rose <- function(secom_data_balance){
  secom_rose<-ROSE(class~.,secom_data_balance,seed = 10)$data
  return(secom_rose)
}



#----------------------------------------------------------------

##Function :: SMOTE

# secom_balance.smote <- function(secom_data_balance){
#   set.seed(2000)
#   secom_smote <- DMwR::SMOTE(class ~ ., secom_data_balance, perc.over = 750, perc.under=115)
#   return(secom_smote)
# }
# 
# #----------------------------------------------------------------
# 
# ##Function :: ADASYN
# 
# #Performing ADASYN
# secom_balance.adasyn <- function(secom_data_balance){
#   class_feature <- subset(secom_data_balance, select=c("class"))
#   attribute_feature <- subset(secom_data_balance, select=-c(class))
#   secom_adasyn <- ADAS(attribute_feature,class_feature)$data
#   #Changing class as factor
#   secom_adasyn$class <- as.factor(secom_adasyn$class)
#   
#   #Moving the class column to first column
#   secom_adasyn <- secom_adasyn[,c(ncol(secom_adasyn),1:(ncol(secom_adasyn)-1))]
#   return(secom_adasyn)
# }


#-----------------------------------------------------------------
## Outlier identification and imputation
#-----------------------------------------------------------------

#-------- Outlier identification and replace

#Replacing outliers with NA Values
# Input: secom.data.5
# Output: secom.data.6
# Outlier function call
secom.data.6 <- secom_outlier(secom.data.5)

#-------- Boxplot of feature 10 as sample

boxplot(secom.feature.rem2$feature010, col = "mediumpurple1", xlab = "Feature 10", cex.main=1, outcex=1, outpch=21, outbg = "darkgreen", outcol = "darkgreen", boxcol = "forestgreen", main = "Boxplot of Feature 10",medcol = "red",whiskcol = "blue")
boxplot(secom.data.6$feature010, xlab = "Feature 10", col = "mediumpurple1",outcex=1, outpch=21, outbg = "red", boxcol = "forestgreen", main = "Boxplot after replacement with NA",medcol = "red",whiskcol = "indianred4")

summary(secom.feature.rem2$feature010)
summary(secom.data.6$feature010)



#--------KNN Imputation

# Input: secom.data.6
# Output: secom.data.7

# KNN function call
secom.data.7 <- secom_impute.knn(secom.data.6)

# checking for any missing values after replacement 
sum(is.na(secom.data.7))

## Adding back classfor feature selection methods

secom.data.7$class<-secom.training$class

#----------------------------------------------------------------
## Feature Selection
#----------------------------------------------------------------

#-------- Boruta Feature Selection method

# Input: secom.data.7
# Output: secom.selected.features_boruta


#Performing Boruta
set.seed(2000)
secom.Boruta <- Boruta(class~.,data = secom.data.7, doTrace = 2, maxRuns = 300)
plot(secom.Boruta)
secom.selected.features_boruta.names<-getSelectedAttributes(TentativeRoughFix(secom.Boruta))
print(secom.selected.features_boruta.names)

#Plot for Importance
plotImpHistory(secom.Boruta, colCode = c("green", "yellow", "red", "blue"), col = NULL,
               type = "l", lty = 1, pch = 0, xlab = "Classifier run")


#features for algorithm
secom.selected.features_boruta.alg <- as.simple.formula(secom.selected.features_boruta.names, "class")
print(secom.selected.features_boruta.alg)

#Save dataset with selected features
secom.selected.features_boruta <-secom.data.7[, c("class",secom.selected.features_boruta.names)]
secom.selected.features_boruta$class<-as.factor(secom.selected.features_boruta$class)
View(secom.selected.features_boruta)

#-------- Chi Square Feature selection method
# Input: secom.data.7
# Output: secom.selected.features_chisq


# chi square test
# weights_cs <- chi.squared(class~., secom.data.7)
# View(weights_cs)
# 
# #Selecting the cut off value based on high weightage of features
# chisq_cutoff <- 21
# secom.selected.features_chisq.names <- cutoff.k(weights_cs, chisq_cutoff)
# print(secom.selected.features_chisq.names)
# 
# #features for algorithm
# secom.selected.features_chisq.alg <- as.simple.formula(secom.selected.features_chisq.names, "class")
# print(secom.selected.features_chisq.alg)
# 
# 
# #Save dataset with selected features
# secom.selected.features_chisq <-secom.data.7[, c("class",secom.selected.features_chisq.names)]
# secom.selected.features_chisq$class<-as.factor(secom.selected.features_chisq$class)
# View(secom.selected.features_chisq)

#-------- Gain Ratio Feature selection method
# Input: secom.data.7
# Output: secom.selected.features_gn_ratio

# secom.data.7$class<-as.factor(secom.data.7$class)
# 
# #Checking features weight
# weights_gr <- gain.ratio(class~., secom.data.7)
# View(weights_gr)
# 
# #Selecting the cut off value based on high weightage of features
# gn.ratio_cutoff <- 20
# secom.selected.features_gn_ratio.names <- cutoff.k(weights_gr, gn.ratio_cutoff)
# print(secom.selected.features_gn_ratio.names)
# 
# #features for algorithm
# secom.selected.features_gn_ratio.alg <- as.simple.formula(secom.selected.features_gn_ratio.names, "class")
# print(secom.selected.features_gn_ratio.alg)
# 
# #Save dataset with selected features
# secom.selected.features_gn_ratio <-secom.data.7[, c("class",secom.selected.features_gn_ratio.names)]
# secom.selected.features_gn_ratio$class<-as.factor(secom.selected.features_gn_ratio$class)
# View(secom.selected.features_gn_ratio)


#----------------------------------------------------------------


# #== Bar chart for number of Features after each data processing method

# secom.ncols<-c(ncol(secom.train.data),ncol(secom.feature.rem1),ncol(secom.feature.rem2),ncol(secom.selected.features_chisq)-1,ncol(secom.selected.features_gn_ratio)-1,ncol(secom.selected.features_boruta)-1)
# secom.plot.ncols<-barplot(secom.ncols, names.arg = c("Original data","High NAs","Near 0 Variance", "Chi square", "Gain ratio", "Boruta"),ylab = "Number of features", col = viridis(6),ylim = c(0,650),main = "Number of Features Remaining", cex.main=1.2)
# text(secom.plot.ncols, y=secom.ncols, secom.ncols, pos = 3)


#----------------------------------------------------------------
## Case Balancing
#----------------------------------------------------------------

#ROSE Plots and Case table
secom.train.rose <- secom_balance.rose(secom.selected.features_boruta)
print(table(secom.train.rose$class))
prop.table(table(secom.train.rose$class))

print(table(secom.selected.features_boruta$class))
prop.table(table(secom.selected.features_boruta$class))

#Plot showing case changes after performing ROSE
plot(secom.selected.features_boruta$feature003, secom.selected.features_boruta$feature060, main="Before ROSE resampling",
     col=as.numeric(secom.selected.features_boruta$class), pch=20, xlab="feature 003", ylab="feature 060")
legend("topleft", c("Majority class","Minority class"), pch=20, col=1:2)

plot(secom.train.rose$feature003, secom.train.rose$feature060, main="After ROSE resampling",
     col=as.numeric(secom.train.rose$class), pch=20, xlab="feature 003", ylab="feature 060")
legend("topleft", c("Majority class","Minority class"), pch=20, col=1:2)

#----------------------------------------------------------------


#SMOTE Plots and Case table

# secom.train.smote <- secom_balance.smote(secom.selected.features_boruta)
# print(table(secom.train.smote$class))
# prop.table(table(secom.train.smote$class))
# 
# colors <- c("deeppink1", "seagreen4")
# colors <- colors[as.numeric(secom.train.smote$class)]
# plot(secom.train.smote$feature008,secom.train.smote$feature060,main="After SMOTE resampling",
#      col=colors, pch=20, cex = 1.2, xlab="feature 008", ylab="feature 060")
# legend("topleft", c("Majority class","Minority class"), pch=20, col=c("deeppink", "seagreen4"))
# 
# View(secom.train.smote)
# #----------------------------------------------------------------

# #ADASYN Plots and Case table
# 
# secom.train.adasyn <- secom_balance.adasyn(secom.selected.features_boruta)
# print(table(secom.train.adasyn$class))
# prop.table(table(secom.train.adasyn$class))
# View(secom.train.adasyn)
# 
# colors <- c("gold", "darkblue")
# colors <- colors[as.numeric(secom.train.adasyn$class)]
# plot(secom.train.adasyn$feature008,secom.train.adasyn$feature060,main="After ADASYN resampling",
#      col=colors, pch=20, xlab="feature 008", ylab="feature 060")
# legend("topleft", c("Majority class","Minority class"), pch=20, col=c("gold", "darkblue"))




#----------------------------------------------------------------
## TEST dataset preparation
#----------------------------------------------------------------

# Outlier removal with 3s boundry and replace with NAs

secom.test.prepare.1 <- secom_outlier(secom.test.data)

# KNN imputation

secom.test_impute <- secom_impute.knn(secom.test.prepare.1)

# checking for any missing values after replacement 
sum(is.na(secom.test_impute))

#Adding test class
secom.test.prepare.3 <- data.frame(class = secom.test$class, secom.test_impute)

#----------------------------------------------------------------
## Modeling
#----------------------------------------------------------------

#-------- Predict function
# Contain confusion matrix and AUC graphs

predict_function <- function(model,testset,threshold){
  random_predict<- predict(model,testset,type="prob")
  random_out<-data.frame(actual = secom.test.model$class%>%as.factor(), predict = random_predict)
  random_out$predict<-ifelse(random_out$predict.1 > threshold, 1, 0)%>%as.factor()
  #print(random_predict$predict)
  x <- confusionMatrix(random_out$predict,random_out$actual,positive = "0")
  print(x)
  cat("F1 score: ",MLmetrics::F1_Score(random_out$actual,random_out$predict),"\n")
  pre<-prediction(random_out$predict.1, random_out$actual)
  perf<-performance(pre,measure = "tpr", x.measure = "fpr")
  plot(perf, col="red")
  # AUC
  auc<-performance(pre, "auc")
  auc<-auc@y.values[[1]]
  print("Area under the curve (AUC)")
  print(auc)
  legend(x = 0.5, y= 0.3,legend = round(auc,3), title = "AUC")
  abline(0,1)
  return()
}

#----------------------------------------------------------------- 
# Random Forest - ROSE - BORUTA
#-----------------------------------------------------------------  

# Modeling & Predicting for Boruta and Rose

# Run ROSE case balancing on train dataset

secom.train.rose <- secom_balance.rose(secom.selected.features_boruta)

# Final train and Test datssets
secom.train.model <- secom.train.rose
secom.test.model <- secom.test.prepare.3

#-- Random Forest 
## Random forest Tuning parameter
# Set tuning parameters
mtry <- 5
ntree <-500
#Threshold for probility of predict function
# set threshold 
threshold_B_Rose <- .65

set.seed(2000)
rf_model <-randomForest(class~., data=secom.train.model,importance = TRUE, mtry =mtry ,ntree = ntree)
print(rf_model)

#-------- Predict

set.seed(2000)
predict_function(rf_model,secom.test.model,threshold_B_Rose)

#----------------------------------------------------------------- 
# Random Forest - SMOTE - BORUTA
#----------------------------------------------------------------- 

# # Modeling & Predicting for Boruta and SMOTE
# 
# # Run SMOTE case balancing on train dataset
# secom.train.smote <- secom_balance.smote(secom.selected.features_boruta)
# 
# # Final train and Test datssets
# secom.train.model <- secom.train.smote
# secom.test.model <- secom.test.prepare.3
# 
# #-- Random Forest 
# 
# ## Random forest Tuning parameters
# # Set tuning parameters
# mtry <- 3
# ntree <-500
# #Threshold for probility of predict function
# # set threshold 
# threshold_B_SMOTE <- .49
# 
# set.seed(2000)
# rf_model2 <-randomForest(class~., data=secom.train.model,importance = TRUE, mtry =mtry ,ntree = ntree)
# print(rf_model)
# 
# #-------- Predict
# 
# 
# predict_function(rf_model2,secom.test.model,threshold_B_SMOTE)

#----------------------------------------------------------------- 
# Random Forest - ADYSYN - BORUTA
#----------------------------------------------------------------- 
# Modeling & Predicting for Boruta and SMOTE
 
# Run ADYSYN case balancing on train dataset

# secom.train.adasyn <- secom_balance.adasyn(secom.selected.features_boruta)

# Final train and Test datssets
# secom.train.model <- secom.train.adasyn
# secom.test.model <- secom.test.prepare.3

# #-- Random Forest 
# 
# ## Random forest Tuning parameters
# # Set tuning parameters
# mtry <- 5
# ntree <-500
# #Threshold for probility of predict function
# # set threshold 
# threshold_B_ADYSYN <- .55

# set.seed(2000)
# rf_model3 <-randomForest(class~., data=secom.train.model,importance = TRUE, mtry =mtry ,ntree = ntree)
# print(rf_model)
# 
# #-------- Predict
# 
# 
# predict_function(rf_model3,secom.test.model,threshold_B_ADYSYN)

#----------------------------------------------------------------- 
# Naive Bayes - ROSE - BORUTA
#-----------------------------------------------------------------
# Modeling & Predicting for Boruta and Rose

# Run ROSE case balancing on train dataset

# secom.train.rose <- secom_balance.rose(secom.selected.features_boruta)
# 
# # Final train and Test datssets
# secom.train.model <- secom.train.rose
# secom.test.model <- secom.test.prepare.3
# 
# 
# # Naive Bayes function
# func_naiveBayes<- function(secom_data.train,secom_data.test,threshold){
#   set.seed(222)
#   x = secom_data.train[,-1]
#   y = secom_data.train$class
#   
#   nb_model <- train(x = x,
#                     y = y,
#                     method = "nb",
#                     trControl=trainControl(method='cv',number=10),
#                     preProc = c("BoxCox", "scale")
#                     )
#   print(nb_model)
#   random_predict<- predict(nb_model,secom_data.test,type="prob")#"raw"
#   
#   conf_pred<-data.frame(actual = secom_data.test$class%>%as.factor(),predict = random_predict)
#   conf_pred$predict<-ifelse(conf_pred$predict.1>threshold, 1, 0)%>%as.factor()
#   cat("F1 score: ",MLmetrics::F1_Score(conf_pred$actual,conf_pred$predict),"\n")
#   roc_over <- roc.curve(conf_pred$actual, conf_pred$predict.1, plotit = T, main ="ROC curve")
#   print(roc_over)
#   caret::confusionMatrix(conf_pred$predict,conf_pred$actual, positive = "0")
# }
# 
# 
# #Model building & predict
# func_naiveBayes(secom.train.model,secom.test.model,.65)
