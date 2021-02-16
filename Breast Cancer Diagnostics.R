library(glmnet)
library(spls)
library(leaps)
library(MASS)

library(ISLR)
library(corrplot)
library(ggplot2)


library(mvtnorm)
library(e1071)
library(splines)
library(xtable)
library(class)

library(bestglm)
library(gglasso)
library(msgl)
library(neuralnet)

library(tidyverse)

library(corrplot)

# Source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

data <- read.csv("C:/Users/DELL/Desktop/Breast Cancer Diagnostics/data/data.csv")
head(data)
dim(data)

y <- vector(length = nrow(data))

for (i in 1:length(y)) {
  if (data$diagnosis[i] == 'M') {
    y[i] = 1
  }
  else {
    y[i] = 0
  }
}

data$y <- y

data <- data[-c(1, 2, ncol(data)-1)]

head(data)

n <- nrow(data)

set.seed(1)
testIndex <- sample(1:n, floor(n/2), replace = FALSE)

testData <- data[testIndex,]
trainData <- data[-testIndex,]


#Standardize THE VARIABLES

sd <- vector(length = ncol(trainData) - 1)
mean <- vector(length = ncol(trainData) - 1)
for (i in 1:(ncol(trainData) - 1)) {
  x <- trainData[,i]
  sd[i] <- sd(x)
  mean[i] <- mean(x)
  
  trainData[,i] <- ((x - mean[i]) / sd[i])
}



for (i in 1:(ncol(testData) - 1)) {
  x <- testData[,i]
  
  testData[,i] <- ((x - mean[i]) / sd[i])
}


########################################
######## PLOTTING CORRELATION ##########
########################################
dataOrd <- data[,1:30]
corr <- cor(dataOrd)
corrplot(corr, method = "color")

##################################################
############ LOGISTIC REGRESSION #################
##################################################



modLog <- glm(y ~ ., data = trainData, family = binomial)

summary(modLog)
#summary(modLog)$coefficients[1]

predLog <- 1*(predict(modLog, newdata = testData, type = "response") > 0.5)
mean(predLog != testData$y)



##################################################
############ ELASTIC NET #########################
##################################################


set.seed(5)

alpha.vec <- seq(0, 1, by = 0.1)


trainDataMat <- data.matrix(trainData)

testDataMat <- data.matrix(testData)

x <- trainDataMat[,1:(ncol(trainDataMat)-1)]
y <- trainData$y

xtest <- testDataMat[,1:(ncol(testDataMat)-1)]
ytest <- testData$y

nTest <- nrow(testData)


K <- 10
randOrder = sample(1:nrow(x), nrow(x), replace=FALSE)

kGroups = list()
for (k in 1 : K) {
  kGroups[[k]] = ((k-1)*nrow(x)/10 + 1) : (k*nrow(x)/10)
}

errorMat = matrix(NA, K, length(alpha.vec))

trainDataMat <- data.matrix(trainData)

testDataMat <- data.matrix(testData)

for (k in 1 : K) {
  ## Split the data into training/testing
  testIndex = randOrder[kGroups[[k]]]
  
  trainingX = x[-testIndex,]
  testX = x[testIndex,]
  
  trainingY = y[-testIndex]
  testY = y[testIndex]
  
  for (kk in 1:length(alpha.vec)) {
    modEN <- cv.glmnet(x = trainingX, y = trainingY, alpha = alpha.vec[kk], family = "binomial", intercept = TRUE)
    
    
    probEN<- predict(modEN, newx = testX, type = "response")
    
    predEN <- rep(0, nrow(testX))
    predEN[probEN > 0.5] <- 1
    
    
    #predEN <- predict(modEN, newx = testX)
    
    errorRate <- mean(predEN != testY)
    
    errorMat[k, kk] <- errorRate
  }
  
}

avg.error.rates <- vector(length = ncol(errorMat))

for (i in 1:ncol(errorMat)) {
  avg.error.rates[i] <- mean(errorMat[,i])
}

plot(alpha.vec, avg.error.rates, type = "l", ylab = "Mean Squared Error", xlab = "alpha", axes = TRUE)

opt.index <- which.min(avg.error.rates)
opt.alpha <- alpha.vec[opt.index]

modEN <- cv.glmnet(x = x, y = y, alpha = opt.alpha, family = "binomial")


summary(modEN)

coef(modEN, s="lambda.min")


probEN <- predict(modEN, newx = testDataMat[,1:(ncol(testDataMat)-1)], type = "response")

predictEN <- rep(0, nrow(testData))
predictEN[probEN > 0.5] <- 1

mean(predictEN != testData$y)


##################################################
############## LASSO #############################
##################################################


modEN <- cv.glmnet(x = x, y = y, alpha = 1, family = "binomial")


summary(modEN)

coef(modEN, s="lambda.min")

probEN <- predict(modEN, newx = testDataMat[,1:(ncol(testDataMat)-1)], type = "response")

predictEN <- rep(0, nrow(testData))
predictEN[probEN > 0.5] <- 1

mean(predictEN != testData$y)

#############################################################
################ GROUP LASSO ################################
#############################################################

#Grouping the summary statistics (mean, se, worst) together into groups of 3, e.g.
# group 1: radius_mean, radius_se, radius_worst
# group 2: texture_mean, texture_se, texture_worst
# etc.


ord.index <- vector(length = 31)

counter <- 1
for (i in 1:10) {
  ord.index[3 * (i-1)+1] <- (i-1) + 1
  ord.index[3 * (i-1)+2] <- (i-1) + 11
  ord.index[3 * (i-1)+3] <- (i-1) + 21
}

ord.index[31] <- 31

dataOrd <- data[,ord.index][1:30]
corr <- cor(dataOrd)
corrplot(corr, method = "color")

groups <- rep(1:10, each = 3)

trainDataOrd <- trainData[,ord.index]
testDataOrd <- testData[,ord.index]


trainDataMat <- data.matrix(trainDataOrd)

testDataMat <- data.matrix(testDataOrd)


modGroupLasso <- cv.gglasso(y = 2 * trainData$y - 1, x = trainDataMat[,1:(ncol(trainDataMat)-1)], nlambda = 20, loss = "logit", group = groups, nfolds = 5, pred.loss = "misclass")

summary(modGroupLasso)

coef(modGroupLasso, s="lambda.min")


predGroupLasso <- predict(modGroupLasso$gglasso.fit, newx = testDataMat[,1:(ncol(testDataMat)-1)], type = "class")

predGroupLasso <- (predGroupLasso + 1) / 2

mean(predGroupLasso != testData$y)




#############################################################
################ SPARSE GROUP LASSO #########################
#############################################################



#######################################################


set.seed(5)

alpha.vec <- seq(0, 1, by = 0.1)


x <- trainDataMat[,1:(ncol(trainDataMat)-1)]
y <- trainData$y

xtest <- testDataMat[,1:(ncol(testDataMat)-1)]
ytest <- testData$y

nTest <- nrow(testData)

K <- 10
randOrder = sample(1:nrow(x), nrow(x), replace=FALSE)

kGroups = list()
for (k in 1 : K) {
  kGroups[[k]] = ((k-1)*nrow(x)/10 + 1) : (k*nrow(x)/10)
}

errorMat = matrix(NA, K, length(alpha.vec))

trainDataMat <- data.matrix(trainDataOrd)

testDataMat <- data.matrix(testDataOrd)

lambda.vec <- c(1, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4)

for (k in 1 : K) {
  print(k)
  ## Split the data into training/testing
  testIndex = randOrder[kGroups[[k]]]
  
  trainingX = x[-testIndex,]
  testX = x[testIndex,]
  
  trainingY = y[-testIndex]
  testY = y[testIndex]
  
  for (kk in 1:length(alpha.vec)) {
    modSG <- msgl::cv(x = trainDataMat[,1:(ncol(trainDataMat)-1)], classes = trainData$y, grouping = groups, lambda = lambda.vec, alpha = alpha.vec[kk])
    
    lambda.opt <- lambda.vec[which.min(Err(modSG))]
    
    optSG <- msgl::fit(x = trainDataMat[,1:(ncol(trainDataMat)-1)], classes = trainData$y, grouping = groups, lambda = c(1, lambda.opt), alpha = alpha.vec[kk], intercept =  TRUE)
    optSG
    
    pred <- predict(optSG, x = testDataMat[,1:(ncol(testDataMat)-1)], lambda = c(1,lambda.opt), intercept = TRUE)
    
    errorRate <-  mean(pred$classes[,2] != testData$y)
    errorMat[k, kk] <- errorRate
  }
  
}

avg.error.rates <- vector(length = ncol(errorMat))

for (i in 1:ncol(errorMat)) {
  avg.error.rates[i] <- mean(errorMat[,i])
}

plot(alpha.vec, avg.error.rates, type = "l", ylab = "Error Rate", xlab = "alpha", axes = TRUE)

opt.index <- which.min(avg.error.rates)
opt.alpha <- alpha.vec[opt.index]


#################### OPTIMAL alpha = 0.3

alpha.opt <- 0.3

modSG <- msgl::cv(x = trainDataMat[,1:(ncol(trainDataMat)-1)], classes = trainData$y, grouping = groups, lambda = lambda.vec, alpha = alpha.opt)

lambda.opt <- lambda.vec[which.min(Err(modSG))]

optSG <- msgl::fit(x = trainDataMat[,1:(ncol(trainDataMat)-1)], classes = trainData$y, grouping = groups, lambda = c(1, lambda.opt), alpha = alpha.opt, intercept =  TRUE)

coefMat <- Matrix(optSG$beta[[2]], sparse = FALSE)
coeffs <- coefMat[2,]
coeffs[abs(coeffs) > 1e-10]

pred <- predict(optSG, x = testDataMat[,1:(ncol(testDataMat)-1)], lambda = c(1,lambda.opt), intercept = TRUE)

errorRate <-  mean(pred$classes[,2] != testData$y)
errorRate                  


######################################################
########### NEURAL NET ###############################
######################################################

model.net <- neuralnet(as.factor(y) ~ ., data = trainData, hidden = c(10, 5), err.fct = "ce", threshold = 1e-1, stepmax = 1e6, linear.output = FALSE)

probNN <- predict(model.net, newdata =  testData[,1:(ncol(testData)-1)])

pred <- vector(length = nrow(probNN))
for (ff in 1:nrow(probNN)) {
  if (probNN[ff,1] > 0.5) {
    pred[ff] <- 0
  }
  else {
    pred[ff] <- 1
  }
}

mean(pred != testData$y)

plot(model.net)

####################################################
################# ENSEMBLE METHODS #################
####################################################

opt.alpha <- 0.3

modEN <- cv.glmnet(x = x, y = y, alpha = opt.alpha, family = "binomial")

probEN <- predict(modEN, newx = testDataMat[,1:(ncol(testDataMat)-1)], type = "response")


model.net <- neuralnet(as.factor(y) ~ ., data = trainData, hidden = c(10), err.fct = "ce", threshold = 1e-1, stepmax = 1e6, linear.output = FALSE)

probNN <- predict(model.net, newdata =  testData[,1:(ncol(testData)-1)])
probNN <- 1 - probNN[,1]
probComb <-  (0.6 * probEN + 0.4* probNN)

pred <- rep(0, nrow(testData))
pred[probComb > 0.5] <- 1

mean(pred != testData$y)



###################################################
################# META CLASSIFIER ####################
###################################################


set.seed(1)
n <- nrow(trainData)
validIndex <- sample(1:n, floor(n/4), replace = FALSE)

trainData.2 <- data[validIndex,]
trainData.1 <- data[-validIndex,]


trainDataMat.1 <- data.matrix(trainData.1)
trainDataMat.2 <- data.matrix(trainData.2)

x.1 <- trainDataMat.1[,1:(ncol(trainDataMat.1)-1)]
x.2 <- trainDataMat.2[,1:(ncol(trainDataMat.2)-1)]

y.1 <- trainData.1$y
y.2 <- trainData.2$y

opt.alpha <- 0.3

modEN <- cv.glmnet(x = x.1, y = y.1, alpha = opt.alpha, family = "binomial")
probEN <- predict(modEN, newx = trainDataMat.2[,1:(ncol(trainDataMat.2)-1)], type = "response")


model.net <- neuralnet(as.factor(y) ~ ., data = trainDataMat.1, hidden = c(10, 5), err.fct = "ce", threshold = 1e-1, stepmax = 1e6, linear.output = FALSE)

probNN <- predict(model.net, newdata =  trainDataMat.2[,1:(ncol(trainDataMat.2)-1)])
probNN <- 1 - probNN[,1]

probData <- data.frame(probEN, probNN, trainData.2$y)
names(probData) <- c("probEN", "probNN", "y")

metaModLog <- glm(y ~ probEN + probNN, data = probData, family = "binomial")
metaLDA <- lda(y ~ probEN + probNN, data = probData)
metaQDA <- qda(y ~ probEN + probNN, data = probData)


###################################################3

opt.alpha <- 0.3

modEN.test <- cv.glmnet(x = x, y = y, alpha = opt.alpha, family = "binomial")
probEN.test <- predict(modEN, newx = testDataMat[,1:(ncol(testDataMat)-1)], type = "response")


model.net.test <- neuralnet(as.factor(y) ~ ., data = trainData, hidden = c(10, 5), err.fct = "ce", threshold = 1e-1, stepmax = 1e6, linear.output = FALSE)
probNN.test <- predict(model.net, newdata =  testData[,1:(ncol(testData)-1)])

probNN.test <- 1 - probNN.test[,1]
probData.test <- data.frame(probEN.test, probNN.test)

names(probData.test) <- c("probEN", "probNN")


predMetaLog <- 1*(predict(metaModLog, newdata = probData.test, type = "response") > 0.5)
mean(predMetaLog != testData$y)

predMetaLDA <- as.numeric(predict(metaLDA, newdata=probData.test)$class) - 1
mean(predMetaLDA != testData$y)

predMetaQDA <- as.numeric(predict(metaQDA, newdata=probData.test)$class) - 1
mean(predMetaQDA != testData$y)


#######################################################
################ AVOIDING FALSE NEGATIVES #############
#######################################################

#0 - Benign
#1 - Malignant


opt.alpha <- 0.3

modEN <- cv.glmnet(x = x, y = y, alpha = opt.alpha, family = "binomial")
probEN <- predict(modEN, newx = testDataMat[,1:(ncol(testDataMat)-1)], type = "response")

pred <- rep(0, nrow(testData))
pred[probEN > 0.2] <- 1

mean(pred - testData$y < 0)
mean(pred != testData$y)


#######################################################
############## BOOTSTRAP ##############################
#######################################################

modLog <- glm(y ~ 0 + radius_mean + radius_se + radius_worst + texture_mean + texture_worst + smoothness_worst  +
                concavity_mean  + concavity_worst + concave.points_mean + concave.points_se + concave.points_worst + symmetry_worst
              , data = trainData, family = binomial)

summary(modLog)$coefficients
#summary(modLog)$coefficients[1]

predLog <- 1*(predict(modLog, newdata = testData, type = "response") > 0.5)
mean(predLog != testData$y)


#Nonparametric Bootstrap
N <- 1000
p <- 12 #Number of parameterss
n <- nrow(trainData)
estBoot <- matrix(nrow = N, ncol = p)


for (i in 1:N) {
  samp <- sample(1:n, n, replace=TRUE)
  DataBoot <- trainData[samp,]

  modLogBoot <- glm(y ~ 0 + radius_mean + radius_se + radius_worst + texture_mean + texture_worst + smoothness_worst  +
                  concavity_mean  + concavity_worst + concave.points_mean + 
                  concave.points_se + concave.points_worst + symmetry_worst,
                data = DataBoot, family = "binomial")
  
  estBoot[i,] <- summary(modLogBoot)$coefficients[1:p]
}

#second method - percentile method
CI <- matrix(nrow = p, ncol = 2)
for (i in 1:p) {
  q <- quantile(estBoot[,i], c(0.025, 0.975))
  CI[i,]<- c(q[1], q[2])

}
CI
