Practical Machine Learning Writeup - Qualitative Activity Recognition of Weight Lifting Exercises
========================================================

The goal of this assignment is to predict the manner in which people that used the personal activity tracking devices exercised, as represented by the data in the weight lifting exercise datasets. The **training** and **testing** datasets were provided by a human activity recognition research group (see http://groupware.les.inf.puc-rio.br/har)


## Data

Training and testing datasets were available for download from the following URLs:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

Once the files were downloaded and placed in a working directory, we loaded the files using read.csv method

```r
pml_training <- read.csv("pml-training.csv", na.strings=c("NA",""), header=TRUE)
pml_testing <- read.csv("pml-testing.csv", na.strings=c("NA",""), header=TRUE)
train_cols <- colnames(pml_training)
test_cols <- colnames(pml_testing)
```
We compared the columns, and determined that they are all pretty much the same except for the last column,
which is classe in training set, and problem_id in testing set

```r
all.equal(train_cols[1:length(train_cols)], test_cols[1:length(test_cols)])
```

```
## [1] "1 string mismatch"
```

## Processing

First, we imported libraries that are used in the code (had to install "rattle" and "rpart.plot" packages)

```r
library(caret); library(rattle); library(rpart.plot); library(randomForest)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
## Rattle: A free graphical interface for data mining with R.
## Version 3.3.0 Copyright (c) 2006-2014 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
## Loading required package: rpart
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

Looking at the training and testing set columns, we saw that columns 1 through 7 are useless in our predictions, so we got rid of those
along with columns that contained NAs

```r
pml_training_slimmed <- pml_training[, !apply(is.na(pml_training), 2, any)]
pml_training_slimmed <- pml_training_slimmed[,8:length(colnames(pml_training_slimmed))]
pml_testing_slimmed <- pml_testing[, !apply(is.na(pml_testing), 2, any)]
pml_testing_slimmed <- pml_testing_slimmed[,8:length(colnames(pml_testing_slimmed))]
colnames(pml_training_slimmed)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "classe"
```

```r
colnames(pml_testing_slimmed)
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"     "problem_id"
```

## Selecting predictors

To determine whether any predictors in the data are low variance, we examined all predictors using near zero variability
and identified predictors that had near zero variability value FALSE

```r
nzv_training <- nearZeroVar(pml_training_slimmed, saveMetrics=TRUE)
nzv_testing <- nearZeroVar(pml_testing_slimmed, saveMetrics=TRUE)
high_val_pred <- row.names(nzv_training[nzv_training$nzv == FALSE,])

length(high_val_pred)
```

```
## [1] 53
```

```r
length(colnames(pml_testing_slimmed))
```

```
## [1] 53
```
It turned out that all columns in the slimmed training set we produced are high value predictors

## Prediction models and parameters

Since we had a monster machine, we could perform training models on the entire training set (without further splitting it).
Next, we performed classification tree evaluation on the training set:

```r
set.seed(31231)
modFit <- train(classe ~ ., data = pml_training_slimmed, method="rpart")
```

```
## Loading required namespace: e1071
```

```r
print(modFit, digits=3)
```

```
## CART 
## 
## 19622 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 19622, 19622, 19622, 19622, 19622, 19622, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp      Accuracy  Kappa   Accuracy SD  Kappa SD
##   0.0357  0.516     0.3716  0.0231       0.0365  
##   0.0600  0.406     0.1935  0.0614       0.1020  
##   0.1152  0.345     0.0947  0.0365       0.0544  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0357.
```

```r
print(modFit$finalModel, digits=3)
```

```
## n= 19622 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 19622 14000 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130 17977 12400 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -34 1578    10 A (0.99 0.0063 0 0 0) *
##      5) pitch_forearm>=-34 16399 12400 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 440 13870  9950 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 124 8643  5130 A (0.41 0.18 0.18 0.17 0.061) *
##         21) roll_forearm>=124 5227  3500 C (0.077 0.18 0.33 0.23 0.18) *
##       11) magnet_dumbbell_y>=440 2529  1240 B (0.032 0.51 0.043 0.22 0.19) *
##    3) roll_belt>=130 1645    14 E (0.0085 0 0 0 0.99) *
```
We used fancyRpartPlot to display visually


```r
fancyRpartPlot(modFit$finalModel)
```

![plot of chunk unnamed-chunk-7](figure/unnamed-chunk-7.png) 

Further, we ran the model against the training set and examined the confusion matrix 
to evaluate accuracy and the confidence intervals

```r
pml_train_predict <- predict(modFit, newdata=pml_training_slimmed)
print(confusionMatrix(pml_train_predict, pml_training_slimmed$classe), digits=4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5080 1581 1587 1449  524
##          B   81 1286  108  568  486
##          C  405  930 1727 1199  966
##          D    0    0    0    0    0
##          E   14    0    0    0 1631
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4956          
##                  95% CI : (0.4885, 0.5026)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3407          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9104  0.33869  0.50468   0.0000  0.45218
## Specificity            0.6339  0.92145  0.78395   1.0000  0.99913
## Pos Pred Value         0.4970  0.50850  0.33040      NaN  0.99149
## Neg Pred Value         0.9468  0.85310  0.88225   0.8361  0.89008
## Prevalence             0.2844  0.19351  0.17440   0.1639  0.18382
## Detection Rate         0.2589  0.06554  0.08801   0.0000  0.08312
## Detection Prevalence   0.5209  0.12889  0.26638   0.0000  0.08383
## Balanced Accuracy      0.7721  0.63007  0.64431   0.5000  0.72565
```

## Accuracy and Out of Sample Error

As we could observe above, the accuracy was quite low (0.4956) and when attempted to use cross validation and preprocessing, it produced no effect on the accuracy. 
The accuracy remained low at the same number. This means the out of sample error is 1 - 0.4956 = 0.5044

```r
set.seed(31231)
modFit <- train(classe ~ ., preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data = pml_training_slimmed, method="rpart")
print(modFit, digits=3)
```

```
## CART 
## 
## 19622 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 14717, 14717, 14717, 14715 
## 
## Resampling results across tuning parameters:
## 
##   cp      Accuracy  Kappa   Accuracy SD  Kappa SD
##   0.0357  0.521     0.3777  0.0324       0.0518  
##   0.0600  0.429     0.2309  0.0724       0.1217  
##   0.1152  0.323     0.0596  0.0452       0.0688  
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.0357.
```

```r
pml_train_predict_preproc <- predict(modFit, newdata=pml_training_slimmed)
print(confusionMatrix(pml_train_predict_preproc, pml_training_slimmed$classe), digits=4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5080 1581 1587 1449  524
##          B   81 1286  108  568  486
##          C  405  930 1727 1199  966
##          D    0    0    0    0    0
##          E   14    0    0    0 1631
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4956          
##                  95% CI : (0.4885, 0.5026)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3407          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9104  0.33869  0.50468   0.0000  0.45218
## Specificity            0.6339  0.92145  0.78395   1.0000  0.99913
## Pos Pred Value         0.4970  0.50850  0.33040      NaN  0.99149
## Neg Pred Value         0.9468  0.85310  0.88225   0.8361  0.89008
## Prevalence             0.2844  0.19351  0.17440   0.1639  0.18382
## Detection Rate         0.2589  0.06554  0.08801   0.0000  0.08312
## Detection Prevalence   0.5209  0.12889  0.26638   0.0000  0.08383
## Balanced Accuracy      0.7721  0.63007  0.64431   0.5000  0.72565
```

Next, we decided to train using  random forest methodology to see if we can improve the accuracy, using pre-processing and cross validation. Based on what we know
about random forest prediction model (which is they're the most accurate), we expect the out of sample error to become close to zero.

```r
set.seed(31231)
modFit <- train(classe ~ ., method="rf", preProcess=c("center", "scale"), trControl=trainControl(method = "cv", number = 4), data=pml_training_slimmed)
print(modFit, digits=3)
```

```
## Random Forest 
## 
## 19622 samples
##    52 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## Pre-processing: centered, scaled 
## Resampling: Cross-Validated (4 fold) 
## 
## Summary of sample sizes: 14717, 14717, 14717, 14715 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    0.994     0.992  0.00161      0.00204 
##   27    0.994     0.993  0.00171      0.00217 
##   52    0.990     0.987  0.00336      0.00425 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

```r
pml_train_predict_rf <- predict(modFit, newdata=pml_training_slimmed)
print(confusionMatrix(pml_train_predict_rf, pml_training_slimmed$classe), digits=4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5580    0    0    0    0
##          B    0 3797    0    0    0
##          C    0    0 3422    0    0
##          D    0    0    0 3216    0
##          E    0    0    0    0 3607
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9998, 1)
##     No Information Rate : 0.2844     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Random forest model (though it took a long time to process), significantly improved accuracy (to 1.0)! This means our out of sample error was 0.0, which means 100% accuracy.
Very promising!

## Conclusion

The random forest model was run agains the slimmed test dataset, and produced the following predictions.

```r
pml_test_predict_rf <- predict(modFit, newdata=pml_testing_slimmed)
print(pml_test_predict_rf)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```




