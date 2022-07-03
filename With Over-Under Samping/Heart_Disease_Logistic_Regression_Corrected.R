#
# Creator:        Andrew Disher
# Affiliation:    UMASS Dartmouth
# Course:         CIS-530 Advanced Data Mining
# Assignment:     Final Project
# Date:           5/2/2022
#
# TASK: Fit a binomial logistic regression model and perform diagnostics on the Heart Disease Data Set. Then 
#       evaluate the performance of the final model. 
#
# Data Source: 
#
# Heart Disease Data Set
# https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease


# Packages
library(car) # For cumsum() function; computes the cumulative sum of a vector
library(caret) # For Confusion matrix creation
library(pROC) # For creating an ROC curve and calculating Area Under Curve
library(ResourceSelection) # For goodness of fit test for the Logistic Regression model
library(ROSE) # For under/oversampling to correct for imbalanced class issue


# -------------------------------------------------------------------------
# Import the data ---------------------------------------------------------
# -------------------------------------------------------------------------

Heart_Disease_Data_Set <- read.csv("~/UMASS Dartmouth/Classes/Spring 2022/Data Mining/Project/Data Sets/Heart Disease Data Set/HeartDiseaseCleaned/HeartDiseaseCleaned.csv")

# Remove the first column (X ~ it is an ID column)
Heart_Disease_Data_Set <- Heart_Disease_Data_Set[, -1]


# -------------------------------------------------------------------------
# Create Training/Testing Data Sets ---------------------------------------
# -------------------------------------------------------------------------

# Split Data into Training and Testing in R 
sample_size = floor(0.8*nrow(Heart_Disease_Data_Set))
set.seed(666)

# Randomly split data
picked = sample(seq_len(nrow(Heart_Disease_Data_Set)), size = sample_size)

# Store the Training and Testing data in their respective data frames
Training_Data <- Heart_Disease_Data_Set[picked, ]
Test_Data <- Heart_Disease_Data_Set[-picked, ]

# Obtain the class proportions in a table
prop.table(table(Training_Data$HeartDisease))

# 91.4% don't have heart disease, 
# 8.6% do have heart disease

# NOTE: This class imbalance will severely affect the ability of our model to accuractely predict if someone
#       has heart disease. We should correct for this.


# -------------------------------------------------------------------------
# Correct imbalanced class issue ------------------------------------------
# -------------------------------------------------------------------------

# Use a combination of over and under sampling to acquire a more balanced data set. 
Training_Data <- ovun.sample(HeartDisease~., data=Training_Data, method = "both",
                             p = 0.47, # Probability of resampling from the rare class (has heart disease)
                             seed = 666,
                             N = 255836)$data

# Check the distribution of the response variable in the new data set. 
prop.table(table(Training_Data$HeartDisease))

# -------------------------------------------------------------------------
# Create a preliminary logistic regression model using all predictors -----
# -------------------------------------------------------------------------

Model1 <- glm(HeartDisease ~., data = Training_Data, family=binomial(link = "logit"))
summary(Model1)

# NOTE: Many of the predictor variables used in this model are significant. Some variables, however, are not. 
#       For example, the predictors RaceOther and RaceWhite (these are actually just levels
#       for the variable `Race`, since dummy variables were created), and `Physical Activity` were all deemed
#       insignificant. Additionally, one level for the `Diabetic` predictor is borderline significant.
#
#       We will exclude the variable`Physical Activity`; However, we shall still include the dummy variables 
#       regarding race, since they are level of the same predictor `Race`. To include the variable at all, 
#       we must include all of its levels. 


Model2 <- glm(HeartDisease ~., data = subset(Training_Data, select = -c(PhysicalActivity)), family=binomial(link = "logit"))
summary(Model2)

# Goodness of fit test for model 
hoslem.test(Training_Data$HeartDisease, fitted(Model2))
# X-squared = 1104.7, df = 8, p-value < 2.2e-16



# -------------------------------------------------------------------------
# Computing Evaluation Metrics --------------------------------------------
# -------------------------------------------------------------------------

# Using Model with 6 principal components, predict the response variable diagnosis with test set
Test_Predictions <- round(predict(Model2, 
                                  subset(Test_Data, select = -c(HeartDisease, PhysicalActivity)),
                                  type = 'response'))

# Produce a confusion matrix for the predictions
Confusion_Matrix <- confusionMatrix(data = as.factor(Test_Predictions), reference = as.factor(Test_Data$HeartDisease), 
                                    positive = c("1"))
Confusion_Matrix
Confusion_Matrix$byClass
Confusion_Matrix$table

# Accuracy = 77.39 

# NOTE: The value 1 is the positive result, i.e. a value of 1 corresponds to a malignant tumor, and 0 
#       corresponds to a benign tumor. Therefore, the confusion matrix above should be changed to reflect 
#       this in our report.


# Creating an ROC curve and calculating area under curve (using test predictions before rounding)
roc_score = roc(Test_Data$HeartDisease, 
                predict(Model2, 
                        subset(Test_Data, select = -c(HeartDisease, PhysicalActivity)),
                        type = 'response'))
roc_score$auc # 0.8427
plot(roc_score ,main ="ROC curve -- Heart Disease -- Logistic Regression ", legacy.axes = TRUE)


# Sensitivity/Recall
Recall = 4092/(4092+1355) # 0.7512392

# Precision
Precision = 4092/(4092+13104) # 0.2379623

# F1-Measure (weight Recall and Precision equally)
2*((Precision*Recall)/((1*Precision) + Recall)) # 0.3614362

# F2-Measure (weight Recall Higher than Precision)
5*((Precision*Recall)/((4*Precision) + Recall)) # 0.5248307


# Computing the cost of our final model

# create a cost matrix to visualize
matrix(c(0, 1, 100, -1), nrow = 2, ncol = 2)

# Computing cost
0*45408 + 1*13104 + 100*1355 + -1*4092 # 144512


