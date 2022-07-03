#
# Creator:        Andrew Disher
# Affiliation:    UMASS Dartmouth
# Course:         CIS-530 Advanced Data Mining
# Assignment:     Homework 3 Part 2
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


# -------------------------------------------------------------------------
# Create a preliminary logistic regression model using all predictors -----
# -------------------------------------------------------------------------

Model1 <- glm(HeartDisease ~., data = Training_Data, family=binomial(link = "logit"))
summary(Model1)

# NOTE: Many of the predictor variables used in this model are significant. Some variabels, however, are not. 
#       For example, the predictors `Physical Health`, RaceOther and RaceWhite (these are actually just levels
#       for the variable `Race`, since dummy variables were created), and `Physical Activity` were all deemed
#       insignificant. Additionally, one level for the `Diabetic` predictor is borderline significant.
#
#       We will exclude the variables `Physical Health` and `Physical Activity`; after all, it makes sense that
#       neither should be in the model since they essentially represent the same metric (more or less) of physical
#       well being. However, we shall still include the dummy variables regarding race, since they are level of 
#       the same predictor `Race`. To include the variable at all, we must include all of its levels. This is
#       also the case for the predictor `Diabetic`
#       


Model2 <- glm(HeartDisease ~., data = subset(Training_Data, select = -c(PhysicalHealth, PhysicalActivity)), family=binomial(link = "logit"))
summary(Model2)

# Goodness of fit test for model 
hoslem.test(Training_Data$HeartDisease, fitted(Model2))
# X-squared = 285.61, df = 8, p-value < 2.2e-16


# -------------------------------------------------------------------------
# Computing Evaluation Metrics --------------------------------------------
# -------------------------------------------------------------------------

# Using Model with 6 principal components, predict the response variable diagnosis with test set
Test_Predictions <- round(predict(Model2, 
                                  subset(Test_Data, select = -c(HeartDisease, PhysicalHealth, PhysicalActivity)),
                                  type = 'response'))

# Produce a confusion matrix for the predictions
Confusion_Matrix <- confusionMatrix(data = as.factor(Test_Predictions), reference = as.factor(Test_Data$HeartDisease), 
                                    positive = c("1"))
Confusion_Matrix
Confusion_Matrix$byClass
Confusion_Matrix$table

# Accuracy = 0.9167  

# NOTE: The value 1 is the positive result, i.e. a value of 1 corresponds to a malignant tumor, and 0 
#       corresponds to a benign tumor. Therefore, the confusion matrix above should be changed to reflect 
#       this in our report.


# Creating an ROC curve and calculating area under curve (using test predictions before rounding)
roc_score = roc(Test_Data$HeartDisease, 
                predict(Model2, 
                        subset(Test_Data, select = -c(HeartDisease, PhysicalHealth, PhysicalActivity)),
                        type = 'response'))
roc_score$auc # 0.8422
plot(roc_score ,main ="ROC curve -- Heart Disease -- Logistic Regression ",legacy.axes = TRUE)


# Sensitivity/Recall
Recall = 588/(588+4859) # 0.1079493

# Precision
Precision = 588/(588+468) # 0.5568182

# F1-Measure (weight Recall and Precision equally)
2*((Precision*Recall)/((1*Precision) + Recall)) # 0.1808396

# F2-Measure (weight Recall Higher than Precision)
5*((Precision*Recall)/((4*Precision) + Recall)) # 0.128699


# Computing the cost of our final model

# create a cost matrix to visualize
matrix(c(0, 1, 100, -1), nrow = 2, ncol = 2)

# Computing cost
0*58044 + 1*468 + 100*4859 + -1*588 # 485780





