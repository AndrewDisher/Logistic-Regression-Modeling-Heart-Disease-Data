This is a readMe file for the project, which contains an application of logistic regression classification to a heart disease data set.

------------------------------
The Data Set -----------------
------------------------------

The data set can be found at the following link, along with a descriptions regarding the variables contained within it:
https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease


The data set contains a response variable, and many predictors variables (features). The response variable is a binary variable representing 
whether an individual has heart disease or not. A value of 1 corresponds to a positive result (the individual has heart disease) and a value 
of 0 corresponds to a negative (the individual does not have heart disease). 

The are a variety of other variables that were deemed key indicators by the CDC, and each one can be found described in detail when viewing the 
Kaggle link provided above. 

-------------------------------
The Model ---------------------
-------------------------------

The model fit in this project was a simple binary logistic regression model, which was fit and evaluated using 

1) a Hoslem-Lemshowe test for goodness of fit
2) a confusion matrix showing the results of predictions (on test data)
3) sensitivity and specificity metrics (and others).

It is important to note that the sensitivity of our model was of the highest importance among the different metrics used to evaluate in (3), since 
failing to detect cases of heart disease would be more detrimental than falsely classifying a negative case as positive, in this health related setting. 

-------------------------------
Additional Info ---------------
-------------------------------

There are two code folders for this project, on labeled 'without-under-over-sampling' and one labeled 'with-under-over-sampling'. Each contains the code for 
fitting and evaluating the logistic regression model, but with one important difference: the use of under-oversampling to balance the data set. Typically, 
data can either be undersampled to reduce the number of instances/observations that are identified with the most populous class (negative cases), or oversampled 
to increase the number of observations of the less populous class (postive cases). In the first code folder, we neglected to use any under- or oversampling
technique to the data set for comparitive purposes. In the second folder, we applied a combination of undersampling and oversampling to the data set to balance 
the classes.

It was found that the under-oversampling technique improved the sensitivity of the logistic regression model drastically, at the cost of specificity. 



