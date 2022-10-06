# Random Forest
Random Forests are defined as an ensemble of decision trees, where the added step of randomness when building each tree. Using Information Gain, we pick the highest gain to split.

$$entropy = - \sum_{j}^{i} P(x_i) logP(x_i)$$


$$Information\ \ Gain = 1 - entropy$$

**For the completion of Data Mining Techniques** 

## Overview
According to the World Stroke Organization, stroke is a leading cause of death in the United States, and there is one new stroke patient every 3 seconds globally in 2022. As a result, it is significant to predict stroke based on relevant factors. Data mining algorithms include K nearest neighbors (KNN), support vector machine (SVM), Naive Bayes, decision tree, random forest, and logistic regression.  

My respective work compared my random forest code to sci-kit learn's random forest.   
Results:
Method | Accuracy | Precision 0 | Precision 1 | Recall 0 | Recall 1 | F1-Score 0 | F1-Score 1
-------|-------|-------|-------|-------|-------|-------|-------
Random Forest (scratch) | 0.89 | 0.96 | 0.15 | 0.93 | 0.24 | 0.94 | 0.18
Random Forest (sklearn w/o SMOTE) | 0.94 | 0.94 | 0.00 | 1.00 | 0.00 | 0.94 | 0.00
Random Forest (sklearn w/ SMOTE) | 0.89 | 0.96 | 0.16 | 0.93 | 0.27 | 0.94 | 0.20

Imbalanced data poses a difficult task for many classification models. When facing imbalanced scenarios, the traditional models often provide good coverage of the majority class, whereas the minority class are distorted. SMOTE is applied to the data to balance the minoirty class (positive stroke) from the majority class (negative stroke). You can see that recall of positive stroke is 0% without SMOTE but increases to 27% once applied. Although accuracy decreased when SMOTE is applied, the model is now able to classify positive strokes. 

# Data Structure

Variables| Variable Type | Role
---------|---------------|-----
age| Numerical |Input
hypertension |Categorical |Input
heart_disease| Categorical |Input
avg_glucose_level| Numerical |Input
bmi| Numerical |Input
stroke| Categorical| ***Target***
gender_Female| Categorical |Input
gender_Male |Categorical |Input
ever_married_No| Categorical| Input
ever_married_Yes |Categorical| Input
work_type_Govt_job| Categorical| Input
work_type_Never_worked| Categorical |Input
work_type_Private| Categorical| Input
work_type_Self-employed| Categorical| Input
work_type_children| Categorical| Input
Residence_type_Rural| Categorical |Input
Residence_type_Urban| Categorical |Input
smoking_status_Unknown| Categorical| Input
smoking_status_formerly| Categorical| Input
smoking_status_never| Categorical| Input
smoking_status_smokes|Categorical |Input

