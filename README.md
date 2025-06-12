## TITANIC SURVIVAL PREDICTION
## Problem Statement 
The RMS Titanic sank after hitting an iceberg, causing 1,502 deaths out of 2,224 on board. The central question: Which types of people were more likely to survive?
Kaggle frames it as a binary classification: predict whether each passenger survived (1) or not (0), based on attributes like age, gender, class, fare, cabin, and port of embarkation
##  Motivation & Context
Historical and educational interest: The Titanic data is ideal for introducing data science workflows and supervised learning
Learning challenges: Although popular, the dataset can mislead without careful thought—for instance, age data inconsistencies (mixed age at death vs. boarding) and limited features that may not directly reflect survival-driving factors .
Real-world implications: It raises critical questions about fairness and data bias: survival often depended on gender, class, family ties, and even port of embarkation—factors deeply rooted in early 20th-century social norms 
##  Dataset Overview
Fields include:

PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked 
Common issues:

~20% missing Age, ~77% missing Cabin, a couple missing Embarked entries .

Inconsistencies in reported age values (survivor ages may reflect year of death) 
Slight class imbalance: ~38% survived, ~62% did not 

##  Challenges in Modeling
Limited feature set: Data was not collected with machine learning in mind. Critical variables influencing survival (e.g., physical location on the ship, timing, crew decisions) are missing 
medium.com
Data leakage risk: Some solutions achieve perfect accuracy by cheating—e.g., using test labels or overly powerful derived features—thus not generalizing 
Data inconsistency: Age inaccuracies and missing values require cautious preprocessing .
Overfitting danger: High metrics on training sets may not reflect real-world performance; proper validation is imperative .

## Modeling Pipeline

## 1. Data Import
The project begins with loading the Titanic dataset, which includes features like Survived (target), Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, and Embarked 
 This forms the core data for predictive modeling.

## 2. Data Cleaning
Handling missing values: Age, Cabin, and Embarked often contain gaps. Common strategies include imputing medians/modes or creating "Missing" indicators (e.g., MissingCabin) .
Duplicates and consistency: Remove duplicated rows and standardize categorical entries.
Outliers: Detect outliers using statistical methods (e.g., IQR) for numerical features like Fare and Age .
## 3. Preprocessing & Feature Engineering
Categorical encoding: Turn variables like Sex, Embarked, and Pclass into numerical form through one-hot or target encoding 
Feature extraction:
Derive Title from passenger names (Mr, Mrs, Miss) .
Create FamilySize by summing SibSp and Parch 
Calculate FarePerPerson, i.e., Fare divided by family size 
Feature selection: Use techniques such as Random Forest importance or recursive feature elimination to refine feature set 
Scaling: Standardize numerical features—critical for SVC performance
## 4. Exploratory Data Analysis (EDA)
Visualizations: Use histograms, boxplots, or heatmaps to examine distributions and relationships .
Correlation & group analysis: Examine survival rates by categorical groupings (e.g., sex, class, embarked) 
Imbalance handling: Note that survival (~38%) vs. non-survival (~62%) is slightly imbalanced. Use stratified sampling, SMOTE, or class weighting as needed 
## 5. Model Training
Train and compare three classifiers:
Logistic Regression
Provides baseline with interpretability (coefficients show feature effects) 
Often achieves ~79–80% accuracy in Titanic scenarios
Random Forest
Ensemble of decision trees with robust handling of mixed data types and no need for scaling 

Achieves ~81–82% accuracy, with fine-tuned versions in 84% range .

Hyperparameters like depth, n_estimators, max features, and leaf size can be tuned via grid/random search .

Support Vector Classifier (SVC)
Finds the margin-maximizing decision boundary; kernels enable non-linear classification 

## 6. Model Validation & Evaluation
Train-test split: Commonly 70–80% training and 20–30% testing .

Performance metrics: Track accuracy, precision, recall, F1-score, and ROC-AUC to get a holistic view of model effectiveness
.

Cross-validation: Use K-fold (5‑10) CV to stabilize performance estimates .

Results comparison:

Logistic Regression: ~79–80%

Random Forest: ~81–84%

SVC: up to ~84.7% when tuned 
## 7. Model Selection & Comparison
Choose based on priorities:
Accuracy-centric: SVC or Random Forest often outperform Logistic Regression.
