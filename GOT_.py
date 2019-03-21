# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 17:15:46 2019

@author: adhish
"""

# Loading Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


file = 'GOT_character_predictions.xlsx'

got = pd.read_excel(file)


# Column names
got.columns


# Displaying the first rows of the DataFrame
print(got.head())


# Dimensions of the DataFrame
got.shape


# Information about each variable
got.info()


# Descriptive statistics
got.describe().round(2)


###############################################################################
# Part 1: Imputing Missing Values
###############################################################################

print(
      got
      .isnull()
      .sum()
      )

for col in got:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    
    if got[col].isnull().any():
        got['m_'+col] = got[col].isnull().astype(int)
        
        
###Imputing mising Values 
###Title
fill = 'No title'

got['title'] = got['title'].fillna(fill)   


###Culture
fill='No Culture'
got['culture']=got['culture'].fillna(fill)

###Mother
fill='Unknown Mother'
got['mother']=got['mother'].fillna(fill)

###Father
fill='Unknown Father'
got['father']=got['father'].fillna(fill)

###Heir
fill= 'Unknown Heir'
got['heir']=got['heir'].fillna(fill)

##House 
fill='No house'
got['house']=got['house'].fillna(fill)

##Spouse 
fill='No Spouse'
got['spouse']=got['spouse'].fillna(fill)

###Age 
got['age'].mean()
got['name'][got['age']<0] 

#There are 2 cases where the age is negative. Further research is required 
# Rhaego was son of Khal Drogo and Daenerys. He was never born, which mean his age is 0
# Doreah age in the book is 25 
# fixing the data in age 

got.loc[431,'age']=0
got.loc[432,'age']=25
#checking the mean again
got['age'].mean() 
#Filling the age with 
fill=got['age'].mean()
got['age']=got['age'].fillna(fill)


##isAliveMother, Father , Heir and Spouse NA will be filled with -1 
fill= -1 
got['isAliveMother']=got['isAliveMother'].fillna(fill)
got['isAliveFather']=got['isAliveFather'].fillna(fill)
got['isAliveHeir']=got['isAliveHeir'].fillna(fill)
got['isAliveSpouse']=got['isAliveSpouse'].fillna(fill)

#Checking one more time for NA
print(
      got
      .isnull()
      .sum()
      )
#Birthdate is measured with unknown method so I am not going to use it in my model 

for val in enumerate(got.loc[ : , 'popularity']):
    
    if val[1] <= 0.3:
        got.loc[val[0], 'popular'] = 0



for val in enumerate(got.loc[ : , 'popularity']):
    
    if val[1] > 0.3:
        got.loc[val[0], 'popular'] = 1

###############################################################################
# Part 2: EDA
###############################################################################

###Culture: There are many different values for the same culture, so I will group them up
##Grouping culture 
cult = {
    'Summer Islands': ['summer islands', 'summer islander', 'summer isles'],
    'Ghiscari': ['ghiscari', 'ghiscaricari',  'ghis'],
    'Asshai': ["asshai'i", 'asshai'],
    'Lysene': ['lysene', 'lyseni'],
    'Andal': ['andal', 'andals'],
    'Braavos': ['braavosi', 'braavos'],
    'Dorne': ['dornishmen', 'dorne', 'dornish'],
    'Myrish': ['myr', 'myrish', 'myrmen'],
    'Westermen': ['westermen', 'westerman', 'westerlands'],
    'Westeros': ['westeros', 'westerosi'],
    'Stormlander': ['stormlands', 'stormlander'],
    'Norvos': ['norvos', 'norvoshi'],
    'Northmen': ['the north', 'northmen'],
    'Wildling': ['wildling', 'first men', 'free folk'],
    'Qarth': ['qartheen', 'qarth'],
    'Reach': ['the reach', 'reach', 'reachmen'],
    'Ironborn': ['ironborn', 'ironmen'],
    'Mereen': ['meereen', 'meereenese'],
    'RiverLands': ['riverlands', 'rivermen'],
    'Vale': ['vale', 'valemen', 'vale mountain clans']
}

def get_cult(value):
    value = value.lower()
    v = [k for (k, v) in cult.items() if value in v]
    return v[0] if len(v) > 0 else value.title()
got.loc[:, "culture"] = [get_cult(x) for x in got["culture"]]


##Title and Alive or dead 
got.boxplot(column = ['isAlive'],
                by = ['m_title'],
                vert = False,
                patch_artist = False,
                meanline = False,
                showmeans = True)



#############################################
##### HISTOGRAMS do not tell the complete story, hence were ignored in the EDA
############################################

###############################################################################
# Qualitative Variable Analysis (Box Plots + Violin Plots)
###############################################################################
        
# Violin Plots


f,ax=plt.subplots(2,2,figsize=(17,15))
sns.violinplot("popular", "isNoble", hue="isAlive", data=got ,split=True, ax=ax[0, 0],color='green')
ax[0, 0].set_title('Noble and Popular vs Mortality')
ax[0, 0].set_yticks(range(2))

sns.violinplot("popular", "male", hue="isAlive", data=got ,split=True, ax=ax[0, 1],color='blue')
ax[0, 1].set_title('Male and Popular vs Mortality')
ax[0, 1].set_yticks(range(2))

sns.violinplot("popular", "isMarried", hue="isAlive", data=got ,split=True, ax=ax[1, 0],color='pink')
ax[1, 0].set_title('Married and Popular vs Mortality')
ax[1, 0].set_yticks(range(2))


sns.violinplot("popular", "book1_A_Game_Of_Thrones", hue="isAlive", data=got ,split=True, ax=ax[1, 1],color='yellow')
ax[1, 1].set_title('Book_1 and Popular vs Mortality')
ax[1, 1].set_yticks(range(2))


plt.show()

#################################################
### The VIOLIN PLOTS have the most interesting insights, which are explained in the report
#################################################


###############################################################################
# Part 3: Machine Learning Model 
###############################################################################
########################
# Working with Categorical Variables
########################

# Scalling our Data 

from sklearn.preprocessing import StandardScaler # standard scaler

# Removing the target variable. It is (generally) not necessary to scale that.

got_features = got.drop(['name','title','culture','dateOfBirth',
                           'mother','father','heir','house','spouse'],axis=1)

# Instantiating a StandardScaler() object
scaler = StandardScaler()


# Fitting the scaler with our data
scaler.fit(got_features)


# Transforming our data after fit
X_scaled = scaler.transform(got_features)


# Putting our scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

# Adding labels to our scaled DataFrame
X_scaled_df.columns = got_features.columns

########################
#Train_test_split 
########################

# Importing new libraries
from sklearn.model_selection import train_test_split # train/test split
from sklearn.neighbors import KNeighborsRegressor # KNN for Regression
import sklearn.metrics # more metrics for model performance evaluation
from sklearn.model_selection import cross_val_score # k-folds cross validation

got_data   = X_scaled_df

got_target = got.loc[:, 'isAlive']

X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


###############################################################################
#KNN
###############################################################################

 
# Creating two lists, one for training set accuracy and the other for test
# set accuracy
training_accuracy = []
test_accuracy = []



# Building a visualization to check to see  1 to 50
neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # Building the model
    clf = KNeighborsRegressor(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train)
    
    # Recording the training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # Recording the generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


# Plotting the visualization
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show() 


#print(testaccuracy.index(max(test_accuracy))) -> to index lists
print(test_accuracy.index(max(test_accuracy))) 

# The best results occur when k = 3.



# Building a model with k =3
knn_reg = KNeighborsRegressor(algorithm = 'auto',
                              n_neighbors = 3)



# Fitting the model based on the training data
knn_reg_fit = knn_reg.fit(X_train, y_train)


# Scoring the model
y_score_knn_optimal = knn_reg.score(X_test, y_test)



# The score is directly comparable to R-Square
print(y_score_knn_optimal)



# Generating Predictions based on the optimal KNN model
knn_reg_optimal_pred_train = knn_reg_fit.predict(X_train)

knn_reg_optimal_pred_test = knn_reg_fit.predict(X_test)

#AUC Score 
from sklearn.metrics import roc_auc_score
print('Training AUC Score:',roc_auc_score(
        y_train,knn_reg_optimal_pred_train),round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,knn_reg_optimal_pred_test),round(4))


# Cross-Validation on knn (cv = 3)

cv_tree_3 = cross_val_score(knn_reg_fit,
                             got_data,
                             got_target,
                             cv = 3)


print(cv_tree_3)


print(pd.np.mean(cv_tree_3).round(3))

###############################################################################
# Decision Tree
###############################################################################
from sklearn.tree import DecisionTreeRegressor # Regression trees

got_data   = got.drop(['isAlive','name','title','culture','dateOfBirth',
                           'mother','father','heir','house','spouse'],axis=1)


got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


# Full tree.
tree_full = DecisionTreeRegressor(random_state = 508)
tree_full.fit(X_train, y_train)

print('Training Score', tree_full.score(X_train, y_train).round(4))
print('Testing Score:', tree_full.score(X_test, y_test).round(4))


########################
# Making Adjustment to the model
########################

tree_leaf_50 = DecisionTreeRegressor(criterion = 'mse',
                                     min_samples_leaf = 50,
                                     random_state = 508)

tree_leaf_50.fit(X_train, y_train)

print('Training Score', tree_leaf_50.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_50.score(X_test, y_test).round(4))



# Defining a function to visualize feature importance

########################
def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')
########################

########################
# Tree with the important features
########################
plot_feature_importances(tree_leaf_50,
                         train = X_train,
                         export = True)



got_data   = got.drop(['isAlive',
                       'name',
                       'title',
                       'culture',
                       'dateOfBirth',
                        'mother',
                        'father',
                        'heir',
                         'house',
                        'spouse',
                        'm_age',
                        'm_isAliveSpouse',
                        'm_isAliveHeir',
                        'm_isAliveFather',
                        'm_isAliveMother',
                        'm_spouse',
                        'm_heir',
                        'm_mother',
                        'm_father',
                        'm_culture',
                        'numDeadRelations',
                        'age',
                        'isNoble',
                        'isMarried',
                        'isAliveSpouse',
                        'isAliveHeir',
                        'isAliveFather',
                        'isAliveMother'
                        ],axis=1)




got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)

######################

tree_leaf_50 = DecisionTreeRegressor(criterion = 'mse',
                                     min_samples_leaf = 50,
                                     random_state = 508)

tree_leaf_50.fit(X_train, y_train)

print('Training Score', tree_leaf_50.score(X_train, y_train).round(4))
print('Testing Score:', tree_leaf_50.score(X_test, y_test).round(4))


#AUC Score for Decision Tree
# Generating Predictions based on the optimal c_tree model
d_tree_optimal_fit_train = tree_leaf_50.predict(X_train)

d_tree_optimal_fit_test = tree_leaf_50.predict(X_test)

print('Training AUC Score:',roc_auc_score(
        y_train,d_tree_optimal_fit_train).round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,d_tree_optimal_fit_test).round(4))

y_test.to_excel("Decision Tree Final prediction.xlsx")
# Cross-Validation on tree_leaf_50 (cv = 3)

cv_tree_3 = cross_val_score(tree_leaf_50,
                             got_data,
                             got_target,
                             cv = 3)


print(cv_tree_3)


print(pd.np.mean(cv_tree_3).round(3))


###############################################################################
# Classification Tree
###############################################################################
got_data   = got.drop(['isAlive',
                       'name',
                       'title',
                       'culture',
                       'dateOfBirth',
                        'mother',
                        'father',
                        'heir',
                         'house',
                        'spouse',
                        'm_age',
                        'm_isAliveSpouse',
                        'm_isAliveHeir',
                        'm_isAliveFather',
                        'm_isAliveMother',
                        'm_spouse',
                        'm_heir',
                        'm_mother',
                        'm_father',
                        'm_culture',
                        'numDeadRelations',
                        'age',
                        'isNoble',
                        'isMarried',
                        'isAliveSpouse',
                        'isAliveHeir',
                        'isAliveFather',
                        'isAliveMother'
                        ],axis=1)


got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


#########################
# Building Classification Trees
#########################

from sklearn.tree import DecisionTreeClassifier # Classification trees

c_tree = DecisionTreeClassifier(random_state = 508)

c_tree_fit = c_tree.fit(X_train, y_train)

###########################
# Hyperparameter Tuning with GridSearchCV
############################

from sklearn.model_selection import GridSearchCV

########################
# Optimizing for two hyperparameters
########################


# Creating a hyperparameter grid
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 500)

param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space}

# Building the model object one more time
c_tree_2_hp = DecisionTreeClassifier(random_state = 508)

# Creating a GridSearchCV object
c_tree_2_hp_cv = GridSearchCV(c_tree_2_hp, param_grid, cv = 3)


# Fit it to the training data
c_tree_2_hp_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", c_tree_2_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", c_tree_2_hp_cv.best_score_.round(4))

# Building a tree model object with optimal hyperparameters
c_tree_optimal = DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 508,
                                        max_depth = 4,
                                        min_samples_leaf = 31)

c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)

#AUC Score 
# Generating Predictions based on the optimal c_tree model
c_tree_optimal_fit_train = c_tree_optimal_fit.predict(X_train)

c_tree_optimal_fit_test = c_tree_optimal_fit.predict(X_test)

print('Training AUC Score:',roc_auc_score(
        y_train,c_tree_optimal_fit_train).round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,c_tree_optimal_fit_test).round(4))

# Cross-Validation on c_tree_optimal (cv = 3)

cv_tree_3 = cross_val_score(c_tree_optimal,
                             got_data,
                             got_target,
                             cv = 3)

print(cv_tree_3)

print(pd.np.mean(cv_tree_3).round(3))

###############################################################################
# Logistic Regression 
###############################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(solver = 'lbfgs',
                            C = 1)
logreg_fit = logreg.fit(X_train, y_train)

logreg_pred = logreg_fit.predict(X_test)

print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4))

#AUC Score 
# Generating Predictions based on the optimal model
logreg_fit_train = logreg_fit.predict(X_train)

logreg_fit_train_test = logreg_fit.predict(X_test)

print('Training AUC Score:',roc_auc_score(
        y_train,logreg_fit_train).round(4))
print('Testing AUC Score:',roc_auc_score(
        y_test,logreg_fit_train_test).round(4))

# Visualizing a confusion matrix

print(confusion_matrix(y_true = y_test,
                       y_pred = logreg_pred))

import seaborn as sns

labels = ['Alive', 'Dead']

cm = confusion_matrix(y_true = y_test,
                      y_pred = logreg_pred)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'Greys')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()

# Cross-Validation on c_tree_optimal (cv = 3)

cv_tree_3 = cross_val_score(logreg_fit,
                             got_data,
                             got_target,
                             cv = 3)

print(cv_tree_3)

print(pd.np.mean(cv_tree_3).round(3))

###############################################################################
# Random Forest
###############################################################################
# Loading new libraries
from sklearn.ensemble import RandomForestClassifier

got_data   = got.drop(['isAlive',
                       'name',
                       'title',
                       'culture',
                       'dateOfBirth',
                        'mother',
                        'father',
                        'heir',
                         'house',
                        'spouse',
                        'm_age',
                        'm_isAliveSpouse',
                        'm_isAliveHeir',
                        'm_isAliveFather',
                        'm_isAliveMother',
                        'm_spouse',
                        'm_heir',
                        'm_mother',
                        'm_father',
                        'm_culture',
                        'numDeadRelations',
                        'age',
                        'isNoble',
                        'isMarried',
                        'isAliveSpouse',
                        'isAliveHeir',
                        'isAliveFather',
                        'isAliveMother'
                        ],axis=1)


got_target = got.loc[:, 'isAlive']


X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.1,
            random_state = 508)


# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)


# Are our predictions the same for each model? 
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()



# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)

########################
# Parameter tuning with GridSearchCV
########################

from sklearn.model_selection import GridSearchCV


# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}



# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)



# Fit it to the training data
full_forest_cv.fit(X_train, y_train)



# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))


########################
# Building Random Forest Model Based on Best Parameters
########################

rf_optimal = RandomForestClassifier(bootstrap = False,
                                    criterion = 'entropy',
                                    min_samples_leaf = 16,
                                    n_estimators = 1100,
                                    warm_start = True)



rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)


print('Training Score', rf_optimal.score(X_train, y_train).round(4))
print('Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test  = rf_optimal.score(X_test, y_test)


#AUC Score 
# Generating Predictions based on the optimal Random Forest model
rf_optimal_pred_train = rf_optimal.predict(X_train)

rf_optimal_pred_test = rf_optimal.predict(X_test)

print('Training AUC Score:',roc_auc_score(
        y_train,rf_optimal_pred_train).round(3))
print('Testing AUC Score:',roc_auc_score(
        y_test,rf_optimal_pred_test).round(3))

# Cross-Validation on rf_optimal (cv = 3)

cv_tree_3 = cross_val_score(rf_optimal,
                             got_data,
                             got_target,
                             cv = 3)

print(cv_tree_3)

print(pd.np.mean(cv_tree_3).round(3))



###############################################################################
# Part 4: Model Results 
###############################################################################


#AUC Score KNN
print('KNN Training AUC Score:',roc_auc_score(
        y_train,knn_reg_optimal_pred_train),round(4))
print('KNN Testing AUC Score:',roc_auc_score(
        y_test,knn_reg_optimal_pred_test),round(4))

#AUC Score for Decision Tree
print('Decision Tree Training AUC Score:',roc_auc_score(
        y_train,d_tree_optimal_fit_train).round(4))
print('Decision Tree Testing AUC Score:',roc_auc_score(
        y_test,d_tree_optimal_fit_test).round(4))


#AUC Score for Classification Tree
print('Classification Tree Training AUC Score:',roc_auc_score(
        y_train,c_tree_optimal_fit_train).round(4))
print('Classification Tree Testing AUC Score:',roc_auc_score(
        y_test,c_tree_optimal_fit_test).round(4))

#AUC Score for Randon Forest

print('Randon Forest Training AUC Score:',roc_auc_score(
        y_train,rf_optimal_pred_train).round(3))
print('Randon Forest Testing AUC Score:',roc_auc_score(
        y_test,rf_optimal_pred_test).round(3))


##export the Decision tree result to excel

finalprediction = c_tree_optimal_fit_test
finalprediction.astype('float64')

# Had to convert the optimal AUC prediction column values to float 

Predictions = pd.DataFrame({
        'Actual' : y_test,
        'Decision Tree Prediction' : finalprediction})

Predictions.to_excel('decision_tree_final.xlsx')
