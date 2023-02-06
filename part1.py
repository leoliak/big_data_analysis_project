# -*- coding: utf-8 -*-

"""
@author: Leonidas Liakopoulos
"""

import pandas as pd
import numpy as np
import sys
from impyute.imputation.cs import fast_knn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



# --------------------------------------------------------------------------- #

"""
    Functions section for implementing all functionalities.
"""


class Classifiers(object):
    def __init__(self, train_data, train_labels, models, hyperTune=True):
        self.train_data=train_data
        self.train_labels=train_labels
        self.models = models
        self.construct_all_models(hyperTune)
        
    
    def construct_all_models(self, hyperTune):
        if hyperTune:
            for name, candidate_hyperParam in self.models.items():
                self.models[name] = self.train_with_hyperParamTuning(candidate_hyperParam[0], name,candidate_hyperParam[1])
            print ('\nTraining process finished\n')
          
            
    def train_with_hyperParamTuning(self, model, name, param_grid):
        #grid search method for hyper-parameter tuning
        grid = GridSearchCV(model, param_grid, cv=10, scoring='accuracy', n_jobs=-1)
        grid.fit(self.train_data, self.train_labels)
        print(
            '\nThe best hyper-parameter for  {} is {}, mean accuracy through 10 Fold test is {} \n'\
            .format(name, grid.best_params_, round(100*grid.best_score_,2)))

        model = grid.best_estimator_
        score = grid.best_score_
        parameters = grid.best_params_
        # train_pred = model.predict(self.train_data)
        # print('{} train accuracy = {}\n'.format(name,100*(train_pred == self.train_labels).mean()))
        return model, score, parameters


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()
    print("\n\n")
    
    
def hyperparameter_tuning(X_tr, y_tr, X_test, y_test):
    ## Define models for hyperparameter optimization
    models = {
                'SVMR':[SVC(probability=True), dict(kernel=['rbf'], gamma=np.logspace(-3, 1, 5), C=np.arange(0.01, 3.01, 0.2))],
                'SVML':[SVC(probability=True), dict(kernel=['linear'], gamma=np.logspace(-3, 1, 5), C=np.arange(0.01, 3.01, 0.2))],
                'KNN':[KNeighborsClassifier(), dict(n_neighbors=np.arange(1, 20))],
                'Random_Forest':[RandomForestClassifier(), dict(n_estimators = np.arange(200,1000,100), max_depth = np.linspace(20, 50, 7, endpoint=True))]
             }
    print('Start training phase..')
    classifiers = Classifiers(X_tr, y_tr, models, True)
    return classifiers


def dataset_preprocess(df_original, strategy = 'mean'):
    """
    Function that preprocess given dataset based on various tasks that were being asked.

    Strategies that have been implemented are:
        - Mean:   Replace NaN values with mean of each feature
        - Median: Replace NaN values with Median of each feature
        - knn:    Replace NaN values with closest k-centroid of each feature
        - drop_instance: Delete row if NaN detected.
        - drop_feature:  Delete whole feature if NaN exists.
    """
    
    df = df_original.copy()
    # Preprocess all nominal data to numerical for classification tasks
    df.replace({'Yes':1,'No':0}, inplace=True)
    df['fried'] = df['fried'].map({'Non frail':0,'Pre-frail':1,'Frail':2})
    df['gender'] = df['gender'].map({'F':0,'M':1})
    df['vision'] = df['vision'].map({'Sees well':0,'Sees moderately':1,'Sees poorly':2})
    df['audition'] = df['audition'].map({'Hears well':0,'Hears moderately':1,'Hears poorly':2})
    df['balance_single'] = df['balance_single'].map({'>5 sec':0,'<5 sec':1})
    df['gait_optional_binary'] = df['gait_optional_binary'].map({False:0,True:1})
    df['sleep'] = df['sleep'].map({'No sleep problem':0,'Occasional sleep problem':1,'Permanent sleep problem':2})
    df['health_rate'] = df['health_rate'].map({'5 - Excellent':0,'4 - Good':1,'3 - Medium':2,'2 - Bad':3,'1 - Very bad':4})
    df['health_rate_comparison'] = df['health_rate_comparison'].map({'5 - A lot better':0,'4 - A little better':1,'3 - About the same':2,'2 - A little worse':3,'1 - A lot worse':4})
    df['activity_regular'] = df['activity_regular'].map({'> 5 h per week':0,'> 2 h and < 5 h per week':1,'< 2 h per week':3,0:4})
    df['smoking'] = df['smoking'].map({'Never smoked':0,'Past smoker (stopped at least 6 months)':1,'Current smoker':2})  
    
    # Remove erroneous data
    df.replace(dict.fromkeys([999,'Test not adequate','test non realizable'], np.nan), inplace=True)
    
    if strategy == 'mean':
        df = df.fillna(df.mean())
    elif strategy == 'median':
        df = df.fillna(df.median())
    elif strategy == 'knn':
        integer_list=[col for col in df  if df[col].dtype=='int64']
        colist = list(df)
        df = fast_knn(df.values, 400)
        df = pd.DataFrame(df, columns=colist) 
        for c in integer_list:  
            df[c] = df[c].apply(lambda x: round(x, 0))
    elif strategy == 'drop_instance':
        df.dropna(thresh=30, axis='rows')
        df = df.fillna(df.mean())
    elif strategy == 'drop_feature':
        df.dropna(thresh=270, axis='columns')
        df = df.fillna(df.mean())
    else:
        print("Please select a correct strategy")
        sys.exit(1)
    return df


def train_test_set_preparation(df, split):
    labels = df['fried']
    labels = labels.values
    df = df.drop(['fried'], axis=1)  # Extract labels
    feats = df.columns
    
    # Initialize Standard Scaler
    st_scaler = preprocessing.StandardScaler()
    st_scaler.fit(df)
    data = st_scaler.transform(df)  # Transform DF data
    
    # Split sets
    X_tr, X_test, y_tr, y_test = train_test_split(data, labels, test_size = split, random_state = 42)

    # Print datasets size
    unique, counts = np.unique(y_tr, return_counts=True)
    print("Train set")
    print(dict(zip(unique, counts)))
    
    unique, counts = np.unique(y_test, return_counts=True)
    print("Test set")
    print(dict(zip(unique, counts)))
    
    return X_tr, X_test, y_tr, y_test, labels, feats



def evaluation(cls_dict, X_test, y_test):  
    for idx, classifier in enumerate(cls_dict):
        model = cls_dict[classifier]
        y_pred = model.predict(X_test)

        ## Confusion Matrix
        c_m = metrics.confusion_matrix(y_test, y_pred)
        
        print("##--##--"*10)
        print('\n')
        print('{} Comfusion Matrix:'.format(classifier))
        print_cm(c_m,['Frail', 'Prefrail', 'Nonfrail'])
        

        ## Calculate Metrics for Classifier
        precision = precision_score(y_test, y_pred, average=None)
        recall = recall_score(y_test, y_pred, average=None)
        f1 = f1_score(y_test, y_pred, average=None)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Classifier: {}".format(classifier))
        print('F1 Score:   {}'.format(f1))
        print('Precision:  {}'.format(precision))
        print('Recall:     {}'.format(recall))
        print('Accuracy:   {} %'.format(round(100*float(accuracy),6)))
    
    
    
    
# --------------------------------------------------------------------------- #


"""
    Main section for running all functionalities.
"""


# Read data as DataFrame
df_original = pd.read_csv('data/clinical_dataset.csv', delimiter=';')
strategy = 'drop_feature' # Pick one from -> ['mean', 'median', 'knn', 'drop_instance', 'drop_feature']

# Data preprocessing.
df = dataset_preprocess(df_original, strategy)
df.to_csv('data/clinical_processed.csv'.format(strategy), index=False)

# Delete certain key features that contains much information for classification task.
l=['weight_loss','exhaustion_score','gait_speed_slower','grip_strength_abnormal','low_physical_activity']
for lab in l:
    if lab in df:
        df = df.drop(lab, axis=1)
        
# Data split and model training using HPO
split = 0.2        
X_tr, X_test, y_tr, y_test, labels, feats = train_test_set_preparation(df, split)
classifiers = hyperparameter_tuning(X_tr, y_tr, X_test, y_test)

# Load trained classification models for evaluation
models = classifiers.models
cls_dict = {
             'kNN':           models['KNN'][0],
             'SVM_Linear':    models['SVML'][0],
             'SVM_RBF':       models['SVMR'][0],
             'Random_Forest': models['Random_Forest'][0]
           }

# Classifiers evaluation
evaluation(cls_dict, X_test, y_test)