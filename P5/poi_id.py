
# coding: utf-8

# In[4]:

#!/usr/bin/python



import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append("../tools/")


from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data







### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#financial features: ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 
#                     'bonus', 'restricted_stock_deferred', 'deferred_income', 
#                     'total_stock_value', 'expenses', 'exercised_stock_options', 
#                     'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

#email features: ['to_messages', 'email_address', 'from_poi_to_this_person', 
#                 'from_messages', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi']
#               (units are generally number of emails messages; 
#                notable exception is ‘email_address’, which is a text string)
    
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 
                 'bonus', 'restricted_stock_deferred', 'deferred_income', 
                 'total_stock_value', 'expenses', 'exercised_stock_options', 
                 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 
                 'from_this_person_to_poi', 'shared_receipt_with_poi']


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Store to my_dataset for easy export below.
#my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, remove_NaN=False, 
#                     remove_all_zeroes=False, remove_any_zeroes=False, 
#                     sort_keys=False)

#labels, features = targetFeatureSplit(data)

#Uncomment next line to see shape of the data and the number of POIs in the dataset
#print data.shape, sum(labels)
    

#this next block of code identifies all NaN values in the data
#as well as identify the amount of data
nan_count = 0
total_count = 0
feature_list = []
first_time=True
for i, (nm, it) in enumerate(data_dict.items()):
    if i > 0:
        first_time=False
    for x, y in it.items():
        if first_time:
            feature_list.append(x)
        total_count+=1
        if y == 'NaN':
            nan_count+=1
            
#Uncomment this next line to print out the number of individuals in the dataset as well as
#the percentage NaN
#print nan_count, total_count-21, nan_count/((total_count-21)*1.) #subtracting fields from 'TOTAL'
    
    
    
    
### Task 2: Remove outliers

#1) convert the dict into a pandas DF
#2) drop the TOTAL column and replace NaN with -1
#3) convert POI labels to 0/1 and drop email address
#4) keep values that are within 3 std of mean
#5) get names for those excluded and counts
from scipy import stats
df = pd.DataFrame()
df = df.from_dict(data_dict, orient='index')
df = df.drop(['TOTAL'], axis=0)
df = df.replace("NaN",np.nan)
#print df.count()
#print('')
df = df.fillna(-1)




#labels = df['poi'].map({False:0, True:1})
features = df.drop(['email_address'], axis=1)

#df = pd.DataFrame(data, columns=[features_list])
features_poi = features[features['poi']==1]
features_no_poi = features[features['poi']==0]
features_no_poi = features_no_poi.drop(['poi'], axis=1)

features_clean = features_no_poi[(np.abs(stats.zscore(features_no_poi)) < 3).all(axis=1)]
fc_names = list(features_clean.index)
f_names = list(features_no_poi.index)
excluded_names = set(fc_names).union(f_names) - set(fc_names).intersection(fc_names)

#6 of the 33 excluded individuals were POIs (~18%)... 
#total population of POIs in the entire dataset is 18 of 145 (~12%)...

#print excluded_names, len(excluded_names)
#excluded 26 outliers that are not POIs


features_clean['poi'] = 0
features_new = pd.concat([features_clean, features_poi], axis=0)
#print features_new.shape, features_new.columns



### Task 3: Create new feature(s)


#new features here... exclude from final model
features_new['sum'] = features_new.sum(axis=1)
features_new['mean'] = features_new.mean(axis=1)



#put dataframe back into a dictionary so it can be pickled appropriately
my_dataset = features_new.to_dict('index')
labels = features_new['poi']
features_new = features_new.drop(['poi'], axis=1)
#print features_new.shape


### Task 4: Try a varity of classifiers

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, SelectFpr
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

 
    
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


from sklearn.preprocessing import MinMaxScaler
minmaxscaler = MinMaxScaler(feature_range=(-1,1))


select = SelectKBest(k=5)
#clf_2 = RandomForestClassifier()
#clf_1 = ExtraTreesClassifier(criterion='gini', n_estimators=50, min_samples_split=5, max_depth=4)
#clf_2 = KNN(n_neighbors=9)
#clf_1 = SVC(C=.01, kernel='sigmoid', gamma=.001)
clf_1 = LinearSVC(C=10., loss='squared_hinge', class_weight='balanced')
#clf_1 = LogisticRegression()
#clf_1 = KMeans(n_clusters=2)



C_range = np.logspace(-2, 10, 13)
#gamma_range = np.logspace(-9, 3, 13)

steps = [('feature_selection', select),
         ('minmax', minmaxscaler),
         #('logitistic', clf_1)]
         ('svc', clf_1)]
         #('kmeans', clf_1),
         #('extra_trees', clf_1)]
         #('random_forest', clf_2)]
        #('knn', clf_2)]

        

param_grid = {"feature_selection__k":[2, 5, 9, 14, 'all'], 
              #"kmeans__n_clusters":[2,3,4,5,6,7,8],
              "svc__C":C_range,
              "svc__loss":['hinge', 'squared_hinge'],
              "svc__class_weight":['balanced', None]
              #"svc__kernel":["sigmoid", "rbf"]
              #"knn__n_neighbors":[3, 5, 7, 9]
              #"random_forest__n_estimators":[50, 100, 200],
              #"random_forest__min_samples_split":[2, 3],
              #"random_forest__criterion":["gini", "entropy"],
              #"extra_trees__n_estimators":[50, 100, 200],
              #"extra_trees__min_samples_split":[2, 3, 5],
              #"extra_trees__criterion":["gini", "entropy"],
              #"extra_trees__max_depth":[4,6,8]
             }

cv = StratifiedShuffleSplit(labels, n_iter=10, test_size=0.33, random_state=42)
#clf = GridSearchCV(Pipeline(steps), param_grid, cv=5)
clf = Pipeline(steps)

#X_train, X_test, y_train, y_test = train_test_split(features_clean, labels, 
#                                                    test_size=0.5, 
#                                                    random_state=42)

X = features_new.as_matrix()



for train_index, test_index in cv:
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = labels[train_index], labels[test_index]



clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print "Cross validated score (cv=5): {}".format(np.mean(cross_val_score(clf, 
                                                                        X_train, 
                                                                        y_train, 
                                                                        cv=5)))
print('')
print "Confusion Matrix: "
print confusion_matrix(y_test, preds)
print('')
print "Precision: {}".format(precision_score(y_test, preds))
print "Recall:    {}".format(recall_score(y_test, preds))



#if using gridsearch
#clf.best_params_


#if running pipeline
#zip(features_clean.columns[select.get_support()], select.scores_)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:




# In[11]:

print zip(features_clean.columns[select.get_support()], select.scores_)


# In[ ]:




# In[ ]:



