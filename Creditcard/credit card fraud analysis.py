#this is my new project

#importing related libraries
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report
import sys
import numpy as np
import pandas as pd
import matplotlib as plt1
import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import sklearn


#Check the versions of the packages

print('Python : {}'.format(sys.version))
print('Numpy : {}'.format(np.__version__))
print('Pandas : {}'.format(pd.__version__))
print('Matplotlib : {}'.format(plt1.__version__))
print('sklearn : {}'.format(sklearn.__version__))
print('Scipy : {}'.format(scipy.__version__))
print('Seaborn : {}'.format(sns.__version__))


#Importing the dataset

all_data = pd.read_csv('/Users/anilyavuz/python deneme/my-first-project/Creditcard/credit:card.csv')

#explore the dataset
print(all_data.columns)
print(all_data.shape)
print(all_data.describe())

sample_data = all_data.sample(frac=0.1,  random_state=1)
print(sample_data.shape)
# pd.__dict__
# dir()
sample_data.hist(figsize=(20, 20))
#plt.show()
#determinew number of fradulent transactions


k1 = sample_data['Class']
Fraud = sample_data[k1 == 1]
Valid = sample_data[k1 == 0]
# outlier_fraction= len('Fraud') /float(len(Valid))
outlier_fraction = len(Fraud)/float(len(Valid))
print("Fraud : {}".format(len(Fraud)))
print('VALID : {}'.format(len(Valid)))
print('Outlier Fraction : {}'.format(outlier_fraction))
#correlation matrix
corr_matrix = sample_data.corr()
fig = plt.figure(figsize=(20, 20))
sns.heatmap(sample_data.corr(), vmax=0.8, square=True)
# plt.show()

#Get all the columns from the dataframe
columns = sample_data.columns.to_list()

#Removing the Target variable Class
columns = [col for col in columns if col not in ["Class"]]


#store the variable we are predicting on
Target = "Class"
X = sample_data[columns]
Y = sample_data['Class']

# APPLYING ALGORITHMS TO THE PROJECT NOW

#Define a random_state
state = 1
#outlier detection Methods

Classifiers = {

    "Isolation Forest" : IsolationForest(max_samples=len(X),
                                        contamination=outlier_fraction,
                                        random_state=1),
    "Local Outlier Factor" : LocalOutlierFactor(n_neighbors=20,
                                              contamination= outlier_fraction)  
                }

#Fit the model
n_outliers = len(Fraud)
print(list(enumerate(Classifiers.items())))
print(Classifiers)

for i, (clf_name,clf) in enumerate(Classifiers.items()):
   #Fit data and mark the outliers
    if clf_name == "Local Outlier Factor":
        
        Y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
        
        
    else:
        
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        Y_pred = clf.predict(X)
        
#replace the numbers with 0 for Valid and 1 for Fraud
Y_pred[Y_pred==0]=1
Y_pred[Y_pred==-1] =1
n_errors = (Y_pred != Y).sum()
#run classification metrics
print('{}:{}'.format(clf_name, n_errors))
print(accuracy_score(Y, Y_pred))
print(classification_report(Y,Y_pred))
