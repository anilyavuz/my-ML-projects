#Importing related modules
import pandas as pd
import sys
import quandl
import sklearn as sk
import math,datetime
import numpy as np
import inspect
from inspect import getmembers
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
import time
import pickle
style.use('ggplot')
# #Checking if modules are imported
# checking = ['pandas', 'quandl', 'sklearn', 'math', 'numpy', 'sys',
#             'preprocessing', 'cross_validate', 'sklearn', 'svm',
#             'LinearRegression', 'model_selection', 'linear_model']
# for key in sys.__package__:
#     if key in sys.__package__:
#         print('{} is imported'.format(key))
#     else:
#         print('{} is not imported'.format(key))
# print('sklearn version : {}'.format(sk.__version__))
# inspect.getmembers(pd, inspect.isclass)




# defining the dataset
# all_data = quandl.get('WIKI/GOOGL')
csv_name = 'WIKIGOOGL.csv'

# all_data.to_csv(r'/Users/anilyavuz/python deneme/my-first-project/{}'.format(csv_name), index=True)
# print(all_data.head())
all_data1 = pd.read_csv(
    r'/Users/anilyavuz/python deneme/my-first-project/quandl google stock/WIKIGOOGL.csv')
                        # index_col='Date', 
                        # parse_dates=True

print(all_data1.head())

df = all_data1[['Date','Adj. Open', 'Adj. High',
               'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df.columns = ['Date','open', 'high', 'low', 'close', 'volume']
df['HL_PCT'] = ((df['high'] - df['close'])*100/df['close'])
df['PCT_change'] = (df['close'] - df['open'])/df['open']*100

print(df.head)
df = df[['Date','close', 'HL_PCT', 'PCT_change', 'volume']]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index('Date', inplace=True)
print(df.head())
print(df.tail())

#####Important adjustments
forecast_col = 'close'
df = df.fillna(-99999)
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

#Show only na values due to shift






X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]
print(df[df.isna().any(axis=1)])
print(df[df.isna()])


df = df.dropna()
Y = np.array(df['label'])


X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2)

#testing the algoruthm
clf =LinearRegression()
clf.fit(X_train,Y_train)
# #Save your trained classifier to faster interpretation
# with open('linear regression','wb') as f:
#     pickle.dump('clf',f)
    
# pickle_in = open('linear regression','rb')
# clf= pickle.load(pickle_in)
    
accuracy = clf.score(X_test,Y_test)
forecast_set= clf.predict(X_lately)
print(forecast_set,accuracy, forecast_out)

# ##Testing a new algorithm
# clf =svm.SVR()
# clf.fit(X_train, Y_train)
# accuracy = clf.score(X_test, Y_test)
# print(accuracy, forecast_out)

# ##Testing a new algorithm
# clf = svm.SVR(kernel='poly')
# clf.fit(X_train, Y_train)
# accuracy = clf.score(X_test, Y_test)
# print(accuracy, forecast_out)

# ##Testing a new algorithm
# clf =LinearRegression(njobs=-1)
# clf.fit(X_train, Y_train)
# accuracy = clf.score(X_test, Y_test)
# print(accuracy, forecast_out)

# ##Testing a new algorithm
# clf = svm.SVR(kernel='poly')
# clf.fit(X_train, Y_train)
# accuracy = clf.score(X_test, Y_test)
# print(accuracy, forecast_out)

# ##Testing a new algorithm
# clf = svm.SVR(kernel='poly')
# clf.fit(X_train, Y_train)
# accuracy = clf.score(X_test, Y_test)
# print(accuracy, forecast_out)

# ##Testing a new algorithm
# clf = svm.SVR(kernel='poly')
# clf.fit(X_train, Y_train)
# accuracy = clf.score(X_test, Y_test)
# print(accuracy, forecast_out)

###Graph part
df['forecast']=np.\
    
print(df.dtypes)

last_date= df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day= 86400
next_unix= last_unix + one_day
print(last_date,next_unix)
for i in forecast_set:
    nextdate= datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[last_date]= [np.nan for k in range(len(df.columns)-1)] +[i]


df['close'].plot() 
df['forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Forecast')
# plt.show()
