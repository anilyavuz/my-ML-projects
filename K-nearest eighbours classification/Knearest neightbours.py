import pandas as pd
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
accuracies_knn = []
for i in range(25):
    df = pd.read_csv('/Users/anilyavuz/python deneme/my-first-project/K-nearest eighbours classification/breast-cancer-wisconsin.data',
                    header= 0)

    # df1 = df.isna().any(axis=1)
    # df3 = df[df1]
    # df2 = df[df.isna()]
    # dfF = df.isna()
    # df1[0]=1
    
    df.replace('?',-99999,inplace = True)
    df.drop(['id'],1, inplace= True)
    X= np.array(df.drop(['class'],1))
    Y= np.array(df['class'])


    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)


    clf= neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)

    accuracy = clf.score(X_test,y_test)
    accuracies_knn.append(accuracy)
print(sum(accuracies_knn)/len(accuracies_knn))
    # print(accuracy)
    # new_array = np.array([[4, 3, 2, 3, 3, 4, 3, 3, 2]])
    # # new_array=new_array.reshape(len(new_array), -1)
    # prediction = clf.predict(new_array)

    # print(prediction)





# creating our own Knearest 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
from matplotlib import style
from collections import Counter
import warnings
import random
style.use('fivethirtyeight')
# style.available

dataset = {'k':[[2,1],[3,1],[3,2]], 'r':[[5,6],[6,7],[7,8]] }

new_features= [10,7]

[[plt.scatter(ii[0],ii[1],s=100,c=i) for ii in dataset[i]] for i in dataset]
plt.scatter( new_features[0],new_features[1])
plt.show()


    #KNN Algos
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is value less than total voting groups')

    distances = []
    for groups in data:
        for features in data[groups]:
            euclidian = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidian, groups])

    votes = [i[1] for i in sorted(distances)[:k]]
    # print(Counter(votes))
    votes_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    return votes_result, confidence
# k_nearest_neighbors(dataset,new_features,k=3)







# ########Testing the real data with my algorithm
# -------------------------------------------------
# ---------------------------------------------------
# -----------------------------------------------

#IMporting what we need
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import random


#KNN Algos
# def k_nearest_neighbors(data, predict, k=3):
#     if len(data) >= k:
#         warnings.warn('K is value less than total voting groups')

#     distances = []
#     for groups in data:
#         for features in data[groups]:
#             euclidian = np.linalg.norm(np.array(features) - np.array(predict))
#             distances.append([euclidian, groups])

#     votes = [i[1] for i in sorted(distances)[:k]]
#     # print(Counter(votes))
#     votes_result = Counter(votes).most_common(1)[0][0]
#     confidence = Counter(votes).most_common(1)[0][0]/k
#     return votes_result, confidence
accuracies=[]
for i in range(25):
    df = pd.read_csv('/Users/anilyavuz/python deneme/my-first-project/K-nearest eighbours classification/breast-cancer-wisconsin.data')
    df.replace('?',-99999,inplace=True)
    df.drop(['id'],1,inplace= True)

    full_data= df.astype(float).values.tolist()
    random.shuffle(full_data)
    # print(full_data[:5])
    test_size= 0.2
    train_set= {2:[],4:[]}
    test_set= {2:[],4:[]}
    train_data = full_data[:-int(test_size * len(full_data))]
    test_data= full_data[-int(test_size * len(full_data)):]
    # print(test_data[:10])



    for i in train_data:
        train_set[i[-1]].append(i[:-1])
        
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0 
    total = 0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data,k=5)
            if group == vote:
                correct+=1
            else:
                print(confidence)
            total+=1
            
        accuracy= correct/total
        accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies))

            
        
     