#Go down until 'code starts here' part
# #######################################################
########################################################
# import pandas as pd
# import numpy as np
# from sklearn import preprocessing, neighbors, svm
# from sklearn.model_selection import cross_validate
# from sklearn.model_selection import train_test_split

# # accuracies_knn = []
# # for i in range(25):
# df = pd.read_csv('/Users/anilyavuz/python deneme/my-first-project/K-nearest eighbours classification/breast-cancer-wisconsin.data',
#                 header= 0)

# # df1 = df.isna().any(axis=1)
# # df3 = df[df1]
# # df2 = df[df.isna()]
# # dfF = df.isna()
# # df1[0]=1

# df.replace('?',-99999,inplace = True)
# df.drop(['id'],1, inplace= True)
# X= np.array(df.drop(['class'],1))
# Y= np.array(df['class'])


# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# clf= svm.SVC(kernel='linear')
# clf.fit(X_train,y_train)

# accuracy = clf.score(X_test,y_test)
#     # accuracies_knn.append(accuracy)
# # print(sum(accuracies_knn)/len(accuracies_knn))
# print(accuracy)
# print(2)
# a=[1,2,3,4,5]
# #################### Random codees###############
# # 
# # # for i in np.array(a):
    
# #     print(i, end='')
    
    
# # dict={'isalnum()':isalnum,'isalpha()':isalpha,'isdigit()':isdigit,'islower()':islower,'isupper()':isupper}
# # print(dict[2])
# # list[isalnum]
# # dict.keys().list[0]()
########################################################
########################################################
########################################################
########################################################
# Code Starts here

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class support_vector_machine:
    def __init__(self,visualization =True):
        self.visualization = visualization
        self.colors={1:'r',-1:'b'}
        if self.visualization :
            self.fig=plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train data 
    def fit(self,data):
        
        self.data=data
        # {||w||: [w,b]}
        opt_dict= {}
        transforms=[[1,1],
                    [1,-1],
                    [-1,1],
                    [-1,-1]]    
        all_data=[]
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.min_feature_value=min(all_data)
        self.max_feature_value= max(all_data)
        all_data=None
        
        ### support vectors yi((xi.w)+b) =1 
        ### keep stepping until you reach to yi((xi.w)+b) =1.01
        
        
        step_sizes = [self.min_feature_value  * 0.1
                    #   ,self.min_feature_value *0.01,
                      #point of expense
                    #   self.min_feature_value *0.001
                    ]
        #extremely expensive
        b_range_multiplier=5
        #we do not need to take as small of steps 
        #with b(wx+b) s we do w
        b_multiplier =5
        latest_optimum = self.max_feature_value*10
        
        
        #beginning the steps
        for step in step_sizes:
            w = [latest_optimum, latest_optimum]
            optimized = False
            while optimized==False:

                for b in np.arange(-1*(self.max_feature_value*b_range_multiplier),
                                self.max_feature_value*b_range_multiplier,
                                b_multiplier*step
                                ):
                    for transformation in transforms:
                        w_t=transformation*w
                        found_option = True
                        #WEAKEST LINK IN THE SVM FUNDEMENTALLY IN GENERAL 
                        # SMO ATTEMTS TO FIX IT 
                        # yi(xi.w)+b=>1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi(np.dot(w_t,xi)+b)>= 1:
                                    found_option=False
                            
                        if found_option==True:
                            opt_dict[np.linalg.norm(w_t)]= [w_t,b]
                                
                    
                if w[0] <0:
                    optimized=True
                    print('optimized a step')
                else:
                    #if w = [5,5]
                    #step=1
                    #w-1=[4,4]
                    w=w-step
            norms= sorted([i for i in opt_dict])
            #||w|| = [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+ step*2
            
            
    
    #####################test set
    def predict(self,features):
        
        # sign(x.v+b)
        classification =np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification != 1 and self.visualization:
            self.ax.scatter(features[0],features[1], s=200, marker = '*', c=self.colors[classification])        
        return classification
    
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],c=self.colors[i])for x in data_dict[i]]for i in data_dict]
        
        #hyperplane x.w+b
        # v = x.w+b  
        #psv =1 
        #nsv= -1
        #decision plane = 0
        
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v)/w[1]
            
        datarange=(self.min_feature_value*0.9,self.max_feature_value*1.1)
        hypr_x_min = datarange[0]
        hypr_x_max= datarange[1]
        #x.w+b = 1
        #psv (positive) hyperplane
        psv1= hyperplane(hypr_x_min,self.w,self.b,1)
        psv2 =hyperplane(hypr_x_max, self.w, self.b, 1)
        self.ax.plot([hypr_x_min,hypr_x_max],[psv1, psv2])
        
        #x.w+b = -1
        #nsv negative hyperplane
        nsv1 = hyperplane(hypr_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hypr_x_max, self.w, self.b, -1)
        self.ax.plot([hypr_x_min, hypr_x_max], [nsv1, nsv2])

        #x.w+b = 0
        #db hyperplane
        db1 = hyperplane(hypr_x_min, self.w, self.b, 0)
        db2 = hyperplane(hypr_x_max, self.w, self.b, 0)
        self.ax.plot([hypr_x_min, hypr_x_max], [db1, db2])

        plt.show()


data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8],]),
             1: np.array([[5, 1], [6, -1], [7, 3],])}

svm = support_vector_machine()
svm.fit(data=data_dict)
svm.visualize()
