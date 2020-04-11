from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.available
style.use('Solarize_Light2')
# xs= np.array([1,2,3,4,5,6], dtype = np.float64)
# ys= np.array([5,4,6,5,6,7], dtype= np.float64)

#Defining best fit slope and intercept

def best_fit_slope_and_intercept(xs,ys):
    
    m= (((mean(xs)*mean(ys))- mean(xs*ys)) / 
        ((mean(xs)**2) - mean(xs**2)))
    b= mean(ys)- m* mean(xs)
    return m, b

#calculating squared error function
def squared_error(ys_line, ys_orig):
    return sum((ys_line-ys_orig)**2)

def coefficient_of_determination (ys_line, ys_orig):
    squared_error_regr= squared_error(ys_line, ys_orig)
    y_mean_list = [mean(ys_orig) for _ in ys_orig]
    squared_error_orig = squared_error(ys_orig, y_mean_list)
    return 1 - (squared_error_regr/squared_error_orig)

#Creatong random datasets

def create_dataset(how_many, variance,step, correlation = False):
    val= 1
    ys= []
    for i in range(how_many):
        ys.append(val+ random.randrange(-variance, +variance))
    
        if correlation=='pos' and correlation :
            val += step
        elif correlation and correlation == 'neg' :
            val -=step
    xs=[i for i in range(len(ys))]
    
    return np.array(xs, dtype= np.float64), np.array(ys, dtype= np.float64)


xs,ys = create_dataset(40, 3, 3, correlation = 'pos')


m, b= best_fit_slope_and_intercept(xs, ys)
predict_X= 8
predictt_Y = (m * predict_X) + b 
regression_line = [m*x +b for x in xs]

print(m,b)

r_squared = coefficient_of_determination(regression_line, ys)
print(r_squared)

plt.scatter(xs,ys,c = 'red')
plt.plot(xs,regression_line)
plt.scatter(predict_X,predictt_Y, s=100,color='red')
plt.show()


