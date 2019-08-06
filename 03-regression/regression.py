import csv
import numpy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from typing import List
# Your code for regression and R2 here
       

def linear_regression_least_squares(x: List[float], y: List[float]):
    x2 = [x_i**2 for x_i in column]
    xy = [x_i*y_i for x_i, y_i in zip(column,target)]
    mean_x = sum(column)/len(column)
    mean_y = sum(target)/len(target)
    mean_x2 = sum(x2)/len(x2)
    mean_xy = sum(xy)/len(xy)
    var_x = mean_x2 - (mean_x*mean_x)
    cov_xy = mean_xy -( mean_x*mean_y)        

    # Regression
    beta = cov_xy/var_x
    alpha = mean_y - (beta*mean_x)
    
    return alpha,beta

def predict(alpha,beta,x_i):
    return beta*x_i+alpha

def r_squared(alpha: float, beta: float, x: List[float], y: List[float]):
    SS_tot = 0
    SS_res = 0
    mean_y = sum(target)/len(target)
    for i in range(len(y)):
        SS_tot += (y[i]-mean_y)**2
        SS_res +=  (y[i] - predict(alpha,beta,x[i]))**2
    R2 = 1- (SS_res/SS_tot)
    return R2
 
if __name__ == "__main__":
    # Here, we load the boston dataset
    boston = csv.reader(open('boston.csv'))  # The boston housing dataset in csv format
    # First line contains the header, short info for each variable
        


    header = boston.__next__()  # In python2, you might need boston.next() instead
    # Data will hold the 13 data variables, target is what we are trying to predict
    data, target = [], []
    for row in boston:
        # All but the last are the data points
    
        data.append([float(r) for r in row[:-1]])
        # The last is the median house value we are trying to predict
        target.append(float(row[-1])) 
    # Now, use the dataset with your regression functions to answer the exercise questions
    print("Names of the columns")
    print(header)
    print("First row of data ->variable to predict")
    print(data[0], " -> ", target[0])
    print(data[1], " -> ", target[1])


    # Plot, regression here
    # Example of writing out the R2.txt file, with 0.0 guess for coefficient of correlation
    fout = open('results.txt', 'w')
    for i in range(13):
        column = [row[i] for row in data]  # get the column
        target  # target is always the median house value

        # Visualization
        '''
        data = [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]
        '''
        plt.scatter(column,target)
        plt.xlabel(header[i])
        plt.ylabel("Predict Price")
        plt.savefig(str(i)+".png")
        plt.clf()   

        alpha, beta = linear_regression_least_squares(column, target)
        R2 = r_squared(alpha,beta,column,target)

        fout.write('%f,%f,%f\n' % (alpha, beta, R2))  # One line per variable
    fout.close()        
