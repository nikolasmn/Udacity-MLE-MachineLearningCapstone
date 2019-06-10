import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of displ}ay() for DataFrames
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
import plotly.offline as py

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
import time

# Import supplementary visualization code visuals.py
import visuals as vs
import utils as utl


def model_execution(X, y):
    
    X_train, X_test, y_train, y_test = training_testing(X,y)
    features_x_coefficients, lm = fit_model(X_train, X_test, y_train, y_test)
    error_valuation(X_test,y_test,lm)
    model_viz(X_train, y_train, X_test, y_test, lm)
    


def training_testing(X, y):
    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.3, 
                                                    random_state = 0)

    # Show the results of the split
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    return X_train, X_test, y_train, y_test



def fit_model(X_train, X_test, y_train, y_test):
    #this creates a linearregression object    
    lm = LinearRegression()
    lm.fit(X_train,y_train)

    print("Estimated intercept coefficiente", lm.intercept_)
    print("Number of coefficients:", len(lm.coef_))

    r_sq = lm.score(X_test, y_test)
    print("coefficient of determination:", r_sq)

    features_x_coefficients = pd.DataFrame(list(zip(X_train.columns, lm.coef_)), columns = ["features", "estimatedCoefficients"])
    
    
    return features_x_coefficients, lm 


def error_valuation(X_test, y_test, lm):
    #this creates a linearregression object    
    #lm = LinearRegression()
    #lm.fit(X,y)
    y_pred = lm.predict(X_test)
    print("Mean Absolute Error MAE:", metrics.mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error MSE:",metrics.mean_squared_error(y_test, y_pred))
    print("R Squared:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

def model_viz(X_train, y_train, X_test, y_test, lm):
    # plot prediction and actual data
    model_viz_1(X_test, y_test, lm)
    model_viz_2(X_test, y_test, lm)
    model_viz_3(X_test, y_test, lm)
    model_viz_41(X_train, y_train, X_test, y_test, lm)
    model_viz_42(X_train, y_train, X_test, y_test, lm)
    
def model_viz_1(X_test, y_test, lm):
    # plot prediction and actual data
    y_pred = lm.predict(X_test) 
    plt.plot(y_test, y_pred, '.')

    # plot a line, a perfit predict would all fall on this line
    x = np.linspace(0,100, 100)
    y = x
    plt.plot(x, y)
    plt.show()
    
def model_viz_2(X_test, y_test, lm):
    result = y_test - lm.predict(X_test)
    df = pd.DataFrame(result, dtype='int')
    sns.distplot(df['overall'], bins = 10)
    plt.title('Original Overall and Predict', fontsize=18)
    plt.xlabel('Overall difference)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    
def model_viz_3(X_test, y_test, lm):
    y_pred = lm.predict(X_test) 
    y_test_ = np.array(list(y_test))
    y_pred_ = np.array(y_pred)
    df = pd.DataFrame({'Actual': y_test_.flatten(), 'Predicted': y_pred_.flatten()})
    #df
    df1 = df.head(50)
    df1.plot(kind='bar',figsize=(10, 8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title('Original Overall and Predict', fontsize=18)
    plt.xlabel('Overall difference)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.show()
    
def model_viz_41(X_train, y_train, X_test, y_test, lm):
    plt.scatter(lm.predict(X_train[:10]), lm.predict(X_train[:10]) - y_train[:10], c='b', s=40, alpha=0.5)
    plt.scatter(lm.predict(X_test[:10]),lm.predict(X_test[:10]) - y_test[:10], c='g', s=40)
    plt.hlines(y=0, xmin=0, xmax=100)
    plt.title('residual plot using training (blue) and test (green) data')
    plt.ylabel('residuals')

def model_viz_42(X_train, y_train, X_test, y_test, lm):
    plt.scatter(lm.predict(X_train), lm.predict(X_train) - y_train, c='b', s=40, alpha=0.5)
    plt.scatter(lm.predict(X_test),lm.predict(X_test) - y_test, c='g', s=40)
    plt.hlines(y=0, xmin=0, xmax=100)
    plt.title('residual plot using training (blue) and test (green) data')
    plt.ylabel('residuals')
    
    
    
def advanced_RFE(X, y, ts, nf):   

    mdl = LinearRegression()
    #Initializing RFE model
    rfe = RFE(mdl, 56)
    #Transforming data using RFE
    X_rfe = rfe.fit_transform(X, y)  
    #Fitting the data to model
    mdl.fit(X_rfe, y)
    #print(rfe.n_features_)
    #print(rfe.support_)
    #print(rfe.ranking_)
    
    features_x_score = pd.DataFrame(list(zip(X.columns, rfe.ranking_,rfe.support_)), 
                       columns = ["features", "ranking","support"])
    
    print(features_x_score.sort_values(by=['support'], ascending=[False]).head(10))

    tic_Total_Processing = time.clock()
    tic_Ite_Processing = time.clock()
    
    final_list = []
    feature_list = []
    for z in range(1,nf):
        nof_list=np.arange(1,z)            
        high_score=0
        #Variable to store the optimum features
        nof=0           
        score_list =[]
        for n in range(len(nof_list)):
            tic_Ite_Processing = time.clock()
            #X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = ts , random_state = 0)
            model = LinearRegression()
            rfe = RFE(model,nof_list[n])
            X_train_rfe = rfe.fit_transform(X_train,y_train)
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe,y_train)
            score = model.score(X_test_rfe,y_test)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]
        print("Optimum number of features: %d" %nof)
        print("Score with %d features: %f" % (nof, high_score))
        for w in range(1,z):
            feature_list.append(X.columns[w])
        
        feature_list_final = list(set(feature_list))
    
        final_list.append([X.columns[z],z, nof, high_score, feature_list_final])
        toc_Ite_Processing = time.clock()
        Iteration_Time = toc_Ite_Processing - tic_Ite_Processing
        print("# Iteration " + str(z) + " processing time:" + str(Iteration_Time))

    toc_Total_processing = time.clock()
    print("# Total processing time:", toc_Total_processing - tic_Total_Processing)    
    
    features_x_rfe = pd.DataFrame(data=final_list, columns=["features","id","number_features","higher_score","feature_list"])
    
    features_x_rfe['feature_list'] = features_x_rfe['feature_list'].astype(str).str.replace("[", "").str.strip()
    features_x_rfe['feature_list'] = features_x_rfe['feature_list'].astype(str).str.replace("]", "").str.strip()
    features_x_rfe['feature_list'] = features_x_rfe['feature_list'].astype(str).str.replace("'", "").str.strip()
    
    return features_x_rfe