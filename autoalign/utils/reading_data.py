'''
Project: deep-sted-autoalign
Created on: Thursday, 31st July 2020 10:47:12 am
--------
@author: hmcgovern
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def read_data(path):
    
    df = pd.read_json(path)
    # print(df.head())
    # print(df['gt'].head()) # range index 0,35, 1
    
    # print(df.columns) 
    # want a bar plot of the prior and post corr 
    # and a line plot of each of the gt vs. preds
    # also want to calculate the MSE 
    # df.plot(x=df.index, y='init_corr')
    df2 = pd.DataFrame(df, columns=['init_corr', 'corr'])
    df2.plot.bar()
    plt.show()
    # print(df.loc[:, 'gt'].head().to_numpy())
    # mean_squared_error(df['gt'].to_numpy(), df['preds'].to_numpy())
    # print(df.head())
    # df3 = pd.DataFrame(df, columns=['gt', 'preds'])
    # df3.plot.line()
    # plt.show()
    



if __name__=="__main__":
    read_data('testrun1000.txt')