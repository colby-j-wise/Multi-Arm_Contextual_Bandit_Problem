"""
Colby Wise
LinUCB1 Multi-Arm Bandit Problem
Re-inforcement Learning

Data helper functions that return the correct
column or row from a pandas dataframe
"""

import pandas as pd


"""
Read in CSV data file and remove 'timestamp' column
@param:
    file - file directory
@return:
    data - pandas dataframe of action & context vectors
"""
def getData(file):
        data = pd.read_csv(file, sep=" ", header=None)
        data = data.loc[:,:101]
        return data 

"""
Helps sequentially retrieve a context vector at each time step
in the algorithm
@param:
    data - pandas df
    idx - time step (row) wanted
@return:
    x - context vector for given time step
"""
def getContext(data, idx):
    return data.loc[idx, 2:]

"""
Helps sequentially retrieve an action vector at each time step
in the algorithm
@param:
    data - pandas df
    idx - time step (row) wanted
@return:
    action - action vector for given time step
"""
def getAction(data, idx):
    return data.loc[idx, 0]

"""
Helps sequentially retrieve a reward vector at each time step
in the algorithm
@param:
    data - pandas df
    idx - time step (row) wanted
@return:
    reward - reward vector for given time step
"""
def getReward(data, idx):
    return data.loc[idx, 1]