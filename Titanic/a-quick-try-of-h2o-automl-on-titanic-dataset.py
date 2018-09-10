# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#data cleaning and feature engineering 
def get_name_prefix(data):
    prefix = pd.Series(np.ones(data.shape[0]), index=data.index)
    data['Prefix'] = prefix
    data.loc[data.Name.str.contains('Miss.', regex=False), 'Prefix'] = 2
    data.loc[data.Name.str.contains('Mrs.', regex=False), 'Prefix'] = 3
    data.loc[data.Name.str.contains('Mr.', regex=False), 'Prefix'] = 4
    
# https://stackoverflow.com/a/42523230
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        del df[each]
        df = pd.concat([df, dummies], axis=1)
    return df

def normalize(df, mean, std):
    """
    @param df pandas DataFrame
    @param mean pandas Series of column values mean
    @param std pandas Series of column values standard deviation
    """
    for i in range(mean.size):
        df[mean.index[i]] = (df[mean.index[i]] - mean[0]) / std[0] 

def process_data(data):
    # get prefix data
    get_name_prefix(data)
    # remove name and ticket
    data.drop(['Ticket', 'Name'], inplace=True, axis=1)
    # sex
    data.loc[data.Sex != 'male', 'Sex'] = 0;
    data.loc[data.Sex == 'male', 'Sex'] = 1;
    # cabin
    data.Cabin.fillna('0', inplace=True)
    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4
    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5
    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6
    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7
    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8
    # embarked
    data.Embarked.fillna(0, inplace=True)
    data.loc[data.Embarked == 'C', 'Embarked'] = 1
    data.loc[data.Embarked == 'Q', 'Embarked'] = 2
    data.loc[data.Embarked == 'S', 'Embarked'] = 3
    data.fillna(-1, inplace=True)
    
    data = one_hot(data, ('Pclass', 'Sex', 'Cabin', 'Embarked', 'Prefix'))
    return data.astype(float)

#load data
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')

train = process_data(train_raw)
test = process_data(test_raw)

data_mean = train[['Age','Fare','SibSp','Parch']].mean(axis=0)
data_std = train[['Age','Fare','SibSp','Parch']].std(axis=0)

normalize(train, data_mean, data_std)
normalize(test, data_mean, data_std)

test, train = test.align(train, axis=1, fill_value=0)

#start H2O 
import h2o
from h2o.automl import H2OAutoML

h2o.init()

#load data as h2o frames
train = h2o.H2OFrame(train)
test = h2o.H2OFrame(test)

#drop passengerId from data set
passId = test['PassengerId']
train = train.drop('PassengerId',axis =1)
test = test.drop('PassengerId',axis =1)

#identify predictors and labels
x = train.columns
y = 'Survived'
x.remove(y)

#for binary classification, lables should be a factor
train[y] = train[y].asfactor()

# Run AutoML
aml_ti = H2OAutoML(max_runtime_secs= 120,max_models= 10, seed= 7,nfolds= 10)
aml_ti.train(x = x, y = y,
          training_frame = train)
          
#check the leaderboard
lb_ti = aml_ti.leaderboard
lb_ti

#prediction
pred = aml_ti.leader.predict(test)

#save predict results to submission form
pred_df = pred.as_data_frame()
pred_res = pred_df.predict
passId_df = passId.as_data_frame()
res_ti = pd.concat([passId_df,pred_res],axis=1,ignore_index = True)
res_ti.columns = ['PassengerId','Survived']
res_ti.to_csv('mypred.csv',index=False)

#http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html