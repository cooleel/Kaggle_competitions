#EAD 

import numpy as np
import pandas as pd

import os
print(os.listdir('./'))
import cv2
import json
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams['font.size'] =15
import seaborn as sns
from collections import Counter
from PIL import Image
import math

input_dir = './'


#helper functions
def classid2label(class_id):
    category, *attribute = class_id.split('_')
    return category, attribute


def print_dict(dictionary, name_dict):
    print

def json2df(json_file):
    df = pd.DataFrame()
    for index, el in enumerate(json_file):
        for key, val in el.items():
            df.loc[index, key] = val
    return df



#Check Text Data
train_df = pd.read_csv(input_dir + 'train.csv')
train_df.head()

#check the label descriptions
with open(input_dir + 'label_descriptions.json') as f:
    label_description = json.load(f)
print('This dataset information:')
print(json.dumps(label_description['info'], indent=2))

#format catagories
category_df = json2df(label_description['categories'])
category_df['id'] = category_df['id'].astype(int)
category_df['level'] = category_df['level'].astype(int)

#format attributes
attribute_df = json2df(label_description['attributes'])
attribute_df['id'] = attribute_df['id'].astype(int)
attribute_df['level'] = attribute_df['level'].astype(int)