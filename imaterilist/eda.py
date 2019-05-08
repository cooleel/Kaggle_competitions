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
    category, *attribute = class_id.split(' ')
    return category, attribute


def print_dict(dictionary, name_dict):
    print

def json2df(json_file):
    df = pd.DataFrame()
    for index, el in enumerate(json_file):
        for key, val in el.items():
            df.loc[index, key] = val
    return df

def print_img_with_labels(img_name, labels, category_name_dict, attribute_name_dict, ax):
    img = np.asarray(Image.open(input_dir + 'train/' + img_name))
    label_interval = (img.shape[0] * 0.9)/ len(labels)
    ax.imshow(img)
    for num, attributed_id in enumerate(labels):
        x_pos = img.shape[1] *1.1
        y_pos = (img.shape[0]* 0.9)/len(labels) * (num +2) + (img.shape[0]*0.1)
        if(num ==0):
            ax.text(x_pos, y_pos-label_interval*2, 'category', fontsize=12)
            ax.text(x_pos, y_pos-label_interval, category_name_dict[attributed_id], fontsize=12)
            if(len(labels) >1):
                ax.text(x_pos, y_pos, 'attribute', fontsize=12)
            else:
                ax.text(x_pos, y_pos, attribute_name_dict[attributed_id], fontsize=12)


def print_img(img_name, ax):
    img_df = train_df[train_df.ImageId == img_name]
    labels = list(set(img_df['ClassId'].values))
    print_img_with_labels(img_name, labels, category_name_dict, attribute_name_dict, ax)

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

print(f'We have {len(category_df)} categories, and {len(attribute_df)} attributes.')
print('Each label has ID, name, supercategory and level')

#no level information

image_label_num_df = train_df.groupby('ImageId')['ClassId'].count()
'''
A lot of images have more than 1 labels, that should be just categories. 
'''
print(f'There are {len(train_df)} imageIDs in train_df and among them there are {len(train_df.ImageId.unique())} unique images.')



#visualize
fig, ax = plt.subplots(figsize = (25,7))
x = image_label_num_df.value_counts().index.values
y = image_label_num_df.value_counts().values
z = sorted(zip(x,y))
x, y = zip(*z)
index = 0
x_list=[]
y_list=[]
for i in range(1, max(x)+1):
    if (i not in x):
        x_list.append(i)
        y_list.append(0)
    else:
        x_list.append(i)
        y_list.append(y[index])
        index +=1
for i,j in zip(x_list, y_list):
    ax.text(i-1, j, j, ha='center', va='bottom', fontsize=13)
sns.barplot(x=x_list, y=y_list, ax=ax)
ax.set_xticks(list(range(0, len(x_list), 5)))
ax.set_xticklabels(list(range(1,len(x_list),5)))
ax.set_title('Number of labels per image')
ax.set_xlabel('Number of labels')
ax.set_ylabel('Counts')
plt.show()


#check the image data
print(f"THe number of training images is {len(os.listdir('./train'))}")
print(f"The number of testing images is {len(os.listdir('./test'))}")

#check the image size
image_shape_df = train_df.groupby('ImageId')['Height','Width'].first()

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,5))
ax1.hist(image_shape_df.Height, bins=100)
ax1.set_title('Height distribution')
ax2.hist(image_shape_df.Width, bins=100)
ax2.set_title('Width distribution')
plt.show()
plt.savefig('image_size_distribution.jpg')


category_name_dict = {}
for i in label_description["categories"]:
    category_name_dict[str(i["id"])] = i["name"]
attribute_name_dict = {}
for i in label_description["attributes"]:
    attribute_name_dict[str(i["id"])] = i["name"]



counter_category = Counter()
counter_attribute = Counter()
for class_id in train_df['ClassId']:
    category, attribute = classid2label(class_id)
    counter_category.update([category])
    counter_attribute.update([attribute])

#check the min and max images
img_name_min = image_shape_df.Height.idxmin()
height, width = image_shape_df.loc[img_name_min, :]
print(f'Miniman height image is {img_name_min},\n(H,W) = ({height},{width})')
fig, ax = plt.subplot()
print_img(img_name_min, ax)