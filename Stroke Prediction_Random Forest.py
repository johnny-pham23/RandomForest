#!/usr/bin/env python
# coding: utf-8

# In[72]:


#!/usr/bin/env python
# coding: utf-8

# # Understanding the data

# id: unique identifier
# 
# gender: "Male", "Female" or "Other"
# 
# age: age of the patient
# 
# hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
# 
# heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
# 
# ever_married: "No" or "Yes"
# 
# work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
# 
# Residence_type: "Rural" or "Urban"
# 
# avg_glucose_level: average glucose level in blood
# 
# bmi: body mass index
# 
# smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"
# 
# stroke: 1 if the patient had a stroke or 0 if not
# 
# Note: "Unknown" in smoking_status means that the information is unavailable for this patient

# # Importing Packages


# In[73]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif


# In[74]:


# # Data Cleaning


# In[75]:


df=pd.read_csv("healthcare-dataset-stroke-data.csv")


# In[76]:


df.head()


# In[77]:


type(df)


# In[78]:


list(df.keys())


# In[79]:


df.isnull().sum()


# In[80]:


# #### Checking for null values in dataset


# In[81]:


df.info()


# In[82]:


# Categorical : gender, ever_married, work_type, residence_type, smoking_status
# 
# Numerical : age, hypertension, heart_disease, avg_glucose_level, bmi
# 
# hypertension & heart_disease have int dtypes, but we can check out that they are in categorical style


# In[83]:


df.describe().transpose()


# In[84]:


# #### Body mass index (BMI) is defined as person's weight measurement into proportion to his/her weight. In other words, it is obtained by dividing the person's weight by the square of his/her height. BMI = body weight (kg) / (height(m) x height(m))


# In[85]:


df["bmi"].fillna(df["bmi"].median(), inplace=True)


# In[86]:


# #### Fill in NA in BMI column with the median.


# In[87]:


df.isna().sum()


# In[88]:


df = df.drop("id", axis=1)
df.head(5)


# In[89]:


##### Identify the unique elements in gender column for encoding. 


# In[90]:


df.gender.value_counts()


# In[91]:


#We only have 1 "other" gender so we can remove it 


# In[92]:


df = df[df.gender != 'Other']


# In[93]:


df=pd.get_dummies(df,columns=["gender"])


# In[94]:


df


# In[95]:


set(df.ever_married)


# In[96]:


df.ever_married.value_counts()


# In[97]:


df=pd.get_dummies(df,columns=["ever_married"])


# In[98]:


df


# In[99]:


set(df.work_type)


# In[100]:


df.work_type.value_counts()


# In[101]:


df=pd.get_dummies(df,columns=["work_type"])


# In[102]:


df


# In[103]:


set(df.Residence_type)


# In[104]:


df.Residence_type.value_counts()


# In[105]:


df=pd.get_dummies(df,columns=["Residence_type"])


# In[106]:


df


# In[107]:


set(df.smoking_status)


# In[108]:


df.smoking_status.value_counts()


# In[109]:


df=pd.get_dummies(df,columns=["smoking_status"])


# In[110]:


df


# In[111]:


df.info()


# In[112]:


df.shape

# ### Correlation


# In[113]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,fmt='.2')


# In[114]:


# ### Finding what features have more correlation with stroke


# In[115]:


classifier = SelectKBest(score_func=f_classif,k=5)
fits = classifier.fit(df.drop('stroke',axis=1),df['stroke'])
x=pd.DataFrame(fits.scores_)
columns = pd.DataFrame(df.drop('stroke',axis=1).columns)
fscores = pd.concat([columns,x],axis=1)
fscores.columns = ['Attribute','Score']
fscores.sort_values(by='Score',ascending=False)


# In[116]:


important_cols=fscores[fscores['Score']>50]['Attribute']
print(important_cols)


# In[117]:


# # Normalization


# In[118]:


#look at normalization again. 
from sklearn.preprocessing import MinMaxScaler


# In[119]:


#normalization range between 0-1
scaler=MinMaxScaler(copy=True, feature_range=(0,1))
x = scaler.fit_transform(df)


# In[120]:


# # Separate the Target Column from the Rest of the Dataset


# In[121]:


#drop the y (target) column from the dataset
x=df.drop("stroke",axis=1)


# In[122]:


x


# In[123]:


x.shape


# In[124]:


y=df.stroke


# In[125]:


y


# In[126]:


print('x\n', x[:10])
print('y\n', y[:10])


# In[127]:


# # Separate the Data into Train and Test Sets

from sklearn.model_selection import train_test_split


# In[128]:


#change the dataframe just with important features(coloumn) to stroke
x=df[important_cols]


# In[129]:


X_train,X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=42)


# In[130]:


X_train


# In[131]:


df.isnull().sum()


# In[132]:


# ## Balancing Data Set with Smote function
# 
# conda install -c conda-forge imbalanced-learn


# In[133]:


#SMOTE for Imbalanced Classification
from imblearn.over_sampling import SMOTE

oversample = SMOTE()

X_resampled, y_resampled  = oversample.fit_resample(X_train, y_train)


# In[134]:


X_train, y_train = X_resampled, y_resampled
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[135]:



from scipy import stats

#global functions
def bootstrap_samples(X_train,y_train):
n_samples = X_train.shape[0]
#make random choice between 0 and the number of samples
inxs = np.random.choice(n_samples, size = n_samples, replace = True)
return X_train[inxs], y_train[inxs]




class RandomForest:
def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
    self.n_trees = n_trees
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.n_feats = n_feats
    #stores the trees
    self.trees = []
    

    
   
def fit(self, X_train, y_train):
    self.trees = []
    #use decision tree to create trees. for loop to create multiple trees
    for _ in range(self.n_trees):
        tree = DecisionTree(min_samples_split = self.min_samples_split,
                            max_depth = self.max_depth, 
                            n_feats = self.n_feats)
        X_train_sample, y_train_sample = bootstrap_samples(X_train, y_train)
        tree.dtfit(X_train_sample, y_train_sample)
        self.trees.append(tree)
        
        
def predict(self, X_test):
    tree_preds = np.array([tree.predict(X_test) for tree in self.trees])
    #swap axis to get majority load
    #[[1111] [0000] [1111]]
    #[[101] [101] [101] [101]]
    tree_preds = np.swapaxes(tree_preds, 0, 1)
    
    #predict most common label with in each tree
    y_pred = [_most_common_label_RF(tree_preds) for tree_pred in tree_preds]
    return np.array(y_pred)

#counts labels and returns most common
def _most_common_label_RF(self, tree_preds):
    most_common = stats.mode(tree_preds)[0][0]
    return most_common
def entropy(y):
hist = np.bincount(y)
ps = hist / len(y)
return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
def __init__(
    self, feature=None, threshold=None, left=None, right=None, *, value=None
):
    #stores information of a node
    self.feature = feature
    self.threshold = threshold
    self.left = left
    self.right = right
    self.value = value
#
def is_leaf_node(self):
    return self.value is not None


class DecisionTree:
#stopping criteria
def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    self.n_feats = n_feats
    self.root = None

#takes X_train and y_train and begins to fit the tree
def dtfit(self, X_train, y_train):
    self.n_feats = X_train.shape[1] if not self.n_feats else min(self.n_feats, X_train.shape[1])
    self.root = self._grow_tree(X_train, y_train)

def predict(self, X_test):
    return np.array([self._traverse_tree(x, self.root) for x in X_test])

def _grow_tree(self, X_train, y_train, depth=0):
    n_samples, n_features = X_train.shape
    n_labels = len(np.unique(y_train))

    # stopping criteria
    #checking for max depth
    #checking if there is only 1 class present
    #if samples are less than the minimun samples split
    if (
        
        depth >= self.max_depth            
        or n_labels == 1            
        or n_samples < self.min_samples_split
    ):
        #if any of those criteria happens return value of the leaf node in line 21
        leaf_value = self._most_common_label(y_train)
        return Node(value=leaf_value)
    #if we don't meet stopping criteria
    feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

    # greedy search
    best_feat, best_thresh = self._best_criteria(X_train, y_train, feat_idxs)

    # grow the children that result from the split
    left_idxs, right_idxs = self._split(X_train[:, best_feat], best_thresh)
    #continue growing left node
    left = self._grow_tree(X_train[left_idxs, :], y_train[left_idxs], depth + 1)
    #continue growing right node
    right = self._grow_tree(X_train[right_idxs, :], y_train[right_idxs], depth + 1)
    return Node(best_feat, best_thresh, left, right)

def _best_criteria(self, X_train, y_train, feat_idxs):
    best_gain = -1
    #create split index and threshold
    split_idx, split_thresh = None, None
    
    for feat_idx in feat_idxs:
        #collects all samples of feat_idx
        X_column = X_train[:, feat_idx]
        #makes sure to only go over unique thresholds
        thresholds = np.unique(X_column)
        #go through all possible thresholds
        for threshold in thresholds:
            #calculate information gain
            gain = self._information_gain(y_train, X_column, threshold)
            #
            if gain > best_gain:
                best_gain = gain
                split_idx = feat_idx
                split_thresh = threshold

    return split_idx, split_thresh

def _information_gain(self, y_train, X_column, split_thresh):
    # parent loss
    parent_entropy = entropy(y_train)

    #create our split left and right indxs
    left_idxs, right_idxs = self._split(X_column, split_thresh)

    #imediately return info gain = 0 if any of indx are empty
    if len(left_idxs) == 0 or len(right_idxs) == 0:
        return 0

    #compute the weighted avgerage of the child entropy
    num_samples = len(y_train)
    num_samples_left, num_samples_right = len(left_idxs), len(right_idxs)
    entropy_left, entropy_right = entropy(y_train[left_idxs]), entropy(y_train[right_idxs])
    child_entropy = (num_samples_left / num_samples) * entropy_left + (num_samples_right / num_samples) * entropy_right

    # information gain is difference in loss before vs. after split
    ig = parent_entropy - child_entropy
    return ig

def _split(self, X_column, split_thresh):
    #ask question if left indx less than or equal to split_treshold
    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
    #ask question if right indx greater than split_treshold
    right_idxs = np.argwhere(X_column > split_thresh).flatten()
    return left_idxs, right_idxs

def _traverse_tree(self, x, node):
    if node.is_leaf_node():
        return node.value

    if x[node.feature] <= node.threshold:
        return self._traverse_tree(x, node.left)
    return self._traverse_tree(x, node.right)

#counts labels and returns most common
def _most_common_label_RF(self, y_train):
    most_common = stats.mode(y_train)[0][0]
    return most_common
  


# In[1]:


import time
start_time = time.time()
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    
    
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
   
    

clf = RandomForest(n_trees = 3)
clf.fit(X_train.values, y_train.values)

y_pred = clf.predict(X_test.values)
acc = accuracy(y_test, y_pred)

print(classification_report(y_test, y_pred))
print("Accuracy:", acc)
print("--- %s seconds ---" % (time.time() - start_time))
    


# In[ ]:


#cited sources
#https://www.youtube.com/watch?v=Oq1cKjR8hNo
#https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/random_forest.py
#https://github.com/python-engineer/MLfromscratch/blob/master/mlfromscratch/decision_tree.py


# In[ ]:




