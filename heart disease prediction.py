#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[68]:


heart_data = pd.read_csv("C:\\Users\\Meher Vaishanavi\\OneDrive\\Desktop\\heart_disease_data (1).csv" )


# In[32]:


heart_data.head()


# In[33]:


heart_data.tail()


# In[34]:


heart_data.shape


# In[35]:


heart_data.isnull().sum()


# In[36]:


heart_data.describe()


# In[37]:


heart_data['target'].value_counts()


# In[38]:


X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']


# In[39]:


print(X)


# In[40]:


print(Y)


# In[41]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[42]:


print(X.shape, X_train.shape, X_test.shape)


# In[43]:


model = LogisticRegression()


# In[44]:


model.fit(X_train, Y_train)


# In[60]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[61]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[62]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[63]:


print('Accuracy on Test data : ', test_data_accuracy)


# In[64]:


cm = confusion_matrix(X_test_prediction,Y_test)
print(cm)


# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt     

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['No', 'Yes']); ax.yaxis.set_ticklabels(['No', 'Yes']);


# In[66]:


print(classification_report(X_test_prediction,Y_test))


# In[56]:


from sklearn.decomposition import PCA
# creating covariance matrix
CVM = PCA(n_components=13)
# calculating eigen values
CVM.fit(heart_data)


# In[57]:


#calculate variance ratios
variance = CVM.explained_variance_ratio_
#cumulative sum of variance explained with [n] features
var=np.cumsum(np.round(variance, decimals=3)*100)
var


# In[58]:


# plotting of variance explained
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.ylim(30,100.5)
plt.style.context('seaborn-whitegrid')
plt.plot(var)


# In[59]:


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')


# In[ ]:





# In[ ]:




