#!/usr/bin/env python
# coding: utf-8

# # SMS Spam Collection Dataset

# ![spam%20image.png](attachment:spam%20image.png)

# **SMS Spam**
# 
# **SMS spam (sometimes called cell phone spam) is any junk message delivered to a mobile phone as text messaging through the Short Message Service (SMS). The practice is fairly rare in North America, but has been common in differnt countries for years.**
# 

# # IMPORTING THE LIBRARIES

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import string
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # LOADING THE DATASET

# In[9]:


data = pd.read_csv('spam_messages.csv', encoding='ISO-8859-1')


# In[10]:


data.head()


# In[11]:


data


# In[13]:


data.mean()


# In[14]:


data.describe().transpose()


# In[8]:


data.describe()


# In[15]:


data.info()


# In[10]:


data.shape   #5572 rows and 5 columns in our dataset


# In[16]:


data.value_counts()


# In[17]:


data.dtypes


# In[18]:


data.columns


# **Checking Null Values**

# In[19]:


data.isnull().sum()


# In[20]:


data.isnull().any()


# In[21]:


data.isnull().all()


# **So we need to drop the columns that are : Unnamed:2, Unnamed:3, Unnamed:4**

# In[22]:


data=data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)


# In[23]:


data


# **For betterment of columns(v1,v2) we can rename them respectively.**

# In[19]:


data=data.rename({'v1':'Class','v2':'Message'},axis=1)
             


# In[20]:


data.head()


# In[21]:


data.columns


# # Exploratory Data Analysis

# In[22]:


plt.figure(figsize=(6,6))

x= data.Class.value_counts()
sns.countplot(x= "Class",data= data)


# In[23]:


plt.figure(figsize=(8,12))

label= ["Class","Message"]

plt.pie(x.values, labels= label ,autopct= "%1.1f%%") # visualizing using pie
plt.show()   


# In[24]:


import nltk
import scikitplot as skplt
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
STOPWORDS = stopwords.words('english')


# In[25]:


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^0-9a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(word for word in text.split() if word not in STOPWORDS)
    return text


# In[26]:


data['clean_text'] = data['Message'].apply(clean_text)
data.head()


# In[27]:


X = data['clean_text']
y = data['Class']


# In[28]:


# importing the PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize,word_tokenize
ps=PorterStemmer
words=word_tokenize('clean_text')


# In[29]:


#importing the CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#lemmatizer=WordNetLemmatizer()


# In[30]:


#define a function to get rid of stopwords present in the messages
def message_text_process(mess):
    # Check characters to see if there are punctuations 
    no_punctuation=[char for char in mess if char not in string.punctuation]
    # now form the sentence
    no_punctuation=''.join(no_punctuation)
    # Now eliminate any stopwords
    return[word for word in no_punctuation.split() if word.lower() not in stopwords.words('english')]


# In[31]:


# to verify that function is working
data['Message'].head(5).apply(message_text_process)


# In[32]:


# start text processing with vectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[33]:


# bag of words by applying the function and fit the data(message) into it
bag_of_words_transformer=CountVectorizer(analyzer=message_text_process).fit(data['Message'])


# In[34]:


# print the length of bag of words stored in vocabulary_attribute
print(len(bag_of_words_transformer.vocabulary_))


# In[35]:


#store bag of words for messages using transform method
message_bagofwords=bag_of_words_transformer.transform(data['Message'])


# In[36]:


#apply tfidf transformer and fit the bag of words into it(transformed version)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(message_bagofwords)


# In[37]:


#print shape of tfidf
message_tfidf=tfidf_transformer.transform(message_bagofwords)
print(message_tfidf.shape)


# In[38]:


# choose naive bayes model to detect the spam and fit the tfidf data into it
from sklearn.naive_bayes import MultinomialNB
spam_detection_model=MultinomialNB().fit(message_tfidf,data['Class'])


# In[39]:


# check model for prediction and expected value say for message#2 and message#5
message=data['Message'][4]
bag_of_words_for_message=bag_of_words_transformer.transform([message])
tfidf=tfidf_transformer.transform(bag_of_words_for_message)

print('predicted',spam_detection_model.predict(tfidf)[0])

#print('expected',data.response[4])


# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


# In[41]:


message=data['Message'][4]
# check model for prediction and expected value say for message#2 and message#5
bag_of_words_for_message=bag_of_words_transformer.transform([message])
tfidf=tfidf_transformer.transform(bag_of_words_for_message)

print('predicted',spam_detection_model.predict(tfidf)[0])
#print('expected',data.label[4])


# In[42]:


#importing PCA for the dimensionality reduction 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA


# In[43]:


#function for the model building and prediction
def Model(model, X, y):
#training and testing the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=30)
    # model building using CountVectorizer and TfidfTransformer
    pipeline_model = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', model)])
    pipeline_model.fit(x_train, y_train)
    
    


    y_pred = pipeline_model.predict(x_test)
    y_probas =pipeline_model.predict_proba(x_test)
    skplt.metrics.plot_roc(y_test,y_probas,figsize=(12,8),title_fontsize=12,text_fontsize=16)
    plt.show()
    skplt.metrics.plot_precision_recall(y_test,y_probas,figsize=(12,8),title_fontsize=12,text_fontsize=16)
    plt.show()
    print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
    print("Classification Report is:\n",classification_report(y_test, y_pred))
    print('Accuracy:', pipeline_model.score(x_test, y_test)*100)
    print("Training Score:\n",pipeline_model.score(x_train,y_train)*100)
    



# # Model Building

# # 1. Logistic Regression

# In[44]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
Model(model, X, y)



# **So we get a accuracy score of 96.19 % using LogisticRegression**

# # 2. KNeighborsClassifier

# In[45]:


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=7)
Model(model,X,y)


# **So we get a accuracy score of 90.16 % using KNeighborsClassifier**

# # 3. SVC

# In[46]:


from sklearn.svm import SVC
model = SVC(probability=True )
Model(model, X, y)


# **So we get a accuracy score of 97.84 % using SVC**

# # 4. Naive Bayes

# In[47]:


from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
Model(model, X, y)


# **So we get a accuracy score of 96.69 % using Naive Bayes**

# # 5. DECISION TREE CLASSIFIER

# In[48]:


from sklearn import tree
tree_clf = tree.DecisionTreeClassifier(max_depth=6, random_state=123,criterion='entropy')
Model(tree_clf,X,y)



# **So we get a accuracy score of 94.90 % using DecisionTreeClassifier**

# # 6. RandomForestClassifier

# In[49]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
Model(model, X, y)


# **So we get a accuracy score of 97.63 % using RandomForestClassifier**

# # 7. AdaBoostClassifier

# In[50]:


from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(base_estimator = None)
Model(model, X, y)


# **So we get a accuracy score of 97.55 % using AdaBoostClassifier**

# # 8. Gradient Boosting Classifier

# In[51]:


from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
Model(model, X, y)



# **So we get a accuracy score of 97.70 % using Gradient Boosting Classifier**

# # 9. XGBClassifier

# In[52]:


from xgboost import XGBClassifier

xgb =XGBClassifier(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
Model(model, X, y)


# **So we get a accuracy score of 97.70 % using XGBClassifier**

# # 10. ExtraTreesClassifier

# In[53]:


from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier(n_estimators=100, random_state=0)
Model(model,X,y)


# **So we get a accuracy score of 97.27 % using ExtraTreesClassifier**

# # 11. Bagging Classifier

# In[54]:


from sklearn.ensemble import BaggingClassifier
from sklearn import tree
model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1))
Model(model,X,y)


# **So we get a accuracy score of 96.26 % using Bagging Classifier**

# **Conclusion :**
# **We get a good accuracy score of 98 % using Random Forest , Ada Boost and Extra Trees Classifier**

# 
