
# coding: utf-8

# In[395]:


import pandas as pd
from sklearn import preprocessing


# In[533]:


df = pd.read_csv('movie_metadata 2.csv')


# In[534]:


df.head()


# In[397]:


df[['director_name','director_facebook_likes']].drop_duplicates().sort_values(by='director_facebook_likes',ascending=False )


# In[398]:


df['Profit'] = df['gross']- df['budget']
df[['movie_title','Profit']].sort_values(by='Profit',ascending=False)


# In[399]:


def decolorize(x):
    if x=='Color':
        return 1
    else:
        return 0
    
df.color = df.color.apply(decolorize)


# In[400]:


def splitit(x):
    if x!='nan':
        return str(x).split('|')
    else:
        return


# In[401]:


df.genres = df.genres.apply(splitit)
df.plot_keywords = df.plot_keywords.apply(splitit)


# In[535]:


def delister(ind,val):
    print ind, val
#     import pdb; pdb.set_trace()
    for keyword in tcols: 
        if keyword in val:
            print keyword, val, ind
            t[keyword.lower()].iloc[ind] = 1
    return val


# In[536]:


temp = df[['genres']]
temp = temp.reset_index()

#creating t matrix with all genres
tcols =  list(set(df.genres.sum())) #since its a list, sum concatenates all the elements in the column
t = pd.DataFrame(np.zeros(shape=(len(temp),len(tcols))), columns=[x.lower() for x in tcols]) #creates a matrix 
#with dimensions number of rows in the dataframe and number of genres.
 
temp.apply(lambda row: delister(row['index'], row['genres']), axis=1)


# In[373]:


# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit_transform(t.as_matrix().T)
# # t


# In[372]:


# pcoutput = pd.DataFrame(pca.components_).T


# In[451]:


dfcat = pd.get_dummies(df[[u'language', u'country',u'content_rating',u'title_year']])


# In[507]:


dfnum = pd.DataFrame(preprocessing.scale(df[[u'num_critic_for_reviews', u'duration',u'director_facebook_likes', u'actor_3_facebook_likes',
       u'actor_1_facebook_likes', u'gross', u'num_voted_users', u'cast_total_facebook_likes', u'facenumber_in_poster', 
        u'num_user_for_reviews',u'budget', u'actor_2_facebook_likes',
       u'imdb_score',u'movie_facebook_likes']].fillna(0)), columns=[u'num_critic_for_reviews', u'duration',u'director_facebook_likes', u'actor_3_facebook_likes',
       u'actor_1_facebook_likes', u'gross', u'num_voted_users', u'cast_total_facebook_likes', u'facenumber_in_poster', 
        u'num_user_for_reviews',u'budget', u'actor_2_facebook_likes',
       u'imdb_score',u'movie_facebook_likes'] )


# In[508]:


print dfnum.shape
print dfcat.shape
print pcoutput.shape
preprocdf = pd.concat([dfnum,dfcat,t],axis=1)


# In[509]:


list(dfnum.columns)


# In[476]:


def profitmargin(x):
    if x>Y.describe()['75%']:
        return 4
    elif x>Y.describe()['50%']:
        return 3
    elif x>Y.describe()['25%']:
        return 2
    else:
        return 1
    
df.Profit = df.Profit.fillna(0).apply(profitmargin)


# In[510]:


import numpy as np
import statsmodels.api as sm

Y = df.Profit.fillna(0)
X = preprocdf.fillna(0)
X = sm.add_constant(X)


# In[511]:


Yfull = np.array(Y)
Xfull = X.as_matrix()


# In[512]:


X


# In[513]:


model = sm.OLS(Y,X)
results = model.fit()


# In[514]:


from sklearn.cross_validation import train_test_split
pred_train, pred_test, tar_train, tar_test  = train_test_split(Xfull, Yfull, test_size=.4)


# In[515]:


from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=25)
classifier=classifier.fit(pred_train,tar_train)
predictions=classifier.predict(pred_test)


# In[516]:


from sklearn.cross_validation import train_test_split
#from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics
from sklearn.ensemble import ExtraTreesClassifier


# In[517]:


sklearn.metrics.accuracy_score(tar_test, predictions)


# In[519]:


model = ExtraTreesClassifier()
model.fit(pred_train,tar_train)
print(model.feature_importances_)


# In[520]:


pd.DataFrame(zip(preprocdf.columns,model.feature_importances_)).sort_values(by=1, ascending=False)


# In[540]:


trees=range(20)
accuracy=np.zeros(20)


# In[541]:


for idx in range(len(trees)):
   classifier=RandomForestClassifier(n_estimators=idx + 1)
   classifier=classifier.fit(pred_train,tar_train)
   predictions=classifier.predict(pred_test)
   accuracy[idx]=sklearn.metrics.accuracy_score(tar_test, predictions)


# In[542]:


import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')
plt.plot(trees, accuracy)


# In[531]:


import xgboost 
from sklearn.grid_search import GridSearchCV


# In[530]:


get_ipython().system('')

