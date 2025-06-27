#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


data = pd.read_csv("netflix_titles.csv")


# In[35]:


data.head(10)


# In[36]:


data.shape


# In[37]:


data.describe()


# In[38]:


data.info()


# In[39]:


data.isnull().sum()


# In[40]:


data["director"].fillna("Unknown",inplace = True)


# In[41]:


data["director"].isnull().sum()


# In[42]:


data["cast"].fillna("Unknown",inplace = True)


# In[43]:


data["cast"].isnull().sum()


# In[44]:


data["country"].fillna("Unknown",inplace = True)


# In[45]:


data["country"].isnull().sum()


# In[46]:


data["date_added"].fillna("Januanry 1,2025",inplace = True)


# In[47]:


data["date_added"].isnull().sum()


# In[48]:


data["rating"].fillna("unrated",inplace = True)


# In[49]:


data["rating"].isnull().sum()


# In[50]:


data["duration"].fillna(value = "mean",inplace = True)


# In[51]:


data["duration"].isnull().sum()


# In[52]:


data.isnull().sum()


# In[55]:


data["date_added"] = pd.to_datetime(data["date_added"],errors = "coerce")
data['date_added'] = data['date_added'].dt.strftime('%d-%m-%Y')


# In[56]:


data.head(10)


# In[57]:


data["type"].value_counts()


# In[60]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["type"] = le.fit_transform(data["type"])


# In[61]:


data.head()


# In[64]:


data.drop(columns = "show_id",inplace = True)


# In[65]:


data.head()


# In[67]:


data.shape


# In[72]:


get_ipython().system('pip install nltk')



# In[73]:


get_ipython().system('pip install spacy')


# In[78]:


import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[80]:


text_columns = ['title', 'cast', 'listed_in', 'description']
for col in text_columns:
    data[col] = data[col].fillna('').astype(str)  # handle nulls


# In[86]:


def clean_and_tokenize(text):
    # lowercase and remove punctuation
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return tokens


# In[87]:


stop_words = set(stopwords.words('english'))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]


# In[88]:


lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]


# In[89]:


def preprocess_text(text):
    tokens = clean_and_tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return tokens

for col in text_columns:
    data[f'{col}_tokens'] = data[col].apply(preprocess_text)
    data[f'{col}_clean'] = data[f'{col}_tokens'].apply(lambda tokens: ' '.join(tokens))


# In[90]:


data.head(10)


# In[91]:


data = data.drop(columns=['title_tokens', 'cast_tokens', 'listed_in_tokens', 'description_tokens'])


# In[92]:


data.head(10)


# In[93]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=500)
X_desc = tfidf.fit_transform(data['description_clean'])  # sparse matrix


# In[94]:


from sklearn.preprocessing import MultiLabelBinarizer

# split into list of genres
data['genres'] = data['listed_in_clean'].apply(lambda x: x.split())

mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(data['genres'])


# In[95]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['country_encoded'] = le.fit_transform(data['country'])
data['rating_encoded'] = le.fit_transform(data['rating'])


# In[98]:


data.head(10)


# In[103]:


final_columns = [
    'type', 'title_clean', 'cast_clean', 'listed_in_clean',
    'description_clean', 'genres', 'release_year', 'duration',
    'country', 'rating', 'country_encoded', 'rating_encoded'
]

final_data = data[final_columns]


# In[104]:


final_data.head(10)


# In[105]:


final_data.to_csv("netflix_final_preprocessed.csv", index=False)


# In[ ]:




