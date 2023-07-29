import numpy as np
import pandas as pd


from sklearn.feature_extraction.text import CountVectorizer
import string

import nltk
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords

df = pd.read_csv("/Users/sohamshirke/Documents/Data Science/Codes/first_batch.csv")
df.columns
df = df.drop(columns=['Unnamed: 0', 'URL', 'PUBLISHER', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

def text_cleaning(a):
 remove_punctuation = [char for char in a if char not in string.punctuation]
 #print(remove_punctuation)
 remove_punctuation=''.join(remove_punctuation)
 #print(remove_punctuation)   
 return [word for word in remove_punctuation.split() if word.lower() not in stopwords.words('english')]

print(df.iloc[:,0].apply(text_cleaning))



from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_cleaning).fit(df['TITLE']) 
  
#print(len(bow_transformer.vocabulary_))   
bow_transformer.vocabulary_


title_bow = bow_transformer.transform(df['TITLE'])

print(title_bow)

X = title_bow.toarray()
print(X)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer().fit(title_bow)
print(tfidf_transformer)

title_tfidf=tfidf_transformer.transform(title_bow)
print(title_tfidf)# got tfidf values for whole vocabulary
print(title_tfidf.shape)


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(title_tfidf,df['CATEGORY'])

all_predictions = model.predict(title_tfidf)
print(all_predictions)

#Printing the confusion matrix of our prediction
from sklearn.metrics import confusion_matrix

confusion_matrix(df['CATEGORY'], all_predictions)