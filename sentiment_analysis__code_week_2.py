#Problem statement
#Develop a sentiment analysis model to classify restarunt reviews as positve or negative

#Description
# with the rapid growth of online platforms for sharing opinions and reviews,restarunts often rely
#on the customer feedback to imporve their services and attract   a new customers.
# Analyzing the sentiment of these reviews can provide valuable insights into customer satisfaction.

pip install pandas

import pandas as pd

data = pd.read_csv('Reviews.csv')

data

data.head() # Top 5 rows of the data set

data.tail() # Last 5 rows of the data set

data.info() # information of the dat set like , data type , memory usage

data.describe() # stastical information of the data set

#checking the null values of the data set
data.isnull().sum()

data.duplicated()

#checking the value counts
value_counts = data['Liked'].value_counts()
print(value_counts)

pip install matplotlib

import matplotlib.pyplot as plt
import seaborn as sns

value_counts.plot(kind = 'bar' , color = ['blue', 'green'])
plt.title("Sentiment value counts")
plt.xlabel('Liked')
plt.ylabel('Count')
plt.xticks(ticks=[0,1] , labels=['Postive','Negative'],rotation=0)
plt.show()

from wordcloud import WordCloud

combined_text = " ".join(data['Review'])
wordcloud = WordCloud(width = 800 , height = 400 ,background_color = 'white').generate(combined_text)
plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.title('Word Cloud of Reviews')
plt.show()

from collections import Counter

target_words = ['food','place','restaurant']
all_words = " ".join(data['Review']).lower().split()
word_counts = Counter(all_words)
target_word_counts = {word:word_counts[word] for word in target_words}
plt.figure(figsize=(8,6))
plt.bar(target_word_counts.keys(),target_word_counts.values() , color = ['blue','green','orange'])
plt.xlabel('words')
plt.ylabel('Frequenecy')
plt.title('Frequency of specific words in Reviews')
plt.show()

#Text preprocessing

#convert a data set into lower case
lowercased_text = data['Review'].str.lower()

print(lowercased_text)

#tokinization
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')

data['Tokens'] = data['Review'].apply(word_tokenize)

print(data['Tokens'])

data.info()

data['Review'].value_counts()

import string

data['Review'] = data['Review'].str.replace(f"[{string.punctuation}]"," ",regex = True)

print(data['Review'])

data['Review'].value_counts()

#Removing the stop words like this, is , are ,was
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

data['Tokens'] = data['Review'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])

print(data['Tokens'])

#stemming
#stemming is the process of reducing the a word into root or base word form by removig suffix
#example : driving stemmed is drive

#Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

data['stemmed'] = data['Review'].apply(lambda x: ' '.join([stemmer.stem(word) for word in word_tokenize(x)]))

print(data['stemmed'])

#Lemmatization
#Lemmatization is the process transforming a word into its base or dictionary form
#example is better is lemmtized to good

import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

lemmatizer = WordNetLemmatizer()

data['Lemmatized'] = data['Review'].apply(lambda x :' '.join([lemmatizer.lemmatize(word , pos = wordnet.VERB) for word in word_tokenize(x)]))

print(data['Lemmatized'])

#Removing the numbers from reviews
import re
data['No_Numbers'] = data['Review'].apply(lambda x : re.sub(r'\d+',' ' ,x))

print(data['No_Numbers'])

#removing special characters like @ # %,*
data['cleaned'] = data['Review'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]','' ,x))

print(data['cleaned'])

#expanding method
# don't eat food in this hotel , when we apply expanted text it will convert into do not eat food in this hotel

!pip install contractions
import contractions
data['Expanded'] = data['Review'].apply(contractions.fix)

print(data['Expanded'])

#Removing emojis
!pip install emoji

import emoji
data['emoji'] = data['Review'].apply(emoji.demojize)

print(data['emoji'])

# removing liks from review_ text
# food is good vist www.abchotel.in

!pip install beautifulsoup4

from bs4 import BeautifulSoup

data['cleaned'] = data['Review'].apply(lambda x: BeautifulSoup(x,"html.parser").get_text())

print(data['cleaned'])

#TF IDF VECTORIZER -Interview damn Sure Question
#TF -Term Frequency
#IDF - Inverse Doument Frequency

"""convert Text----> into matrices
for that each term should have frequency Below is some example for this...
"""

from sklearn.feature_extraction.text import TfidfVectorizer

data={'Reviews':["The sun is bright", "the sky is blue","The sun in the bright","we can see the shining sun, the bright son"],'Other_data':[1,2,3]}
#Let this be our sample data for this example
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data)
print(tfidf_matrix.toarray())
print(tfidf_vectorizer.get_feature_names_out())

"""TF IDF only on 1coln= Review coln of dataset:: down---------------->"""

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer()
X = vectorizer.fit_transform(data['Review'])
#Here data is not eg ones startiong from Reviews.csv file
#Arrays cant have text values but matices can - Sometimes
print(X.toarray())

#Naive buyes Classification method-
#Building ML model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer= TfidfVectorizer()
X = vectorizer.fit_transform(data['Review'])
y= data['Liked']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train,X_test,y_train,y_test)

model= MultinomialNB()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)

#prdeict sentiment for new review
def predict_sentiment(new_review):
  cleaned_review=preprocess_text(new_review)
  X_new= vectorizer.transform([cleaned_review])
  return model.predict(X_new)[0]

new_reviews = input("Enter a review: ")
for review in new_reviews:
  sentiment = predict_sentiment(review)
  sentiment_label='Positive' if sentiment == 1 else 'Negative'
  print(f"Review: '{review}'\nPredicted Sentiment: {sentiment_label}\n")
