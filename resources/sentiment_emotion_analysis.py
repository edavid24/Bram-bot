from textblob import TextBlob
import csv
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import spacy
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import RocCurveDisplay
import sklearn.metrics
from sklearn.metrics import f1_score
import sklearn.preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_transformer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDClassifier
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import cross_validate
import text2emotion as te



"""
This is an algorithm that reads in sentences and a given political lean of the sentences
it analyses the character average of words, part of speech / num words, sentiment, and emotions
it then takes that information and creates a prediction model to try and predict the political lean of other sentences
It has around 70% accuracy with the dataset it was given
"""



# adding the stop words (words to ignore what classifying)
my_stop_words = text.ENGLISH_STOP_WORDS
myWords = []
for word in my_stop_words:
    myWords.append(word)

#reading in the csv file
data = pd.read_csv('b.csv',encoding='cp437')
#filling in missing data
data = data.fillna("brand")

#gathering data
numberOfWords = []
charAvg = []
tempdata = []
i = 0
PartsOfSpeach = []
sentiment = []
allTweets = data['text'].tolist()
anger = []
fear = []
happy = []
sad = []
suprise = []
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentAnalyzer

sia = SentimentIntensityAnalyzer()
for line in allTweets:
    temp = line
    # adding the sentiment of the line to the array
    sentiment.append(sia.polarity_scores(line)["compound"])
    #a dictionary that holds the emotion for the line of code
    dict = te.get_emotion(line)
    # adding each emotion to their own dictionary
    anger.append(dict['Angry'])
    fear.append(dict['Fear'])
    happy.append(dict['Happy'])
    sad.append(dict['Sad'])
    suprise.append(dict['Surprise'])
    #removing links are replacing them with a place holder Http replacing all hashtags with # 
    #looking for an removing links
    if 'http' in temp:
        s = list(temp)
        for j in range (len(temp)-4):
            if(s[j]=='h'and s[j+1]=='t'and s[j+2]=='t' and s[j+3]=='p'):
                for k in range(j+4,len(temp)):
                    if s[k]==' ':
                        break
                    else:
                        s[k] = ' '
        temp = "".join(s)
    #looking for and removing hashtags
    if '#' in temp:
        s = list(temp)
        for j in range (len(temp)-4):
            if(s[j]=='#'):
                for k in range(j+1,len(temp)):
                    if s[k]==' ':
                        break
                    else:
                        s[k] = ' '
        temp = "".join(s)


    numWords = 0
    charTotal = 0
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(temp)
    pos = {}
    #looking word by word
    for token in doc:
        #print()
        if '#' or 'http' not in token.text:
            if token.pos_ in pos.keys():
                #print(token.pos)
                pos[token.pos_] = pos[token.pos_] + 1
            else:
                pos[token.pos_] = 1
            #print(token.text, token.lemma_, token.pos_, token.tag_)
            numWords = numWords+1
            charTotal = charTotal + len(token.text)
        else:
            charTotal = charTotal + (charTotal/numWords)
            numWords = numWords+1
    #calculating the average characters per word
    charTotal = charTotal/numWords
    #adding all of the new information 
    PartsOfSpeach.append(pos)
    charAvg.append(charTotal)
    numberOfWords.append(numWords)
    #data['text'][i]=temp
    tempdata.append(temp)
    i = i+1
data['text']=tempdata
parts = {}
bgs = {}
#print(PartsOfSpeach[0])
print(sentiment[0])
#calculating the average part of speech by words
#note: each part of speech will be stored as its own column in the data table 
for i in range (len(PartsOfSpeach)):
    #print(PartsOfSpeach[i])
    #each individual object
    for j in PartsOfSpeach[i].keys():
        if j not in parts.keys():
            temp = []
            for k in range (i):
                temp.append(0)
            temp.append(PartsOfSpeach[i][j]/numberOfWords[i])
            parts[j] = temp
        else:
            parts[j].append(PartsOfSpeach[i][j]/numberOfWords[i]) 
    for j in parts.keys():
        if j not in PartsOfSpeach[i].keys():
            #print(parts.keys())
            parts[j].append(0)
#print(parts)





#creating and filling the dataframe
d = {'pos': data['pos'],'text':data['text']}
df = pd.DataFrame(data=d)
df['word_count'] = numberOfWords
df['charAvg'] = charAvg
df['sentiment'] = sentiment
df['anger'] =anger 
df['fear'] =fear
df['happy'] = happy 
df['sad'] = sad 
df['suprise'] =suprise 


#adding the parts of speech to the dataframe
posList = ['PROPN','NOUN','VERB','ADV','PUNCT','INTJ','ADP','PART','CCONJ','SCONJ','SYM','NUM','DET','AUX','X']
tempPos = []
for i in range (len(PartsOfSpeach)):
    tempPos.append(0)
for group in posList:
    if group in parts.keys():
        df[group] = parts[group]
    else:
        df[group]=tempPos

print(df.head())
ds = df
print(ds.head())
#x is everyline in the dataframe minus the political lean (pos)
x = df.drop('pos', axis='columns')
#y is the political lean row
y=df['pos']
#splitting the data into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)






#working with text and nums
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer
#vectorizer_params = dict(ngram_range=(1, 2), min_df=5, max_df=0.8,stop_words=myWords)
#creating the pipeline that the data will follow
#note: strings have to be turned into integers so each word will become a column in the dataframe with 1 in it existed in the row and 0 if it did not
TweetTextProcessor = Pipeline(steps=[
    ("squeez", FunctionTransformer(lambda x: x.squeeze())),
    ("vect", CountVectorizer(ngram_range=(1, 3),stop_words=myWords)),
    ("tfidf", TfidfTransformer()),
    ("toarray", FunctionTransformer(lambda x: x.toarray())),
])

preprocessor = ColumnTransformer(transformers=[
    ('tweet', TweetTextProcessor, ['text']),
    #('dict', dictionary, ['pos'])
    ])
#uses the K nearest neighbor classifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
    #('classifier', SVC())
])




#k-fold cross validation finding the precision accuracy and recall of predictions
scoring = ['precision_macro', 'recall_macro','accuracy']
cvals = cross_validate(pipeline, x, y, cv=5,scoring = scoring)
#print(cvals)
kvals = [1,2,3,4,5]
d = {'K Vals': kvals}
df = pd.DataFrame(data=d)
df['precision'] = cvals['test_precision_macro']
df['accuracy'] = cvals['test_accuracy']
df['recall'] = cvals['test_recall_macro']


#printing the results
print(df.head())


sum = 0
for i in cvals['test_precision_macro']:
    sum = sum+i
print("average precision: ",(sum/len(cvals['test_precision_macro'])))
sum = 0
for i in cvals['test_accuracy']:
    sum = sum+i
print("average accuracy: ",(sum/len(cvals['test_accuracy'])))
sum = 0
for i in cvals['test_recall_macro']:
    sum = sum+i
print("average recall: ",(sum/len(cvals['test_recall_macro'])))



