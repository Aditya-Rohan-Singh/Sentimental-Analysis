import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

def process_tweet(tweet):
    #Remove old style retweet text "RT"
    new_tweet = re.sub(r'^RT[\s]','', tweet)
    new_tweet=re.sub(r'b','',new_tweet)
    #Remove hyperlinks
    new_tweet = re.sub(r'https?:\/\/.*[\r\n]*','', new_tweet)
    #Remove hastags
    new_tweet = re.sub(r'#','',new_tweet)
        
    # instantiate tokenizer class
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    
    # tokenize tweets
    tweet_tokens = tokenizer.tokenize(new_tweet)    
        
    #Import the english stop words list from NLTK
    stopwords_english = stopwords.words('english') 
    stopwords1 = set(STOPWORDS)
    stopwords1.update(["br", "href","https","t","co","c","b'RT","b'","'","neg","b","neg'"])
    
    #Creating a list of words without stopwords
    clean_tweets = []
    for word in tweet_tokens:
        if word not in stopwords_english and word not in string.punctuation and word not in stopwords1 :
            clean_tweets.append(word)
        
    #Instantiate stemming class
    stemmer = PorterStemmer()
    
    #Creating a list of stems of words in tweet
    stem_words = []
    for word in clean_tweets:
        stem_word = stemmer.stem(word)
        stem_words.append(stem_word)
        
    return stem_words

def frequency_builder(tweets, label_train):
    label_train_list = np.squeeze(label_train).tolist()
  
    freqs = {}
    for y, tweet in zip(label_train_list, sample_train):
        for word in process_tweet(tweet):
                pair = (word, y)
                freqs[pair] = freqs.get(pair, 0) + 1
    return(freqs)

def sigmoid_func(S1,B):
    return 1/(1+np.exp(-S1@B))

def extract_features(tweet, freqs):
    # process_tweet tokenizes, stems, and removes stopwords
    final_tweet = process_tweet(tweet)
    # 3 elements in the form of a 1 x 3 vector
    x = np.zeros((1, 3)) 
    
    #bias term is set to 1
    x[0,0] = 1 
        
    # loop through each word in the list of words
    for word in final_tweet:
        # increment the word count for the positive label 1
        x[0,1] += freqs.get((word,1),0)
        
        # increment the word count for the negative label 0
        x[0,2] += freqs.get((word,0),0)
        
    assert(x.shape == (1, 3))
    return x

def gradient_Descent(x, y, B, alpha, num_iters):
    m = len(x)
    for i in range(0, num_iters):
        s = sigmoid_func(x,B)
        B = B - (alpha/m)*np.dot(x.T, s-y)
    return B

def predict_tweet(tweet, freqs, B):
    x = extract_features(tweet, freqs)
    y_pred = sigmoid_func(x,B)
    return y_pred

#Load data we got from Tweets_Api.py
df = pd.read_csv('Sentiment_tweets.csv')
#df = pd.read_csv('Testing.csv')
df.head()

sample = df.Tweet # first p-1 columns are features
label = df.Sentiment # last column is label

n=len(sample)
n_train = int(n*0.80)
n_test = int(n*0.20)

sample_train = sample[:n_train]
sample_test = sample[n-n_test:]


label_train = label[0:n_train]
label_test = label[n-n_test:]

#Create frequency table based on sample data
freqs=frequency_builder(sample_train,label_train)

#Building training and testing sets
positive = df[df['Sentiment'] == 1]
negative = df[df['Sentiment'] == 0]

#Positive sets
sample_positive=np.array(positive.Tweet)
label_positive=np.array(positive.Sentiment)

n1=len(sample_positive)
n_train_pos = int(n1*0.80)
n_test_pos = int(n1*0.20)

sample_train_positive=sample_positive[0:n_train_pos]
sample_test_positive=sample_positive[n1-n_test_pos:]

label_train_positive=label_positive[0:n_train_pos]
label_test_positive=label_positive[n1-n_test_pos:]

#Negative sets
sample_negative=np.array(negative.Tweet)
label_negative=np.array(negative.Sentiment)

n2=len(sample_negative)
n_train_neg = int(n2*0.80)
n_test_neg = int(n2*0.20)

sample_train_negative=sample_negative[0:n_train_neg]
sample_test_negative=sample_negative[n2-n_test_neg:]

label_train_negative=label_negative[0:n_train_neg]
label_test_negative=label_negative[n2-n_test_neg:]

label_train_negative=np.array(label_train_negative)
label_test_negative=np.array(label_test_negative)

#Combine positive and negative data set
f1=[sample_train_positive,sample_train_negative]
train_sample=np.concatenate(f1)

f2=[sample_test_positive,sample_test_negative]
test_sample=np.concatenate(f2)

f3=[label_test_positive,label_test_negative]
label_test=np.concatenate(f3)


# combine positive and negative labels
train_y = np.append(np.ones((len(label_train_positive), 1)), np.zeros((len(label_train_negative), 1)), axis=0)
test_y = np.append(np.ones((len(label_test_positive), 1)), np.zeros((len(label_test_negative), 1)), axis=0)


X = np.zeros((len(train_sample), 3))
for i in range(len(train_sample)):
    X[i, :]= extract_features(train_sample[i], freqs)

Y = train_y
B=np.zeros((3, 1))
alpha=0.00001
B=gradient_Descent(X, Y, B, alpha, 1500)

Label_pred = []
for tweet in test_sample:
    y_pred = predict_tweet(tweet, freqs, B)
    if y_pred > 0.50:
        Label_pred.append(1)
    else:
        Label_pred.append(0)
        
Label_pred = np.array(Label_pred)
test_y = test_y.reshape(-1)
accuracy = np.sum((test_y == Label_pred).astype(int))/len(test_sample)
mse_test_N=mean_squared_error(label_test,Label_pred)
print(accuracy)
print(mse_test_N)