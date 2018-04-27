#!pip install tweepy
#!pip install textblob
#!python -m textblob.download_corpora
#!pip install nltk
#!pip install numpy
#!pip install sklearn
#!pip install scipy
#!pip install wordcloud
#!pip install keras
#!pip install tensorflow
#!pip install pandas
#!pip install google-cloud
#!pip install scipy


print( "Load Modules..." )

# -*- coding: utf-8 -*-
import re
import csv
import tweepy
import unicodedata
import json
import nltk
from nltk.stem.porter import *
from nltk.tokenize import TweetTokenizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.misc import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Convolution1D, Flatten, Dropout, LSTM, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard
from scipy import *
from scipy.sparse import *
from textblob import TextBlob

print( "Class definitions...")
#Class Definiton for the keywords that will be used to fetch the tweets
class keywords:
    def __init__(self):
        self.keywords = ['AlaskaAir','Allegiant','AmericanAir','Delta','FlyFrontier','HawaiianAir','@united','JetBlue','SouthwestAir','SpiritAirlines','VirginAmerica','SunCountryAir']
        
    def getKeyWords(self):
        return self.keywords

#Class Definition to pull the twitter tweets
class pullData:      
    def __init__(self,key,secret,maxTweets,tweetsPerQry,keywords):
        self.key = key   
        self.secret = secret
        self.maxTweets = maxTweets
        self.tweetsPerQry = tweetsPerQry          
        self.keywords = keywords
        self.api = ''
        self.auth = ''
        
    def printParams(self):
        print('Parameters set to...')
        print('key...',self.key)
        print('secret...',self.secret)
        print('maxTweets...',self.maxTweets)
        print('tweetsPerQry...',self.tweetsPerQry)
        print('keywords...',self.keywords)
        
    def connect(self):
        self.auth = tweepy.AppAuthHandler(self.key,self.secret )  
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
        return True
        
        if (not self.api):
            print ("Can't Authenticate")
            return False
        
    def downloadData(self):
        for word in self.keywords:
            print( "Downloading Tweets for the keyword: ", word )
            fName = 'tweets_' + word + '.txt'
            sinceId = None
            max_id = -1
            tweetCount = 0
            tweet_dict = []
    
            print("Downloading max {0} tweets".format(maxTweets))

            while tweetCount < self.maxTweets:
                try:
                    if (max_id <= 0):
                        if (not sinceId):   
                            new_tweets = self.api.search(q=word, count=self.tweetsPerQry)
                        else:
                            new_tweets = self.api.search(q=word, count=self.tweetsPerQry,since_id=sinceId)
                    else:
                        if (not sinceId):
                            new_tweets = self.api.search(q=word, count=self.tweetsPerQry,
                                    max_id=str(max_id - 1))
                        else:
                            new_tweets = self.api.search(q=word, count=self.tweetsPerQry,
                                    max_id=str(max_id - 1),since_id=sinceId)
                    if not new_tweets:
                        print("No more tweets found")
                        break            
                        
                    for tweet in new_tweets:
                        tweet_dict.append(tweet._json)
 
                    tweetCount += len(new_tweets)
                    print("Downloaded {0} tweets".format(tweetCount))
                    max_id = new_tweets[-1].id
                
                except tweepy.TweepError as e:
                    # Just exit if any error
                    print("some error : " + str(e))
                    break
            
            with open(fName, 'w', encoding='utf8', errors='replace') as f:   
                json.dump(tweet_dict, f, ensure_ascii=False)


#Populate Text Files
from google.cloud import storage
client = storage.Client()

def getFile(fname):
    bucket = client.get_bucket('surisrb1')
    file = storage.Blob('notebooks/'+fname, bucket)

    file = file.download_as_string()
    file = file.decode() if isinstance(file,bytes) else file
    return file

effect_word = getFile('EffectWordNet.tff')
subj_clues = getFile('subjclueslen1-HLTEMNLP05.tff')
negation_cues = getFile('negation_cues.txt')
stopwords = getFile('stopwords.txt')

united = getFile('tweets_@united.txt')
alaska = getFile('tweets_AlaskaAir.txt')
allegiant = getFile('tweets_Allegiant.txt')
american_air = getFile('tweets_AmericanAir.txt')
delta = getFile('tweets_Delta.txt')
froniter = getFile('tweets_FlyFrontier.txt')
hawaii = getFile('tweets_HawaiianAir.txt')
jetblue = getFile('tweets_JetBlue.txt')
south_west = getFile('tweets_SouthwestAir.txt')
spirit = getFile('tweets_SpiritAirlines.txt')
sun_country = getFile('tweets_SunCountryAir.txt')
virgin = getFile('tweets_VirginAmerica.txt')


#Class definition for JsonParser

class JsonParser:
    def loadData(self,fname):
        #with open(fname, encoding='utf8', errors='replace') as json_data:
        d = json.loads(fname)
        return d
        

#Class definition to Clean the tweets and get the sentiment
class TweetCleaner: 
    def __init__(self,stopwords_fname):
        self.stopwords_fname = stopwords_fname
        self.negation_cues = self.get_negation_cues('gs://surisrb1/notebooks/negation_cues.txt')
        self.sentiment_fnames = ["EffectWordNet.tff", "subjclueslen1-HLTEMNLP05.tff"]
        self.emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
        self.url_pattern = re.compile(r'https?:\/\/.*\b')
        self.handle_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        
        #Populate the Stop Words
        self.stopwords = set()
        self.populateStopWords(self.stopwords_fname)
        
        #populate the wordtoEffectMap
        self.wordToEffectMap = {}
        self.buildWordToEffectMap(self.sentiment_fnames)
        
        #Initialize nltk classes
        self.stemmer = PorterStemmer()
        self.tokenizer = TweetTokenizer()
    
    #---------Create the Stop words Set---------#
    def populateStopWords(self,fname):
        for line in stopwords.split("\n") :
            self.stopwords.add(line.strip())
    
    def get_negation_cues(self, fname):
        cues_fname = negation_cues
        neg = []
        for line in cues_fname.split("\n"):
            if not line:
                continue
            line = line.strip()
            neg.append(line)
        return neg
    
    #---------Build the word to +/- Effect Map----------#
    def buildWordToEffectMap(self, sentiment_files):
        #read sentiment files into dictionary of words and positive or negative sentiment
        for file in sentiment_files:
                if(file == "EffectWordNet.tff"):
                    f = effect_word.split("\n")
                    for line in f:
                        #02279615	+Effect	profiteer	 make an unreasonable profit, as on the sale of difficult to obtain goods 
                        if( line.strip() == '' ):
                            continue
                        
                        words = line.split('\t')
              
                        
                        effect = words[1]
                        effect_val = 0
                        if '+' in effect:
                            effect_val = 1
                        elif '-' in effect:
                            effect_val = -1
                        else:
                            effect_val = 0
                        list_of_words = []
                        if (',' in words[2]):
                            list_of_words = words[2].split(',')
                        else:
                            list_of_words.append(words[2])
                        for word in list_of_words:
                            self.wordToEffectMap[word] = effect_val
                elif(file == "subjclueslen1-HLTEMNLP05.tff"):
                    f = subj_clues
                    for line in f.split("\n"):
                        #type=weaksubj len=1 word1=abandoned pos1=adj stemmed1=n priorpolarity=negative
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        if( line.strip() == '' ):
                            continue
                        words = line.split(' ')
                        if ( len(words) == 0):
                            continue
                        effect_val = words[5]
                        #print(effect_val)
                        word_val = words[2]
                        word = word_val.split("=")[1]
                        effect = effect_val.split("=")[1]
                        eff = 0
                        if (effect == "positive"):
                            eff = 1
                        elif (effect == "negative"):
                            eff = -1
                        elif (effect == "neutral"):
                            eff = 0
                        self.wordToEffectMap[word] = eff
        
    #---------Clean the tweets---------#
    def cleanTweet(self,tweet):
        tweet = tweet.lower()
        
        #1. remove emojis
        tweet = re.sub(self.emoji_pattern, r'', tweet)
        
        #remove URLS
        tweet = re.sub(self.url_pattern, r'', tweet)

        #convert @mentions to AT_USER
        tweet = re.sub(self.handle_pattern, r'AT_USER', tweet)
        
        #convert #tags to HASH_TAG
        tweet = re.sub(self.hashtag_pattern, r'HASH_TAG', tweet)
        
        tweet_list = self.tokenizer.tokenize(tweet)
        
        #2. remove the stop words
        words_filtered = []
        for word in tweet_list:
            if (word not in self.stopwords ):
                words_filtered.append( word )
              
        #3. Stem the words
        #words_stemmed = [self.stemmer.stem(word) for word in words_filtered]
        
        _tweet = ""
        for word in words_filtered:
            _tweet += " " + word
        
        #4. return the fitered tweet
        return _tweet
             
    #--------- negative: 0 , neutral : 1 , positive : 2 ---------#
    def getSentiment(self,tweet):
        tweet = tweet.lower()
        tweetBlob = TextBlob(tweet)
        if (tweetBlob.sentiment.polarity > 0 ):
            return 2
        elif ( tweetBlob.sentiment.polarity < 0):
            return 0
        else:
            return 1


#Cleans all the tweets , builds the tweet set and adds the sentiment to the tweets
class BuildFeatureSet: 
    
    def __init__(self,n,stopwords_fname,max_features_count=5000): # n : ngram for the tweets
        self.ngrams = n

        #Lexicon related variables
        self.lexicon = {}
        self.inverse_lexicon = { }
        self.ngram_count = 0
        self.max_features_count = max_features_count
        self.tweetCleaner = TweetCleaner(stopwords_fname)
        self.tweet_count = 0
        self.tweet_map = {}
        
        #Training Dataset
        self.data = {}

    def addToLexicon(self,words):
        for word in words:
            if ( word not in self.lexicon and ( self.ngram_count <  self.max_features_count ) ):
                self.lexicon[word] = self.ngram_count #Assign a unique number for the word seen
                self.inverse_lexicon[self.ngram_count] = word
                self.ngram_count = self.ngram_count + 1
                #print( 'Lexicon: ',self.ngram_count,word)
    
    def isTweetReply(self,tweet):
        if ( tweet[0].lower() == 'r' and  tweet[1].lower() == 't' ):
            return True
        else:
            return False
    
    def get_tweet_map(self):
        return self.tweet_map
        
    def isTweetFromAirline(self, tweet, airline_handle):
        if( tweet['user']['screen_name'] == airline_handle):
            return True
        else:
            return False
        
    #Add the tweet to the lexicon set
    def addTweet(self,tweet):
        
        self.tweet_count += 1
        self.tweet_map[self.tweet_count] = tweet
        
        #1. get the sentiment for the tweet
        sentiment = self.tweetCleaner.getSentiment(tweet)
        
        #2. Clean the Tweet
        tweet = self.tweetCleaner.cleanTweet(tweet)
        
        #3. get the ngrams for the tweet
        _ngrams = nltk.ngrams(tweet.split(), self.ngrams)
            
        #4. Add the ngrams to the lexicon dictionary
        words = list(_ngrams)
        self.addToLexicon( words )
        
        #5. Add this tweet row to the training set
        self.addToTrainingData(words,sentiment)
    
    #Build the Feature set for all the tweets
    def addToTrainingData(self,ngrams,sentiment):
        row = np.zeros(self.max_features_count + 1 ) # last feature is the label 
        
        for word in ngrams:
            if ( word in self.lexicon ):
                row[ self.lexicon[word] ] = row[ self.lexicon[word] ] + 1 #Increase the count of the word 

        row [ self.max_features_count ] = sentiment
        self.data[ len(self.data)  ]  = row 
    
    def getFeatures(self):
        print ( len(self.data) )
        features = np.zeros(( len(self.data), self.max_features_count ))
        
        #features = np.array(list(self.data.values()))
        
        # features = csr_matrix(list(result.items()), dtype=int8)
        for index in self.data:
            features[index][:] =  self.data[index][:-1]
            
        return features
    
    def getLabels(self):
        labels = np.zeros( len(self.data) )
        
        for index in self.data:
            labels[index] = ( int(self.data[index][self.max_features_count]) )
            
        return labels
    
    def getHeaders(self):
        headers = []
        for i in range(self.max_features_count):
            headers.append( self.inverse_lexicon[i] )
        return headers
            

#Train Classifiers
class Classifiers:
    def __init__(self,ngrams,stopwords_fname,f_airlines,split,noOfFeatures):
        self.ngrams = ngrams
        self.f_airlines = f_airlines
        self.split  = split
        self.noOfFeatures=noOfFeatures

        #1.Define Custom Classes required for the class
        self.features = BuildFeatureSet(self.ngrams, stopwords_fname, max_features_count = self.noOfFeatures)
        self.jsonParser = JsonParser()

        #2. Build the feature set
        print ('Building the feature set')
        
        for file in f_airlines:
            print ('Parsing tweets from the file...')
            tweets =  self.jsonParser.loadData(file)
            #airline_handle = file.split("_")[1]
            i = 0
            for tweet in tweets:
                if ( i > 25000 ):
                    break
                #1. Skip if the tweet the is a reply to an existing tweet
                if ( self.features.isTweetReply(tweet['text']) == False): # and self.features.isTweetFromAirline(tweet, airline_handle) == False ):
                    self.features.addTweet ( tweet['text']  ) 
                    i=i+1
        
        #3. Prepare the train and test datasets
        print ( 'Preparing the training and the test set...')
        self.X = self.features.getFeatures()
        self.y = self.features.getLabels()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size= self.split,test_size=1- self.split)
        print ( 'Done preapring the train and test set...Ready to train classifiers')
        
        #4. Define Variables for classifiers performance
        #4a. Logistic Regression
        self.logreg_train_acc = 0
        self.logreg_test_acc = 0
        
        #4b. Gaussian Naive Bayes
        self.nb_train_acc = 0
        self.nb_test_acc = 0
        
        #4c. LibLinearSVC
        self.svc_train_acc = 0
        self.svc_test_acc = 0
        
        #4d. Decision Tree
        self.dt_train_acc = 0
        self.dt_test_acc = 0

        #4e.  Ada Boost
        self.adaboost_train_acc = 0
        self.adaboost_test_acc = 0
        
        #4f.
        self.rf_train_acc = 0
        self.rf_test_acc = 0
        
        #4g.
        self.ltsm_train_acc = 0
        self.ltsm_test_acc = 0
    
    def applyPCA(self,n):
        print ( 'Applying PCA on X' )
        pca = PCA(n_components=n)
        #print ( pca.explained_variance_ratio_ )
        self.X = pca.fit_transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, train_size= self.split,test_size=1- self.split)
        
    def printResults(self):
        print ( 'Results for Logistic Regression')
        print( '(train,test):(',self.logreg_train_acc,self.logreg_test_acc,')' )
        print ( 'Results for Gaussian Naive Bayes')
        print( '(train,test):(',self.nb_train_acc,self.nb_test_acc,')' )
        print ( 'Results for LibLinear SVM')
        print( '(train,test):(',self.svc_train_acc,self.svc_test_acc,')' )
        print ( 'Results for Decision Tree')
        print( '(train,test):(',self.dt_train_acc,self.dt_test_acc,')' )
        print ( 'Results for Adaboost')
        print( '(train,test):(',self.adaboost_train_acc,self.adaboost_test_acc,')' )
        print ( 'Results for Random Forests')
        print( '(train,test):(',self.rf_train_acc,self.rf_test_acc,')' )
        print ( 'Results for LTSM')
        print( '(train,test):(',self.ltsm_train_acc,self.ltsm_test_acc,')' )
    
    def getResults(self):
        return [self.logreg_train_acc,self.logreg_test_acc, 
                self.nb_train_acc,self.nb_test_acc, 
                self.svc_train_acc,self.svc_test_acc,  
                self.dt_train_acc,self.dt_test_acc, 
                self.adaboost_train_acc,self.adaboost_test_acc, 
                self.rf_train_acc,self.rf_test_acc]
                #,self.ltsm_train_acc,self.ltsm_test_acc]
    
    def runClassifiers(self):
            #1. Logistic regression classifier
            print('Training Logistic Regression Classifiers')
            
            C = [0.1,0.001,10]

            logreg = GridSearchCV(LogisticRegression(max_iter=100),cv = 5, param_grid= {"C" : C},verbose=5,n_jobs=-1)            
            logreg.fit(self.X_train, self.y_train)

            self.logreg_train_acc = accuracy_score( self.y_train,logreg.predict(self.X_train) )
            self.logreg_test_acc = accuracy_score( self.y_test,logreg.predict(self.X_test) )
            
            #2. Naive Bayes
            print('Training Gaussian Naive Bayes Classifier')
            nb = GaussianNB()
            nb.fit(self.X_train, self.y_train)

            self.nb_train_acc = accuracy_score( self.y_train,nb.predict(self.X_train) )
            self.nb_test_acc = accuracy_score( self.y_test,nb.predict(self.X_test) )
            
            #3.LibLinear SVM
            print( 'Training SVM Lib Linear Classifier')
            svc = GridSearchCV(LinearSVC(max_iter=200),cv = 5, param_grid= {"C" : C},verbose=5,n_jobs=-1)
            svc.fit(self.X_train, self.y_train)

            self.svc_train_acc =  accuracy_score(self.y_train,svc.predict(self.X_train)) 
            self.svc_test_acc = accuracy_score(self.y_test,svc.predict(self.X_test)) 
            
            #4.Decision Tree
            print( 'Training Decision Tree Classifier' )
            max_depth = [1,3,5,7]
            dt = GridSearchCV(DecisionTreeClassifier(),cv = 5, param_grid= {"max_depth" : max_depth},verbose=5,n_jobs=-1)
            dt.fit(self.X_train, self.y_train)
            
            self.dt_train_acc =  accuracy_score(self.y_train,dt.predict(self.X_train)) 
            self.dt_test_acc = accuracy_score(self.y_test,dt.predict(self.X_test)) 
            
            #5. AdaBoost
            print( 'Training AdaBoost' )
            grd = GradientBoostingClassifier(n_estimators=50,max_depth=5,random_state=42)
            grd.fit(self.X_train, self.y_train)
            self.adaboost_train_acc =  accuracy_score(self.y_train,grd.predict(self.X_train)) 
            self.adaboost_test_acc = accuracy_score(self.y_test,grd.predict(self.X_test)) 
            
            #6. Random Forests
            print( 'Training Random Forests' )
            rf = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42,n_jobs=-1)
            rf.fit(self.X_train, self.y_train)
            self.rf_train_acc =  accuracy_score(self.y_train,rf.predict(self.X_train)) 
            self.rf_test_acc = accuracy_score(self.y_test,rf.predict(self.X_test)) 
            
            #7.Deep net LTSM
            #print( 'Training LTSM Deep Net' )
            #embedding_vecor_length = 300
            #model = Sequential()
            #top_words = self.X_train.shape[1]
            #max_review_length = self.X_train.shape[1]
            #model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))

            ## Convolutional model (3x conv, flatten, 2x dense)
            #model.add(Dropout(0.2))
            #model.add(Convolution1D(64, 5, activation='relu'))
            #model.add(MaxPooling1D(pool_size=4))
            #model.add(LSTM(128))
            #model.add(Dense(3,activation='softmax'))

            ## Log to tensorboard
            #tensorBoardCallback = TensorBoard(log_dir='./logs', write_graph=True)
            #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            #model.fit(self.X_train, self.y_train, epochs=3, batch_size=64)

            ## Evaluation on the test set
            #self.ltsm_train_acc =  model.evaluate(self.X_train, self.y_train, verbose=0)[1]
            #self.ltsm_test_acc =  model.evaluate(self.X_test, self.y_test, verbose=0)[1]

files = [alaska,allegiant,united,american_air,delta,froniter,hawaii,jetblue,south_west,spirit,sun_country,virgin]
noOfFeatures = [5000]

headers=['logreg_train','logreg_test','nb_train','nb_test','svm_train','svm_test','dt_train','dt_test','adaboost_train','adaboost_test'
        ,'rf_train','rf_test'] 
results=[]

#for featureLen in noOfFeatures:
#    results = []
#    for i in range(3):
#        print( 'Generate results for ', i, '-gram Classifiers')
#        clf = Classifiers(i+1,'stopwords.txt',files,0.75,featureLen)
#        clf.runClassifiers()
#        results.append( clf.getResults() )
    
#    print( 'Results (features): (',featureLen,'):' )
#    df = pd.DataFrame(results, columns=headers)
#    print( df)
    
print ( 'Results After Performing PCA ')
for featureLen in noOfFeatures:
    results = []
    for i in range(3):
        print( 'Generate results for ith gram Classifiers')
        clf = Classifiers(i+1,'stopwords.txt',files,0.75,featureLen)
        clf.applyPCA(300) #Keeping the top 20 important features
        clf.runClassifiers()
        results.append( clf.getResults() )
    
    print( 'Results (features): (',featureLen,'):' )
    df = pd.DataFrame(results, columns=headers)
    print( df)
    df.to_csv( str(i)+'_'+'res'+'.csv' )



