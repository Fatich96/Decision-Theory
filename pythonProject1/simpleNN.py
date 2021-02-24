#Libraries
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords,twitter_samples
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer,word_tokenize
from nltk import FreqDist,classify,NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.activations import *
from keras import layers
from keras.callbacks import ModelCheckpoint
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re,string
import numpy as np

tt=TweetTokenizer()
#Define modules for pre-process of data.
#Remove noise removes URL ,Tweeter nicknames,punctuaries and stopwords 
def remove_noise(tweet_tokens,stop_words=()):

    cleaned_tokens=[]

    for token,tag in pos_tag(tweet_tokens):
        token=re.sub('http[s]?://(?:[a-zA-Z][0-9]|[$-_@.&+#]|[!*\(\),]|'\
                     '(?:%[0-9a-fA-F][0-9a-fA-F]))+','',token)
        token=re.sub("(@[A-Za-z0-9_]+)","",token)

        print("token",token)

        if tag.startswith("NN"):
            pos='n'
        elif tag.startswith("VB"):
            pos='v'
        else:
            pos='a'

        lmtzr=WordNetLemmatizer()
        token=lmtzr.lemmatize(token,pos)

        if len(token)>0 and token not in  string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


#Run the main programme
if __name__ == '__main__':

    #Inputs the dataframes with pandas
    positive_tweets =pd.read_csv("~/Desktop/DT-Datasets-part3/Project 7/SocialMedia_Positive.csv")
    negative_tweets =pd.read_csv("~/Desktop/DT-Datasets-part3/Project 7/SocialMedia_Negative.csv")
    #Tokenizing strings in the column "Text" of the dataframes 
    positive_tweet_tokens= positive_tweets["Text"].apply(tt.tokenize)
    negative_tweet_tokens = negative_tweets["Text"].apply(tt.tokenize)
    
    #Merging vertically the dataframes under the name df
    df= pd.concat([positive_tweets,negative_tweets])
    #Plot the amounts of data of the column "Sentiment" to check if they are biased
    sns.set_theme()
    sns.countplot(x='Sentiment',data=df)
    plt.show()
    

    #Import the list of the english stopwords
    stop_words=stopwords.words("english")

    #Creates two empty lists 
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    
    #put the clean data of the in those lists
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens,stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens,stop_words))





    #Merging the two lists
    dataset =  positive_cleaned_tokens_list + negative_cleaned_tokens_list
    
    #Counts and print the length of the larger sentence in our dataset
    max_ln = len(dataset[0])
    for sentence in dataset:
        print(sentence)
        if max_ln < len(sentence):
            max_ln = len(sentence)
    print(max_ln)

    #create a y variable with the list of the sentiments
    y=df['Sentiment']
    #transform "Positive" in 1's and "Negative" in 0's in the y list
    y=np.array(list(map(lambda x: 1 if  x== 'positive' else 0,y)))

    #split Data for training and testing randomly
    x_train,x_test,y_train,y_test = train_test_split(dataset,y, test_size=0.20,random_state=42)


    max_words=max_ln
    #Tokenizing data
    tokenizer=Tokenizer(num_words=max_words)
    #Create a vocbulary that corresponds numbers in data
    tokenizer.fit_on_texts(x_train)
    x_train=tokenizer.texts_to_sequences(x_train)
    x_test=tokenizer.texts_to_sequences(x_test)
    print (x_train)
    print(x_test)

    #Takes the vocabulary size in order to create the matrix of the embending layer
    vocab_size=len(tokenizer.word_index)+1
    print(vocab_size)
    
    #The data should come as a vectors of the same size, we put 0 at the end of the shorter data
    x_train=pad_sequences(x_train,padding='post',maxlen=max_words)
    x_test=pad_sequences(x_test,padding='post',maxlen=max_words)

    #Create the model imput 3 Layers
    model1 = Sequential()
    embedding_layer=layers.Embedding(vocab_size,max_words,input_length=max_words,trainable=False)
    model1.add(embedding_layer)
    model1.add(layers.Flatten())
    model1.add(layers.Dense(1,activation='relu'))
    model1.compile(optimizer='adam',loss='binary_crossentropy', metrics=['acc'])
    
    #Print Model's summary
    print(model1.summary())
    #Fiting and training the model in a 100 epochs, the NN inputs data in batches of 34(Input Layer)
    history = model1.fit(x_train, y_train, batch_size=34, epochs=100, verbose=1, validation_split=0.2)
    
     #Evaluate Model
    score = model1.evaluate(x_test, y_test, verbose=1)
    
    #Print Accuracy and loss
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    #Plot an accuracy table
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    #Plot an error table
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
