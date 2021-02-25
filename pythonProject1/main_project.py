from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import TweetTokenizer,word_tokenize
from nltk import FreqDist,classify,NaiveBayesClassifier
import pandas as pd
import re,string,random

tt=TweetTokenizer()
#Define modules for pre-process of data.
def remove_noise(tweet_tokens,stop_words=()):

    cleaned_tokens=[]

    for token,tag in pos_tag(tweet_tokens):
        token=re.sub('http[s]?://(?:[a-zA-Z][0-9]|[$-_@.&+#]|[!*\(\),]|'\
                     '(?:%[0-9a-fA-F][0-9a-fA-F]))+','',token)
        token=re.sub("(@[A-Za-z0-9_]+)","",token)

        #print("token",token)

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

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

#Preparing data for input
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token,True] for token in tweet_tokens)

#Run the main programme
if __name__ == '__main__':
    
    #Inputs the dataframes with pandas
    positive_tweets =pd.read_csv("~/Desktop/DT-Datasets-part3/Project 7/SocialMedia_Positive.csv")
    negative_tweets =pd.read_csv("~/Desktop/DT-Datasets-part3/Project 7/SocialMedia_Negative.csv")
    #Tokenizing strings in the column "Text" of the dataframes 
    positive_tweet_tokens= positive_tweets["Text"].apply(tt.tokenize)
    negative_tweet_tokens = negative_tweets["Text"].apply(tt.tokenize)
    
    stop_words=stopwords.words("english")

    #Creates two empty lists 
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    
    #put the clean data of the in those lists
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens,stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens,stop_words))
        
    
    #Call all the positive words
    all_pos_words=get_all_words(positive_cleaned_tokens_list)
    #Take the frequencies of every word in tweets datafram and print the 10 most common
    freq_dist_pos=FreqDist(all_pos_words)

    print("10 Most common words:",freq_dist_pos.most_common(10))
    
    #Prepare data for the input (create a list of dictionary sentences)
    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    
    #Create list that contains lists that contains our dictionary sentences and the string "possitive"
    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]
    #Create list that contains lists that contains our dictionary sentences and the string "negative"
    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]
    #Merging the list of data
    dataset = positive_dataset + negative_dataset
    #Randomize their position
    random.shuffle(dataset)
    #split dataset in 80% training and 20% as testing
    value= 0.8*len(dataset)+1
    train_dataset=dataset[:int(value)]
    test_dataset=dataset[int(value):]


    #Call and train Naives Bayes classifier
    classifier = NaiveBayesClassifier.train(train_dataset)
    #Check and print the accuracy with the testing data
    print("Accuracy is:", classify.accuracy(classifier, test_dataset))
    #Show the 10 more important words
    print(classifier.show_most_informative_features(10))
    #Create and run a testing tweet
    custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."
    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))









