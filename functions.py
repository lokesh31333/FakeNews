import re
import string
import pandas  as pd
import numpy as np
import pickle
import json

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download(['stopwords', 'punkt', 'wordnet', 'omw-1.4'])

# Suppress warnings.
import warnings
warnings.filterwarnings("ignore")

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix

# Deep learning
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

# LIME
from lime.lime_text import LimeTextExplainer

# Chat GPT.
import openai
from dotenv import load_dotenv, find_dotenv             # Read local .env file.
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, BaseOutputParser
from langchain.chains import LLMChain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate,SystemMessagePromptTemplate

# Relevant news.
import feedparser
import urllib.parse
from bs4 import BeautifulSoup

# Project modules.
import configuration as conf

# Initialize lemmatizer.
lemmatizer = WordNetLemmatizer()

# Initialize stemmer.
stemmer = PorterStemmer()

# Initialize stopword.
stopwords_english = stopwords.words('english')

# Load the tokenizer from the pickle file
tokenizer_pickle_file = conf.tokenizer_pickle_file

with open(tokenizer_pickle_file, 'rb') as file:
    loaded_tokenizer = pickle.load(file)

# Load the model.
model = tf.keras.models.load_model(conf.trained_model)

# Specify classes.
class_names=['Real','Fake']

# Initialize LIME text explainer.
explainer= LimeTextExplainer(class_names=class_names)

# Function to decontract words.
def decontract(text):
    text = re.sub(r"won[’']t", "will not", text)
    text = re.sub(r"can[’']t", "can not", text)
    text = re.sub(r"n[’']t", " not", text)
    text = re.sub(r"[’']re", " are", text)
    text = re.sub(r"[’']s", " is", text)
    text = re.sub(r"[’']d", " would", text)
    text = re.sub(r"[’']ll", " will", text)
    text = re.sub(r"[’']t", " not", text)
    text = re.sub(r"[’']ve", " have", text)
    text = re.sub(r"[’']m", " am", text)
    return text


# Function to clean the text.
def process_text(text):

    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', str(text))

    # decontract words
    text=decontract(text)

    tokens = word_tokenize(text)

    texts_clean = []
    # Create word mapping for word highlighter.
    word_mapping = {}

    for word in tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation+'...'):  # remove punctuation

            lemma_word = lemmatizer.lemmatize(word, "v")  # Lemmatize word
            stem_word = stemmer.stem(lemma_word)

            # Store the mapping from the stemmed word to the original word
            word_mapping[stem_word] = word

            texts_clean.append(stem_word)


    cleaned_text = " ".join(texts_clean)
    return cleaned_text, word_mapping

# Function to tokenize and pad text.
def tokenize_pad_text(news_text):

    list_tokenized_news_text= loaded_tokenizer.texts_to_sequences(pd.Series(news_text))
    padded_news_text = pad_sequences(list_tokenized_news_text, maxlen=conf.max_news_length, truncating=conf.truncating, padding=conf.padding)

    return padded_news_text

# Function to predict news.
def make_prediction(news_text):

    # make prediction.
    prediction = model.predict(news_text)

    if prediction[0][0] > 0.5:
        result = 'Fake News'
    else:
        result = 'Real News'

    return result


# Function to predict predicted probabilities.
def predict_proba(news_text):

    list_tokenized_news = loaded_tokenizer.texts_to_sequences(pd.Series(news_text))
    padded_news_text = pad_sequences(list_tokenized_news, maxlen=conf.max_news_length)

    fake_prob=model.predict(padded_news_text)

    probabilities = np.column_stack((1 - fake_prob, fake_prob))

    return probabilities

# Function to get key for the given dictionary value.
def get_key_for_value(value, word_mapping):

    # Create a reverse mapping with values as keys and keys as values
    reverse_mapping = {v: k for k, v in word_mapping.items()}

    # Look up the value to get the corresponding key
    key_value = reverse_mapping.get(value)

    return key_value

# Function to highlight text.
def text_highlighter(text, word_weights, word_mapping):

    words = text.split()
    highlighted_text = []

    for word in words:
        # Lower case word.
        word_lower = word.lower()

        # Remove special characters.
        word_lower = re.sub(r'[^\w\s]', '', str(word_lower))

        word_key = get_key_for_value(word_lower, word_mapping)
        weight = word_weights.get(word_key, 0)

        if weight > 0:
            if weight > 0.020:
                highlight_style = "background-color: #329932;"
            elif weight > 0.015:
                highlight_style = "background-color: #4ca64c;"
            elif weight > 0.010:
                highlight_style = "background-color: #66b266;"
            elif weight > 0.005:
                highlight_style = "background-color: #7fbf7f;"
            elif weight > 0.001:
                highlight_style = "background-color: #99cc99;"
        elif weight < 0:
            if weight < -0.020:
                highlight_style = "background-color: #ff1919;"
            elif weight < -0.015:
                highlight_style = "background-color: #ff3232;"
            elif weight < -0.010:
                highlight_style = "background-color: #ff6666;"
            elif weight < -0.005:
                highlight_style = "background-color: #ff9999;"
            elif weight < -0.001:
                highlight_style = "background-color: #ffcccc;"
        else:
            highlight_style = ""

        highlighted_word = f'<span class="highlight" style="{highlight_style}">{word}</span>'

        highlighted_text.append(highlighted_word)

    modified_text = " ".join(highlighted_text)

    return modified_text


# Function for LIME explanation.
def lime_explanation(text, news_type):

    exp = explainer.explain_instance(str(text), predict_proba, num_features=conf.num_features)

    # print ('\n'.join(map(str, exp.as_list(label=1))))

    word_weights = {}

    # for word, weight in exp.as_list(label=1):
    #     word_weights[word] = weight

    if news_type == 'Fake News':
        # Extract up to 10 positive word weights for fake news
        count_positive = 0
        for word, weight in exp.as_list(label=1):
            if weight > 0 and count_positive < 10:
                word_weights[word] = weight
                count_positive += 1

        # If positive weights are less than 10, add remaining negative word weights
        for word, weight in exp.as_list(label=1):
            if weight < 0 and len(word_weights) < 10:
                word_weights[word] = weight

    elif news_type == 'Real News':
        # Extract up to 10 negative word weights for real news
        count_negative = 0
        for word, weight in exp.as_list(label=1):
            if weight < 0 and count_negative < 10:
                word_weights[word] = weight
                count_negative += 1

        # If negative weights are less than 10, add remaining positive word weights
        count_positive = 0
        for word, weight in exp.as_list(label=1):
            if weight > 0 and len(word_weights) < 10:
                word_weights[word] = weight

    # Invert the extracted word weights
    inverted_word_weights = {word: -weight for word, weight in word_weights.items()}

    return inverted_word_weights


# Sentiment and Tone analysis with chat GPT.

template = "You are a helpful assistant who can analyse news text and find out sentiment and tone of the news. \
A user will pass news text and you have to find out sentiment from this list [positive, negative, neutral]. \
You have to also find out tone of the given news text from this list [happy, sad, angry, joyful, fearful, cautious]. \
ONLY return a python directory containing sentiment and tone"

templateForNewsType = "You are a helpful assistant who can analyse news text and find out if it is belongs to which category.\
A user will pass news text and you have to find out If news belongs to which category in one word from this list[sports, politics, technology, entertainment, health,\
 \science, business, world news, education, real estate,\
\ culture, travel, environment, fashion, art, lifestyle, \
\ crime, religion, food, health & fitness, cinema, local news,\
\ nation news, automotive, economy, weather, law, music, books,\
\ events]. \
            "

templateForAIDetection = "You are a helpful assistant who can analyse news text and find out if it is Genereated by AI or not.\
A user will pass news text and you have to find out If news was genereated by AI or not from this list [AI, Human, Might be AI, Might be Human]. \
            "

templateForIntentDetection =" You are a helpful assistant who can analyse Author's Intent for which news article belongs.\
A user will pass news text and you have to find out Author's Intentin single word from this list [Analysis, Opinion, Reportage, Interview, Review]. \
            "

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
system_message_prompt_AiDetection = SystemMessagePromptTemplate.from_template(templateForAIDetection)
system_message_prompt_NewsType = SystemMessagePromptTemplate.from_template(templateForNewsType)
system_message_prompt_NewsIntent = SystemMessagePromptTemplate.from_template(templateForIntentDetection)
_ = load_dotenv(find_dotenv())

llm = ChatOpenAI()

class commaSeparatedSentimentAndTone(BaseOutputParser):
    def parse(self, text):
        json_data = json.loads(text)
        sentiment = json_data['sentiment']
        tone = json_data['tone']
        return sentiment, tone

# Function to find out sentiment and tone of the news.
def sentiment_tone_analysis(p_news_text):
    sentiment = None
    tone = None
    err_msg = None
    human_template = "{news_text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm = llm,
                     prompt = chat_prompt,
                     output_parser = commaSeparatedSentimentAndTone()
                )

    try:
        sentiment, tone  = chain.run(p_news_text)
        # return sentiment, tone
    except Exception as e:
        # Handle timeout error, e.g. retry or log
        print(f"Error occurred while calling OpenAI API\n{e}")
        pass

    return sentiment, tone, err_msg

def ai_detection(p_news_text):
    err_msg = None
    result = None
    human_template = "{news_text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt_AiDetection, human_message_prompt])
    chain = LLMChain(llm = llm,
                     prompt = chat_prompt,
                    #  output_parser = commaSeparatedSentimentAndTone()
                )

    try:
        result  = chain.run(p_news_text)
        # return sentiment, tone
    except Exception as e:
        # Handle timeout error, e.g. retry or log
        print(f"Error occurred while calling OpenAI API\n{e}")
        pass
    print(result)
    return result, err_msg

def NewsType_detection(p_news_text):
    err_msg = None
    result = None
    human_template = "{news_text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt_NewsType, human_message_prompt])
    chain = LLMChain(llm = llm,
                     prompt = chat_prompt,
                    #  output_parser = commaSeparatedSentimentAndTone()
                )

    try:
        result  = chain.run(p_news_text)
        # return sentiment, tone
    except Exception as e:
        # Handle timeout error, e.g. retry or log
        print(f"Error occurred while calling OpenAI API\n{e}")
        pass
    print(result)
    return result, err_msg


def NewsIntent_detection(p_news_text):
    err_msg = None
    result = None
    human_template = "{news_text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt_NewsIntent, human_message_prompt])
    chain = LLMChain(llm = llm,
                     prompt = chat_prompt,
                    #  output_parser = commaSeparatedSentimentAndTone()
                )

    try:
        result  = chain.run(p_news_text)
    except Exception as e:
        # Handle timeout error, e.g. retry or log
        print(f"Error occurred while calling OpenAI API\n{e}")
        pass
    print(result)
    return result, err_msg

# Function to parse rss news feed to find relevant news on internet.
def fetch_rss_feed(search_query):
    # Format the search query to be URL-friendly
    formatted_query = urllib.parse.quote(search_query)

    # Construct the RSS feed URL with the formatted query
    rss_url = f"http://news.google.com/news?q={formatted_query}&output=rss"

    # Parse the RSS feed
    feed = feedparser.parse(rss_url)

    news_data = {
        'news_search_title': feed.feed.title,
        'news_search_url': feed.feed.link,
        'news_list': []
    }

    if feed.entries:
        for entry in feed.entries:
            # Parse HTML summary to extract plain text
            soup = BeautifulSoup(entry.description, "html.parser")
            plain_text_summary = soup.get_text(separator="\n")
            plain_text_summary = plain_text_summary.split("\n")[0]

            title = ""
            source = ""
            try:
                title = entry.title.split(" - ")[0]
                source = entry.title.split(" - ")[1]
            except:
                pass

            news_entry = {
                'entry_title': title,
                'source': source,
                'summary': plain_text_summary,
                'url': entry.link
            }
            news_data['news_list'].append(news_entry)

    return news_data


# Function to clean the text.
def clean_text(text):

    # remove hyperlinks
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', str(text))

    return text


