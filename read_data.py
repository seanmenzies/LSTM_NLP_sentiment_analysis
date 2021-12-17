import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import re

# import data as df
fname = r'C:\Users\Sean-work\OneDrive\Coding\PycharmProjects\NLP_btcsentanalysis\reviews\Reviews.csv'
df = pd.read_csv(fname)


# visualise reviews by rating
def visualise_ratings(df):
    fig = px.histogram(df, x="Score")
    fig.update_layout(title_text='Product Scores')
    fig.show()


# generate wordcloud
def generate_wordcloud(df):
    nltk.download('stopwords')
    stwds = set(stopwords.words('English'))
    stwds.update(["br", "href"])

    text_input = " ".join(t for t in df.Text)
    wordcloud = WordCloud(stopwords=stwds).generate(text_input)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(r'C:\Users\Sean-work\OneDrive\Coding\PycharmProjects\NLP_btcsentanalysis\wordcloud.png')
    plt.show()


# create 'sentiment' df column based on user score
def gen_sentiment(df):
    df = df[df['Score'] != 3]
    df['Sentiments'] = 0
    df['Sentiments'].loc[df['Score'] < 3] = -1
    df['Sentiments'].loc[df['Score'] > 3] = 1
    return df


# clean data
def remove_punctuation(text):
    text = re.sub('[.,:;\'\"!?]', '', text)
    return text


df['Text'] = df['Text'].apply(remove_punctuation)
df['Summary'] = df['Summary'].apply(remove_punctuation)
