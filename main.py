import openai
import requests
import streamlit as st
from bs4 import BeautifulSoup
from newspaper import Article
import nltk
import pandas as pd
from pygooglenews import GoogleNews
import spacy

nltk.download('punkt')

# Use your OpenAI API key to access GPT-3
openai.api_key = st.secrets["openai_api_key"]

# Use streamlit to create a text input for the user's query
query = st.text_input("Enter your news query:")

# Set the parameters for the GPT-3 model
model_engine = "text-davinci-003"
prompt = f"Find top 20 related search terms for news based on keyword: {query}"
max_tokens = 1024
n = 5
stop = None
temperature = 0.7

# Send the query to GPT-3
completions = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=max_tokens,
    n=n,
    stop=stop,
    temperature=temperature,
)

# Use streamlit to display the related search terms
st.write("Related search terms:")
related_result = completions.choices[0].text.split("\n")[2:]
new_related_result = set()
for item in related_result:
    new_item = item.strip("1234567890. ")
    new_related_result.add(new_item)
st.write(new_related_result)

# use the search terms to query for related news
new_related_list = list(new_related_result)
gn = GoogleNews()
news_result = gn.search(
    " OR ".join(new_related_list),
)

nlp = spacy.load('en_core_web_sm')

for article in news_result["entries"][:20]:
    article_link = article["link"]
    if article_link:
        try:
            # Article details
            article = Article(article_link)
            article.download()
            article.parse()
            article.nlp()
            article_title = article.title
            article_publish_date = article.publish_date
            article_text = article.text
            article_summary = article.summary
            article_image = article.top_image
            st.markdown(f'[{article_title}]({article_link})')
            if article_image:
                st.image(article_image)
            st.write(article_summary)
            st.write(article_publish_date)

            # NER
            doc = nlp(article_text)
            ents = [(e.text, e.label_) for e in doc.ents]
            df = pd.DataFrame(ents, columns=["Entity", "Label"])
            df = df.drop_duplicates()
            df = df.loc[df['Label'].isin(['PERSON', 'ORG', 'PRODUCT'])]
            st.write(df)

        except:
            article_text = "Unable to extract article text."
    else:
        article_text = "Unable to extract article text."

