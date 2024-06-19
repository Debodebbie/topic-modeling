from typing import List
import pandas as pd
import pickle
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from transformers import GPT2LMHeadModel, GPT2Tokenizer


seed = 321 

def lda_model(df: pd.DataFrame, num_topics: int=10, summarize: bool = False) -> List[str]:
    """LDA topic modeling. The generated topic words can be used by a generative LLM to create coherent topics.

    Args:
        df (pd.DataFrame): tweets 
        num_topics (int, optional): Number of topics to be outputed. Defaults to 10.
        summarize (bool, optional): Summarize the topics using a LLM. Defaults to False.

    Returns:
        List: list of topics (strings)
    """

    token_per_tweet =  [tokens.split() for tokens in df['final']] 
    dictionary = Dictionary(token_per_tweet)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    dictionary.token2id
    tweets_bow = [dictionary.doc2bow(tweet.split()) for tweet in df['final']]
    tweets_lda = LdaModel(tweets_bow,
                        num_topics = num_topics,
                        id2word = dictionary,
                        random_state = seed,
                        passes=10)

    sentences = []
    for i in range(tweets_lda.num_topics):
        top_words = tweets_lda.show_topic(i) 
        topic_words_string = " ".join([word for word, _ in top_words])
        sentences.append(topic_words_string)

    # Summarize using gpt
    summaries = []
    if summarize:
        for s in sentences:
            summaries.append(summarize_topic(s))

        return summaries
    
    return sentences

def summarize_topic(sentence: List) -> str:
    """Creates a summary of a list of strings using gpt-2.

    Args:
        sentence (List): List of strings (topic words)

    Returns:
        str: summary
    """

    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    prompt = f"Summarize the following tweet:\n\n{sentence}\n\nSummary:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    summary_ids = model.generate(inputs, max_length=100, num_return_sequences=1)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

def bertopic_model(df: pd.DataFrame) -> List[str]:
    """Topic modeling using BERTopic and a sentence transformer

    Args:
        df (pd.DataFrame): tweets

    Returns:
        List[str]: list of topics (strings)
    """

    model_name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    embedding_model = SentenceTransformer(model_name)
    text_list = df['final'].tolist()
    topic_model = BERTopic(embedding_model=embedding_model)
    _, _ = topic_model.fit_transform(text_list)
    topic_df = topic_model.get_topic_info()
    topic_df['Name'] = topic_df['Name'].str.replace(r'^[-]?\d+_','', regex=True).str.replace('_',' ')

    # Save file to pickle
    with open("data/topic_model.pkl", "wb+") as f:
        pickle.dump(topic_model, f)

    print("Topic model saved in data/topic_model.pk")

    return topic_df['Name'].to_list()
