# imports
import pandas as pd
from IPython.display import display
import stanza 
import statistics
import spacy
import dataframe_image as dfi
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from wordcloud import WordCloud


en_nlp = spacy.load('en_core_web_sm')
zh_nlp = spacy.load('zh_core_web_sm')

# article file paths
zh_filepath = "data/ai_zh_overview.tsv"
en_filepath = "data/ai_en_overview.tsv"

# article dataframes
zh_news_content = pd.read_csv(zh_filepath, sep="\t", header = 0, keep_default_na=False, encoding="utf-8")
en_news_content = pd.read_csv(en_filepath, sep="\t", header = 0, keep_default_na=False, encoding="utf-8")

# list of dataframes to loop over
news_content_list = [zh_news_content, en_news_content]

def save_metadata(zh_news_content, en_news_content):
    """
    :param zh_news_content: overview of 100 Chinese articles converted into pandas df
    :param en_news_content: overview of 100 English articles converted into pandas df
    
    This function gets some basic statistics on the articles and saves the results to a csv file.
    
    """
    # get statistics
    zh_metadata = zh_news_content.describe()
    en_metadata = en_news_content.describe()

    # save to csv
    zh_metadata.to_csv("outputs/zh_metadata.csv", index=False)
    en_metadata.to_csv("outputs/en_metadata.csv", index=False)
    
    # import to image
    #dfi.export(metadata_en, "outputs/en_metadata.png")
    #dfi.export(metadata_zh, "outputs/zh_metadata.png")

# call function
save_metadata(zh_news_content, en_news_content)

def get_content_statistics(news_content, nlp):   
    """
    :param news_content: overview of 100 articles converted into pandas df
    :param nlp: define spacy language depending on the language of news_content
    :returns: list of data for converting into dataframe
    
    This function extracts content statistics from the text of the articles.
    
    """
    # define empty lists for df column
    data = []

    # define empty lists for token counting
    texts = []
    tokens = []
    sents = []
    lemmas = []
    pos_noun = []
    pos_verb = []
    pos_adj = []
    pos_adv = []
    pos_prep = []
    pos_pron = []
    pos_proper = []
    pos_det = []

    # iterate over all articles and add them to list
    current_article = news_content["Text"]
    for item in current_article:
        texts.append(item)
    
    # iterate over text
    # process with spacy
    # add pos_tags to designated lists
    for text in texts:
        doc = nlp(text)
        for sent in doc.sents:
            sents.append(sent)
        for token in doc:
            tokens.append(token.text)
            lemmas.append(token.lemma)
            if token.pos_ == 'NOUN':
                pos_noun.append(token.pos_)
            elif token.pos_ == 'DET':
                pos_det.append(token.pos_)
            elif token.pos_ == 'ADJ':
                pos_adj.append(token.pos_)
            elif token.pos_ == 'ADP':    
                pos_prep.append(token.pos_)
            elif token.pos_ == 'PROPN':
                pos_proper.append(token.pos_)
            elif token.pos_ == 'PRON': 
                pos_pron.append(token.pos_)
            elif token.pos_ == 'VERB' or token.pos_=='AUX':
                pos_verb.append(token.pos_)
            elif token.pos_ == 'ADV':
                pos_adv.append(token.pos_)

    # Total number of sentences
    total_sents = len(sents) 

    # Total number of tokens
    total_tokens = len(tokens) 

    # Total number of lemmas
    total_lemmas = len(set(lemmas)) 

    # calculate average sentence length
    length_sents = []
    for sent in sents:
        l = len(sent)
        length_sents.append(l)

    sent_mean = statistics.mean(length_sents)

    # calculate average token length
    length_tokens = []
    for token in tokens:
        l = len(token)
        length_tokens.append(l)

    token_mean = statistics.mean(length_tokens)

    data.append("{:.0f}".format(total_tokens))
    data.append(total_lemmas)
    data.append(total_sents)
    data.append("{:.2f}".format(token_mean))
    data.append("{:.2f}".format(sent_mean))
    data.append(len(pos_noun))
    data.append(len(pos_verb))
    data.append(len(pos_adv))
    data.append(len(pos_prep))
    data.append(len(pos_pron))
    data.append(len(pos_proper))
    data.append(len(pos_det))
    data.append(len(pos_adj))
    
    return data

# function call
zh_content_data = get_content_statistics(zh_news_content, zh_nlp)
en_content_data = get_content_statistics(en_news_content, en_nlp)


def create_content_table(en_content_data, zh_content_data):
    """
    :param en_content_data: list of content statistic values taken from 100 English articles
    :zh_content_data: list of content statistic values taken from 100 Chinese articles
    :returns: list of data for converting into dataframe
    
    This function converts content statistics into pandas df and saves to png.
    
    """    
    # create dataframe
    index_list = ['Total nr. of tokens', 'Total nr. of unique lemmas', 'Total nr. of sentences', 'Average token lengths', 'Average nr. of tokens p/sent', 'Total nr. of nouns', 'Total nr. of verbs', 'Total nr. of adjectives', 'Total nr. of adverbs', 'Total nr. of pronouns', 'Total nr. of proper nouns', 'Total nr. of prepositions', 'Total nr. of determiners']                                 
    d = {'zh': zh_content_data, 'en': en_content_data}
    df = pd.DataFrame(d, index=index_list)
    
    # output to screen
    print()
    print("Content statistics:")
    print(df)
    print()
    
create_content_table(en_content_data, zh_content_data)


def create_author_table(news_content, outputfile): 
    """
    :param news_content: overview of 100 articles converted into pandas df
    :outputfile: filepath to output csv
    
    This function gets the author names and frequencies from all articles and saves the results to csv.
    
    """    
    authors = news_content["Author"]

    # Count how often each author occurs
    frequencies = Counter(authors)

    # loop over frequency dict and get author names
    author_keys = frequencies.keys()
    authors = []
    for item in author_keys:
        authors.append(item)

    # loop over frequency dict and get frequency values
    frequency_values = frequencies.values()
    values = []
    for item in frequency_values:
        values.append(item)

    # add data to pandas df
    d = {'Authors': authors, 'Frequency': values}
    df = pd.DataFrame(d)
    
    df.to_csv(outputfile, index=False)  
      
create_author_table(zh_news_content, "outputs/zh_authors.csv")
create_author_table(en_news_content, "outputs/en_authors.csv")



def preprocess_tokens(language='en'):
    
    """
    : para language: your target language
    : para number: most common words
    
    This function preprocesses the tokens across all documents. We lowercase the tokens and 
    remove all the digits, stopwords, and punctuation. 
    
    """

    article_file =  "data/ai_"+language+"_overview.tsv"
    content = pd.read_csv(article_file, sep="\t", header = 0, keep_default_na=False)

    # Prepare the nlp pipeline
    
    stanza.download(language)
    nlp = stanza.Pipeline(language, processors='tokenize')
    
    current_article = content['Text']
    tokenized = []
    token_frequencies = Counter()
        
    for text in current_article:
        
        processed_article = nlp(text)
        sentences = processed_article.sentences 
        
        
        if language == 'en':
            
            for sentence in sentences:
                mystopwords = stopwords.words('english')
                all_tokens =[token.text.lower() for token in sentence.tokens ]
                outs = [re.sub(r'[^\w\s]','',all_token) for all_token in all_tokens 
                       if not all_token in mystopwords and not all_token.isdigit()]
                out = [out for out in outs if not out == '' and not out =='s']
                tokenized.append(out)    
                token_frequencies.update(out)  
        else:

            for sentence in sentences:
                stopword = []
                stopwords_path = 'chinese_stopword.txt'
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(line)>0:
                            stopword.append(line.strip())
                all_tokens =[token.text for token in sentence.tokens]
                outs = [re.sub(r'[^\w\s]','',all_token) for all_token in all_tokens 
                        if not all_token in stopword and not all_token.isdigit()]
                out = [out for out in outs if not out == '']
                tokenized.append(out)
                token_frequencies.update(out)
    
        
    return tokenized, token_frequencies

language = 'en'
processed_tokens_en, counter_en = preprocess_tokens(language)
language = 'zh'
processed_tokens_zh, counter_zh = preprocess_tokens(language)



def calculate_frequent_tokens(counter_en, counter_zh, ntoken):
    """
    :param counter_en: English data counter dict that maps tokens to frequency
    :param counter_zh: Chinese data counter dict that maps tokens to frequency
    
    This function helps you calculate n number of most frequent words across all documents. 
    
    """
    # print 20 most common tokens
    print()
    print(f"20 most common English tokens:\n {counter_en.most_common(ntoken)}")
    print()
    print(f"20 most common Chinese tokens:\n {counter_zh.most_common(ntoken)}")
    print()
    
# function call
calculate_frequent_tokens(counter_en, counter_zh, 20)


def word_relations(processed_tokens_en, processed_tokens_zh):
    """
    :param processed_tokens_en: English data counter dict that maps tokens to frequency
    :param processed_tokens_zh: Chinese data counter dict that maps tokens to frequency
    
    This function trains a Word2Vec model on our data and performs some simple word similarity operations.
    
    """    
    # train models on English and Chinese data
    mymodel_en = Word2Vec(processed_tokens_en, min_count=3, window = 8)
    mymodel_zh = Word2Vec(processed_tokens_zh, min_count=2, window = 8)
  
    # calculate top 10 most similar words to our key terms Chinese
    data_similar_zh = mymodel_zh.wv.most_similar('數據')
    privacy_similar_zh = mymodel_zh.wv.most_similar('隱私')
    security_similar_zh = mymodel_zh.wv.most_similar('資安')

    # calculate top 10 most similar words to our key terms English
    data_similar_en = mymodel_en.wv.most_similar('data')
    privacy_similar_en = mymodel_en.wv.most_similar('privacy')
    security_similar_en = mymodel_en.wv.most_similar('security')

    # calculate similarity scores between our key terms Chinese
    print()
    print("Similarity scores Chinese:")
    print(f"Data - Privacy: {mymodel_zh.wv.similarity('數據','隱私')}")
    print(f"Security - Privacy: {mymodel_zh.wv.similarity('資安','隱私')}")
    print(f"Data - Security: {mymodel_zh.wv.similarity('數據','資安')}")
    
    print()
    
    # calculate similarity scores between our key terms English
    print("Similarity scores English:")
    print(f"Data - Privacy: {mymodel_en.wv.similarity('data','privacy')}")
    print(f"Security - Privacy: {mymodel_en.wv.similarity('security','privacy')}")
    print(f"Data - Security: {mymodel_en.wv.similarity('data','security')}")
    print()
    
    return data_similar_zh, privacy_similar_zh, security_similar_zh, data_similar_en, privacy_similar_en, security_similar_en

# function call
data_similar_zh, privacy_similar_zh, security_similar_zh, data_similar_en, privacy_similar_en, security_similar_en = word_relations(processed_tokens_en, processed_tokens_zh)


def create_word_relation_table(data_similar_zh, privacy_similar_zh, security_similar_zh, data_similar_en, privacy_similar_en, security_similar_en):
    """
    :params: top 10 most similar words per key term per language
    
    This creates a pandas df showing word similarity relations.
    
    """    
    # get only the words from the tuple with list comprehension
    data_en = [tup[0] for tup in data_similar_en]
    privacy_en = [tup[0] for tup in privacy_similar_en]
    security_en = [tup[0] for tup in security_similar_en]
    data_zh = [tup[0] for tup in data_similar_zh]
    privacy_zh = [tup[0] for tup in privacy_similar_zh]
    security_zh = [tup[0] for tup in security_similar_zh]

    # create pandas df
    df = pd.DataFrame()
    df['Data en'] = data_en
    df['Privacy en'] = privacy_en
    df['Security en'] = security_en
    df['Data zh'] = data_zh
    df['Privacy zh'] = privacy_zh
    df['Security zh'] = security_zh
    
    # print results
    print()
    print("Top 10 most similar words to key terms: Data, Privacy and Security")
    print()
    display(df)
    print()

# function call
similar_df = create_word_relation_table(data_similar_zh, privacy_similar_zh, security_similar_zh, data_similar_en, privacy_similar_en, security_similar_en)


def tt_ratio(token_frequencies, word):
    """
    : para language: your target language
    
    This function helps you calculate the word frequency 
    across the whole document. We lowercase the token and 
    remove all the digits, stopwords, and punctuations. 
    
    """
    freqs = token_frequencies.items()
    num_tokens = sum(token_frequencies.values())
    
    for key, val in freqs:
        if key in word:
            print(key,val,'num of tokens in total:',num_tokens)
            tt_ratio = val/num_tokens 
            print(key, "%.4f" % tt_ratio)
            
    
# Print the type token ratio with 4 decimals
print()
word = ['security', 'privacy', 'data']
tt_ratio(counter_en, word)
word = ['資安', '隱私', '數據']
tt_ratio(counter_zh, word)
print()


def get_en_sentiments(en_news_content):
    """
    :param en_news_content: overview of 100 English articles converted into pandas df
    :returns: list of sentences and sentiments to convert into pandas df
    
    This function analyzes the sentiment of the english articles.
    
    """
    nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

    sentence = []
    sentiment = []

    for text in en_news_content["Text"]:
        doc = nlp(text)
        for sent in doc.sentences:
            sentence.append(sent.text)
            if sent.sentiment == 0:
                sentiment.append('negative')
            elif sent.sentiment == 1:
                sentiment.append('neutral')
            elif sent.sentiment == 2:
                sentiment.append('positive')
                
    return sentence, sentiment
            

def get_zh_sentiments(zh_news_content):
    """
    :param zh_news_content: overview of 100 Chinese articles converted into pandas df
    :returns: list of sentences and sentiments to convert into pandas df
    
    This function analyzes the sentiment of the english articles.
    
    """    
    nlp = stanza.Pipeline(lang='zh', processors='tokenize,sentiment')

    sentence = []
    sentiment = []

    for text in zh_news_content["Text"]:
        doc = nlp(text)
        for sent in doc.sentences:
            sentence.append(sent.text)
            if sent.sentiment == 0:
                sentiment.append('negative')
            elif sent.sentiment == 1:
                sentiment.append('neutral')
            elif sent.sentiment == 2:
                sentiment.append('positive')
                
    return sentence, sentiment

en_sentence, en_sentiment = get_en_sentiments(en_news_content)
zh_sentence, zh_sentiment = get_zh_sentiments(zh_news_content)


def create_sentiment_df(en_sentence, en_sentiment, zh_sentence, zh_sentiment):
    """
    :param sentence:  list of sentences in all document
    :param sentiments: list of corresponding sentiments
    
    This function helps you calculate the word frequency 
    across the whole document. We lowercase the token and 
    remove all the digits, stopwords, and punctuations. 
    
    """
    d = {'Sentence': en_sentence, 'Sentiment': en_sentiment}
    eng_df = pd.DataFrame(d)

    d = {'Sentence': zh_sentence, 'Sentiment': zh_sentiment}
    zh_df = pd.DataFrame(d)
    
    return eng_df, zh_df

# function call
eng_df, zh_df = create_sentiment_df(en_sentence, en_sentiment, zh_sentence, zh_sentiment)

# print plots
print()
print("plots")
print(eng_df.Sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors = ['tab:green', 'tab:orange', 'tab:blue'], legend=True))
print(zh_df.Sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors = ['tab:blue', 'tab:orange', 'tab:green'], legend=True))


def preprocess(article):
    processed_article = nlp.process(article)
    all_lemmas = []
    

    for s in processed_article.sentences: 
        mystopwords = set(stopwords.words('english'))
        lemmas = [word.lemma.lower() for word in s.words]
        clean_lemmas = [re.sub(r'[^\w\s]','',lemma) for lemma in lemmas 
                        if not lemma in  mystopwords]
        clean_lemma = [lemma for lemma in clean_lemmas if not lemma =='' ]
        all_lemmas.extend(clean_lemma)
    return all_lemmas
        
def preprocess_zh(article):
    processed_article = nlp.process(article)
    all_lemmas = []
    
    for s in processed_article.sentences: 
        stopword = []
        stopwords_path = 'chinese_stopword.txt'
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line)>0:
                    stopword.append(line.strip())
        lemmas = [word.lemma.lower() for word in s.words]
        clean_lemmas = [re.sub(r'[^\w\s]','',lemma) for lemma in lemmas 
                        if not lemma in stopword]
        clean_lemma = [lemma for lemma in clean_lemmas if not lemma =='']
        all_lemmas.extend(clean_lemma)

            
    return all_lemmas

def create_article(language):
    tsv_file = "data/ai_"+language+"_overview.tsv"
    news_content = pd.read_csv(tsv_file, sep="\t", keep_default_na=False, header=0)
    

    # We filter out empty articles
    news_content = news_content[news_content["Text"].str.len() >0 ]
    articles = news_content["Text"]
    return articles

language1 = 'en'
language2 = 'zh'
nlp = stanza.Pipeline(language1, processors='tokenize,pos,lemma')
articles_en = create_article(language1)
nlp = stanza.Pipeline(language2, processors='tokenize,pos,lemma')
articles_zh = create_article(language2)




def tf_idf_cluster(preprocess, articles):
    vectorizer = TfidfVectorizer(use_idf=True, tokenizer=preprocess)
    tf_idf = vectorizer.fit_transform(articles)
    all_terms = vectorizer.get_feature_names()
    num_clusters = 3
    km = KMeans(n_clusters=num_clusters)
    km.fit(tf_idf)
    clusters = km.labels_.tolist()
    return tf_idf, all_terms, clusters

tf_idf, all_terms, clusters_en = tf_idf_cluster(preprocess, articles_en)
tf_idf_zh, all_term_zh, clusters_zh = tf_idf_cluster(preprocess_zh, articles_zh)


num_keywords = 25

def get_top_tfidf_features(row, terms, top_n=25):
    top_ids = np.argsort(row)[::-1][:top_n]
    top_features = [terms[i] for i in top_ids]
    return top_features, top_ids

#for english 
keywords = []
keyword_ids = []
for i in range(0, tf_idf.shape[0]):
    row = np.squeeze(tf_idf[i].toarray())
    top_terms, top_ids= get_top_tfidf_features(row, all_terms, top_n=num_keywords)
    keywords.append(top_terms)
    keyword_ids.append(top_ids)
    
# for chinese
keywords_zh = []
keyword_ids_zh = []
for i in range(0, tf_idf_zh.shape[0]):
    row = np.squeeze(tf_idf_zh[i].toarray())
    top_terms, top_ids= get_top_tfidf_features(row, all_term_zh, top_n=num_keywords)
    keywords_zh.append(top_terms)
    keyword_ids_zh.append(top_ids)    

font = 'data/SourceHanSansTW-Regular.otf'
def wordcloud_cluster_byIds(clusterId, clusters, keywords, version='en'):
    words = []
    stopword = ['的','','們',"在","一","人"]
    for i in range(0, len(clusters)):
        if clusters[i] == clusterId:
            for word in keywords[i]:
                if word not in stopword:
                    words.append(word)
#     print(words)
    # Generate a word cloud based on the frequency of the terms in the cluster
    if version == 'en':
        wordcloud = WordCloud(max_font_size=60, relative_scaling=.8).generate(' '.join(words))
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig('en'+str(clusterId)+".png")
    else:
        wordcloud = WordCloud(max_font_size=60, relative_scaling=.8, font_path = font).generate(' '.join(words))
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig('zh'+str(clusterId)+".png")
    
for i in range(3): 
    wordcloud_cluster_byIds(i, clusters_en, keywords)
    wordcloud_cluster_byIds(i, clusters_zh, keywords_zh, version='zh')

print("loading for wiki2vec - chinese")
word_vectors = KeyedVectors.load_word2vec_format("data/wiki.zh.vec")
print("Similarity scores Chinese - wiki2vec:")
print(f"Data - Privacy: {word_vectors.similarity('數據','隱私')}")
print(f"Security - Privacy: {word_vectors.similarity('安全','隱私')}")
print(f"Data - Security: {word_vectors.similarity('數據','安全')}")
print()

word_vectors = KeyedVectors.load_word2vec_format("data/wiki-news-300d-1M.vec")
print("Similarity scores English - wiki2vec:")
print(f"Data - Privacy: {word_vectors.similarity('data','privacy')}")
print(f"Security - Privacy: {word_vectors.similarity('security','privacy')}")
print(f"Data - Security: {word_vectors.similarity('data','security')}")
print()