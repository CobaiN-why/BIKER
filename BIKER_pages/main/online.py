import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.dirname(current_directory) + os.path.sep + ".")
sys.path.append(root_path)

from algorithm import recommendation
from preprocess import read_data
from lxml import etree
from nltk.stem import SnowballStemmer
from algorithm import similarity
from nltk.tokenize import WordPunctTokenizer
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import gensim
import cPickle as pickle
from bs4 import BeautifulSoup
import util
import time


w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed') # pre-trained word embedding

with open('../data/idf','rb') as input_file:
    try:
        idf=pickle.load(input_file)
    except EOFError:
        idf=None

# idf = pickle.load(open('../data/idf')) # pre-trained idf value of all words in the w2v dictionary
questions = pickle.load(open('../data/api_questions_pickle_new', 'rb')) # the pre-trained knowledge base of api-related questions (about 120K questions)
questions = recommendation.preprocess_all_questions(questions, idf, w2v) # matrix transformation
#print len(questions)
javadoc = pickle.load(open('../data/javadoc_pickle_wordsegmented','rb')) # the pre-trained knowledge base of javadoc
javadoc_dict_classes = dict()
javadoc_dict_methods = dict()
recommendation.preprocess_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,idf,w2v) # matrix transformation
parent = dict() # In online mode, there is no need to remove duplicate question of the query


print 'loading data finished'

mrr = 0.0
tot = 0.0

while True:

    query = raw_input()

    query_words = WordPunctTokenizer().tokenize(query.lower())
    if query_words[-1] == '?':
        query_words = query_words[:-1]
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]
    #print query_words
    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)
    print len(questions)
    top_questions = recommendation.get_topk_questions(query, query_matrix, query_idf_vector, questions, 50, parent)
    recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_methods,-1)


    pos = -1
    for i,api in enumerate(recommended_api):
        print 'Rank',i+1,':',api
        recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)
        if i==4:
            break


    # query = 'Java Fastest way to read through text file with 2 million lines?'
    # query = 'How to round a number to n decimal places in Java'
    # query = 'run linux commands in java code'
    # query = 'How to remove single character from a String'
    # query = 'How to initialise an array in Java with a constant value efficiently'
    # query = 'How to generate a random permutation in Java?'
