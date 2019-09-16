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
print 'begin'
w2v = gensim.models.Word2Vec.load('../data/w2v_model_stemmed') # pre-trained word embedding

with open('../data/idf','rb') as input_file:
    try:
        idf=pickle.load(input_file)
    except EOFError:
        idf=None

# idf = pickle.load(open('../data/idf')) # pre-trained idf value of all words in the w2v dictionary
questions = pickle.load(open('../data/api_questions_pickle_new', 'rb')) # the pre-trained knowledge base of api-related questions (about 120K questions)
questions = recommendation.preprocess_all_questions(questions, idf, w2v) # matrix transformation
javadoc = pickle.load(open('../data/javadoc_pickle_wordsegmented','rb')) # the pre-trained knowledge base of javadoc
javadoc_dict_classes = dict()
javadoc_dict_methods = dict()
recommendation.preprocess_javadoc(javadoc,javadoc_dict_classes,javadoc_dict_methods,idf,w2v) # matrix transformation
parent = dict() # In online mode, there is no need to remove duplicate question of the query

print 'loading data finished'

mrr = 0.0
tot = 0.0

from flask import Flask, redirect, url_for, request, render_template
app = Flask(__name__)

@app.route('/result/<query_x>')
def get_result(query_x):
    query_to = query_x

    query_words = WordPunctTokenizer().tokenize(query_to.lower())
    if query_words[-1] == '?':
        query_words = query_words[:-1]
    query_words = [SnowballStemmer('english').stem(word) for word in query_words]

    query_matrix = similarity.init_doc_matrix(query_words, w2v)
    query_idf_vector = similarity.init_doc_idf_vector(query_words, idf)

    top_questions = recommendation.get_topk_questions(query_to, query_matrix, query_idf_vector, questions, 50, parent)
    recommended_api = recommendation.recommend_api(query_matrix, query_idf_vector, top_questions, questions, javadoc,javadoc_dict_methods,-1)

    pos = -1
    # covert print to page
    five_apis = []
    methods_descriptions_five_texts = []
    titles_dict = {}
    code_snippets_dict = {}

    for i,api in enumerate(recommended_api):
        print 'Rank',i+1,':',api
        five_apis.append(api)
        methods_descriptions_pure_text,titles,code_snippets = recommendation.summarize_api_method(api,top_questions,questions,javadoc,javadoc_dict_methods)
        methods_descriptions_five_texts.append(methods_descriptions_pure_text)
        tot = 0
        titles_last = {}
        code_snippets_last = []
        for title in titles:
            if tot == 3:
                break
            if len(code_snippets[title[0]])>0:
                tot+=1
                title_id = -1
                for question in questions:
                    if title[0] == question.title:
                        title_id = question.id
                        break
                titles_last[title[0]] = title_id
        if tot < 3:
            for title in titles:
                if tot == 3:
                    break
                if len(code_snippets[title[0]])==0:
                    tot+=1
                    title_id = -1
                    for question in questions:
                        if title[0] == question.title:
                            title_id = question.id
                            break
                    titles_last[title[0]] = title_id
        tot = 0
        for title in titles:
            if tot == 3:
                break
            if len(code_snippets[title[0]]) > 0:
                tot += 1
                code_snippets_last.append(code_snippets[title[0]][0])
        titles_dict[i] = titles_last
        code_snippets_dict[i] = code_snippets_last

        # methods_descriptions_pure_text, titles, code_snippets to echo in the pages
        # watch recommendation.summarize_api_method
        if i==4:
            break

    # query = 'Java Fastest way to read through text file with 2 million lines?'
    # query = 'How to round a number to n decimal places in Java'
    # query = 'run linux commands in java code'
    # query = 'How to remove single character from a String'
    # query = 'How to initialise an array in Java with a constant value efficiently'
    # query = 'How to generate a random permutation in Java?'
    # print five_apis, methods_descriptions_five_texts, titles_dict, code_snippets_dict

    return render_template('result.html', uquery = query_to , result_apis = five_apis, methods_descriptions = methods_descriptions_five_texts, result_titles = titles_dict, result_code = code_snippets_dict)


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/seeresult')
def seeresult():
    return render_template('result.html')

@app.route('/index', methods = ['POST','GET'])
def query():
	if request.method == 'POST':
		qestion_query = request.form['query'].replace("/","")
		if qestion_query:
			return redirect(url_for('get_result',query_x = qestion_query))
		else:
			return redirect(url_for('index'))
	if request.method == 'GET':
		qestion_query = request.args.get['query'].replace("/","")
		if qestion_query:
			return redirect(url_for('get_result',query_x = qestion_query))
		else:
			return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='58888')
