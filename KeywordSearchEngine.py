import numpy as np
import pandas as pd
import os
import string
import nltk
import math
from textblob import TextBlob as tb
from collections import Counter

from nltk.corpus import brown
from nltk.corpus import reuters

from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer


def prepare_dataset(filename, flag_load_data_pickle = False, ...
                                                        flag_save_data = False):
    if flag_load_data_pickle:
        data = pd.read_pickle(filename)
    else:
        data = pd.read_csv(filename)

        exclude = string.punctuation + '—'

        stop_words = stopwords.words('english')

        snowball_stemmer = SnowballStemmer("english")
        temp = []
        for i,s in enumerate(data['content']):
            s = s.translate(str.maketrans('', '', exclude))
            s = word_tokenize(s)
            s = [x.lower() for x in s]
            s = [w for w in s if not w in stop_words]
            s = [snowball_stemmer.stem(w) for w in s]
            print(i)
            #print(str(s) + '\n~\n')
            temp.append(s)
        data['words'] = temp

        if flag_save_data:
            data.to_pickle("./data/data_words1.pkl")

    return data



def tf(word, doc):
    return doc.count(word) / len(doc)

def n_containing(word, doclist):
    return sum(1 for doc in doclist if word in doc)

def idf(word, doclist):
    return math.log(len(doclist) / (0.01 + n_containing(word, doclist)))

def tfidf(word, doc, doclist):
    return (tf(word, doc) * idf(word, doclist))



def generate_reverse_index(data, flag_load_reverse_index = False, ...
                                        flag_save_reverse_index = False ):
    filename = 'data/worddic.npy'

    if flag_load_reverse_index:
        worddic = np.load(filename)
    else:
        unique_words = list(set([item for sublist in ...
                        data['words'].values.flatten() for item in sublist]))

        worddic = {}

        d = data['words']
        for i,doc_words in enumerate(d):
            #print(i) uncomment for progress checking
            for word in unique_words:
                if word in doc_words:
                    positions = list(np.where(np.array(doc_words) == word)[0])
                    idfs = tfidf(word,doc_words,d)
                    try:
                        worddic[word].append([i,positions,idfs])
                    except:
                        worddic[word] = []
                        worddic[word].append([i,positions,idfs])

        if flag_save_reverse_index:
            np.save(filename, worddic)

    return worddic


def search(searchsentence):
    try:
            # split sentence into individual words
        searchsentence = searchsentence.lower()
        try:
            words = searchsentence.split(' ')
        except:
            words = list(words) #why???
        enddic = {}
        idfdic = {}
        closedic = {}

        # remove words if not in worddic
        words = [x for x in words if x in list(worddic.keys())]
        numwords = len(words)

        # make metric of number of occurances of all words in each doc & largest total IDF
        for word in words:
            for indpos in worddic[word]:
                index = indpos[0]
                amount = len(indpos[1])
                idfscore = indpos[2]
                enddic[index] = amount
                idfdic[index] = idfscore
                fullcount_order = sorted(enddic.items(), key=lambda x:x[1], ...
                                                                reverse=True)
                fullidf_order = sorted(idfdic.items(), key=lambda x:x[1], ...
                                                                reverse=True)


        # make metric of what percentage of words appear in each doc
        combo = []
        alloptions = {k: worddic.get(k, None) for k in (words)}
        for worddex in list(alloptions.values()):
            for indexpos in worddex:
                for indexz in indexpos:
                    combo.append(indexz)
        comboindex = combo[::3]
        combocount = Counter(comboindex)
        for key in combocount:
            combocount[key] = combocount[key] / numwords
        combocount_order = sorted(combocount.items(), key=lambda x:x[1], ...
                                                                reverse=True)

        # make metric for if words appear in same order as in search
        if len(words) > 1:
            x = []
            y = []
            for record in [worddic[z] for z in words]:
                for index in record:
                     x.append(index[0])
            for i in x:
                if x.count(i) > 1:
                    y.append(i)
            y = list(set(y)) #articles where > 2 search words appear

            closedic = {}
            for wordbig in [worddic[x] for x in words]:
                for record in wordbig:
                    if record[0] in y:
                        index = record[0]
                        positions = record[1]
                        try:
                            closedic[index].append(positions)
                        except:
                            closedic[index] = []
                            closedic[index].append(positions) #dict with articles where >2 search words appear, doc: [word positions]

            x = 0
            fdic = {}
            for index in y:
                csum = []
                for seqlist in closedic[index]:
                    while x > 0:
                        secondlist = seqlist
                        x = 0
                        sol=[]
                        for i in firstlist:
                            for aux in range(1,5):
                                if i + aux in secondlist:
                                    sol.append(1/aux)

                        csum.append(sol)
                        fsum = [item for sublist in csum for item in sublist]
                        fsum = sum(fsum)
                        fdic[index] = fsum
                        fdic_order = sorted(fdic.items(), key=lambda x:x[1], ...
                            reverse=True) #higher score for consecutive words
                    while x == 0:
                        firstlist = seqlist
                        x = x + 1
        else:
            fdic_order = [(-1,-1)]


        return(searchsentence,words,fullcount_order,combocount_order,fullidf_order,fdic_order)

    except:
        return(None)

def rank(term):
    results = search(term)

    if results:
        # get metrics
        num_score = results[2]
        per_score = results[3]
        tfscore = results[4]
        order_score = results[5]

        final_candidates = []

        # rule1: if high word order score & 100% percentage terms then put at top position
        try:
            first_candidates = []

            for candidates in order_score:
                if candidates[1] > 1:
                    first_candidates.append(candidates[0])

            second_candidates = []

            for match_candidates in per_score:
                if match_candidates[1] == 1:
                    second_candidates.append(match_candidates[0])
                if match_candidates[1] == 1 and match_candidates[0] in first_candidates:
                    final_candidates.append(match_candidates[0])

        # rule2: next add other word order score which are greater than 1

            t3_order = first_candidates[0:3]
            for each in t3_order:
                if each not in final_candidates:
                    final_candidates.insert(len(final_candidates),each)

        # rule3: next add top td-idf results
            final_candidates.insert(len(final_candidates),tfscore[0][0])
            final_candidates.insert(len(final_candidates),tfscore[1][0])

        # rule4: next add other high percentage score
            t3_per = second_candidates[0:3]
            for each in t3_per:
                if each not in final_candidates:
                    final_candidates.insert(len(final_candidates),each)

        #rule5: next add any other top results for metrics
            othertops = [num_score[0][0],per_score[0][0],tfscore[0][0],order_score[0][0]]
            for top in othertops:
                if top not in final_candidates:
                    if top > -1:
                        final_candidates.insert(len(final_candidates),top)

        # unless single term searched, in which case just return
        except:
            othertops = [num_score[0][0],num_score[1][0],num_score[2][0],per_score[0][0],tfscore[0][0]]
            for top in othertops:
                if top not in final_candidates:
                    final_candidates.insert(len(final_candidates),top)

        final_candidates_score = []
        for temp in final_candidates[0:20]:
            a = sum([item[1] for item in num_score if item[0] == temp]+
                    [item[1] for item in per_score if item[0] == temp]+
                    [item[1] for item in tfscore if item[0] == temp]+
                    [item[1] for item in order_score if item[0] == temp])
            final_candidates_score.append(a)

        return(len(final_candidates),final_candidates[0:20], final_candidates_score)
    else:
        return(0,[],[])



def keyword_search_engine(query, flag_load_data_pickle = False, ...
                    flag_save_data = False , flag_load_reverse_index = False,...
                    flag_save_reverse_index = False):

    data_file = "data/articles1.csv"

    data = prepare_dataset( data_file, flag_load_data_pickle, flag_save_data)

    worddic = generate_reverse_index(data, flag_load_reverse_index, ...
                                                    flag_save_reverse_index):

    output = rank(query)

    return output







if __name__ == "__main__":

    query = ["Trump", "turtle fossil", "United States of America"]

    for q in query:
        o = keyword_search_engine(q, flag_load_data_pickle = True, ...
                    flag_save_data = False , flag_load_reverse_index = True,...
                    flag_save_reverse_index = False)
        print(str(q) + ": " + str(o))
