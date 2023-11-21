import sys
import os
import argparse
import math
from loader import load_dir
import string
from collections import Counter
from nltk.util import ngrams, bigrams
from nltk import ConditionalFreqDist
from nltk.probability import FreqDist

def load_data(rootdir):
    '''
    Loads all the data by searching the given directory for
    subdirectories, each of which represent a class.  Then returns a
    dictionary of class vs. text document instances.
    '''

    classdict = {}
    for classdir in os.listdir(rootdir):
        fullpath = os.path.join(rootdir, classdir)
        print("Loading from {}.".format(fullpath))
        if os.path.isdir(fullpath):
            classdict[classdir] = load_dir(fullpath)
    return classdict


def count_of_words(corpus):
    '''
    Counts the total number of words in the corpus.
    '''

    all_words_counter=0
    unique_words=[]
    unique_words_n_counts={}
    for all_lists in corpus.values(): #since we are dealing with a dictionnary?
        for each_lyric_list in all_lists:
            for word in each_lyric_list:
                if word not in unique_words:
                    unique_words.append(word)
                all_words_counter+=1
    return ('Total of words:', all_words_counter, 'Total unique words:', len(unique_words))

def bigram_analysis(corpus):
    '''
    Calculates the frequency of 2-grams in the corpus.
    '''
    list_of_all_bigrams=[]

    for all_lists in corpus.values():
        for each_lyric_list in all_lists:
            bi_grams=list(bigrams(each_lyric_list))
            for bigram in bi_grams:
                list_of_all_bigrams.append(bigram)
    bigram_count=dict(FreqDist(list_of_all_bigrams))
    most_common=Counter(list_of_all_bigrams).most_common(20)

    return most_common


def ngram_analysis(sequence,n):

    '''
    Calculates the frequency of n-grams for any given value of n. 
    '''
    
    list_of_all_ngrams=[]

    for all_lists in corpus.values():
        for each_lyric_list in all_lists:
            N_grams=list(ngrams(each_lyric_list,n))
            for n_gram in N_grams:
                list_of_all_ngrams.append(n_gram)
    ngram_count=dict(FreqDist(list_of_all_ngrams))
    most_common=Counter(ngram_count).most_common(20)

    return most_common


def cond_prob(word,historyword):

    '''Estimate the conditional probability of a specific word occurring after another word.'''

    list_of_all_bigrams=[]
    
    for all_lists in corpus.values():
        for each_lyric_list in all_lists:
            bi_grams=list(bigrams(each_lyric_list))
            for bigram in bi_grams:
                list_of_all_bigrams.append(bigram)

    bigrams_cfd=dict(ConditionalFreqDist(list_of_all_bigrams))
    return bigrams_cfd[historyword].freq(word)
    
               
def calc_prob(classdict, classname, word):
    '''
    Calculates p(classname|word) given the corpus in classdict, which is a directory holding the class directories.
    '''

    #first calculating the number of times the word occurs in the entire corpus

    list_of_all_words_in_corpus=[]
    list_of_all_words_in_classname=[]

    #Starting with the whole corpus and computing the word distribution.
    #Then, finding out the number of times the feature word occurs in it.

    for all_lists in corpus.values():
        for each_lyric_list in all_lists:
            for words in each_lyric_list:
                list_of_all_words_in_corpus.append(words)
    counts_all=dict(FreqDist(list_of_all_words_in_corpus))  #getting the frequency distribution 
    word_occurances_all=counts_all[word]                    #the number of times the word occurs in the entire corpus

    #Doing the same thing but now computing the word distribution in only the class of interest.
    #Finding out how many times the feature word occurs in it. 

    for each_list in corpus[classname]:
        for w0rd in each_list:
            list_of_all_words_in_classname.append(w0rd)
    counts_in_classname=dict(FreqDist(list_of_all_words_in_classname))
    word_occurances_classname=counts_in_classname[word]

    print("this is the num of all occurances of", word, "-->", word_occurances_all) #just testing out
    print("this is the num of", classname, "occurances of", word, "-->", word_occurances_classname) #just testing out

    #returning the probability
    #number of times the feature word occurs in classname / number of times the feature word occurs in the whole corpus

    print(word_occurances_classname/word_occurances_all)
    return word_occurances_classname/word_occurances_all

if __name__ == "__main__":

    '''
    Entry point for the code. We load the command-line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("filesdir", help="The root directory containing the class directories and text files.")
    parser.add_argument("classname", help="The class of interest.")
    parser.add_argument("feature", help="The word of interest for calculating -log2 p(class|feature).")

    args = parser.parse_args()

    corpus = load_data(args.filesdir)

    print(count_of_words(corpus))

    print(bigram_analysis(corpus))

    print(ngram_analysis(corpus,4))

    print(cond_prob('love','i'))

    print("Number of classes in corpus: {}".format(len(corpus)))
    
    print("Looking up probability of class {} given word {}.".format(args.classname, args.feature))
    prob = calc_prob(corpus, args.classname, args.feature)
    if prob == 0:
        print("-log2 p({}|{}) is undefined.".format(args.classname, args.feature))
    else:
        print("-log2 p({}|{}) = {:.3f}".format(args.classname, args.feature, -math.log2(prob)))
    
