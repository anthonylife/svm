#!/usr/bin/env python
#encoding=utf8

# Function: Perform all the procudures in order.
# Procedure: 1.Extract main text from mails and create word tokens;
#            2.Get the stem of words and remove stop words;
#            3.Word Feature Selection by gini index and frequency
#            4.Compute TF-IDF and output feature file
#            5.Tag extraction and convert features to sparse
#               representaion.
#
# @Date: 10/26/2012

import os
from python.extractText import extractTokens
from python.wordProcess import loadStopwords, filterWords
from python.featureSelection import mkTable, wordSelection
from python.computeFeature import computeTfidf
from python.partition import partitionData
from python.trainTest import train, test, sparseTransform
from python.sparseRep import convertSparse

## Path Setting
# File path setting
Stopword_file = './dictionary/stopwords'
final_feature_file = './features/feature.full'
word_map_file = './features/word.map'
doc_map_file = './features/doc.map'
instance_group_file = './features/feature'
final_cle_feature_file = './features/feature.full.cle'
cate_tag_file = './features/ins_category_tag.txt'
sparse_fea_file = './features/feature.full.sparse.txt'
# Source file root path
Baseball_dir = "./baseball/"
Hockey_dir = "./hockey/"
# Feature file root path
Feature_dir = "./features/"

## Value Setting
# Word Lest frequency setting
word_freq = 5
# Least gini index setting
gini_index = 0.7

#----Start Running-----
# Get source files' lists
baseball_files = map(lambda x: Baseball_dir + x, os.listdir(Baseball_dir))
hockey_files = map(lambda x: Hockey_dir + x, os.listdir(Hockey_dir))

# Procedure 1 and 2
# =================
print 'Start procedure 1 and 2: Extract Main text and pre-processing text data ...'
all_tokens = []
#lm = WordNetLemmatizer()
id_sep = len(os.listdir(Baseball_dir))

stopwordDic = loadStopwords(Stopword_file)
for i, baseball_file in enumerate(baseball_files):
    tokens = extractTokens(baseball_file)
    tokens = filterWords(stopwordDic, tokens)
    all_tokens.append(['-1'] + tokens)

for i, hockey_file in enumerate(hockey_files):
    tokens = extractTokens(hockey_file)
    tokens = filterWords(stopwordDic, tokens)
    all_tokens.append(['1'] + tokens)
print 'Finish procedure 1 and 2.'

# Procedure 3
# ===========
print 'Start procedure 3: Word feature selection ...'
token_table = mkTable(all_tokens)
token_table = wordSelection(token_table, id_sep, word_freq, gini_index)
print 'Finish procedure 3.'

# Procedure 4
# ===========
print 'Start procedure 4: Compute tf-idf value ...'
word_map = {}   # word id mapping
wfd = open(word_map_file, 'w')
for i, word in enumerate(token_table.keys()):
    word_map[word] = str(i+1)
    wfd.write(word + ' ' + str(i+1) + '\n')
wfd.close()

dimensions = len(word_map)  # dimensions of features

final_feature_table = computeTfidf(token_table, all_tokens, word_map)
wfd = open(final_feature_file, 'w')
for final_feature in final_feature_table:
    wfd.write(' '.join(final_feature) + '\n')
wfd.close()
print 'Finish procedure 4.'

# Procedure 5
# ===========
print 'Start procedure 5: Feature clean ...'
os.system('python python/featureClean.py %s %s' % (final_feature_file, final_cle_feature_file))
print 'Tag extraction ...'
os.system('awk \'{print $1}\' %s > %s' % (final_cle_feature_file, cate_tag_file))
print 'Convert features to sparse representation'
convertSparse(final_cle_feature_file, sparse_fea_file)
