import spacy
import numpy as np
import pandas as pd
from spacy.lang.en import English
from multiprocessing import Process
import os
import argparse
import nltk

def multiprocess_function(num_process, function_ref,args):
    jobs = []

    for idx in range(num_process):

        process = Process(target=function_ref, args=(idx,) + args)
        process.daemon = True
        jobs.append(process)
        process.start()

    for i in range(num_process):
        jobs[i].join()

def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]

def evaluate_entity(idx, text_chunk, article_chunk):
    nlp = spacy.load("en_core_web_sm")
    entity_avg = []
    text_chunk_idx = text_chunk[idx]
    article_chunk_idx = article_chunk[idx]
    for i in range(len(text_chunk_idx)):
        line = text_chunk_idx[i]
        doc = nlp(line)
        entity_avg.append(len(set(i.text.lower() for i in doc.ents)))
        #print(doc.ents)
        overlap = 0
        for ent in doc.ents:
            if ent.text in article_chunk_idx[i]:
                overlap += 1 
        print(len(doc.ents), overlap)
    print(np.sum(entity_avg), "\t", len(entity_avg), np.sum(entity_avg)/len(entity_avg))

def evaluate_ngram(idx, n, text_chunk, article_chunk):
    entity_avg = []
    text_chunk_idx = text_chunk[idx]
    article_chunk_idx = article_chunk[idx]
    inter_num = 0
    union_num = 0
    fracs = []
    for i in range(len(text_chunk_idx)):
        line = text_chunk_idx[i]
        summ = nltk.word_tokenize(line)
        article = nltk.word_tokenize(article_chunk_idx[i])
        summ_gram = [" ".join([summ[j+i] for j in range(n)]) for i in range(len(summ)-n+1)]
        article_gram = [" ".join([article[j+i] for j in range(n)]) for i in range(len(article)-n+1)]
        a = set(summ_gram)
        b = set(article_gram)
        len1, len2 = len(a.intersection(b)), len(a.union(b))
        inter_num += len1
        union_num += len2
        fracs.append(len1/len2)
    print("%d-gram"%n, inter_num, union_num, np.mean(fracs))
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_path",type=str)
    parser.add_argument("--article_path",type=str)
    args = parser.parse_args()
    file_gen = open(args.gen_path,'r',encoding="utf8")
    file_article = open(args.article_path,'r',encoding="utf8")
    data = file_gen.readlines()
    adata = file_article.readlines()
    data = [" ".join(i.strip().split()) for i in data]
    adata = [" ".join(i.strip().split()) for i in adata]
    # MultiProcess
    num_process = 4#os.cpu_count() - 2
    print("cpu_num", num_process)
    chunk_data = chunkify(data, num_process)
    chunk_adata = chunkify(adata, num_process)
    multiprocess_function(num_process, evaluate_entity, (chunk_data, chunk_adata))
    #multiprocess_function(num_process, evaluate_ngram, (1, chunk_data, chunk_adata))
    #multiprocess_function(num_process, evaluate_ngram, (2, chunk_data, chunk_adata))
    #multiprocess_function(num_process, evaluate_ngram, (3, chunk_data, chunk_adata))
    #evaluate_entity(data)
