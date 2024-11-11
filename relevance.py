import pandas as pd
import numpy as np
import os 
# from indexing import  BasicInvertedIndex
# from document_preprocessor import RegexTokenizer
# from ranker import Ranker, WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF
import json 
from tqdm import tqdm
# from collections import defaultdict
import time 

"""
Treat search results from the ranking function that are not listed in the file as non-relevant. Thus,
three relevance levels: 0 (non-relevant), 1 (marginally relevant), and 2 (very relevant). 
"""

def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    # Implement MAP
    
    nominator = 0 
    count = 0
    for idx,rel in enumerate(search_result_relevances):
        if idx<=cut_off-1:
            if rel ==1:
                nominator += float(np.sum(search_result_relevances[0:idx+1])/(idx+1))
                count +=1
        else:
            break
    
    if count ==0:
        score = 0
    else:
        score = nominator/np.sum(search_result_relevances)

    return score


def ndcg_score(search_result_relevances: list[float], 
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    # Implement NDCG
    svalues = 0
    ivalues = 0
    n = len(search_result_relevances)
    
    for idx in range(n):
        if idx<=cut_off-1:
            if idx ==0:
                lnv = 1
            else: lnv = np.log2(idx+1)
            
            svalues += search_result_relevances[idx]/lnv
            ivalues += ideal_relevance_score_ordering[idx]/lnv
        else:break
        
    if ivalues == 0:
        return 0
    else:
        return svalues/ivalues
    

# run relevance test by query
def run_relevance_tests(relevance_data_filename,ranker) -> dict:
    # Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.
    
    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    #Load the relevance dataset
    data = read_relevance_data(relevance_data_filename)
    querys = list(set(data['query'].values))
    isrel = dict(data[data['rel'].isin([4,5])].groupby(['query'])['query'].count())
    
    # Run each of the dataset's queries through yxour ranking function
    # For each query's result, calculate the MAP and NDCG for every single query and average them out
    mscores = []
    nscores = []
    map_list = dict()
    ndcg_list = dict()
    
    for query in tqdm(querys,total = len(querys)):
        search_result = ranker.query(query)[:1000]
        docs = [doc[0] for doc in search_result]
        
        rels =[] #ranker score
        map_rels = [] #mapping value from test.csv

        for museum_id in docs:
            rel = data[(data['museum_id']==museum_id) & (data['query']==query)]['rel'].values
            if len(rel)>0:
                rels.append(rel[0])
                if (rel[0] in [1,2,3]):
                    map_rels.append(0)
                else:
                    map_rels.append(1)
            elif len(rel)==0:
                map_rels.append(0)
                rels.append(1)
        
        if query in isrel: ttrels = isrel[query] 
        else: ttrels=0
        
        map_rels.append(ttrels-np.sum(map_rels))
        mscore = map_score(map_rels)
        mscores.append(mscore)
        
        irels = sorted(data[data['query']==query]['rel'].values,reverse=True)[:len(rels)]
        nscore = ndcg_score(rels,irels)
        nscores.append(nscore)

        map_list[query] = mscore
        ndcg_list[query] = nscore
        
        
    return {'map': np.mean(mscores), 'ndcg': np.mean(nscores), 'map_list': map_list, 'ndcg_list': ndcg_list}


def read_stopwords(file_name)->set:
    with open(file_name) as f:
        
        lines = set([line.rstrip() for line in f])
    
    return lines

def read_relevance_data(relevance_data_filename):
    data = pd.read_csv(relevance_data_filename,encoding='unicode_escape')
    
    return data


def read_stopwords(file_name):
    with open(file_name) as f:
        return set([line.rstrip() for line in f])
    
def save_file(file_name,data):
    with open(file_name,'w') as f:
        json.dump(data,f)
    
    f.close()
