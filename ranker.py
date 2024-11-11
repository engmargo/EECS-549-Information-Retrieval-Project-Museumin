"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from indexing import InvertedIndex
import numpy as np
from collections import Counter,defaultdict
# from indexing import Indexer, IndexType
# from tqdm import tqdm
# import time
from sentence_transformers import CrossEncoder
import random

class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                scorer: 'RelevanceScorer', raw_text_dict: dict[int,str]=None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document
        """
        self.index = index
        self.tokenizer = document_preprocessor
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query
        #qtokens is tokens list before stopwords filtering
        #qtokens_count is token counter after wtopwords filtering
        qtokens = self.tokenizer.tokenize(query)
        
        # remove stopwords and get query token count
        qtokens_count = self.get_Counter(qtokens,self.stopwords)

        # 2. Fetch a list of possible documents from the index
        doctokens = self.get_doc_tokens(list(qtokens_count.keys()))

        # 2. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)
        scorelist = []
        
        # start = time.time()
        for id in doctokens.keys():
            scorelist.append((id,self.scorer.score(id,doctokens[id],qtokens_count)))
        # end = time.time()
        # print(f'time:{end-start}')
        # 3. Return **sorted** results as format [{museum_id: 100, score:0.5}, {{museum_id: 10, score:0.2}}]

        sortedscore = sorted(scorelist,key = lambda x:x[1],reverse=True)
        
        return sortedscore
            

    def get_Counter(self,qtokens,stopwords):
        qtokens = [token if token not in stopwords else None for token in qtokens]
        qcount = Counter(qtokens)
        
        return qcount 
    
    
    def get_doc_tokens(self,unique_qtokens:list) ->dict: #unique_qtokens of query
        
        doctokens =defaultdict(lambda:defaultdict(int))
        
        for token in unique_qtokens:
            if token in self.index.vocabulary:
                for idx,item in enumerate(self.index.index[token]):
                    if idx<=1000 and item[0] not in doctokens:
                        doctokens[item[0]] = {key:value for key,value in self.index.document_metadata[item[0]]['tokens_count'].items() if (key in unique_qtokens)and(key!=None)}
                    else:
                        break

        return doctokens

class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, museum_id: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            museum_id: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError
    

class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, museum_id: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        self._name = 'WordCountCosineSimilarity'
        
    def score(self, museum_id: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query
        # generate vector 
        vdoc = [freq for term,freq in query_word_counts.items() if term in doc_word_counts]
        vquery = [doc_word_counts[term] for term,freq in query_word_counts.items() if term in doc_word_counts]

        # 2. Return the score
        score  = np.dot(vdoc,vquery)
        
        return score


class DirichletLM(RelevanceScorer): #input original query_word_counts
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters
        self._name = 'DirichletLM'

    def score(self, museum_id: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        doc_length = self.index.document_metadata[museum_id]['length']
        tokens_count = self.index.statistics['total_token_count']
        qlength = np.sum(list(query_word_counts.values()))

        # 2. Compute additional terms to use in algorithm
        score = 0
        for token,freq in query_word_counts.items():
            if token in doc_word_counts:
                ele = (freq,doc_word_counts[token],len(self.index.index[token])) 
                score += ele[0]*np.log(1+(ele[1]/self.parameters['mu']/(ele[2]/tokens_count)))
                
        score += qlength *np.log(self.parameters['mu']/(doc_length+self.parameters['mu']))
        
        return score
 

class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self._name = 'BM25'
    
    def score(self, museum_id: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        ndocs = self.index.statistics['number_of_documents']
        mean_length = self.index.statistics['mean_document_length']
        doc_length = self.index.document_metadata[museum_id]['length']
 
        # 2. Find the dot product of the word count vector of the document and the word count vector of the query
        
        score = 0
        for token,freq in query_word_counts.items():
            if token in doc_word_counts:
                ele = (freq,doc_word_counts[token],len(self.index.index[token])) 
                # for ele in interseted_tokens:
                a = ((self.k1+1)*ele[1])/\
                            (self.k1*(1-self.b+self.b*doc_length/mean_length)+ele[1])
                b = (self.k3+1)*ele[0]/(self.k3+ele[0])
                c = np.log((ndocs-ele[2]+0.5)/(ele[2]+0.5))
                score += a*b*c

        return score
        

class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']
        self._name = 'PivotedNormalization'

    def score(self, museum_id: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index
        ndocs = self.index.statistics['number_of_documents']
        mean_length = self.index.statistics['mean_document_length']
        doc_length = self.index.document_metadata[museum_id]['length']
        
        # 2. Compute additional terms to use in algorithm
        score = 0
        
        for token,freq in query_word_counts.items():
            if token in doc_word_counts:
                ele = (freq,doc_word_counts[token],len(self.index.index[token]))
                a = (1+np.log(1+np.log(ele[1])))/(1-self.b+self.b*doc_length/mean_length)
                b = np.log((ndocs+1)/ele[2])*ele[0]
                score += a*b
            
        return score


class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        self._name = 'TF_IDF'
        
    def score(self, museum_id: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index
        ndocs = self.index.statistics['number_of_documents']
        
        score = 0
        for token,freq in query_word_counts.items():
            if token in doc_word_counts:
                ele = (freq,doc_word_counts[token],len(self.index.index[token]))
                score += np.log(ele[1]+1)*(np.log(ndocs/ele[2])+1)
        
        return score


class YourRanker(RelevanceScorer):
    # postion index 
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8,'k4':0.75}) -> None:
        # Must be positional indexer
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.k4 = parameters['k4']
        self._name = 'YourRanker'
    
    def score(self, museum_id: int, doc_words: dict, query_word_counts: dict[str, int],query_pos:dict)->float:
        # 1. Get necessary information from index
        ndocs = self.index.get_statistics()['number_of_documents']
        mean_length = self.index.get_statistics()['mean_document_length']
        doc_length = self.index.get_doc_metadata(museum_id)['length']
        
        # 2. Compute additional terms to use in algorithm
        bscore = 0
        for term in doc_words:
            a = doc_words[term][0]
            b = self.index.get_term_metadata(term)['doc_frequency']
            c = query_word_counts[term]
            
            _a = ((self.k1+1)*a)/\
                (self.k1*(1-self.b+self.b*doc_length/mean_length)+a)
            _b  = np.log((ndocs-b+0.5)/(b+0.5))
            _c = (self.k3+1)*c/(self.k3+c)
            
            score = self.possim(doc_words[term][1],query_pos[term],0,len(query_pos[term]))
            score_adj = np.log(1+(self.k4+1)*score/(self.k4+score))
            
            bscore += _a*_b*_c*score_adj
        
        return bscore
        
    
    def possim(self,pos_q:list[int],pos_d:list[int],max:int,len_q0:int) -> int:
        len_q = len(pos_q)
        len_d = len(pos_d)
        
        if len_q == len_d:
            minus = list(map(lambda x:x[0]-x[1],zip(pos_q,pos_d)))
            a = sorted(Counter(minus).values(),reverse=True)[0]
            if max<a:max = a
            
            return max
            
        elif len_q >len_d:
            minus = list(map(lambda x:x[0]-x[1],zip(pos_q[0:len_d],pos_d)))
            a = sorted(Counter(minus).values(),reverse=True)[0]
            if max<a:max = a
            
            return self.possim(pos_q[1:],pos_d,max,len_q0)
            
        elif len_q < len_d:
            minus = list(map(lambda x:x[0]-x[1],zip(pos_q,pos_d[0:len_q])))
            a = sorted(Counter(minus).values(),reverse=True)[0]
            if a>max:
                max = a
            elif a == len_q0 and max ==a:
                max +=max
            
            return self.possim(pos_q,pos_d[1:],max,len_q0)
        
class RandomRanker(RelevanceScorer):
    def __init__(self):
        pass
    
    def score(self,museum_id: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int],hnum = 5)->int:
        return random.randint(1,hnum)

# The CrossEncoderScorer class uses a pre-trained cross-encoder model from the Sentence Transformers package
#             to score a given query-document pair; check README for details
#
# This is not a RelevanceScorer object because the method signature for score() does not match, but it
# has the same intent, in practice
class CrossEncoderScorer:
    '''
    A scoring object that uses cross-encoder to compute the relevance of a document for a query.
    '''
    def __init__(self, raw_text_dict: dict[int, str], 
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        # Save any new arguments that are needed as fields of this class
        self.text = raw_text_dict
        # self.model_name = cross_encoder_model_name
        self.model = CrossEncoder(cross_encoder_model_name,max_length = 500)
        
    def score(self, museum_id: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.
        
        Args:
            museum_id: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # edge case
        # (e.g., museum_id does not exist in raw_text_dict or empty query, both should lead to 0 score)
        if (query == ''):
            return 0
        elif museum_id not in self.text:
            return 0
        else:
        # unlike the other scorers like BM25, this method takes in the query string itself,
        # not the tokens!
        # Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
            
            score = self.model.predict((query,self.text[museum_id]))
            
            return score
