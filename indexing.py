from enum import Enum
from document_preprocessor import Tokenizer
from collections import Counter, defaultdict
import json
import numpy as np
import os
from document_preprocessor import RegexTokenizer
from tqdm import tqdm
import gzip

class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter() # token count
        self.vocabulary = set()  # the vocabulary of the collection
        self.document_metadata = {} # metadata like length, number of unique tokens of the documents

        self.index = defaultdict(list)  # the index "token":[id,frequency, [positions]]
        self.title_index = defaultdict(list)

    
    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def __iter__(self):
        yield {
            "statistics":self.statistics,
            "vocabulary":self.vocabulary,
            "document_metadata":self.document_metadata,
            "index":self.index
        }
        
    def remove_doc(self, museum_id: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            museum_id: The id of the document
        """
        raise NotImplementedError
      
            
    def add_doc(self, museum_id: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            museum_id: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError
        
        

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the museum_id and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            museum_id: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        
        raise NotImplementedError
    
    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError
        
        
class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
    

    def add_doc(self, museum_id: int, tokens: list[str]) -> None:
        tokens_count = Counter(tokens)
        
        n = len(tokens_count)
        # update vocab and freq
        for token, count in tokens_count.items():
            if token is not None:
                # {'token': count}
                self.statistics['vocab'][token] += count
                self.index[token].append((museum_id, count))
                self.vocabulary.add(token)
            else:
                n -=1
        
        # update doc unique_tokens
        self.document_metadata[museum_id] = {'length':len(tokens)}
        self.document_metadata[museum_id].update({'unique_tokens':n})
        self.document_metadata[museum_id].update({'tokens_count':tokens_count})
        # self.document_metadata[museum_id].update({'tokens':tokens})
            
    
    def remove_doc(self, museum_id: int) -> None:
        if museum_id in self.document_metadata:
            
            # delete vocab, update vocab and freq
            freq_count = self.statistics['vocab']
            
            for token,freq in self.get_doct_tokens(museum_id):
                freq_count[token] -= freq
                
                if freq_count[token] ==0:
                    del freq_count[token]
                    del self.index[token]
            
            
            self.vocabulary = set(self.statistics['vocab'].keys())
            
            # delete doc
            del self.document_metadata[museum_id]
    
    def get_doct_tokens(self,museum_id:int):
        index_copy = self.index.copy()
        for token,post in index_copy.items():
            for id,freq in post:
                if id==museum_id:
                    self.index[token].remove((id,freq))
                    yield token,freq


    def get_postings(self, term: str) -> list:
        return self.index[term]
    
    
    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        return {'length':self.document_metadata[doc_id]['length'],'unique_tokens':self.document_metadata[doc_id]['unique_tokens']}
          
    
    def get_term_metadata(self, term: str) -> dict[str, int]:
        term_metadata = dict()
        doc_frequency = len(self.index[term])
        term_count = self.statistics['vocab'][term]
        term_metadata[term] = {'term_count':term_count,'doc_frequency':doc_frequency}
        
        return term_metadata[term]
         
         
    def get_statistics(self) -> dict[str, int]:
        self.statistics['unique_token_count'] = len(self.vocabulary)
        
        total_length=0
        for key,value in self.document_metadata.items():
            total_length += value['length']
        self.statistics['total_token_count'] = total_length 
        
        self.statistics['stored_total_token_count']= int(np.sum(list(dict(self.statistics['vocab']).values())))
        self.statistics['number_of_documents'] = len(self.document_metadata.keys())
        
        if self.statistics['unique_token_count'] ==0:
            self.statistics['mean_document_length'] = 0
        else: self.statistics['mean_document_length'] = total_length/self.statistics['number_of_documents']
        
        return self.statistics
    
    def save(self,index_directory_name:str) -> None:
        dir_path  = os.path.join(os.path.dirname(__file__),index_directory_name)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = ['statistics',\
            'vocabulary',\
                'document_metadata','index']
        data_info = [self.statistics,\
                    list(self.vocabulary),\
                    self.document_metadata,self.index]
        
        for i in range(len(file_name)):
            path  = os.path.join(dir_path,file_name[i]+'.json')
            with open(path, 'w') as f:
                json.dump(data_info[i], f)
            print(data_info[i])
            
            f.close()
        
    def load(self,index_directory_name:str) -> None:
        dir_path  = os.path.join(os.path.dirname(__file__),index_directory_name)
        
        
        file_name = ['statistics',\
            'vocabulary',\
                'document_metadata','index']
        
        path = [os.path.join(dir_path,file_name[i]+'.json') for i in range(len(file_name))]

        with open(path[0], 'r') as f:
            self.statistics = json.load(f)
        
        with open(path[1], 'r') as f:
            self.vocabulary = json.load(f)
        
        with open(path[2], 'r') as f:
            data = json.load(f)
            for k,v in data.items():
                self.document_metadata[int(k)] = v
            # self.document_metadata = json.load(f)
        
        with open(path[3], 'r') as f:
            self.index= json.load(f)
        

class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="wiki_article",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index

        '''
        # If minimum word frequencies are specified, process the collection to get the
        #       word frequencies

        # Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency

        tokenizer = document_preprocessor
        dataset_path = os.path.join(os.path.dirname(__file__),dataset_path)
        ivtindex = BasicInvertedIndex()
        
        tokens_count = Counter()
        if dataset_path[-3:] == '.gz':
            with gzip.open(dataset_path,'rt') as f:
                tokens_count  = Indexer.read_queries_and_fitering(f,max_docs,doc_augment_dict,tokenizer,text_key,minimum_word_frequency)
        elif dataset_path[-6:] == '.jsonl':
            with open(dataset_path,'rt') as f:
                tokens_count = Indexer.read_queries_and_fitering(f,max_docs,doc_augment_dict,tokenizer,text_key,minimum_word_frequency)
        
        #filtering out words with low frequency and append it to stopwords set
        if minimum_word_frequency>0:
            filter_words = set([key for key,value in tokens_count.items() if value < minimum_word_frequency])
            stopwords.update(filter_words)
        
        #add doc
        if dataset_path[-3:] == '.gz':
            with gzip.open(dataset_path,'rt') as f:
                ivtindex = Indexer.insert_data(f,tokenizer,text_key,stopwords,ivtindex,doc_augment_dict)
        elif dataset_path[-6:] == '.jsonl':
            with open(dataset_path,'rt') as f:
                ivtindex = Indexer.insert_data(f,tokenizer,text_key,stopwords,ivtindex,doc_augment_dict)
                
        #caculate statistics
        ivtindex.statistics = ivtindex.get_statistics()
        #sort posting in reverse order
        for token,value in ivtindex.index.items():
            ivtindex.index[token] = sorted(value,key = lambda x:x[1],reverse=True)
            
        return ivtindex

    @staticmethod
    def read_queries_and_fitering(f,max_docs,doc_augment_dict,tokenizer,text_key,minimum_word_frequency)->dict:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            f: the file to be read.
            tokenizer: tokenizer class from document_preprocessor.py.
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            A dict of word counts after being minimum word frequency.

        '''
        
        tokens_count = Counter()
        for idx,line in tqdm(enumerate(f),total = 5716):
            if max_docs == -1 or (idx<max_docs) :
                doc = json.loads(line)
                if text_key == 'wiki_article' and doc_augment_dict != None:
                    for query in doc_augment_dict[doc['museum_id']]:
                        doc[text_key] = doc[text_key]+'\n'+ query #append queries
                
                tokens = tokenizer.tokenize(doc[text_key])
                if minimum_word_frequency>0:
                    # update counter and index
                    for t in tokens:
                        tokens_count[t] += 1
            else:
                break
            
        return tokens_count
    
    @staticmethod
    def insert_data(f,tokenizer,text_key,stopwords,ivtindex,doc_augment_dict) ->InvertedIndex:
        for i,line in tqdm(enumerate(f),total = 5716):
                doc = json.loads(line)
                if text_key == 'text' and doc_augment_dict != None:
                    for query in doc_augment_dict[doc['museum_id']]:
                        doc[text_key] = query+'\n'+doc[text_key] #append queries
                        
                tokens = tokenizer.tokenize(doc[text_key])
                tokens = [token if token not in stopwords else None for token in tokens]
                ivtindex.add_doc(doc['museum_id'], tokens)
                
        return ivtindex
        
