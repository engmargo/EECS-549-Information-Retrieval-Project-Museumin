"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques.
"""
from nltk.tokenize import RegexpTokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import spacy
import re
from spacy.attrs import ORTH

import warnings
warnings.filterwarnings("ignore")

class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        #  Save arguments that are needed as fields of this class
        self.lowercase  = lowercase
        self.multiword_expressions = multiword_expressions
        

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        #stemming words
        # stemmer = SnowballStemmer('english')
        # input_tokens = [stemmer.stem(i) for i in input_tokens]
        
        #lowercasing
        if self.lowercase:
            new_tokens,mwsl = self.lower(input_tokens,self.multiword_expressions)
        else:
            new_tokens,mwsl = input_tokens,self.multiword_expressions
            
        #multiwords: convert mws list to dict 
        if self.multiword_expressions:
            #multiwords: matching
            mwsd = self.trie(mwsl)
            new_tokens = self.multimatch(new_tokens,mwsd)
        
        return new_tokens
    
    
    def lower(self,input_tokens:list[str],multiword_expressions: list[str] = None):
        new_tokens = [i.lower() for i in input_tokens]
        
        if multiword_expressions:
            mwsl = [i.lower() for i in multiword_expressions]
        else:
            mwsl = []

        return new_tokens,mwsl
    
    
    def trie(self,multiword_expressions:list[str]=None)->dict:
        """
        a trie method for multiword_expressions
        
        Args:
            multiword_expressions: A list of strings that should be recognized as single tokens
            If set to 'None' no multi-word expression matching is performed.
            
        Returns:a dictionary of a string 
        Example:
        trie = self.trie(["Taylor Swift"])
        {"Taylor":{"Swift":{Tail:None}}}
        """
        defaultdict = dict()
        for i in multiword_expressions:
            # input = re.split(r"\s|(?=\')",i) #split the sentence by whitespace and  and keep "'s" instead of "s"
            input = i.split()
            defaultdict.update(self.insert(defaultdict,input))
            
        return defaultdict


    def insert(self,maindict:dict,input:list[str]) -> dict:
        """
        Use reinforcement to insert the string into the trie
        """
        subdict=dict()
        if not len(input):
            subdict['Tail'] = None
        elif input[0] in maindict:
            subdict[input[0]]= self.insert(maindict[input[0]],input[1:])
            subdict[input[0]].update(maindict[input[0]])
            maindict[input[0]] = subdict[input[0]]
        elif input[0] not in maindict:
            subdict[input[0]] = self.insert({},input[1:])
        return subdict
    
            
    def multimatch(self,input_tokens:list[str],mwsd:dict) ->list[str]:
        """
        match input_tokens with customized multiple-words
        
        case 1: one token list
        case 2: the consecutive tokens is a substring of a certain multiword but has no perfect match 
        case 3: the consecutive tokens is a substring of a certain multiword with a perfect match
        case 4: the consecutive tokens has no perfect match and doesn't make a part of any multiword
        """
        new_tokens = []
        end_idx = 0
        
        # case 1
        if len(input_tokens) ==1:
            new_tokens = input_tokens
        else:
            while end_idx < len(input_tokens):
                maindict = mwsd.copy()
                
                if (input_tokens[end_idx] in maindict): 
                    subtoken =[]
                    head,tail = end_idx,end_idx
                    
                    while end_idx < len(input_tokens) and input_tokens[end_idx] in maindict :
                        ptoken = input_tokens[end_idx]
                        subtoken.append(ptoken)

                        # mark the tail of a matched multiword
                        if 'Tail' in maindict[ptoken]:
                            tail = end_idx
                        
                        end_idx  +=1
                        maindict = maindict[ptoken]
                        
                    # case 2,case 3
                    # if the previous traversed tokens together as a key has a perfect matched key in dictionary, then add the perfect match
                    if len(subtoken)>0:
                        new_tokens.append(' '.join(subtoken[0:tail-head+1]))
                        
                    end_idx = tail+1
                
                # case 4     
                else:
                    new_tokens.append(input_tokens[end_idx])
                    end_idx +=1
                
                
        return new_tokens

        
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        return self.postprocess(text.split())
        
        
class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # Save a new argument that is needed as a field of this class
        self.token_regex = token_regex
        #  Initialize the NLTK's RegexpTokenizer 
        self.regextokenizer = RegexpTokenizer(self.token_regex)
        

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        return self.postprocess(self.regextokenizer.tokenize(text))
       

class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single words.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        nlp = spacy.load("en_core_web_sm")  
        # contextualSpellCheck.add_to_pipe(nlp)
        
        if self.multiword_expressions:
            if self.lowercase:
               for word in self.multiword_expressions:
                    nlp.tokenizer.add_special_case(word.lower(),[{"ORTH":word.lower()}])
            else:
                for word in self.multiword_expressions:
                    nlp.tokenizer.add_special_case(word,[{"ORTH":word}])
        else:
            if self.lowercase:
                text = text.lower()
        
        tokens = nlp(text)
        tokens_list=[token.text for token in tokens]
        
        return tokens_list


class Doc2QueryAugmenter:
    """
    This class is responsible for generating queries for a document.
    These queries can augment the document before indexing.

    Ref:[1] https://huggingface.co/doc2query/msmarco-t5-base-v1
         [2] Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """
    
    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.
        
        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries
        """
        self.device = torch.device('cpu')  # Do not change this unless you know what you are doing

        # Create the dense tokenizer and query generation model using HuggingFace transformer
        self.model_name = doc2query_model_name
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.
         
            Ensure you take care of edge cases.
            
        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter that gets added before the text.
                Some models like flan-t5 are not fine-tuned to generate queries.
                So we need to add a prompt to instruct the model to generate queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                See the PDF for what you need to do for this part.
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering
        
        Returns:
            A list of query strings generated from the text
        """
        # Note: Feel free to change these values to experiment
        document_max_token_length = 400  # as used in Ref[2]
        top_p = 0.85

        # See https://huggingface.co/doc2query/msmarco-t5-base-v1 for details
        
        # For the given model, generate a list of queries that might reasonably be issued to search
        #       for that document
        
        if len(document)>0:
            if len(prefix_prompt)>0:
                if prefix_prompt[-1] != ':':
                    input_ids = self.tokenizer.encode(prefix_prompt+':'+document,max_length=document_max_token_length, truncation=True, return_tensors='pt')
            
            input_ids = self.tokenizer.encode(prefix_prompt+document,max_length=document_max_token_length, truncation=True, return_tensors='pt')
            outputs = self.model.generate(\
                input_ids=input_ids,\
                max_length=64,\
                do_sample=True,\
                top_p=top_p,\
                num_return_sequences=n_queries)
            
            queries =[]
            for i in range(len(outputs)):
                queries.append(self.tokenizer.decode(outputs[i], skip_special_tokens=True))

            return queries
        else:
            return ''
