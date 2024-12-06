from indexing import InvertedIndex
import numpy as np
from collections import Counter, defaultdict

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

    def __init__(
        self,
        index: InvertedIndex,
        document_preprocessor,
        stopwords: set[str],
        scorer: "RelevanceScorer",
        raw_text_dict: dict[int, str] = None,
        pseudofeedback_num_docs=0,
        pseudofeedback_alpha=0.8,
        pseudofeedback_beta=0.2
    ) -> None:
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
        
        self.pseudofeedback_num_docs = pseudofeedback_num_docs
        self.pseudofeedback_alpha = pseudofeedback_alpha
        self.pseudofeedback_beta = pseudofeedback_beta
        

    def query(
        self,
        query: str,
        filterids:list = []
    ) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        #  Tokenize query
        # qtokens is tokens list before stopwords filtering
        # qtokens_count is token counter after wtopwords filtering
        qtokens = self.tokenizer.tokenize(query)

        # remove stopwords and get query token count
        qtokens_count = self.get_Counter(qtokens, self.stopwords)

        #  Fetch a list of possible documents from the index
        doctokens = self.get_doc_tokens(list(qtokens_count.keys()),filterids)
        

        if len(doctokens)==0:
            return []
        else:
            # ranks for initial query
            scorelist = []
            for id in doctokens:
                scorelist.append((id, self.scorer.score(id, doctokens[id], qtokens_count)))
            
            sortedscore = sorted(scorelist, key=lambda x: x[1], reverse=True)

            if self.pseudofeedback_num_docs > 0:
                pseudo_doc = defaultdict(int)
                num = 0

                for doc in sortedscore:
                    if num < self.pseudofeedback_num_docs:
                        for word, freq in doctokens[doc[0]].items():
                            pseudo_doc[word] += freq
                        num += 1
                    else:
                        break
                    
                weighted_qtokens = {
                    word: freq * self.pseudofeedback_alpha
                    for word, freq in qtokens_count.items()
                }
                weighted_dtokens = {
                    word: freq * self.pseudofeedback_beta / num
                    for word, freq in pseudo_doc.items()
                }
                new_qtokens_count = Counter(weighted_dtokens) + Counter(weighted_qtokens)
                    
                # redo the scoring for new query
                
                doctokens = self.get_doc_tokens(list(new_qtokens_count.keys()))

                scorelist = []
                for id in doctokens.keys():
                    scorelist.append(
                        (id, self.scorer.score(id, doctokens[id], new_qtokens_count))
                    )

                sortedscore = sorted(scorelist, key=lambda x: x[1], reverse=True)

            # 3. Return **sorted** results as format [(100, 0.5), (10, 0.2), ...]
            return sortedscore

    def get_Counter(self, qtokens, stopwords):
        qtokens = [token if token not in stopwords else None for token in qtokens]
        qcount = Counter(qtokens)

        return qcount

    def get_doc_tokens(self, unique_qtokens: list,filterids:list = []) -> dict:  # unique_qtokens of query
        doctokens = defaultdict(lambda: defaultdict(int))

        for token in unique_qtokens:
            if token in self.index.vocabulary:
                for idx, item in enumerate(self.index.index[token]):
                    if idx <= 1000 and item[0] not in doctokens:
                        is_checked = True if (len(filterids)==0) or (item[0] in filterids) else False
                        if is_checked:
                            doctokens[item[0]] = {
                                key: value
                                for key, value in self.index.document_metadata[item[0]][
                                    "tokens_count"
                                ].items()
                                if (key in unique_qtokens) and (key != None)
                            }
                    else:
                        break

        return doctokens


class RelevanceScorer:
    """
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    """

    # Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(
        self,
        docid: int,
        doc_word_counts: dict[str, int],
        query_word_counts: dict[str, int],
    ) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
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

    def score(
        self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]
    ) -> float:
        """
        Scores all documents as 10.
        """
        return 10

class RandomScore(RelevanceScorer):
    def __init__(self):
        self._name = "RandomScore"
    
    def score(self,
        docid: int,
        doc_word_counts: dict[str, int],
        query_word_counts: dict[str, int])->float:
        
        return float(random.randint(1,10000))
    
#   Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        self._name = "WordCountCosineSimilarity"

    def score(
        self,
        docid: int,
        doc_word_counts: dict[str, int],
        query_word_counts: dict[str, int],
    ) -> float:
        #  Find the dot product of the word count vector of the document and the word count vector of the query
        # generate vector
        vdoc = [
            freq for term, freq in query_word_counts.items() if term in doc_word_counts
        ]
        vquery = [
            doc_word_counts[term]
            for term, freq in query_word_counts.items()
            if term in doc_word_counts
        ]

        #  Return the score
        score = np.dot(vdoc, vquery)

        return score


#   Implement DirichletLM
class DirichletLM(RelevanceScorer):  # input original query_word_counts
    def __init__(self, index: InvertedIndex, parameters: dict = {"mu": 2000}) -> None:
        self.index = index
        self.parameters = parameters
        self._name = "DirichletLM"

    def score(
        self,
        docid: int,
        doc_word_counts: dict[str, int],
        query_word_counts: dict[str, int],
    ) -> float:
        #  Get necessary information from index
        doc_length = self.index.document_metadata[docid]["length"]
        tokens_count = self.index.statistics["total_token_count"]
        qlength = np.sum(list(query_word_counts.values()))

        #  Compute additional terms to use in algorithm
        score = 0
        for token, freq in query_word_counts.items():
            if token in doc_word_counts:
                ele = (freq, doc_word_counts[token], len(self.index.index[token]))
                score += ele[0] * np.log(
                    1 + (ele[1] / self.parameters["mu"] / (ele[2] / tokens_count))
                )

        score += qlength * np.log(
            self.parameters["mu"] / (doc_length + self.parameters["mu"])
        )

        return score


#   Implement BM25
class BM25(RelevanceScorer):
    def __init__(
        self, index: InvertedIndex, parameters: dict = {"b": 0.75, "k1": 2, "k3": 8}
    ) -> None:
        self.index = index
        self.b = parameters["b"]
        self.k1 = parameters["k1"]
        self.k3 = parameters["k3"]
        self._name = "BM25"

    def score(
        self,
        docid: int,
        doc_word_counts: dict[str, int],
        query_word_counts: dict[str, int],
    ) -> float:
        #  Get necessary information from index
        ndocs = self.index.statistics["number_of_documents"]
        mean_length = self.index.statistics["mean_document_length"]
        doc_length = self.index.document_metadata[docid]["length"]

        #  Find the dot product of the word count vector of the document and the word count vector of the query

        score = 0
        for token, freq in query_word_counts.items():
            if token in doc_word_counts:
                ele = (freq, doc_word_counts[token], len(self.index.index[token]))
                # for ele in interseted_tokens:
                a = ((self.k1 + 1) * ele[1]) / (
                    self.k1 * (1 - self.b + self.b * doc_length / mean_length) + ele[1]
                )
                b = (self.k3 + 1) * ele[0] / (self.k3 + ele[0])
                c = np.log((ndocs - ele[2] + 0.5) / (ele[2] + 0.5))
                score += a * b * c

        return score

# # Implement Personalized BM25
# class PersonalizedBM25(RelevanceScorer):
#     def __init__(
#         self,
#         index: InvertedIndex,
#         relevant_doc_index: InvertedIndex,
#         parameters={"b": 0.75, "k1": 1.2, "k3": 8},
#     ) -> None:
#         """
#         Initializes Personalized BM25 scorer.

#         Args:
#             index: The inverted index used to use for computing most of BM25
#             relevant_doc_index: The inverted index of only documents a user has rated as relevant,
#                 which is used when calculating the personalized part of BM25
#             parameters: The dictionary containing the parameter values for BM25

#         Returns:
#             The Personalized BM25 score
#         """
#         self.index = index
#         self.relevant_doc_index = relevant_doc_index
#         self.b = parameters["b"]
#         self.k1 = parameters["k1"]
#         self.k3 = parameters["k3"]
#         self._name = "PersonalizedBM25"

#     def score(
#         self,
#         docid: int,
#         doc_word_counts: dict[str, int],
#         query_word_counts: dict[str, int],
#     ) -> float:
#         # TODO (HW4): Implement Personalized BM25
#         ndocs = self.index.statistics["number_of_documents"]
#         mean_length = self.index.statistics["mean_document_length"]
#         doc_length = self.index.document_metadata[docid]["length"]
#         rndocs = self.relevant_doc_index.statistics['number_of_documents']
        
#         score = 0
#         for token, freq in query_word_counts.items():
#             if token in doc_word_counts:
#                 if token in self.relevant_doc_index.index:
#                     ri = len(self.relevant_doc_index.index[token])
#                 else:
#                     ri = 0
                
#                 ele = (freq, doc_word_counts[token], len(self.index.index[token]),ri)
#                 # for ele in interseted_tokens:
#                 a = ((self.k1 + 1) * ele[1]) / (
#                     self.k1 * (1 - self.b + self.b * doc_length / mean_length) + ele[1]
#                 )
#                 b = (self.k3 + 1) * ele[0] / (self.k3 + ele[0])
#                 c = np.log(
#                     ((ri+0.5)*(ndocs-ele[2]-rndocs+ri+0.5))/\
#                     (ele[2]-ri+0.5)/(rndocs-ri+0.5)
#                 )
#                 score += a * b * c
        
#         return score


#   Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {"b": 0.2}) -> None:
        self.index = index
        self.b = parameters["b"]
        self._name = "PivotedNormalization"

    def score(
        self,
        docid: int,
        doc_word_counts: dict[str, int],
        query_word_counts: dict[str, int],
    ) -> float:
        #  Get necessary information from index
        ndocs = self.index.statistics["number_of_documents"]
        mean_length = self.index.statistics["mean_document_length"]
        doc_length = self.index.document_metadata[docid]["length"]

        #  Compute additional terms to use in algorithm
        score = 0

        for token, freq in query_word_counts.items():
            if token in doc_word_counts:
                ele = (freq, doc_word_counts[token], len(self.index.index[token]))
                a = (1 + np.log(1 + np.log(ele[1]))) / (
                    1 - self.b + self.b * doc_length / mean_length
                )
                b = np.log((ndocs + 1) / ele[2]) * ele[0]
                score += a * b

        return score


#   Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        self._name = "TF_IDF"

    def score(
        self,
        docid: int,
        doc_word_counts: dict[str, int],
        query_word_counts: dict[str, int],
    ) -> float:
        # Get necessary information from index
        ndocs = self.index.statistics["number_of_documents"]

        score = 0
        for token, freq in query_word_counts.items():
            if token in doc_word_counts:
                ele = (freq, doc_word_counts[token], len(self.index.index[token]))
                score += np.log(ele[1] + 1) * (np.log(ndocs / ele[2]) + 1)

        return score

# This is not a RelevanceScorer object because the method signature for score() does not match, but it
# has the same intent, in practice
class CrossEncoderScorer:
    """
    A scoring object that uses cross-encoder to compute the relevance of a document for a query.
    """

    def __init__(
        self,
        raw_text_dict: dict[int, str],
        cross_encoder_model_name: str = "cross-encoder/msmarco-MiniLM-L6-en-de-v1",
    ) -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        #  Save any new arguments that are needed as fields of this class
        self.text = raw_text_dict
        # self.model_name = cross_encoder_model_name
        self.model = CrossEncoder(cross_encoder_model_name, max_length=500)

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        if query == "":
            return 0
        elif docid not in self.text:
            return 0
        else:
            #unlike the other scorers like BM25, this method takes in the query string itself,
            # not the tokens!
            # Get a score from the cross-encoder model
            #     Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed

            score = self.model.predict((query, self.text[docid]))

            return score