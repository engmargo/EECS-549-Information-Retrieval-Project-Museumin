from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import numpy as np


class VectorRanker(Ranker):
    def __init__(
        self, bi_encoder_model_name: str, encoded_docs: ndarray, row_to_docid: list[int],
        pseudofeedback_num_docs=0,
        pseudofeedback_alpha=0.8,
        pseudofeedback_beta=0.2
    ) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding
        """
        model = SentenceTransformer(bi_encoder_model_name)
        self.biencoder_model = model
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid
        self.pseudofeedback_num_docs = pseudofeedback_num_docs
        self.pseudofeedback_alpha = pseudofeedback_alpha
        self.pseudofeedback_beta = pseudofeedback_beta

    def query(
        self,
        query: str,
        filterids:list=[],
        # user_id=None,
    ) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first or an empty list if a query cannot be encoded
            or no results are return
        """
        if len(query) == 0:
            return []
        else:
            #   Encode the query using the bi-encoder
            embedding_query = self.biencoder_model.encode(query)
            #   Score the similarity of the query vector and document vectors for relevance
            
            encoded_docs,row_to_docid,id2row = self.filter(filterids)
                        
            scorelist = self.get_scorelist(embedding_query, encoded_docs,row_to_docid)
            
            if self.pseudofeedback_num_docs > 0:
                reldocs=[]
                for id,doc in enumerate(scorelist):
                    if id<self.pseudofeedback_num_docs:
                        reldocs.append(self.encoded_docs[id2row[doc[0]]])
                    else:
                        break
                
                avg_reldocs = np.mean(reldocs, axis=0)
                new_embedding_query = np.multiply(
                    embedding_query, self.pseudofeedback_alpha
                ) + np.multiply(avg_reldocs, self.pseudofeedback_beta)

                scorelist = self.get_scorelist(new_embedding_query, self.encoded_docs,row_to_docid)

            return scorelist
    
    def get_scorelist(
        self, embedding_query: list, encoded_docs: list,row2id:list[int]
    ) -> list[tuple[int, float]]:

        sscores = util.dot_score(embedding_query, encoded_docs)[0].cpu().tolist()
        # self.biencoder_model.similarity(embedding_query,self.encoded_docs).numpy()
        # Generate the ordered list of (document id, score) tuples
        scorelist = list(zip(row2id, sscores))
        # Sort the list so most relevant are first
        scorelist = sorted(scorelist, key=lambda x: x[1], reverse=True)

        return scorelist
        
    def filter(self,filterids:list[int])->tuple:
        if len(filterids)>0:
            encoded_docs = []
            row_to_docid = self.row_to_docid.copy()

            for row,id in enumerate(self.row_to_docid):
                if id in filterids:
                    encoded_docs.append(self.encoded_docs[row])
                else:
                    row_to_docid.remove(id)
        
            id2row = {id:row for row,id in enumerate(row_to_docid)}
            
            return encoded_docs,row_to_docid,id2row
        else:
            return self.encoded_docs,self.row_to_docid,{id:row for row,id in enumerate(self.row_to_docid)}