"""
Author: Zim Gong
Modified by: Lea Lei
This file is a template code file for the Search Engine. 
"""

import csv
import gzip
import json
import jsonlines
import os
import pickle
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from models import BaseSearchEngine, SearchResponse

# project library imports go here
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType, BasicInvertedIndex
from ranker import *
from l2r import L2RRanker, L2RFeatureExtractor
from vector_ranker import VectorRanker

DATA_PATH = "./data_1101/"
CACHE_PATH = "./__cache__/"

STOPWORD_PATH = DATA_PATH + "stopwords.txt"
DOC2QUERY_PATH = DATA_PATH + "subid_musuemid_query_200.jsonl"
DATASET_PATH = DATA_PATH + "museum_texts.jsonl"
RELEVANCE_TRAIN_PATH = DATA_PATH + "relevance_train.csv"
CHUNK_RELEVANCE_TRAIN_PATH = DATA_PATH + "chunk_relevance_train.csv"
ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH = DATA_PATH + "msmarco-MiniLM-L12-cos-v5.npy"
DOC_IDS_PATH = DATA_PATH + "museum-ids.txt"
CATEGORY_PATH = DATA_PATH + "museum_category.jsonl"

CHUNK_DATASET_PATH = DATA_PATH + "subid_museumid_text_200.jsonl"
CHUNK_ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH = (
    DATA_PATH + "chunks_200_msmarco-MiniLM-L12-cos-v5.npy"
)
CHUNK_IDS_PATH = DATA_PATH + "chunks-ids.text"
CHUNKED_ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH = (
    DATA_PATH + "chunks_200_msmarco-MiniLM-L12-cos-v5.npy"
)

MAIN_INDEX = "body_index"
CHUNK_MAIN_INDEX = "chunk_body_index"
TITLE_INDEX = "title_index"
RAW_TRAIN = "raw_text_dict_train.pkl"
CHUNK_RAW_TRAIN = "chunk_raw_text_dict_train.pkl"

BLOCK_PATH = DATA_PATH + "id2blocks.jsonl"


class SearchEngine(BaseSearchEngine):
    def __init__(
        self,
        max_docs: int = -1,
        ranker: str = "BM25",
        l2r: bool = False,
        aug_docs: bool = False,
        chunk: bool = False,
    ) -> None:
        # 1. Create a document tokenizer using document_preprocessor Tokenizers
        # 2. Load stopwords, network data, categories, etc
        # 3. Create an index using the Indexer and IndexType (with the Wikipedia JSONL and stopwords)
        # 4. Initialize a Ranker/L2RRanker with the index, stopwords, etc.
        # 5. If using L2RRanker, train it here.

        self.l2r = False
        self.chunk = False
        self.l2r_model_name = None

        print("Initializing Search Engine...")
        self.stopwords = set()
        with open(STOPWORD_PATH, "r") as f:
            for line in f:
                self.stopwords.add(line.strip())

        self.doc_augment_dict = None
        if aug_docs:
            print("Loading doc augment dict...")
            self.doc_augment_dict = defaultdict(lambda: [])
            with open(DOC2QUERY_PATH, "r") as f:
                for line in f:
                    data = json.loads(line)
                    if not chunk:
                        self.doc_augment_dict[data["museum_id"]].append(data["query"])
                    elif chunk:
                        self.doc_augment_dict[data["id"]].append(data["query"])

        self.subid2id = None
        if chunk:
            self.subid2id = defaultdict(int)
            with open(CHUNK_DATASET_PATH, "r") as f:
                for line in f:
                    data = json.loads(line)
                    self.subid2id[data["id"]] = data["museum_id"]

        print("Loading categories...")
        docid_to_categories = defaultdict()
        categories_to_docid = defaultdict(list)
        with open(CATEGORY_PATH, "rt") as f:
            for line in tqdm(f):
                data = json.loads(line)
                docid_to_categories[data["museum_id"]] = data["new_category"]
                categories_to_docid[data["new_category"]].append(data["museum_id"])

        self.doc_category_info = defaultdict()
        self.categories_to_docid = defaultdict(list)
        if not chunk:
            self.doc_category_info = docid_to_categories
            self.categories_to_docid = categories_to_docid
        else:
            for subid, mainid in self.subid2id.items():
                self.doc_category_info[subid] = docid_to_categories[mainid]
                self.categories_to_docid[docid_to_categories[mainid]].append(subid)

        del docid_to_categories, categories_to_docid

        print("Loading Blocks...")
        self.blocks = defaultdict(dict)
        with open(BLOCK_PATH, "r") as f:
            for line in f:
                data = json.loads(line)
                self.blocks[data["museum_id"]] = data

        self.state_to_ids = defaultdict(list)
        if not chunk:
            for block in self.blocks.values():
                self.state_to_ids[block["museum_state"]].append(block["museum_id"])
        else:
            for subid, mainid in self.subid2id.items():
                self.state_to_ids[self.blocks[mainid]["museum_state"]].append(subid)

        print("Loading indexes...")
        self.preprocessor = RegexTokenizer("\w+")
        main_path = MAIN_INDEX if not chunk else CHUNK_MAIN_INDEX
        data_path = DATASET_PATH if not chunk else CHUNK_DATASET_PATH
        aug_pre = "aug" if aug_docs else ""

        if not os.path.exists(CACHE_PATH + aug_pre + main_path):
            self.main_index = Indexer.create_index(
                IndexType.BasicInvertedIndex,
                data_path,
                self.preprocessor,
                self.stopwords,
                50,
                max_docs=max_docs,
                doc_augment_dict=self.doc_augment_dict,
                text_key="texts",
            )
            self.main_index.save(CACHE_PATH + aug_pre + main_path)
        else:
            self.main_index = BasicInvertedIndex()
            self.main_index.load(CACHE_PATH + aug_pre + main_path)

        if not os.path.exists(CACHE_PATH + TITLE_INDEX):
            self.title_index = Indexer.create_index(
                IndexType.BasicInvertedIndex,
                data_path,
                self.preprocessor,
                self.stopwords,
                2,
                max_docs=max_docs,
                text_key="museum_name",
            )
            self.title_index.save(CACHE_PATH + TITLE_INDEX)
        else:
            self.title_index = BasicInvertedIndex()
            self.title_index.load(CACHE_PATH + TITLE_INDEX)

        print("Loading raw text dict...")
        if not os.path.exists(CACHE_PATH +  RAW_TRAIN):
            if not os.path.exists(CACHE_PATH):
                os.makedirs(CACHE_PATH)
            self.raw_text_dict = defaultdict()
            data_path = DATASET_PATH if not chunk else CHUNK_DATASET_PATH
            with open(data_path, "rt") as f:
                for line in f:
                    self.raw_text_dict[str(data["museum_id"])] = data["texts"][:500]

            pickle.dump(self.raw_text_dict, open(CACHE_PATH +  RAW_TRAIN, "wb"))
        else:
            self.raw_text_dict = pickle.load(open(CACHE_PATH +  RAW_TRAIN, "rb"))


        print("Loading ranker...")
        self.set_ranker(ranker)
        self.set_l2r(l2r)

        print("Search Engine initialized!")

    def set_ranker(self, ranker: str = "BM25", user_id: int = None,
        pseudofeedback_num_docs:int = 0,
        pseudofeedback_alpha:int = 0.8,
        pseudofeedback_beta:int = 0.2) -> None:
        if ranker == "VectorRanker":
            npy_path = (
                ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH
                if not self.chunk
                else CHUNK_ENCODED_DOCUMENT_EMBEDDINGS_NPY_PATH
            )
            docid_path = DOC_IDS_PATH if not self.chunk else CHUNK_IDS_PATH

            with open(npy_path, "rb") as f:
                self.encoded_docs = np.load(f)
            with open(docid_path, "r") as f:
                self.row_to_docid = [int(line.strip()) for line in f]
            self.ranker = VectorRanker(
                "sentence-transformers/msmarco-MiniLM-L12-cos-v5",
                self.encoded_docs,
                self.row_to_docid,
                pseudofeedback_num_docs,
                pseudofeedback_alpha,
                pseudofeedback_beta
            )
        else:
            if ranker == "BM25":
                self.scorer = BM25(self.main_index)
            elif ranker == "RandomScore":
                self.scorer = RandomScore()
            elif ranker == "WordCountCosineSimilarity":
                self.scorer = WordCountCosineSimilarity(self.main_index)
            elif ranker == "DirichletLM":
                self.scorer = DirichletLM(self.main_index)
            elif ranker == "PivotedNormalization":
                self.scorer = PivotedNormalization(self.main_index)
            elif ranker == "TF_IDF":
                self.scorer = TF_IDF(self.main_index)
            else:
                raise ValueError("Invalid ranker type")
            
            print("check",pseudofeedback_num_docs,"\n")
            self.ranker = Ranker(
                    self.main_index,
                    self.preprocessor,
                    self.stopwords,
                    self.scorer,
                    self.raw_text_dict,
                    pseudofeedback_num_docs,
                    pseudofeedback_alpha,
                    pseudofeedback_beta
            )
        if self.l2r:
            self.pipeline.ranker = self.ranker
        else:
            self.pipeline = self.ranker

    def set_l2r(
        self, l2r: bool = True, aug_docs_ce: bool = False, model_name="LambdaMART",params:dict = None
    ) -> None: 

        if (self.l2r == l2r) and (self.l2r_model_name == model_name):
            return

        if not l2r:
            self.pipeline = self.ranker
            self.l2r = False
        else:
            if aug_docs_ce:
                for docid, text in self.raw_text_dict.items():
                    text = " ".join(self.doc_augment_dict[docid]) + text
                    text = text[:500]
                    self.raw_text_dict[docid] = text

            self.cescorer = CrossEncoderScorer(self.raw_text_dict)
            self.fe = L2RFeatureExtractor(
                self.main_index,
                self.title_index,
                self.doc_category_info,
                self.preprocessor,
                self.stopwords,
                self.cescorer,
            )

            print("Loading L2R ranker...")
            self.pipeline = L2RRanker(
                self.main_index,
                self.title_index,
                self.preprocessor,
                self.stopwords,
                self.ranker,
                self.fe,
                model = model_name,
                params = params
            )

            print("Training L2R ranker...")
            self.pipeline.train(RELEVANCE_TRAIN_PATH)
            self.l2r = True
            self.l2r_model_name = model_name

    def search(
        self, query: str, states: list[str], cates: list[str]
    ) -> list[SearchResponse]:
        # 1. Use the ranker object to query the search pipeline
        # 2. This is example code and may not be correct.
        
        filterids = set()
        if len(states) > 0:
            for state in states:
                filterids.update(self.state_to_ids[state])
            
        filterids2 = set()
        if len(cates) > 0:
            for cate in cates:
                filterids2.update(self.categories_to_docid[cate])
        
        if len(filterids)>0 and len(filterids2)>0:
            filterids = filterids.intersection(filterids2)
        else:
            filterids = filterids.union(filterids2)
        
        del filterids2
        
        
        if len(query) ==0:
            results = [(id,idx) for idx,id in enumerate(filterids)]
        else:
            results = self.pipeline.query(query,list(filterids))
            
        if self.chunk:
            bestrks = {}
            for idx, item in enumerate(results):
                museumid = self.subid2id[item[0]]
                if museumid not in bestrks:
                    bestrks[museumid] = idx
                else:
                    continue
            results = [(id, rk) for id, rk in bestrks.items()]

            del bestrks

        if len(cates) > 0:
            if "all" in cates:
                pass
            else:
                results = [
                    (museumid, score)
                    for museumid, score in results
                    if self.blocks[museumid]["new_category"] in cates
                ]

        return [
            SearchResponse(
                id=idx + 1,
                museum_id=result[0],
                score=result[1],
                museum_name=self.blocks[result[0]]["museum_name"],
                img_urls=self.blocks[result[0]]["img_urls"],
                category=self.blocks[result[0]]["new_category"],
                museum_state=self.blocks[result[0]]["museum_state"],
                location=(
                    self.blocks[result[0]]["location"]
                    if self.blocks[result[0]]["location"]
                    else ""
                ),
                admission=self.blocks[result[0]]["admission"],
                website=self.blocks[result[0]]["web"],
                description=self.blocks[result[0]]["description"].strip(),
            )
            for idx, result in enumerate(results)
        ]


def initialize():
    search_obj = SearchEngine(max_docs=1000, ranker="VectorRanker")
    search_obj.set_l2r(params={"n_estimators":15,"learning_rate":0.3})
    
    return search_obj


def main():
    search_obj = SearchEngine(max_docs=10000)
    search_obj.set_l2r(True)
    query = "What is the capital of France?"
    results = search_obj.search(query)
    print(results[:5])
