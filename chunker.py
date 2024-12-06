from collections import defaultdict
import re
import os
import pickle
# from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter

CACHE_PATH = "./__cache__/"


class Chunker:
    def __init__(
        self,
        idkey: str = "museum_id",
        textkeys: list = ["wiki_article"],
    ):
        """
        Initialize general chuncker class.

        attributes:
            idkey: the document id, e.g. museum_id.
            textkeys: the text keys list.
        """
        self.idkey = idkey
        self.textkeys = textkeys
        self.chunks = []

    def create_chunks(self, dataset: list[dict] = None) -> None:
        self.dataset = defaultdict(list)
        for dt in dataset:
            for key in self.textkeys:
                if (key in dt) and len(dt[key]) > 0:
                    self.dataset[dt[self.idkey]].append(dt[key])

    
    def save(self, outfile: str = "id2text.pkl") -> None:
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)

        if len(self.chunks) == 0:
            raise ValueError("Empty Chunks!")
        else:
            pickle.dump(self.chunks, open(CACHE_PATH + outfile, "wb"))
            print("successfully saved!")

    def load(self, outfile: str = "id2text.pkl") -> None:
        if not os.path.exists(CACHE_PATH + outfile):
            raise ValueError(
                f"chunk file doesn't exist. filename: {CACHE_PATH+outfile}"
            )
        else:
            self.chunks = pickle.load(open(CACHE_PATH + outfile, "rb"))


class BySectionChunker(Chunker):
    """
    Generally, wikipedia articles are well organzied by differenct sections that can be easily recognized by pattern like this "== Legislation =="
    This class utilizes regexp to chunk documents.
    """

    def __init__(self, idkey="museum_id", textkeys: list = ["wiki_article"]):
        super().__init__(idkey, textkeys)
        self._name = "Bysection"

    def create_chunks(self, dataset: list[dict], pattern: str = r"=+.*=") -> None:
        super().create_chunks(dataset)
        self.pattern = pattern
        self.chunks = []
        for id, texts in tqdm(self.dataset.items(), total=len(self.dataset)):
            for text in texts:
                subtexts = [i.strip() for i in re.split(self.pattern, text)]
                for subtext in subtexts:
                    if len(subtext) > 0:
                        self.chunks.append({"museum_id": id, "text": subtext})


# class SSemanticChunker(Chunker):
#     def __init__(
#         self,
#         idkey="museum_id",
#         textkeys=["wiki_article"],
#         model_name="sentence-transformers/all-mpnet-base-v2",
#     ):
#         super().__init__(idkey, textkeys)
#         self.model_name = model_name
#         self._name = "SemanticChunker"
#         self.embed_model = HuggingFaceEmbeddings(model_name=self.model_name)
#         self.splitter = SemanticChunker(embeddings=self.embed_model)
        
#     def create_chunks(self, dataset: list[dict], max_length: int = 500) -> None:
#         super().create_chunks(dataset)
#         self.chunks = []
#         for id, texts in tqdm(self.dataset.items(), total=len(self.dataset)):
#             for text in texts:
#                 if len(text) > max_length:
#                     subtexts = self.splitter.create_documents([text])
#                     for subtext in subtexts:
#                         if len(subtext.page_content) > 0:
#                             self.chunks.append(
#                                 {self.idkey: id, "text": subtext.page_content}
#                             )
#                 else:
#                     self.chunks.append({self.idkey: id, "text": text})

class ByChracterChunker(Chunker):
    def __init__(self, idkey:str = "museum_id", textkeys:list = ["wiki_article"],splitterparams:dict={'chunk_size':1000, 'chunk_overlap':100}):
        super().__init__(idkey, textkeys)
        self.splitter = CharacterTextSplitter(**splitterparams)
        self._name = "ChracterSplitter"
    
    def create_chunks(self, dataset: list[dict]):
        super().create_chunks(dataset)
        page_contents = []
        metadata = []
        for id,texts in self.dataset.items():
            page_contents.append(' '.join(texts))
            metadata.append({self.idkey:id})
        
        subtexts = self.splitter.create_documents(page_contents,metadata)
        self.chunks = []
        for subtext in subtexts:
            self.chunks.append(
                                {self.idkey: subtext.metadata["id"], "text": subtext.page_content}
                            )


class RecursiveChunker(Chunker):
    def __init__(self, idkey = "museum_id", textkeys = ["wiki_article"],**splitterparams):
        super().__init__(idkey, textkeys)
        self.splitter = RecursiveCharacterTextSplitter(**splitterparams)
        self._name = "RecursiveChunker"
    
    def create_chunks(self, dataset: list[dict]):
        super().create_chunks(dataset)
        page_contents = []
        metadata = []
        for id,texts in self.dataset.items():
            page_contents.append(' '.join(texts))
            metadata.append({self.idkey:id})
        
        subtexts = self.splitter.create_documents(page_contents,metadata)
        self.chunks = []
        for subtext in subtexts:
            self.chunks.append(
                                {self.idkey: subtext.metadata["museum_id"], "text": subtext.page_content}
                            )
    
    
