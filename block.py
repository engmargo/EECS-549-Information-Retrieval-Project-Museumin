from collections import defaultdict
import json
import requests
import re
import nltk
from typing import Union

DATA = "./data_1101/"
CATEGORY_PATH = DATA + "museum_category.jsonl"
DEFAULT_IMG = "static/" + "default.jpg"


class Block:
    """
    This class is a user-interface museum information container.

    default block information template:
    museum_name (official website/wiki_url)
    new_category
    museum image
    State,location(coodinate)
    admission
    openning hour
    """

    def __init__(self, template: list[str]):
        for attr in template:
            setattr(self,attr,None)

class BlockRetriver:
    """
    Retrive block given museum_id.
    """

    def __init__(
        self,
        dataset: dict[int, dict],
        template=[
            "museum_id",
            "museum_name",
            "museum_url",
            "img_urls",
            "new_category",
            "museum_state",
            "location",
            "coordinates",
            "admission",
            "opening hour",
            "website",
            "opening hour",
            "description",
        ],
    ):
        self.dataset = dataset
        self.template = template

        for col in template:
            if "new_category" in col:
                self.cateogry = defaultdict(str)
                with open(CATEGORY_PATH, "rt") as f:
                    for line in f:
                        data = json.loads(line)
                        self.cateogry[data["museum_id"]] = data["new_category"]

        if "description" in self.template:
            self.detector = nltk.data.load("tokenizers/punkt/english.pickle")
            self.detector._params.abbrev_types.add("hon")

    def get_block(self, museum_id: int,is_block:bool = False) -> Union[Block,dict]:
        block = Block(self.template)
        setattr(block,"museum_id", museum_id)
        setattr(block,"new_category", self.cateogry[museum_id])
        data = self.dataset[museum_id]
        block = self.set_attrvalue(data, block, 1)

        if "img_urls" in self.template:
            block.img_urls = self.check_img(block.img_urls)
        
        if "description" in self.template:
            block.description = self.get_description(museum_id)
            
        if ("website" in self.template) or ("museum_url" in self.template):
            setattr(block,"web",self.check_wb(block.website,block.museum_url))
            del block.website
            del block.museum_url
            
        if "admission" in self.template:
            block.admission = self.check_admission(block.admission)
                
        if is_block:
           return block
        else:
           return block.__dict__
            

    def set_attrvalue(self, data, block: Block, find_num: int) -> Block:
        if find_num == len(self.template):
            return block
        else:
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in self.template:
                        setattr(block,key, value)
                        find_num += 1
                    else:
                        block = self.set_attrvalue(value, block, find_num)

            return block

    def check_img(self, img_urls: list[str]) -> str:
        if img_urls:
            for url in img_urls:
                response = requests.head(url)
                if "image" in response.headers.get("content-type"):
                    return url

        return DEFAULT_IMG

    def check_wb(self,web:str=None,wiki:str = None)->str:
        wb = web if web else wiki
        return wb
    
    def check_admission(self,admission:Union[str,list])->str:
        '''
        Get the admission range, e.g. $0 ~ $12.00
        '''
        if isinstance(admission,list):
            minp = None
            maxp = 0
            for i,j in admission:
                if j == "Free":
                    minp = 0
                if len(re.findall(r"\d+\.?\d*",j))>0:
                    nums = [float(num) for num in re.findall(r"\d+\.?\d*",j)]
                    for num in nums:
                        if num>maxp:
                            maxp = num
                        if (not minp) or (num<minp):
                            minp = num
                            
            return f"${minp} ~ ${maxp}"
        
        else:
            return "Unavailable"
    def get_description(self, museum_id: int):
        """
        Set first sentence from wiki article as description
        """
        # remove html tags
        text = re.sub(r"<[^>]+>", "", self.dataset[museum_id]["wiki_article"])
        # retrieve first two sentences
        return " ".join(self.detector.tokenize(text)[:2])
