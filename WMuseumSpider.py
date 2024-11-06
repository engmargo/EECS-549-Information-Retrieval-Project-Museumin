from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm
from nltk.stem.porter import *
from selenium.webdriver.chrome.options import Options
import time 
import re
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


class WMuseumSpider:
    def __init__(self,options = Options(),args='headless'):
        self.options = options
        self.options.add_argument(args)
        self.driver = webdriver.Chrome(self.options)
    
    
    def get_single_page_museums_url(self,soup:BeautifulSoup)->list[dict]:
        page_museums_url = []
        webeles = soup.find('ol',{'class':'list list--grid'}).find_all('li')
        for ele in webeles:
            museum_url = {}
            museum_link = ele.find('a',{'class':'list__container'}).get('href')
            museum_name = ele.find('div',{'class':'list__content'}).find('h3').get_text()
            
            score = WMuseumSpider.get_value_or_null(\
                            ele.find('span',{'class':'list__label label yellow left bottom'}).find('strong').get_text(),\
                            '')
            
            img_links = WMuseumSpider.get_value_or_null(ele.find('img').get('src'),'')
            
            full_link = 'https://whichmuseum.com/'+museum_link
        
            museum_url[museum_name] = {'wmuseum_link':full_link,'image':img_links,'score':score}
            
            page_museums_url.append(museum_url)
            
        return page_museums_url


    def get_pages_museums_url(self,page_url:str,museums_urls:list,start_page:int,end_page:int)->list[dict]:
        '''
        args
            page_url: the first page
            museums_urls:
            n: the first n pages to be crawled
        '''
        
        for i in tqdm(range(start_page,end_page+1),total=end_page-start_page+1):
            try:
                if i ==1:
                    self.open_url(page_url)
                else:
                    self.open_url(page_url+f'?page={i}')
            
                soup = BeautifulSoup(self.driver.page_source,'html.parser')
                museums_urls.extend(self.get_single_page_museums_url(soup))
            except:
                print(i)
                break             
            
        return museums_urls

    
    def get_single_museum_info(self,page_url:str)->dict:
        self.open_url(page_url)
        soup = BeautifulSoup(self.driver.page_source,'html.parser')
        
        info = {}
        text = WMuseumSpider.get_value_or_null(\
                soup.find('div',{'class':'museum__description mb-4'}).find('div',{'class':'collapse__content'}).get_text(),\
                '')
        
        info['description'] = text
        
        
        tbs = soup.find_all('div',{'class': 'cell medium-6 large-4'})
        for tb in tbs:
            title = tb.find('h3').get_text()
            if tb.find('table'):
                table = []
                for tr in tb.find_all('tr'):
                    tds = tr.find_all('td')
                    table.append((tds[0].get_text(),tds[1].get_text()))
                
                info[title] = table
                
            else:
                texts = tb.find('p').get_text()
                info[title] = texts
        
        return info
    
    
    def open_url(self,url:str)->None:
        self.driver.get(url)
        WebDriverWait(self.driver,10).until(EC.presence_of_element_located((By.TAG_NAME,"body")))


    @staticmethod
    def get_value_or_null(function,default_value):
        try:
            return function
        except:
            return default_value
    
    
    
if __name__ == '__main__':
    WMuseumSpider = WMuseumSpider()
    page_url ='https://whichmuseum.com/place/united-states-2682'
    museums_urls=[]
    pages = (0,2)
    is_start=True
    WMuseumSpider.get_pages_museums_url(page_url,museums_urls,pages,is_start)
    