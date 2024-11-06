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
import requests
import json 

class WikiSpider:
    def __init__(self,options = Options(),args='headless')-> None:
        
        """ 
        initialize a WikiSpider.
        
        args:
            options: options for webdriver
            driver: webdriver to use

        """
        self.options = options
        self.options.add_argument(args)
        self.driver = webdriver.Chrome(self.options)
        pass
    
    # get museums lists from all states
    def get_states_museums_list(self,wkp_url='https://en.wikipedia.org/wiki/List_of_museums_in_the_United_States') -> dict[int,list]:
        
        '''
        output: dict, state_name: museum_list
        '''
        self.open_url(wkp_url,)
        
        soup = BeautifulSoup(self.driver.page_source,'html.parser')
        states = soup.find('table',{'class':'col-begin'}).find_all('dl')
        
        states_museums_urls = []
        for state in tqdm(states,total = len(states)):
            line = state.find_all('i')[0].find('a')
            # print(line)
            state_name = line.get_text().split('in ')[-1].strip()  # edge_case washington d.c. 
            # print(state_name)
            state_museums_full_url = 'https://en.wikipedia.org'+ line.get('href')
            
            states_museums_urls.append((state_name,state_museums_full_url))
            
        
        time.sleep(10)
        
        states_museums_list={}
        for idx,item in tqdm(enumerate(states_museums_urls),total = len(states_museums_urls)):
            try:
                single_state_museum_list = self.get_single_state_museum_list(item[1])
                states_museums_list[item[0]] = single_state_museum_list
            except:
                print(item[1])
            
            time.sleep(10)
            
            # break
            
        self.driver.quit()
        
        return states_museums_list
    
    
    # get single museum list
    def get_single_state_museum_list(self,state_museum_full_url:str) -> list[dict]:
        
        self.open_url(state_museum_full_url)
        # self.driver.get(state_museum_full_url)
        soup = BeautifulSoup(self.driver.page_source,'html.parser')
   
        table = soup.find('table',{'class':re.compile('jquery-tablesorter')})
        head_trs = table.find('thead').find_all('th')
        cols_title = []
        for tr in head_trs: 
            cols_title.append(tr.get_text().strip())
        # print(cols_title)
        
        body_trs = table.find('tbody').find_all('tr')
        museums = []
        for idx,museum in enumerate(body_trs): 
            museum_row = {}
            
            th_flag = False
            if museum.find('th'):
                #the museum table of some states start with th
                museum_name = museum.find('th',{'scope':'row'}).find('a').get_text().strip()
                museum_link =  'https://en.wikipedia.org'+ museum.find('th').find('a').get('href')
                museum_row[cols_title[0]] = (museum_name,museum_link)
                th_flag = True
            
            museums_cols = museum.find_all('td')
            for idx,col in enumerate(museums_cols):
                url = ''
                text = ''
                
                loc = col.find('a')
                if loc:
                    url = 'https://en.wikipedia.org'+loc.get('href')
                
                text = col.get_text().strip()
                
                start_idx = idx+1 if th_flag else idx
                museum_row[cols_title[start_idx]] = (text,url)
                
            museums.append(museum_row)
        
        
        return museums


    def open_url(self,url:str)->None:
        self.driver.get(url)
        WebDriverWait(self.driver,10).until(EC.presence_of_element_located((By.TAG_NAME,"body")))
        
        
    def get_single_museum_wiki_artile(self,wiki_url:str)->str:
        title = wiki_url.split('/')[-1]
        
        url =(
        'https://en.wikipedia.org/w/api.php?'
        'format=json&action=query&prop=extracts'
        f'&titles={title}&explaintext=True'
        )
        # print(title)
        
        try:
            article =list(requests.get(url).json()['query']['pages'].values())[0]['extract']
        except:
            article = ''
        
        return article
    
    def save_raw_list(self,file_name:str,state_museums_list:dict)->None:
        '''
        args:
            file_name: nema of file
            state_museums_list: [{state1:museum_1(dict)},{state1:museum_2(dict)}]
        
        '''
        
        with open(file_name,'wt') as f:
            idx = 0
            for key,value in state_museums_list.items():
                for museum in value:
                    museum_info = {'museum_id':idx,'museum_state':key}
                    museum_info.update(museum)                                                                      
                    
                    f.write(json.dumps(museum_info)+'\n')
                    idx +=1
                    
                    
    def wiki_article_filter(self,state_museums_list:list[dict])->list[dict]:
        '''
        args:
            state_muesums_list:[{'museum_id':1,'museum_state':michigan,'museum_name':(museum_name_text,museum_name_url)}]
        '''
        
        wikipedia_museum_articles = []
        for museum in tqdm(state_museums_list,total=len(state_museums_list)):
            for col in museum.keys():
                if col in ['Museum Name', 'Museum name', 'Name']:
                    name_col = col
                    break
        
            name = museum[name_col]
            
            if len(name[1])>0: #filter out the museums with wikipedia article urls
                article = self.get_single_museum_wiki_artile(name[1])
                if len(article)>0:
                    wikipedia_museum_articles.append({'museum_id':museum['museum_id'],'museum_name':name[0],'museum_url':name[1],'article':article})
        
        return wikipedia_museum_articles
    
    
    def get_single_wiki_inbodx_info(self,museum:dict) -> dict[str,str]:
        url = museum['museum_url']
        self.open_url(url)
        
        soup = BeautifulSoup(self.driver.page_source,'html.parser')
        
        table = soup.find('table',{'class':'infobox vcard'})
        table_data = {}
        
        if table:
            
            try:
                img_url = 'https:'+table.find('a',{'class':'mw-file-description'}).find('img').get('src')
            except:
                img_url = ''
                                                       
                
            table_data['img_url'] = img_url
            
            ths = table.find_all('th',{'scope':'row'})
            tds = table.find_all('td',{'class':re.compile('infobox-data')})
            
            num = len(ths)
            for i in range(num):
                row_title = ths[i].get_text()

                if tds[i].find('span',{'class':re.compile('geo')}):
                    row_text = tds[i].find('span',{'class':'geo-dec'}).get_text()
                elif row_title == 'Website':
                    try:
                        row_text = tds[i].find('a').get('href')
                    except:
                        row_text = ''
                else:
                    row_text = tds[i].get_text()
                
                table_data[row_title] = row_text
            
        return table_data


    def get_all_wiki_inbodx_info(self,museums:list[dict]) -> list[dict]:
        new_museums_info = []
        for museum in tqdm(museums,total = len(museums)):
            inbodx_info = self.get_single_wiki_inbodx_info(museum)
            museum.update(inbodx_info)
            new_museums_info.append(museum)
            
        self.driver.quit()
        return new_museums_info
    
        
# if __name__ == '__main__':
#     WikiSpider = WikiSpider()
#     states_museums_list = WikiSpider.get_states_museum_list()
        
        