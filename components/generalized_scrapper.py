# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 21:26:54 2023

@title: A Generalized Scrapper component for getting all the Indo-European classics

@author: Madhunil Pachghare
"""
#%%
import bs4 as bs
import urllib.request
import re
import nltk
import chars2vec
import os

import requests
from bs4 import BeautifulSoup
from requests_testadapter import Resp

#%%
def get_links(base_url,start_index,end_index):
    """Getting links for url via index value"""
    urls=[]
    for i in range(start_index,end_index+1):
        temp=base_url+str(i)+".htm"
        urls.append(temp)
    return urls

#%%
class LocalFileAdapter(requests.adapters.HTTPAdapter):
    def build_response_from_file(self, request):
        file_path = request.url[7:]
        with open(file_path, 'rb') as file:
            buff = bytearray(os.path.getsize(file_path))
            file.readinto(buff)
            resp = Resp(buff)
            r = self.build_response(request, resp)

            return r

    def send(self, request, stream=False, timeout=None,
             verify=True, cert=None, proxies=None):

        return self.build_response_from_file(request)

#%%
def scrape_p_1_v1(url):
    """Function to scrape data from FTP url"""
    requests_session = requests.session()
    requests_session.mount('file://', LocalFileAdapter())
    page = requests_session.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    #text = soup.find_all('p')[1].prettify()
    text = soup.find_all('p')
    text = [p.prettify() for p in text]
    text = text[1:-1]
    str_text = ' '.join(text)
    my_list = [i.replace('<br>', '').replace('<br/>', '').replace('<p>', '').replace('</p>', '') for i in str_text.split()]
    return ' '.join(my_list)

#%%
def get_text_in_list_format(urls):
    """Function to append the scrapped raw text in a list"""
    all_text=[]
    for url in urls:
        all_text.append(scrape_p_1_v1(url))
    return all_text

#%% Run this cell to generate the greek classic text the iliad by homer
links_1st = get_links('file://C:/Users/madpa/Indo-European-language-family-analysis-using-NLP-techniques/dataset/public_html/cla/homer/ili/ili0',1,9)
links_2nd = get_links('file://C:/Users/madpa/Indo-European-language-family-analysis-using-NLP-techniques/dataset/public_html/cla/homer/ili/ili',10,24)

links_for_book1 = links_1st + links_2nd


iliad_full = get_text_in_list_format(links_for_book1)


iliad_full_book = ' '.join(iliad_full)


with open("iliad_full_book.txt", "w") as file:
    file.write(iliad_full_book)



#%% Run this cell to generate the indic classic text the rigved
#https://www.sacred-texts.com/hin/rvsan/rv01001.htm
links_1st = get_links('file://C:/Users/madpa/Indo-European-language-family-analysis-using-NLP-techniques/dataset/public_html/rvsan/rv01001',1,101)
#links_2nd = get_links('file://C:/Users/madpa/Indo-European-language-family-analysis-using-NLP-techniques/dataset/public_html/rvsan/homer/ili/ili',10,24)

links_for_book1 = links_1st


rvsan_full = get_text_in_list_format(links_for_book1)


rvsan_full_book = ' '.join(rvsan_full)


with open("rvsan_full_book.txt", "w") as file:
    file.write(rvsan_full_book)

#%%
ili_link = 'file://C:/Users/madpa/Indo-European-language-family-analysis-using-NLP-techniques/dataset/public_html/cla/homer/ili/ili01.htm'

#%%
requests_session_exp = requests.session()
requests_session_exp.mount('file://', LocalFileAdapter())

#%%
page_ili = requests_session_exp.get(ili_link)
soup_ili = BeautifulSoup(page_ili.content, "html.parser")

#%%
text_ili = soup_ili.find_all('p')[1].prettify()
text2_ili = soup_ili.get_text()

#%%
text3_ili = soup_ili.find_all('p')
text3_ili = [p.prettify() for p in text3_ili]
text3_ili = text3_ili[1:-1]

#%%
str_ili = ' '.join(text3_ili)

#%%
ili_list = [i.replace('<br>', '').replace('<br/>', '').replace('<p>', '').replace('</p>', '') for i in text3_ili.split()]
ili_list = ' '.join(ili_list)


#%%
rigveda_link = 'file://C:/Users/madpa/Indo-European-language-family-analysis-using-NLP-techniques/dataset/public_html/hin/rigveda/rv01001.htm'

#%%
page_rigveda = requests_session_exp.get(rigveda_link)
soup_rigveda = BeautifulSoup(page_rigveda.content, "html.parser")

#%%
text_rigveda = soup_rigveda.find_all('p')[1].prettify()

text2_rigveda = soup_rigveda.get_text()
#%%
text3_rigveda = soup_rigveda.find_all('p')
text3_rigveda = [p.prettify() for p in text3_rigveda]
text3_rigveda = text3_rigveda[1:-1]

#%%
rv_list = [i.replace('<br>', '').replace('<br/>', '').replace('<p>', '').replace('</p>', '') for i in text_rigveda.split()]
rv_list = ' '.join(rv_list)

 