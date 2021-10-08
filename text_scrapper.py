# -*- coding: utf-8 -*-
"""
Created on Wed May 12 23:26:35 2021

@author: Madhunil
"""


#%%
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium import webdriver
import time
import pandas as pd

#Initializing the webdriver
options = webdriver.ChromeOptions()

#Uncomment the line below if you'd like to scrape without a new Chrome window every time.
#options.add_argument('headless')

#Change the path to where chromedriver is in your home folder.
driver = webdriver.Chrome(executable_path="D:/NEU/My DS Projects/DS_salary_proj/chromedriver", options=options)
driver.set_window_size(1120, 1000)

url = 'https://www.sacred-texts.com/hin/rigveda/index.htm'
driver.get(url)
text = []