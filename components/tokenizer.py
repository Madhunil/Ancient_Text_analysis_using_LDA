# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 04:50:32 2021

@title: Tokenizer component for Rigved and other Indo European classics

@author: Madhunil Pachghare
"""
#%%
import nltk

#%%
file_content = open("rigved_full_eng.txt",encoding="utf-8").read()
tokens = nltk.word_tokenize(file_content)
#%%
vocabulary = ()
vocabulary = set(tokens)