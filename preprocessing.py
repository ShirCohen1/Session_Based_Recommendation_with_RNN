# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:50:45 2019
@author: s-moh
"""
import numpy as np
import pandas as pd
import datetime

def removeShortSessions(data):
    #delete sessions of length < 1
    sessionLen = data.groupby('SessionID').size() #group by sessionID and get size of each session
    data = data[np.in1d(data.SessionID, sessionLen[sessionLen > 1].index)]
    return data

#delete records of items which appeared less than 5 times
def remove_items(train):
  itemLen = train.groupby('ItemID').size() #groupby itemID and get size of each item
  train = train[np.in1d(train.ItemID, itemLen[itemLen > 4].index)]
  return train

def convert_time(train):
  train['Time']= train.Time.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #Convert time objects to timestamp
  return train
