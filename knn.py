import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import time

class ItemKNN:
    '''
    ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')
    
    Item-to-item predictor that computes the the similarity to all items to the given item.
    
    Similarity of two items is given by:
    
    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}
        
    Parameters
    --------
    n_sims : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')
    
    '''    
    
    def __init__(self, n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionID', item_key = 'ItemID', time_key = 'Time'):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key
        self.sims = dict()
        self.itemid_2_itemindex = dict()


    def fit(self, data):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data[self.item_key].unique()
        n_items = len(itemids) 
       
        #data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemID':np.arange(len(itemids))}), on=self.item_key, how='inner')
      
        l1 = data.ItemID.unique()
        l2 = np.arange(len(l1))
        self.itemid_2_itemindex = dict(zip(l1,l2))
        itemindex_2_itemid = dict(zip(l2,l1))
        def mapitem(row):
          return self.itemid_2_itemindex[row]
          
        data['ItemID'] = data['ItemID'].apply(mapitem)
        sessionids = data[self.session_key].unique()
        #data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionID':np.arange(len(sessionids))}), on=self.session_key, how='inner')
       
        l1 = data.SessionID.unique()
        l2 = np.arange(len(l1))
        sessionid_2_sessionindex = dict(zip(l1,l2))
        sessionindex_2_sessionid = dict(zip(l2,l1))
        def maps(row):
          return sessionid_2_sessionindex[row]
  
        data['SessionID'] = data['SessionID'].apply(maps)
        #recreate based on new range 0 - 37000
        itemids = data[self.item_key].unique()
        supp = data.groupby('SessionID').size()
        session_offsets = np.zeros(len(supp)+1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()
        index_by_sessions = data.sort_values(['SessionID', self.time_key]).index.values
        supp = data.groupby('ItemID').size()
        item_offsets = np.zeros(n_items+1, dtype=np.int32)

        item_offsets[1:] = supp.cumsum()
        index_by_items = data.sort_values(['ItemID', self.time_key]).index.values
        print('Training...')
        for i in tqdm(range(n_items)):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            end = item_offsets[i+1]
            for e in index_by_items[start:end]:
                uidx = data.SessionID.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx+1]
                user_events = index_by_sessions[ustart:uend]
        
                iarray[data.ItemID.values[user_events]] += 1
              
            iarray[i] = 0
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1-self.n_sims:-1]
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        preds = np.zeros(len(predict_for_item_ids))
        sim_list = self.sims[input_item_id]
        mask = np.in1d(predict_for_item_ids, sim_list.index)
        #predict_for_item_ids.index(mask)
        #print(mask[mask == True])
        for idx, i in enumerate(mask):
          if i:
            preds[idx] = sim_list[predict_for_item_ids[idx]]
        #preds[mask] = sim_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)

    def predict(self,session_id,item_id,set_of_ids):
    
        '''
        input:
            - item_id: the item id of the chosen current event. Mapped to range 0 - Number of items
            - set_of_ids : IDs of items for which the recommender should give prediction scores. list/np.array
            
        output: pd.Series of items ID and corresponding prediction score based on popularity model
        '''

        #item_idx = self.itemid_2_itemindex[item_id]
        item_idx = item_id
        #print(f'item {item_id}, mapped to {item_idx}')

        if type(set_of_ids)==type([]):
            set_of_ids = np.array(set_of_ids) 
        pred = np.zeros(len(set_of_ids))


        sim_list = self.sims[item_idx]
        mask = np.in1d(set_of_ids, sim_list.index)
        pred[mask] = sim_list[set_of_ids[mask]]
        return pd.Series(pred,set_of_ids)

    def evaluate_sessions(self, train, test, items=None, cut_off=20):    

      test.sort_values([self.session_key, self.time_key], inplace=True)
      items_to_predict = train[self.item_key].unique()
      evalutation_point_count = 0
      prev_iid, prev_sid = -1, -1
      mrr, recall = 0.0, 0.0
      print('Evaluating...')
      for i in tqdm(range(len(test))):
          sessionid = test[self.session_key].values[i]
          itemid = test[self.item_key].values[i]
          item_idx = self.itemid_2_itemindex[itemid]
          #print(f'on Item {itemid}, mapped to {item_idx}')
          if prev_sid != sessionid:
              prev_sid = sessionid
          else:
              if items is not None:
                  if np.in1d(item_idx, items): items_to_predict = items
                  else: items_to_predict = np.hstack(([item_idx], items))
              preds = self.predict(sessionid, prev_iid, items_to_predict)
              preds[np.isnan(preds)] = 0
              preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
              rank = (preds > preds[item_idx]).sum()+1
              assert rank > 0
              if rank < cut_off:
                  recall += 1
                  mrr += 1.0/rank
              evalutation_point_count += 1
          prev_iid = item_idx
      recall = recall/evalutation_point_count
      mrr = mrr/evalutation_point_count

      return recall,mrr

