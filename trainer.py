import os
import lib
import time
import torch
import numpy as np
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, train_data, eval_data, optim, loss_func, batch_size, model_name = 'RNN'):
        self.model = model
        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.evaluation = lib.Evaluation(self.model, self.loss_func, torch.cuda.is_available(), model_name = model_name) #, k = 20)
        self.device = torch.device('cuda')
        self.batch_size = batch_size
        self.model_name = model_name

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            print('Start Epoch #', epoch)
            train_loss = self.train_epoch(epoch)
            loss, recall, mrr = self.evaluation.eval(self.eval_data, self.batch_size)
            print("Epoch: {}, train loss: {:.4f}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, train_loss, loss, recall, mrr, time.time() - st))
         
    def train_epoch(self, epoch):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden

        hidden = self.model.init_hidden()
        #Create dataloader for minibatch parallel training
        dataloader = lib.DataLoader(self.train_data, self.batch_size)

        if self.model_name == 'RNN':
          for ii, (input, times,  target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
              input = input.to(self.device)
              target = target.to(self.device)
              self.optim.zero_grad()
              hidden = reset_hidden(hidden, mask).detach()
              logit, hidden = self.model(input, hidden)
              # output sampling
              logit_sampled = logit[:, target.view(-1)]
              loss = self.loss_func(logit_sampled)
              losses.append(loss.item())
              loss.backward()
              self.optim.step()
        else: #using improved RNN
          for ii, (input, times, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
              input = input.to(self.device)
              times = times.to(self.device)
              target = target.to(self.device)
              self.optim.zero_grad()
              hidden = reset_hidden(hidden, mask).detach()
              logit, hidden = self.model(input, times, hidden)
              # output sampling
              logit_sampled = logit[:, target.view(-1)]
              loss = self.loss_func(logit_sampled)
              losses.append(loss.item())
              loss.backward()
              self.optim.step()

        mean_losses = np.mean(losses)
        return mean_losses
