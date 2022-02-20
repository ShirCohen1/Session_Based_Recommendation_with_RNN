from torch import nn
import torch
import numpy as np

class GRU4REC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, final_act='tanh',
                 dropout_hidden=.5, dropout_input=0, batch_size=50, embedding_dim=-1, use_cuda=False):
        super(GRU4REC, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.onehot_buffer = self.init_emb()
        self.h2o = nn.Linear(hidden_size, output_size)
        self.create_final_activation(final_act)
        if self.embedding_dim != -1:
            self.look_up = nn.Embedding(input_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        else:
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def init_model(self, sigma):
        if sigma is not None:
          for p in self.parameters():
              if sigma != -1 and sigma != -2:
                  sigma = sigma
                  p.data.uniform_(-sigma, sigma)
              elif len(list(p.size())) > 1:
                  sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
                  if sigma == -1:
                      p.data.uniform_(-sigma, sigma)
                  else:
                      p.data.uniform_(0, sigma)
    
    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input, hidden):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.

        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''

        if self.embedding_dim == -1:
          #this is where embedding of item occurs, want to add here item time spent
            embedded = self.onehot_encode(input)
            if self.training and self.dropout_input > 0: embedded = self.embedding_dropout(embedded)
            embedded = embedded.unsqueeze(0)
        else:
            embedded = input.unsqueeze(0)
            embedded = self.look_up(embedded)

        output, hidden = self.gru(embedded, hidden) #(num_layer, B, H)
        output = output.view(-1, output.size(-1))  #(B,H)
        logit = self.final_activation(self.h2o(output))

        return logit, hidden

    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer, which will be used for producing the one-hot embeddings efficiently
        '''
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input
        Args:
            input (B,): torch.LongTensor of item indices
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1)
        #option 1- add another dim to one_hot to account for ranges of time spent
        #option 2- add input+time spend into the onehot_buffer which makes the encoding all together (need to increase the dim for this)
        return one_hot


    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)
        mask = mask.to(self.device)
        input = input * mask
        return input

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0

        


class GRU4REC_improved(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, final_act='tanh',
                 dropout_hidden=.5, dropout_input=0, batch_size=50, embedding_dim=-1, use_cuda=False):
        super(GRU4REC_improved, self).__init__()
        self.input_size = input_size + 1
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout_hidden = dropout_hidden
        self.dropout_input = dropout_input
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.onehot_buffer = self.init_emb()
        self.h2o = nn.Linear(hidden_size, output_size)
        self.create_final_activation(final_act)
        if self.embedding_dim != -1:
            self.look_up = nn.Embedding(input_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        else:
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def init_model(self, sigma):
        if sigma is not None:
          for p in self.parameters():
              if sigma != -1 and sigma != -2:
                  sigma = sigma
                  p.data.uniform_(-sigma, sigma)
              elif len(list(p.size())) > 1:
                  sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
                  if sigma == -1:
                      p.data.uniform_(-sigma, sigma)
                  else:
                      p.data.uniform_(0, sigma)
                      
    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    def forward(self, input, time, hidden):
        '''
        Args:
            input (B,): a batch of item indices from a session-parallel mini-batch.
            target (B,): torch.LongTensor of next item indices from a session-parallel mini-batch.

        Returns:
            logit (B,C): Variable that stores the logits for the next items in the session-parallel mini-batch
            hidden: GRU hidden state
        '''

        if self.embedding_dim == -1:
          #this is where embedding of item occurs, want to add here item time spent
            embedded = self.onehot_encode(input)
            embedded = self.add_time_to_onehotvector(embedded, time)
            if self.training and self.dropout_input > 0: embedded = self.embedding_dropout(embedded)
            embedded = embedded.unsqueeze(0)
        else:
            embedded = input.unsqueeze(0)
            embedded = self.look_up(embedded)

        output, hidden = self.gru(embedded, hidden) #(num_layer, B, H)
        output = output.view(-1, output.size(-1))  #(B,H)
        logit = self.final_activation(self.h2o(output))

        return logit, hidden

    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer, which will be used for producing the one-hot embeddings efficiently
        '''
        onehot_buffer = torch.FloatTensor(self.batch_size, self.output_size)
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def onehot_encode(self, input):
        """
        Returns a one-hot vector corresponding to the input
        Args:
            input (B,): torch.LongTensor of item indices
            buffer (B,output_size): buffer that stores the one-hot vector
        Returns:
            one_hot (B,C): torch.FloatTensor of one-hot vectors
        """
        self.onehot_buffer.zero_()
        index = input.view(-1, 1)
        one_hot = self.onehot_buffer.scatter_(1, index, 1)
        #option 1- add another dim to one_hot to account for ranges of time spent
        #option 2- add input+time spend into the onehot_buffer which makes the encoding all together (need to increase the dim for this)
        return one_hot

    def add_time_to_onehotvector(self, encoding, times):
        return torch.cat((encoding, times.reshape(self.batch_size,-1)),1)  
    
    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)
        mask = mask.to(self.device)
        input = input * mask
        return input

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        return h0        