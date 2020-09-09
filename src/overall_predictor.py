import sys
import os
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

activation_function_choice = {
    'relu': torch.nn.ReLU(inplace=True),
    'leaky_relu': torch.nn.LeakyReLU(),
    'tanh': torch.nn.Tanh()
}

optimizer_choice = {
    'adam': torch.optim.Adam
}

class overall_network(torch.nn.Module):
    def __init__(self, config):
        # base class initialize
        super(overall_network, self).__init__()
        
        # parameters from config
        self.games = config['game_number']
        self.bp_order = config['ban_pick_order']
        self.use_gpu = config['train_use_gpu']
        self.embedding_size = config['embedding_size']
        self.activation_function = activation_function_choice[config['activation_function']]
        self.optimizer = optimizer_choice[config['optimizer']]
        self.pick_block_dim = config['pick_block_dim']                                         
        self.ban_block_dim = config['ban_block_dim']
        self.current_block_dim = config['current_block_dim']
        self.hidden_block_dim = config['hidden_block_dim']
        self.policy_output_dim = config['policy_output_dim']
        self.value_output_dim = config['value_output_dim']
        self.class_numbers = config['hero_numbers']
        self.ban_considered = config['ban_considered']
        self.libtorch_use_gpu = config['libtorch_use_gpu']       

        self.set_weights()

    def generate_linear_weights(self, container, input_size, dims):
        in_size = input_size
        for dim in dims:
            container.append(torch.nn.Linear(in_size, dim))
            in_size = dim

    def set_weights(self):
        # set weights
        
        # embedding_weight. ***note that the last weight is fixed to 0***
        self.emb_np_weight = np.load('./models/emb_weight.npy')
        temp = np.zeros((1, self.embedding_size))
        self.emb_np_weight = np.concatenate([self.emb_np_weight, temp], axis=0)
        self.emb_np_weight = torch.Tensor(self.emb_np_weight)
        self.embedding_weight = torch.nn.Embedding(self.class_numbers+1, self.embedding_size, _weight=self.emb_np_weight)
        
        # calculate bp orders and numbers
        self.bp_size, self.take_out_dims = self.calculate_total_bp_number()
        self.input_tensor_total_size = self.games*self.bp_size['b'] + self.games*self.bp_size['p'] + self.games * 2
        
        # generate current block weights
        self.current_block_weights = torch.nn.ModuleList()
        input_size = self.embedding_size * self.bp_size['p']
        self.generate_linear_weights(self.current_block_weights, input_size, self.current_block_dim)

        # generate ban feature weights
        self.ban_block_weights = torch.nn.ModuleList()
        input_size = self.embedding_size * self.bp_size['b'] // 2
        self.generate_linear_weights(self.ban_block_weights, input_size, self.ban_block_dim)

        # generate pick feature weights
        self.pick_block_weights = torch.nn.ModuleList()
        input_size = self.embedding_size * self.bp_size['p'] // 2
        self.generate_linear_weights(self.pick_block_weights, input_size, self.pick_block_dim) 

        # generate hidden weights
        self.hidden_weight = torch.nn.ModuleList()
        # ban + pick + winlose + current camp
        if self.ban_considered:
            input_size = self.current_block_dim[-1] + self.ban_block_dim[-1] * self.games * 2 + self.pick_block_dim[-1] * self.games * 2 + self.games * 2
        else:
            input_size = self.current_block_dim[-1] + self.pick_block_dim[-1] * self.games * 2 + self.games * 2
        self.generate_linear_weights(self.hidden_weight, input_size, self.hidden_block_dim)

        # generate policy weights
        self.policy_output_weights = torch.nn.ModuleList()
        input_size = self.hidden_block_dim[-1]
        self.policy_output_dim.append(self.class_numbers) # add a layer
        self.generate_linear_weights(self.policy_output_weights, input_size, self.policy_output_dim)

        # generate value weights
        self.value_output_weights = torch.nn.ModuleList()
        input_size = self.hidden_block_dim[-1]
        self.value_output_dim.append(1) # add a layer
        self.generate_linear_weights(self.value_output_weights, input_size, self.value_output_dim)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        print('weights generated done.')

    def forward(self, inputs):
        # input tensor: [batch_size, best of, bp_total_num]
        if self.ban_considered:
            inputs_current = inputs[:, 0:self.bp_size['p']]
            end = self.bp_size['p']
            inputs_ban_feature = inputs[:, end:self.games*self.bp_size['b']]
            end = self.games*self.bp_size['b']
            inputs_pick_feature = inputs[:, end:end+self.games*self.bp_size['p']]
            end = end + self.games*self.bp_size['p']
            inputs_other_feature = inputs[:, end:end + self.games * 2]
        else:
            inputs_current = inputs[:, 0:self.bp_size['p']]
            end = self.bp_size['p']
            inputs_pick_feature = inputs[:, end:end+self.games*self.bp_size['p']]
            end = end + self.games*self.bp_size['p']
            inputs_other_feature = inputs[:, end:end + self.games * 2]
        # longint to float for concat
        inputs_other_feature = inputs_other_feature.float()
        
        # current outputs
        inputs_current = self.embedding_weight(inputs_current)
        current_outputs = torch.reshape(inputs_current, [-1, self.embedding_size * self.bp_size['p']])
        for layer in self.current_block_weights:
            current_outputs = self.activation_function(layer(current_outputs))

        # first embedding for ban and pick features
        inputs_pick_feature = self.embedding_weight(inputs_pick_feature)
       
        if self.ban_considered:
            inputs_ban_feature = self.embedding_weight(inputs_ban_feature)
        
            # deal with ban feature
            ban_outputs = torch.reshape(inputs_ban_feature, [-1, self.embedding_size * self.bp_size['b'] // 2])
            for layer in self.ban_block_weights:
                ban_outputs = self.activation_function(layer(ban_outputs))
            ban_outputs = torch.reshape(ban_outputs, [-1, self.games*2*self.ban_block_dim[-1]])
        
        # deal with pick feature
        pick_outputs = torch.reshape(inputs_pick_feature, [-1, self.embedding_size * self.bp_size['p'] // 2])
        for layer in self.pick_block_weights:
            pick_outputs = self.activation_function(layer(pick_outputs))
        pick_outputs = torch.reshape(pick_outputs, [-1, self.games*2*self.pick_block_dim[-1]])
        
        # deal with hidden layer
        if self.ban_considered:
            hidden_inputs = torch.cat([current_outputs, ban_outputs, pick_outputs, inputs_other_feature], 1)
        else:
            hidden_inputs = torch.cat([current_outputs, pick_outputs, inputs_other_feature], 1)
        for layer in self.hidden_weight:
            hidden_inputs = self.activation_function(layer(hidden_inputs))
        hidden_outputs = hidden_inputs

        # get value
        value_input = hidden_outputs
        for (layer, dim) in zip(self.value_output_weights, self.value_output_dim):
            if dim != 1:
                value_input = self.activation_function(layer(value_input))
            else:
                value_input = layer(value_input)
        value_output = self.sigmoid(value_input)

        # get policy
        policy_input = hidden_outputs
        for (layer, dim) in zip(self.policy_output_weights, self.policy_output_dim):
            if dim != self.class_numbers:
                policy_input = self.activation_function(layer(policy_input))
            else:
                policy_input = layer(policy_input)
        policy_output = self.softmax(policy_input)
        # print('forward done.')
        
        return value_output, policy_output
        
    def calculate_total_bp_number(self):
        splited_bp_order = self.bp_order.split('-')
        ans = {'b':0, 'p':0, 'total':0}
        take_out_dims = {'b': [[], []],     # ban dims for two players
                         'p': [[], []]}     # pick dims for two players
        now_player = 0
        now_order = 0
        for item in splited_bp_order:
            number=1 if item[1]=='1' else 2
            ans[item[0]] += number
            for i in range(number):
                take_out_dims[item[0]][now_player].append(now_order)
                now_order += 1
            now_player = (now_player+1) % 2
        ans['total'] = ans['b'] + ans['p']
        return ans, take_out_dims

class total_loss(torch.nn.Module):
    # concat losses
    def __init__(self):
        super(total_loss, self).__init__()

    def forward(self, pi, vs, target_pi, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_pi * torch.log(pi), 1))

        return value_loss + policy_loss

class overall_predictor():
    def __init__(self, config):
        self.network  = overall_network(config)
        print(self.network) 
        # train network with multi gpu ***only training***
        # self.network = torch.nn.DataParallel(self.network)
        
        # get parameters from config
        self.l2 = config['l2']
        self.learning_rate = config['learning_rate']
        self.training_epoches = config['training_epoches']
        self.training_batch_size = config['training_batch_size']
        self.libtorch_use_gpu = config['libtorch_use_gpu']

        # if use cuda
        if config['train_use_gpu']:
            self.network.cuda()
        
        # others
        self.optimizer = optimizer_choice[config['optimizer']](self.network.parameters(),
                                          lr=self.learning_rate, 
                                          weight_decay=self.l2)
        self.total_loss = total_loss()
    
    def parallel(self):
        self.network = torch.nn.DataParallel(self.network, device_ids=[0, 1, 2, 3, 4, 5, 6, 7])

    def train(self, example_buffer, batch_size, epoches):
        if batch_size == None:
            batch_size = self.training_batch_size
        for epo in range(1, epoches + 1):
            # label the parameters with trainable
            self.network.train()

            # sample data from buffer
            train_data = random.sample(example_buffer, batch_size)
            
            # deal with train data
            state_batch, p_batch, v_batch = list(zip(*train_data))
            state_batch = torch.Tensor(state_batch).long().cuda()
            p_batch = torch.Tensor(p_batch).cuda()
            v_batch = torch.Tensor(v_batch).cuda()
            # optimizer set to zero
            self.optimizer.zero_grad()

            # inference and calculate loss
            value, pi = self.network(state_batch)
            loss = self.total_loss(pi, value, p_batch, v_batch)

            # backward and optimize
            loss.backward()
            self.optimizer.step()

            # log
            print("EPOCH: {}, LOSS: {}".format(epo, loss.item()))

    def load_model(self, folder='models', file_name='checkpoint'):
        # load model(only network ***but MCTS tree not included***)
        
        # folder = 'models'
        filepath = os.path.join(folder, file_name)
        state = torch.load(filepath)
        self.network.load_state_dict(state['network'])
        self.optimizer.load_state_dict(state['optim'])

    def save_model(self, folder='models', file_name='checkpoint'):
        # save model(only network ***but MCTS tree not included***)
        
        # create folder if not exists
        # folder = 'models'
        if not os.path.exists(folder):
            os.mkdir(folder)

        file_path = os.path.join(folder, file_name)
        state = {'network':self.network.module.state_dict(), 'optimizer':self.optimizer.state_dict()}
        torch.save(state, file_path)

        # the model for C++ should be 
        file_path += '.pt'
        self.network.eval()

        # why? it seems that torchscript is for C++
        if self.libtorch_use_gpu:
            self.network.cuda()
            example = torch.ones((1, self.network.module.input_tensor_total_size), dtype=torch.long).cuda()
        else:
            self.network.cpu()
            example = torch.ones((1, self.network.module.input_tensor_total_size), dtype=torch.long).cpu()

        traced_script_module = torch.jit.trace(self.network.module, example)
        traced_script_module.save(file_path)

        if self.network.module.use_gpu:
            self.network.cuda()
        else:
            self.network.cpu()

    


    
