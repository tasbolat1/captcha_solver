import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import string
from PIL import Image

all_chars = string.ascii_uppercase + '0123456789'
total_chars = len(all_chars)
captcha_length = 8

encoding_dict = {l:e for e,l in enumerate(all_chars)}
decoding_dict = {e:l for l,e in encoding_dict.items()}

class CaptchaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        'Initialize'
        self.path = data_dir
        self.samples = np.genfromtxt(f'{self.path}/all.txt', delimiter=' ', dtype='str')
        
    def to_onehot(self, label):
        onehot = np.zeros((total_chars, captcha_length))
        for column, letter in enumerate(label):
            onehot[encoding_dict[letter], column] = 1
        return onehot.reshape(-1)
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sample_fname = self.samples[index, 0]
        
        if 'original' in self.path:
            im = Image.open(f'{self.path}/some{sample_fname}.png')
            im = torch.FloatTensor( np.array( im.convert('L') ) ) / 255.0
        else:
            im = torch.FloatTensor( np.load(f'{self.path}/{sample_fname}.npy') )/255.0 # normalized
        label = str(self.samples[index, 1])
        target = torch.FloatTensor( self.to_onehot(label) )
        
        if im.shape[0] != 30:
            im = im[:30,:]
        if im.shape[1] != 140:
            im = im[:,:140]
        
        return im, label, target
    
    
class CaptchaDataset2(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        'Initialize'
        self.path = data_dir
        self.samples = np.genfromtxt(f'{self.path}/all.txt', delimiter=' ', dtype='str')
        #self.samples = self.samples[:64,:]
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.samples)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        sample_fname = self.samples[index, 0]
        
        if 'original' in str(self.path):
            im = Image.open(f'{self.path}/some{sample_fname}.png')
            im = torch.FloatTensor( np.array( im.convert('L') ) ) / 255.0
        else:
            im = torch.FloatTensor( np.load(f'{self.path}/{sample_fname}.npy') )/255.0 # normalized
        label = str(self.samples[index, 1])
        
        if im.shape[0] != 30:
            im = im[:30,:]
        if im.shape[1] != 140:
            im = im[:,:140]
            
        im = im.unsqueeze(0)
        
        label_sequence = [encoding_dict[c] for c in label]
        
        while len(label_sequence) < 8:
            label_sequence.append(total_chars)
        
        return im, torch.FloatTensor(label_sequence)#label
    
def plot_sample(x):
    x = x.squeeze()
    plt.imshow(x, cmap='Greys_r')
    plt.show()
    
def decode_from_output(output_vector):
    output_vector = output_vector.reshape(total_chars, captcha_length)
    output_string = ''
    for i in range(captcha_length):
        character_onehot = np.argmax(output_vector[:,i])
        output_string += decoding_dict[character_onehot]
    return output_string


criteria = nn.MSELoss()
def compute_loss(x1,x2):
    batch_size,_ = x1.shape
    x1 = x1.reshape(batch_size, total_chars, captcha_length)
    x2 = x2.reshape(batch_size, total_chars, captcha_length)
    total_loss = 0
    for i in range(captcha_length):
        total_loss += criteria(x1[:,i], x2[:,i])
    return total_loss

def accuracy(x1, x2):
    bs, _ = x1.shape
    correct = 0
    for i in range(bs):
        predicted_label = decode_from_output(x1[i,:].detach().cpu().numpy())
        true_label = x2[i]
        
        if predicted_label == true_label:
            correct += 1
    return correct

def get_string_label(x):
    x_str = ''
    for z in x:
        if z == 36:
            continue
        x_str += decoding_dict[int(z)] 
    return x_str

width = 140

class StackedLSTM(nn.Module):
    def __init__(self, input_size=30, output_size=total_chars+1, hidden_size=512, num_layers=3):
        super(StackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
    def forward(self, inputs, hidden):
        batch_size, seq_len, input_size = inputs.shape
        outputs, hidden = self.lstm(inputs, hidden)
        outputs = self.dropout(outputs)
        outputs = torch.stack([self.fc(outputs[i]) for i in range(width)])
        outputs = F.log_softmax(outputs, dim=2)
        return outputs, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data 
        return (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
    
    
class CaptchaModel(nn.Module):
    def __init__(self, input_size=30, output_size=total_chars+1, hidden_size=64, num_layers=2):
        super(CaptchaModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,2))
        
        self.linear1 = nn.Linear(448, hidden_size)
        
        
        self.drop1 = nn.Dropout()
        
        self.gru = nn.GRU(hidden_size, int(hidden_size/2), bidirectional=True, num_layers=num_layers, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs):
        batch_size, c, seq_len, input_size = inputs.shape
        #print('input', inputs.size())
        
        # convolution part
        x = F.relu(self.conv1(inputs))  # 128, 30, 140
        # print('conv1', x.size())
        x = self.max_pool1(x) # 128, 15, 70
        # print('max1', x.size())
        
        x = F.relu(self.conv2(x)) # 64, 15, 70
        # print('conv2', x.size())
        x = self.max_pool2(x) # 64, 7, 35
        # print('max2', x.size())
        
        # rnn part
        x = x.permute(0,3,1,2) # 35, 64, 7
        # print('permute ', x.size())
        x = x.view(batch_size, x.size(1), -1)
        # print('view ', x.size())
        x = self.linear1(x)
        x = self.drop1(x)
        # print('linear1 ', x.size())
        x,_ = self.gru(x)
        
        x = x.permute(1,0,2)
        # print('gru ', x.size())
        a, bs, b = x.size()
        # x = self.fc(x)
        # print('fc', x.size())
        
        outputs = torch.stack([self.fc(x[i]) for i in range(a)])
        outputs = F.log_softmax(outputs, dim=2)
        
        #print('outputs ', outputs.size())
        return outputs
    
    
class CaptchaModelBNLight(nn.Module):
    def __init__(self, input_size=30, output_size=total_chars+1, hidden_size=64, num_layers=2):
        super(CaptchaModelBNLight, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.batchnorm1 = nn.BatchNorm2d(256)
        
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.batchnorm2 = nn.BatchNorm2d(128)
        
        self.linear1 = nn.Linear(896, hidden_size)
        
        
        self.drop1 = nn.Dropout()
        
        self.gru = nn.GRU(hidden_size, int(hidden_size/2), bidirectional=True, num_layers=num_layers, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs):
        batch_size, c, seq_len, input_size = inputs.shape
        #print('input', inputs.size())
        
        # convolution part
        x = F.relu(self.batchnorm1( self.conv1(inputs)) ) # 128, 30, 140
        # print('conv1', x.size())
        x = self.max_pool1(x) # 128, 15, 70
        # print('max1', x.size())
        
        x = F.relu(self.batchnorm2( self.conv2(x)) )# 64, 15, 70
        # print('conv2', x.size())
        x = self.max_pool2(x) # 64, 7, 35
        # print('max2', x.size())
        
        # rnn part
        x = x.permute(0,3,1,2) # 35, 64, 7
        # print('permute ', x.size())
        x = x.view(batch_size, x.size(1), -1)
        # print('view ', x.size())
        x = self.linear1(x)
        x = self.drop1(x)
        # print('linear1 ', x.size())
        x,_ = self.gru(x)
        
        x = x.permute(1,0,2)
        # print('gru ', x.size())
        a, bs, b = x.size()
        # x = self.fc(x)
        # print('fc', x.size())
        
        outputs = torch.stack([self.fc(x[i]) for i in range(a)])
        outputs = F.log_softmax(outputs, dim=2)
        
        #print('outputs ', outputs.size())
        return outputs

    
class CaptchaModelBN(nn.Module):
    def __init__(self, input_size=30, output_size=total_chars+1, hidden_size=64, num_layers=2):
        super(CaptchaModelBN, self).__init__()
        self.conv1 = nn.Conv2d(1, 512, kernel_size=3, padding=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.batchnorm1 = nn.BatchNorm2d(512)
        
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.batchnorm2 = nn.BatchNorm2d(256)
        
        self.linear1 = nn.Linear(896*2, hidden_size)
        
        
        self.drop1 = nn.Dropout()
        
        self.gru = nn.GRU(hidden_size, int(hidden_size/2), bidirectional=True, num_layers=num_layers, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, inputs):
        batch_size, c, seq_len, input_size = inputs.shape
        #print('input', inputs.size())
        
        # convolution part
        x = F.relu(self.batchnorm1( self.conv1(inputs)) ) # 128, 30, 140
        # print('conv1', x.size())
        x = self.max_pool1(x) # 128, 15, 70
        # print('max1', x.size())
        
        x = F.relu(self.batchnorm2( self.conv2(x)) )# 64, 15, 70
        # print('conv2', x.size())
        x = self.max_pool2(x) # 64, 7, 35
        # print('max2', x.size())
        
        # rnn part
        x = x.permute(0,3,1,2) # 35, 64, 7
        # print('permute ', x.size())
        x = x.view(batch_size, x.size(1), -1)
        # print('view ', x.size())
        x = self.linear1(x)
        x = self.drop1(x)
        # print('linear1 ', x.size())
        x,_ = self.gru(x)
        
        x = x.permute(1,0,2)
        # print('gru ', x.size())
        a, bs, b = x.size()
        # x = self.fc(x)
        # print('fc', x.size())
        
        outputs = torch.stack([self.fc(x[i]) for i in range(a)])
        outputs = F.log_softmax(outputs, dim=2)
        
        #print('outputs ', outputs.size())
        return outputs
