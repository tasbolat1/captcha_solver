import torch
from torch import nn, optim
import torch.nn.functional as F


class CRNN(nn.Module):
    
    def __init__(self, h, w, outputs, num_channels=1, num_layers=2):
        super(CRNN, self).__init__()
        
        self.conv1 = nn.Conv2d(num_channels, 512, kernel_size=3, padding=(1,1))
        self.maxpool1 = nn.MaxPool2d(2)
        self.batchnorm1 = nn.BatchNorm2d(512)
        
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=(1,1))
        self.maxpool2 = nn.MaxPool2d(2)
        self.batchnorm2 = nn.BatchNorm2d(256)
        
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=(1,1))
        self.maxpool3 = nn.MaxPool2d(2)
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 64, kernel_size=3, padding=(1,1))
        self.maxpool4 = nn.MaxPool2d(2)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=(1,1))
        self.maxpool5 = nn.MaxPool2d(2)
        self.batchnorm5 = nn.BatchNorm2d(64)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.output = nn.Linear(245*64, outputs)

        #self.gru = nn.GRU(64, 32, num_layers,
                                #batch_first=True,
                                #bidirectional=True, dropout=0.25)

        #self.output = nn.Linear(64, outputs)
        
    def forward(self, input):
        bs, c, h, w = input.size()
        x  = F.leaky_relu(self.batchnorm1(self.conv1(input)))
        x  = self.maxpool1(x)
        x  = F.leaky_relu(self.batchnorm2(self.conv2(x)))
        
        x  = F.leaky_relu(self.batchnorm3(self.conv3(x)))
        conv  = self.maxpool2(x) # 1, 64, 18, 75
        
        #print(conv.shape)
        x  = F.leaky_relu(self.batchnorm4(self.conv4(x)))
        conv  = self.maxpool2(x) # 1, 64, 18, 75
        
        x  = F.leaky_relu(self.batchnorm5(self.conv5(x)))
        conv  = self.maxpool2(x)
        conv = conv.view(bs, -1)
        #print(conv.shape)
        x = F.relu( self.output(conv) )
        out = self.dropout(x)
        
        #print(x.shape)
        
        # gru features
        #outs, _ = self.gru(x)
        
        #out = F.leaky_relu( self.output( outs.reshape(bs,  -1) ) )
       
        return out 
        
        
        
        
    
    
        
        
        
        
    
    