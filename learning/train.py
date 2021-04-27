#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from utils import CaptchaDataset2, StackedLSTM, total_chars

from itertools import groupby
from pathlib import Path

import argparse
from torch.utils.tensorboard import SummaryWriter

# python train.py --epochs 50 --lr 0.00005 --batch_size 128 --hidden_size 1024 --data_dir ../data2 --version 2

parser = argparse.ArgumentParser("Train VT-SNN models.")

parser.add_argument(
    "--epochs", type=int, help="Number of epochs.", required=True
)
parser.add_argument("--data_dir", type=str, help="Path to data.", required=True)
parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="Path for saving checkpoints.",
    default=".",
)

parser.add_argument("--lr", type=float, help="Learning rate.", required=True)
parser.add_argument("--hidden_size", type=int, help="Size of hidden layer.", required=True)
parser.add_argument("--batch_size", type=int, help="Batch Size.", required=True)
parser.add_argument("--version", type=int, help="Version.", required=True)


args = parser.parse_args()
writer = SummaryWriter(".")

batch_size = args.batch_size #64
lr = args.lr # 0.0001
epochs = args.epochs # 1000
hidden_size = args.hidden_size # 512

train_dataset = CaptchaDataset2(Path(args.data_dir)/ f'train')
val_dataset = CaptchaDataset2(Path(args.data_dir)/ f'test')
test_dataset = CaptchaDataset2(Path(args.data_dir)/ f'original')


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


BLANK_LABEL = total_chars
    
net = StackedLSTM(hidden_size=hidden_size).to(device)



optimizer = torch.optim.Adam(net.parameters(), lr=lr)
criterion = nn.CTCLoss(blank=BLANK_LABEL)


for epoch in range(1, epochs+1):
  # set network to training phase
  net.train()
  
  train_loss, train_correct, train_total = 0, 0, 0
  #batch_size = args.batch_size
  #h = net.init_hidden(batch_size)
  
  # for each batch of training examples
  for batch_index, (inputs, targets) in enumerate(train_loader):
    
      inputs = inputs.to(device)
      
      
      batch_size, channels, height, width = inputs.shape
      #print(batch_size, channels, height, width)
      h = net.init_hidden(batch_size)
      h = tuple([each.data for each in h])
      # reshape inputs: NxCxHxW -> WxNx(HxC)
      inputs = (inputs
                .permute(3, 0, 2, 1)
                .contiguous()
                .view((width, batch_size, -1)))
              
      optimizer.zero_grad()  # zero the parameter gradients
      outputs, h = net(inputs, h)  # forward pass
      
      #print(outputs.shape)

      # compare output with ground truth
      input_lengths = torch.IntTensor(batch_size).fill_(width)
      target_lengths = torch.IntTensor([len(t) for t in targets])
      #print(outputs.shape, targets.shape, input_lengths.shape, target_lengths.shape)
      loss = criterion(outputs, targets, input_lengths, target_lengths)

      loss.backward()  # backpropagation
      #nn.utils.clip_grad_norm_(net.parameters(), 10)  # clip gradients
      optimizer.step()  # update network weights
      
      # record statistics
      prob, max_index = torch.max(outputs, dim=2)
      train_loss += loss.item()
      train_total += len(targets)

      for i in range(batch_size):
          raw_pred = list(max_index[:, i].cpu().numpy())
          #print(len(raw_pred))
          pred = [c for c, _ in groupby(raw_pred) if c != BLANK_LABEL]
          target = list(targets[i].cpu().numpy())
          if pred == target:
              train_correct += 1

      # print statistics every 10 batches
      if (batch_index + 1) % 200 == 0:
          print(f'Epoch {epoch }/{epochs}, ' +
                f'Batch {batch_index + 1}/{len(train_loader)}, ' +
                f'Train Loss: {(train_loss/1):.5f}, ' +
                f'Train Accuracy: {(train_correct/train_total):.5f}')
          writer.add_scalar("loss/train", train_loss, epoch)
          writer.add_scalar("acc/train", train_correct / train_total, epoch)
          
          train_loss, train_correct, train_total = 0, 0, 0
          
  
  # validation
  net.eval()
  val_loss, val_correct, val_total = 0, 0, 0
  
  # for each batch of training examples
  for batch_index, (inputs, targets) in enumerate(val_loader):
    inputs = inputs.to(device)
    batch_size, channels, height, width = inputs.shape
    h = net.init_hidden(batch_size)
    h = tuple([each.data for each in h])
    
    
    
    # reshape inputs: NxCxHxW -> WxNx(HxC)
    inputs = (inputs
              .permute(3, 0, 2, 1)
              .contiguous()
              .view((width, batch_size, -1)))
    
    
            
    outputs, h = net(inputs, h)  # forward pass
    input_lengths = torch.IntTensor(batch_size).fill_(width)
    target_lengths = torch.IntTensor([len(t) for t in targets])
    
    loss = criterion(outputs, targets, input_lengths, target_lengths)
    
    # record statistics
    prob, max_index = torch.max(outputs, dim=2)
    val_loss += loss.item()
    val_total += len(targets)

    for i in range(batch_size):
        raw_pred = list(max_index[:, i].cpu().numpy())
        #print(len(raw_pred))
        pred = [c for c, _ in groupby(raw_pred) if c != BLANK_LABEL]
        target = list(targets[i].cpu().numpy())
        if pred == target:
            val_correct += 1

      
  print(f'Epoch {epoch }/{epochs}, ' +
        f'Val Loss: {(val_loss/1):.5f}, ' +
        f'Val Accuracy: {(val_correct/val_total):.5f}')
  writer.add_scalar("loss/val", val_loss, epoch)
  writer.add_scalar("acc/val", val_correct / val_total, epoch)

  val_loss, val_correct, val_total = 0, 0, 0
  
  # test
  net.eval()
  test_loss, test_correct, test_total = 0, 0, 0
  
  # for each batch of training examples
  for batch_index, (inputs, targets) in enumerate(test_loader):
    inputs = inputs.to(device)
    batch_size, channels, height, width = inputs.shape
    h = net.init_hidden(batch_size)
    h = tuple([each.data for each in h])
    
    
    
    # reshape inputs: NxCxHxW -> WxNx(HxC)
    inputs = (inputs
              .permute(3, 0, 2, 1)
              .contiguous()
              .view((width, batch_size, -1)))
            
    outputs, h = net(inputs, h)  # forward pass
    

    # compare output with ground truth
    input_lengths = torch.IntTensor(batch_size).fill_(width)
    target_lengths = torch.IntTensor([len(t) for t in targets])
    
    loss = criterion(outputs, targets, input_lengths, target_lengths)

    
    # record statistics
    prob, max_index = torch.max(outputs, dim=2)
    test_loss += loss.item()
    test_total += len(targets)

    for i in range(batch_size):
        raw_pred = list(max_index[:, i].cpu().numpy())
        #print(len(raw_pred))
        pred = [c for c, _ in groupby(raw_pred) if c != BLANK_LABEL]
        target = list(targets[i].cpu().numpy())
        if pred == target:
            test_correct += 1
    
  print(f'Epoch {epoch }/{epochs}, ' +
        f'Test Loss: {(test_loss/1):.5f}, ' +
        f'Test Accuracy: {(test_correct/test_total):.5f}')

  writer.add_scalar("loss/test", test_loss, epoch)
  writer.add_scalar("acc/test", test_correct / test_total, epoch)

  test_loss, test_correct, test_total = 0, 0, 0

  torch.save(net.state_dict(), f'models/v{args.version}_{epoch}.pt')

print('Done!')
