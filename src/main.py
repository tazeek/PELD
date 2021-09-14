# coding = utf-8
import pandas as pd
import numpy as np
import torch
import argparse
import os
import datetime
import traceback
import model



# CONFIG
DATA_PATH = '../Dyadic_PELD.tsv'

parser = argparse.ArgumentParser(description='')
args = parser.parse_args()

args.mode         = 3
args.Senti_or_Emo = 'Emotion'
args.loss_function = 'CE' # CE or MSE or Focal
args.device       = 0
args.SEED         = 42
args.MAX_LEN      = 64 
args.batch_size   = 64
args.lr           = 1e-5
args.adam_epsilon = 1e-8
args.epochs       = 50
args.result_name  = args.Senti_or_Emo+'_Mode_'+str(args.mode)+'_'+args.loss_function+'_Epochs_'+str(args.epochs)+'.csv'

## LOAD DATA
from dataload import load_data
train_length, train_dataloader, valid_dataloader, test_dataloader = load_data(args, DATA_PATH)
args.train_length = train_length

## TRAIN THE MODEL
from model import Emo_Generation
from transformers import RobertaConfig, RobertaModel, PreTrainedModel
from train import train_model

model = Emo_Generation.from_pretrained('roberta-base', mode=args.mode).cuda(args.device)
train_model(model, args, train_dataloader, valid_dataloader, test_dataloader)







