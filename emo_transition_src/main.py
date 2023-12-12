# coding = utf-8
from dataload import load_data
from model import Emo_Generation
from train import train_model
import argparse
import model

# CONFIG
DATA_PATH = 'data/Dyadic_PELD.tsv'

parser = argparse.ArgumentParser(description='')
args = parser.parse_args()

# Load the parameter values
args.mode         = 3
args.Senti_or_Emo = 'Emotion'
args.loss_function= 'Focal' # CE or MSE or Focal
args.base         = 'RoBERTa'
args.device       = 0
args.SEED         = 42
args.MAX_LEN      = 256
args.batch_size   = 16
args.lr           = 1e-4
args.adam_epsilon = 1e-8
args.epochs       = 50

args.result_name  = args.Senti_or_Emo + '_Mode_' + \
        str(args.mode) + '_' + args.loss_function + \
        '_Epochs_' + str(args.epochs) + '.csv'

## LOAD DATA
train_length, train_dataloader, valid_dataloader, test_dataloader = load_data(args, DATA_PATH)
args.train_length = train_length

## TRAIN THE MODEL
if args.base == 'RoBERTa':
        model = Emo_Generation.from_pretrained('roberta-base', mode=args.mode).cuda(args.device)
else:
        model = Emo_Generation.from_pretrained('bert-base-uncased', mode=args.mode).cuda(args.device)

train_model(model, args, train_dataloader, valid_dataloader, test_dataloader)








