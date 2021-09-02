import json

#import torch
import numpy as np
from transformers import BertTokenizerFast

UNK_TOKEN = '[UNK]'
CLS_TOKEN = '[CLS]'
EOS_TOKEN = '[SEP]'



class VocabTrans:
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.vocab = {}
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

        # load ...
        self.load_vocab()

    def load_vocab(self):
        with open(self.vocab_path + "vocab.txt", 'r', encoding='utf-8') as f:
            content = f.readlines()
            content = [x.strip('\n') for x in content if len(x.strip('\n')) > 0]
        for i in range(len(content)):
            self.vocab[content[i]] = i
        print('finish load vocabulary ...')

    def trans_sen(self, sentence):
        sen_index = [list(map(self.word2index, sentence.split())) ]
        return self.tokenizer.batch_decode(sen_index)[0]

    def word2index(self, word):
        if word in self.vocab:
            return self.vocab[word]
        return self.vocab[UNK_TOKEN]




def trans_factcc():
    dataset_path = 'data_input/tmp/data-train.jsonl'
    vocab_path = '../checkpoints/torch_bert/old_vocab/' 
    tran = VocabTrans(vocab_path)
    f = open(dataset_path, "r", encoding="utf8")
    lines = f.readlines()
    f.close()
    f = open(dataset_path, "w", encoding="utf8")
    for line in lines:
        item = json.loads(line)
        item["claim"] = tran.trans_sen(item["claim"])
        item["text"] = tran.trans_sen(item["text"])
        item["augmentation_span"] = item["extraction_span"] = (0, len(item["text"].split()))
        print(item)
        f.write(json.dumps(item)+"\n")
    f.close()
        

def trans_summ():
    dataset_path = '/home/LAB/zhuhd/tmp/summ.txt'
    vocab_path = '../checkpoints/torch_bert/old_vocab/' 
    tran = VocabTrans(vocab_path)
    f = open(dataset_path, "r", encoding="utf8")
    lines = f.readlines()
    f.close()
    dataset_path2 = '/home/LAB/zhuhd/tmp/summ2.txt'
    f = open(dataset_path2, "w", encoding="utf8")
    for line in lines:
        line = line.strip()
        line =  tran.trans_sen(line)
        print(line)
        f.write(line+"\n")
    f.close()

if __name__ == "__main__":
    trans_summ()



