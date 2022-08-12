import json
import csv

import pandas as pd
from tqdm import tqdm, trange
import argparse
import random
import math
import sys
import numpy as np
import os
import pdb
import ast
import copy
import codecs
import sys
#csv.field_size_limit(sys.maxsize)

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformers import BartTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BartConfig

from bart_decoder_HHDKS import HHDKS_Enc_np_Dec_p_BartForConditionalGeneration

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from metrics import evaluate_nq
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class GenerationInputExample(object):

    def __init__(self, guid, his, target, doc=None):
        self.guid = guid
        self.his = his
        self.target = target
        self.doc = doc

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures:
    def __init__(self,
                 example_index,
                 his_ids,
                 his_mask,
                 his_len,
                 target_ids,
                 target_labels,
                 target_len,
                 doc_ids,
                 doc_mask,
                 doc_len):

        self.example_index = example_index

        self.his_ids = his_ids
        self.his_mask = his_mask
        self.his_len = his_len

        self.target_ids = target_ids
        self.target_labels = target_labels
        self.target_len = target_len

        self.doc_ids = doc_ids
        self.doc_mask = doc_mask
        self.doc_len = doc_len

class MultiBartQA:
    def __init__(self):
        self.args = self.parse_args()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        self.generator = HHDKS_Enc_np_Dec_p_BartForConditionalGeneration.from_pretrained_multi(self.args, self.args.model_file_path,
                                                                                               use_pretrain=True,
                                                                                               his_num=self.args.his_len)
        self.generator.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(self.args.model_name) # Need to add base to "tokenization_bart.py" when using transformers==2.11.0
        if self.args.use_tb:
            print("using tensorboard")
            self.tb_writer = SummaryWriter(log_dir=self.args.output_dir + '/tb')

    def save(self, num_updates):
        model_to_save = (
            self.generator.module if hasattr(self.generator, "module") else self.generator
        )
        checkpoint = {
            'model': model_to_save.state_dict(),
            'optimizer': self.get_optimizer(),
            'args': self.args
        }
        output_dir = os.path.join(self.args.output_dir, f"checkpoint-{num_updates}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(output_dir, 'model.pt'))

    def parse_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--seed",
                            default=42,
                            type=int,
                            help="Random seed")
        parser.add_argument("--model_name",
                            default='facebook/bart-base',
                            type=str,
                            help="BART model")
        parser.add_argument('--data_dir',
                            type=str,
                            default='data/cmu_dog/',
                            help='path to data_dir')
        parser.add_argument('--output_dir',
                            type=str,
                            default='trained_models/cmu_dog/doha_test/',
                            help='path to save the model')
        parser.add_argument('--log_file_path',
                            type=str,
                            default='trained_models/cmu_dog/doha_test/log.txt',
                            help='Log file')
        parser.add_argument('--model_file_path',
                            type=str,
                            default='./pytorch_model.bin',
                            help='Model file')
        parser.add_argument("--source_max_len",
                            default=512,
                            type=int,
                            help="Max len of source")
        parser.add_argument("--target_max_len",
                            default=128,
                            type=int,
                            help="Max len of target")
        parser.add_argument("--train_batch_size",
                            default=1,
                            type=int,
                            help="Total batch size for training.")
        parser.add_argument("--validation_timing",
                            default=1000,
                            type=int,
                            help="Check dev score after every N updates")
        parser.add_argument("--eval_batch_size",
                            default=2,
                            type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--learning_rate",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--learning_rate2",
                            default=5e-5,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",
                            default=25.0,
                            type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.1,
                            type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int,
                            default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False,
                            default=1.0,
                            type=float)
        parser.add_argument("--do_train",
                            action='store_true',
                            help="Flag to indicate whether to train or not")
        parser.add_argument("--do_eval",
                            action='store_true',
                            help="Flag to indicate whether to train or not")
        parser.add_argument("--do_generate",
                            action='store_true',
                            help="Flag to indicate whether to train or not")
        parser.add_argument('--experiment_type',
                            type=str,
                            default='chat_document',
                            help='Type of input to be fed. Options are '
                                 '[doc_only | chat_document | chat_wizard]')
        parser.add_argument('--continue_train',
                            type=bool,
                            default=False,
                            help='re train or not')
        parser.add_argument('--use_tb',
                            type=bool,
                            default=True,
                            help='use tensorboard or not')
        parser.add_argument('--his_len',
                            type=int,
                            default=3,
                            help='how many history dialogue to use')
        parser.add_argument('--data_set',
                            type=str,
                            default='cmu',
                            help='cmu/wow/d2d/dd')

        return parser.parse_known_args()[0]

    def load_examples(self, data_dir, filename):
        examples = []
        samples = pd.read_csv(data_dir + filename, index_col=0)
        for idx in range(len(samples)):
            his1 = samples.iloc[idx].his1
            his1 = his1 if his1 != "no content" else ""
            his2 = samples.iloc[idx].his2
            his2 = his2 if his2 != "no content" else ""
            his3 = samples.iloc[idx].his3
            his3 = his3 if his3 != "no content" else ""
            try:
                his4 = samples.iloc[idx].his4
                his4 = his4 if his4 != "no content" else ""
                his5 = samples.iloc[idx].his5
                his5 = his5 if his5 != "no content" else ""
            except:
                his4 = ""
                his5 = ""
            his = [his1, his2, his3, his4, his5]
            his = his[:self.args.his_len]
            assert len(his) == self.args.his_len
            tgt = samples.iloc[idx].tgt
            doc = samples.iloc[idx].doc
            examples.append(GenerationInputExample(
                guid=idx,
                his=his,
                target=tgt,
                doc=doc
            ))
        return examples

    def convert_examples_to_features(self, examples):
        config = self.generator.model.config
        features = []
        index = 0

        for e in tqdm(examples, desc='Examples'):
            # Process source information

            all_his = e.his
            all_his_ids, all_his_mask, all_his_len = [], [], []
            his_max_len = int(self.args.source_max_len / self.args.his_len)
            for i, his in enumerate(all_his):
                his = "chat"+str(i+1)+": "+his
                his_tokens = self.tokenizer.tokenize(his)[:his_max_len-2]
                his_ids = [config.bos_token_id] + self.tokenizer.convert_tokens_to_ids(his_tokens) + [config.eos_token_id] # <s> ... </s>
                his_len = len(his_ids)
                his_mask = [1] * his_len

                padding_len = his_max_len - his_len
                his_ids += ([config.pad_token_id] * padding_len)
                his_mask += ([0] * padding_len)

                assert len(his_ids) == his_max_len
                assert len(his_mask) == his_max_len

                all_his_ids.append(his_ids)
                all_his_mask.append(his_mask)
                all_his_len.append(his_len)

            if self.args.experiment_type == 'doc_only':
                document = 'document: ' + e.doc
            elif self.args.experiment_type == 'chat_document':
                document = 'document: ' + e.doc
            elif self.args.experiment_type == 'chat_wizard':
                document = 'document: ' + e.doc
            elif self.args.experiment_type == 'chat+document':
                source = 'chat: ' + " ".join(e.his)
                document = source.strip() + ' document: ' + e.doc
            else:
                print('Unrecongnized argument for experiment type')

            doc_tokens = self.tokenizer.tokenize(document)[:self.args.source_max_len-2]
            doc_ids = [config.bos_token_id] + self.tokenizer.convert_tokens_to_ids(doc_tokens) + [config.eos_token_id] # <s> ... </s>
            doc_len = len(doc_ids)
            doc_mask = [1] * doc_len

            padding_len = self.args.source_max_len - doc_len
            doc_ids += ([config.pad_token_id] * padding_len)
            doc_mask += ([0] * padding_len)

            assert len(doc_ids) == self.args.source_max_len
            assert len(doc_mask) == self.args.source_max_len

            # Process target information

            answer = e.target
            answer_tokens = self.tokenizer.tokenize(answer)[:self.args.target_max_len-1] # -1 for <s> or </s>
            if len(answer_tokens) == 0:
                print(e.target)
                continue
            target_ids = [config.bos_token_id] + self.tokenizer.convert_tokens_to_ids(answer_tokens) # <s> ...
            target_labels = self.tokenizer.convert_tokens_to_ids(answer_tokens) + [config.eos_token_id] # ... </s>
            target_len = len(target_ids)

            padding_len = self.args.target_max_len - target_len
            target_ids += ([config.pad_token_id] * padding_len)
            target_labels += ([-100] * padding_len) # -100 is the default index to be ignored

            assert len(target_ids) == self.args.target_max_len
            assert len(target_labels) == self.args.target_max_len

            f = InputFeatures(
                index,
                all_his_ids,
                all_his_mask,
                all_his_len,
                target_ids,
                target_labels,
                target_len,
                doc_ids,
                doc_mask,
                doc_len
            )
            features.append(f)

            index += 1

        return features

    def init_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if torch.cuda.is_available:
            torch.cuda.manual_seed(self.args.seed)

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.generator.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in self.generator.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

    def get_train_dataloader(self,
                             train_features,
                             train_batch_size):
        all_his_ids = torch.tensor([f.his_ids for f in train_features], dtype=torch.long)
        all_his_mask = torch.tensor([f.his_mask for f in train_features], dtype=torch.long)
        all_his_len = torch.tensor([f.his_len for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_labels = torch.tensor([f.target_labels for f in train_features], dtype=torch.long)
        all_target_len = torch.tensor([f.target_len for f in train_features], dtype=torch.long)
        all_doc_ids = torch.tensor([f.doc_ids for f in train_features], dtype=torch.long)
        all_doc_mask = torch.tensor([f.doc_mask for f in train_features], dtype=torch.long)
        all_doc_len = torch.tensor([f.doc_len for f in train_features], dtype=torch.long)
        train_data = TensorDataset(
            all_his_ids,
            all_his_mask,
            all_his_len,
            all_target_ids,
            all_target_labels,
            all_target_len,
            all_doc_ids,
            all_doc_mask,
            all_doc_len
        )
        train_sampler = RandomSampler(train_data)
        return DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    def get_eval_dataloader(self, dev_features, dev_batch_size):
        all_example_indices = torch.tensor([f.example_index for f in dev_features], dtype=torch.long)
        all_his_ids = torch.tensor([f.his_ids for f in dev_features], dtype=torch.long)
        all_his_mask = torch.tensor([f.his_mask for f in dev_features], dtype=torch.long)
        all_his_len = torch.tensor([f.his_len for f in dev_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in dev_features], dtype=torch.long)
        all_target_labels = torch.tensor([f.target_labels for f in dev_features], dtype=torch.long)
        all_target_len = torch.tensor([f.target_len for f in dev_features], dtype=torch.long)
        all_doc_ids = torch.tensor([f.doc_ids for f in dev_features], dtype=torch.long)
        all_doc_mask = torch.tensor([f.doc_mask for f in dev_features], dtype=torch.long)
        all_doc_len = torch.tensor([f.doc_len for f in dev_features], dtype=torch.long)
        eval_data = TensorDataset(
            all_example_indices,
            all_his_ids,
            all_his_mask,
            all_his_len,
            all_target_ids,
            all_target_labels,
            all_target_len,
            all_doc_ids,
            all_doc_mask,
            all_doc_len
        )
        eval_sampler = SequentialSampler(eval_data)
        return DataLoader(eval_data, sampler=eval_sampler, batch_size=dev_batch_size)
    
    def get_train_batch_data(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        his_ids, his_mask, his_len, target_ids, target_labels, target_len, doc_ids, doc_mask, doc_len = batch

        batch_source_max_len = his_len.max().item()
        batch_target_max_len = target_len.max().item()
        batch_doc_max_len = doc_len.max().item()
        batch_total_tokens = target_len.sum().item()

        his_ids = his_ids[:, :, :batch_source_max_len]
        his_mask = his_mask[:, :, :batch_source_max_len]
        doc_ids = doc_ids[:, :batch_doc_max_len]
        doc_mask = doc_mask[:, :batch_doc_max_len]
        target_ids = target_ids[:, :batch_target_max_len]
        target_labels = target_labels[:, :batch_target_max_len].contiguous()
        
        return his_ids, his_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens

    def get_eval_batch_data(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        idx, his_ids, his_mask, his_len, target_ids, target_labels, target_len, doc_ids, doc_mask, doc_len = batch

        example_indices = idx.tolist()
        batch_source_max_len = his_len.max().item()
        batch_target_max_len = target_len.max().item()
        batch_total_tokens = target_len.sum().item()
        batch_doc_max_len = doc_len.max().item()

        his_ids = his_ids[:, :, :batch_source_max_len]
        his_mask = his_mask[:, :, :batch_source_max_len]
        doc_ids = doc_ids[:, :batch_doc_max_len]
        doc_mask = doc_mask[:, :batch_doc_max_len]
        target_ids = target_ids[:, :batch_target_max_len]
        target_labels = target_labels[:, :batch_target_max_len].contiguous()
        
        return example_indices, his_ids, his_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens

    def his_doc_cross_attn_np(self, his_word, his_utt, doc, his_word_mask, doc_mask):
        bsz, his_word_len, his_utt_len, doc_len = his_word.shape[0], his_word.shape[1], his_utt.shape[1], doc.shape[1]
        one_his_len = int(his_word_len / self.args.his_len)
        his_word_mask = (his_word_mask * -1 + 1).unsqueeze(1).bool()
        doc_mask = (doc_mask * -1 + 1).unsqueeze(1).bool()

        his_word_doc_attn = torch.bmm(doc, his_word.transpose(1, 2))
        if his_word_mask is not None:  # don't attend to padding symbols
            his_word_doc_attn = his_word_doc_attn.view(bsz, doc_len, his_word_len)
            reshaped = his_word_mask
            his_word_doc_attn = his_word_doc_attn.masked_fill(reshaped, float("-inf"))
            his_word_doc_attn = his_word_doc_attn.view(bsz, doc_len, his_word_len)

        his_word_doc_attn = F.softmax(his_word_doc_attn, dim=-1)        # (bs, doc_len, his_word_len)

        his_utt_doc_attn = torch.bmm(doc, his_utt.transpose(1, 2))
        his_utt_doc_attn = F.softmax(his_utt_doc_attn, dim=-1)  # (bs,doc_len,his_utt_len)
        his_utt_doc_attn = his_utt_doc_attn.reshape(bsz, doc_len, his_utt_len, 1)\
            .repeat((1, 1, 1, one_his_len))\
            .reshape(bsz, doc_len, his_utt_len*one_his_len)      # (bs*num_heads, doc_len, his_word_len)
        attn = his_word_doc_attn * his_utt_doc_attn

        # re-mask attn
        if his_word_mask is not None:  # don't attend to padding symbols
            attn = attn.view(bsz, doc_len, his_word_len)
            reshaped = his_word_mask
            attn = attn.masked_fill(reshaped, 0)
            attn = attn.view(bsz, doc_len, his_word_len)

        scaler = (1/(attn.sum(-1) + 1e-18))
        scaler = scaler.reshape(bsz, doc_len, 1).tile(1, 1, his_word_len)
        attn = attn * scaler
        doc_his_attn = attn
        doc_his_attn_output = torch.bmm(doc_his_attn, his_word)

        his_doc_attn = attn
        his_doc_attn = torch.max(his_doc_attn, dim=-1)[0].unsqueeze(-1)  # (bs, doc_len, 1)
        his_doc_attn = his_doc_attn.transpose(1, 2)     # (bs, 1, doc_len)

        if doc_mask is not None:  # don't attend to padding symbols
            his_doc_attn = his_doc_attn.view(bsz, 1, doc_len)
            reshaped = doc_mask
            his_doc_attn = his_doc_attn.masked_fill(reshaped, 0)
            his_doc_attn = his_doc_attn.view(bsz, 1, doc_len)

        scaler_doc = (1/(his_doc_attn.sum(-1) + 1e-18))
        scaler_doc = scaler_doc.reshape(bsz, 1, 1).tile(1, 1, doc_len)
        his_doc_attn = his_doc_attn * scaler_doc            # (bs*num_heads, 1, doc_len)
        his_doc_attn_output = torch.bmm(his_doc_attn, doc).repeat(1, doc_len, 1)
        attn_output = torch.cat([doc_his_attn_output*doc, his_doc_attn_output*doc], dim=-1)  # (bs,doc_len,2*d)
        attn_output = self.generator.model.mlp(attn_output)
        return attn_output

    def encode(self, his_ids, his_mask, doc_ids, doc_mask):

        # (B, N, L) -> (B*N, L) -> (B*N, L, D) -> (B, N*L, D) --> (B, N, L, D)
        # [(B, L1), (B, L2)] --> [(B, L1, D), (B, L2, D)]
        # (B, N, L) -> (B*N, L) -> (B*N, L, D) -> (B, N*L, D) --> Aggregate[(B, N*L, V) + (B, L, V)] --> (B, L, V)
        # (B, N, L) -> (B*N, L) -> (B, N*L)

        his_reps = self.generator.model.encoder(
                                        input_ids=his_ids,
                                        attention_mask=his_mask
                                    )
        his_reps = his_reps[0]

        doc_reps = self.generator.model.encoder(
                                        input_ids=doc_ids,
                                        attention_mask=doc_mask
                                    )
        doc_reps = doc_reps[0]

        bs, his_len, dim = his_reps.shape[0], int(his_reps.shape[1] / self.args.his_len), his_reps.shape[2]
        masked_his_reps = his_reps * his_mask.unsqueeze(2)
        his_utt = masked_his_reps.reshape(bs, self.args.his_len, his_len, dim).mean(2)
        his_utt = self.generator.model.utt_mlp(his_utt)
        doc_reps2 = self.his_doc_cross_attn_np(his_reps, his_utt, doc_reps, his_mask, doc_mask)
        if self.args.experiment_type == 'chat+document':
            final_doc_reps = doc_reps + doc_reps2
        else:
            final_doc_reps = doc_reps2
        return his_reps, final_doc_reps

    def process_his(self, his1_ids, his2_ids, his3_ids):
        his3_ids = his3_ids + his3_ids.eq(2)*-1
        his2_ids = his2_ids + his2_ids.eq(2) * -1 + his2_ids.eq(0) * 1
        his1_ids = his1_ids + his1_ids.eq(0) * 1

        his1_mask = his1_ids.ne(1).int()
        his2_mask = his2_ids.ne(1).int()
        his3_mask = his3_ids.ne(1).int()

        all_his_ids = torch.cat([his3_ids, his2_ids, his1_ids], dim=1)
        all_his_mask = torch.cat([his3_mask, his2_mask, his1_mask], dim=1)

        return all_his_ids, all_his_mask

    def train(self):
        self.init_seed()
        cached_features_devfile = os.path.join(
                self.args.data_dir,
                "cached_Bart_{}_DOHA_HHDKS_dataset_{}_hisLen_{}_task_{}_dev_srcLen{}_tgtLen{}".format(
                    self.tokenizer.__class__.__name__,
                    self.args.data_set,
                    self.args.his_len,
                    self.args.experiment_type,
                    str(self.args.source_max_len),
                    str(self.args.target_max_len),
                ),
            )
        cached_features_trainfile = os.path.join(
                self.args.data_dir,
                "cached_Bart_{}_DOHA_HHDKS_dataset_{}_hisLen_{}_task_{}_train_srcLen{}_tgtLen{}".format(
                    self.tokenizer.__class__.__name__,
                    self.args.data_set,
                    self.args.his_len,
                    self.args.experiment_type,
                    str(self.args.source_max_len),
                    str(self.args.target_max_len),
                ),
        )
        if self.args.data_set == "cmu":
            dev_examples = self.load_examples(self.args.data_dir, 'DOHA_HHDKS_dev.csv')
            train_examples = self.load_examples(self.args.data_dir, 'DOHA_HHDKS_train.csv')
        elif self.args.data_set == "d2d":
            dev_examples = self.load_examples(self.args.data_dir, 'd2d_dev.csv')
            train_examples = self.load_examples(self.args.data_dir, 'd2d_train.csv')
        elif self.args.data_set == "wow":
            dev_examples = self.load_examples(self.args.data_dir, 'wow_test_unseen.csv')
            train_examples = self.load_examples(self.args.data_dir, 'wow_train.csv')

        if os.path.exists(cached_features_devfile):
            dev_features = torch.load(cached_features_devfile)
        else:
            dev_features = self.convert_examples_to_features(dev_examples)
            torch.save(dev_features, cached_features_devfile)
        dev_data = (dev_examples, dev_features)

        if os.path.exists(cached_features_trainfile):
            train_features = torch.load(cached_features_trainfile)
        else:
            train_features = self.convert_examples_to_features(train_examples)
            torch.save(train_features, cached_features_trainfile)

        train_batch_size = int(self.args.train_batch_size / self.args.gradient_accumulation_steps)
        num_train_steps = int(len(train_features) / train_batch_size / self.args.gradient_accumulation_steps * self.args.num_train_epochs)

        optimizer = self.get_optimizer()
        t_total = num_train_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_proportion), num_training_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        train_dataloader = self.get_train_dataloader(train_features, train_batch_size)
        
        self.generator.zero_grad()
        self.generator.train()
        
        num_updates = 0
        curr_loss, curr_total_words = 0, 0

        if self.args.log_file_path is not None:
            f_log = open(self.args.log_file_path, 'w')
        else:
            f_log = None
            
        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                his_ids, his_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens = self.get_train_batch_data(batch)
                all_his_ids = his_ids.reshape(his_ids.size(0), -1)
                all_his_mask = his_mask.reshape(his_mask.size(0), -1)
                all_his_reps, doc_reps = self.encode(all_his_ids, all_his_mask, doc_ids, doc_mask)
                outputs = self.generator(input_ids=None,
                                         attention_mask=(all_his_mask, doc_mask),
                                         encoder_outputs=(all_his_reps, doc_reps),
                                         decoder_input_ids=target_ids,
                                         lm_labels=target_labels,
                                         labels=target_labels)

                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                curr_loss += (loss.item()*batch_total_tokens)
                curr_total_words += batch_total_tokens

                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.generator.zero_grad()
                    num_updates += 1
                    lr1, lr2 = scheduler.get_last_lr()[0], scheduler.get_last_lr()[-1]
                    if (num_updates + 1) % 10 == 0:
                        train_stat_curr = {
                            'step': step,
                            'num_updates': num_updates,
                            'epoch': epoch,
                            'loss': loss.item(),
                            'lr1': lr1,
                            'lr2': lr2,
                            'train_ppl': math.exp(min(curr_loss / curr_total_words, 100))
                        }
                        if self.args.use_tb and (num_updates + 1) % 1000 == 0:
                            self.write_train_info_to_tb(train_stat_curr)
                        print(str(train_stat_curr))
                        sys.stdout.flush()
                        curr_loss, curr_total_words = 0, 0

                    if num_updates % self.args.validation_timing == 0:
                        results = self.evaluate(dev_data, num_updates=num_updates)
                        results["steps"] = step
                        results["num_updates"] = num_updates
                        results["lr1"] = lr1
                        results["lr2"] = lr2
                        if self.args.use_tb:
                            self.write_test_info_to_tb(results)
                        if f_log is not None:
                            f_log.write(str(results))
                            f_log.write('\n')
                            f_log.flush()
                        self.save(num_updates)

        if f_log is not None:
            f_log.close()

    def predict(self, dev_data):

        dev_examples, dev_features = dev_data
        eval_dataloader = self.get_eval_dataloader(dev_features, self.args.eval_batch_size)

        self.generator.eval()

        pred = [None] * len(dev_examples)
        total_eval_loss, total_words = 0, 0

        for batch in tqdm(eval_dataloader, desc="Generating"):
            example_indices, his_ids, his_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens = self.get_eval_batch_data(batch)
            with torch.no_grad():
                # all_his_ids, all_his_mask = self.process_his(his1_ids, his2_ids, his3_ids)
                all_his_ids = his_ids.reshape(his_ids.size(0), -1)
                all_his_mask = his_mask.reshape(his_mask.size(0), -1)
                all_his_reps, doc_reps = self.encode(all_his_ids, all_his_mask, doc_ids, doc_mask)
                outputs = self.generator(input_ids=None,
                                         attention_mask=(all_his_mask, doc_mask),
                                         encoder_outputs=(all_his_reps, doc_reps),
                                         decoder_input_ids=target_ids,
                                         lm_labels=target_labels,
                                         labels=target_labels)
                loss = outputs[0]
                total_eval_loss += (loss.item()*batch_total_tokens)
                total_words += batch_total_tokens
                predicted_ids = self.generator.generate(
                                            input_ids=all_his_mask,
                                            attention_mask=(all_his_mask, doc_mask),
                                            encoder_outputs=(all_his_reps, doc_reps),
                                            num_beams=1,
                                            max_length=self.args.target_max_len,
                                            early_stopping=True,
                                            do_sample=True,
                                            temperature=1.0,
                                            top_k=0,
                                            top_p=0.9,
                                        )

            predicted_ids = predicted_ids.to(self.cpu)
            for i in range(len(example_indices)):
                if pred[example_indices[i]] is not None:
                    continue
                answer = self.tokenizer.decode(
                                    predicted_ids[i].tolist(), 
                                    skip_special_tokens=True, 
                                    clean_up_tokenization_spaces=False
                                )
                pred[example_indices[i]] = answer
            
        self.generator.train()
        return pred, total_eval_loss, total_words

    def evaluate(self, dev_data=None, save_file=True, num_updates=None):

        if dev_data is None:
            cached_features_devfile = os.path.join(
                self.args.data_dir,
                "cached_Bart_{}_DOHA_HHDKS_dataset_{}_hisLen_{}_task_{}_dev_srcLen{}_tgtLen{}".format(
                    self.tokenizer.__class__.__name__,
                    self.args.data_set,
                    self.args.his_len,
                    self.args.experiment_type,
                    str(self.args.source_max_len),
                    str(self.args.target_max_len),
                ),
            )
            if self.args.data_set == "cmu":
                dev_examples = self.load_examples(self.args.data_dir, 'DOHA_HHDKS_dev.csv')
            elif self.args.data_set == "d2d":
                dev_examples = self.load_examples(self.args.data_dir, 'd2d_dev.csv')
            elif self.args.data_set == "wow":
                dev_examples = self.load_examples(self.args.data_dir, 'wow_test_unseen.csv')

            if os.path.exists(cached_features_devfile):
                dev_features = torch.load(cached_features_devfile)
            else:
                dev_features = self.convert_examples_to_features(dev_examples)
                torch.save(dev_features, cached_features_devfile)
        else:
            dev_examples, dev_features = dev_data
        
        pred, total_eval_loss, total_words = self.predict((dev_examples, dev_features))
        results = evaluate_nq(dev_examples, pred, total_eval_loss, total_words)
        if save_file:
            output_dir = self.args.output_dir + "checkpoint-" + str(num_updates)
            os.makedirs(output_dir, exist_ok=True)
            with codecs.open(self.args.output_dir + "checkpoint-" + str(num_updates) + '/dev_predictions.txt', 'w', 'utf-8') as out:
                for p in pred:
                    p = self.clean_text(p)
                    out.write(p + '\n')

            with codecs.open(self.args.output_dir + "checkpoint-" + str(num_updates) + '/dev_reference.txt', 'w', 'utf-8') as out:
                for example in dev_examples:
                    target = self.clean_text(example.target)
                    out.write(target + '\n')

        return results

    def clean_text(self, text):
        text = ' '.join(text.split('\n'))
        text = ' '.join(text.split('\t'))
        text = ' '.join(text.split())
        return text

    def generate(self):
        if self.args.data_set == "wow":
            self.generate_wizard()
        else:
            cached_features_testfile = os.path.join(
                    self.args.data_dir,
                    "cached_Bart_{}_DOHA_HHDKS_dataset_{}_hisLen_{}_task_{}_test_srcLen{}_tgtLen{}".format(
                        self.tokenizer.__class__.__name__,
                        self.args.data_set,
                        self.args.his_len,
                        self.args.experiment_type,
                        str(self.args.source_max_len), 
                        str(self.args.target_max_len), 
                    ),
            )

            if self.args.data_set == "cmu":
                test_examples = self.load_examples(self.args.data_dir, 'DOHA_HHDKS_test.csv')
            elif self.args.data_set == "d2d":
                test_examples = self.load_examples(self.args.data_dir, 'd2d_test.csv')

            if os.path.exists(cached_features_testfile):
                test_features = torch.load(cached_features_testfile)
            else:
                test_features = self.convert_examples_to_features(test_examples)
                torch.save(test_features, cached_features_testfile)

            pred, total_eval_loss, total_words = self.predict((test_examples, test_features))
            with codecs.open(self.args.output_dir + 'predictions.txt', 'w', 'utf-8') as out:
                for p in pred:
                    p = self.clean_text(p)
                    out.write(p + '\n')

            with codecs.open(self.args.output_dir + 'reference.txt', 'w', 'utf-8') as out:
                for example in test_examples:
                    target = self.clean_text(example.target)
                    out.write(target + '\n')
            results = evaluate_nq(test_examples, pred, total_eval_loss, total_words)
            print(str(results))

    def generate_wizard(self):
        cached_features_testfile = os.path.join(
            self.args.data_dir,
            "cached_Bart_{}_DOHA_HHDKS_dataset_{}_hisLen_{}_task_{}_test_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.data_set,
                self.args.his_len,
                self.args.experiment_type,
                str(self.args.source_max_len),
                str(self.args.target_max_len),
            ),
        )

        test_examples = self.load_examples(self.args.data_dir, 'test_seen.tsv')
        if os.path.exists(cached_features_testfile):
            test_features = torch.load(cached_features_testfile)
        else:
            test_features = self.convert_examples_to_features(test_examples)
            torch.save(test_features, cached_features_testfile)

        pred, total_eval_loss, total_words = self.predict((test_examples, test_features))
        with codecs.open(self.args.output_dir + 'predictions_seen.txt', 'w', 'utf-8') as out:
            for p in pred:
                p = self.clean_text(p)
                out.write(p + '\n')

        with codecs.open(self.args.output_dir + 'reference_seen.txt', 'w', 'utf-8') as out:
            for example in test_examples:
                target = self.clean_text(example.target)
                out.write(target + '\n')

        with codecs.open(self.args.output_dir + 'all_results_seen.csv', 'w', 'utf-8') as out:
            writer_ = csv.writer(out, delimiter=',')
            for i in range(len(pred)):
                writer_.writerow([i, test_examples[i].target, pred[i]])

        results = evaluate_nq(test_examples, pred, total_eval_loss, total_words)
        print(str(results))

        cached_features_testfile = os.path.join(
                self.args.data_dir,
                "cached_Bart_{}_DOHA_HHDKS2_hislen_1_task_{}_test_unseen_srcLen{}_tgtLen{}".format(
                    self.tokenizer.__class__.__name__, 
                    self.args.experiment_type,
                    str(self.args.source_max_len), 
                    str(self.args.target_max_len), 
                ),
            )

        test_examples = self.load_examples(self.args.data_dir, 'test_unseen.tsv')
        if os.path.exists(cached_features_testfile):
            test_features = torch.load(cached_features_testfile)
        else:
            test_features = self.convert_examples_to_features(test_examples)
            torch.save(test_features, cached_features_testfile)

        pred, total_eval_loss, total_words = self.predict((test_examples, test_features))
        with codecs.open(self.args.output_dir + 'predictions_unseen.txt', 'w', 'utf-8') as out:
            for p in pred:
                p = self.clean_text(p)
                out.write(p + '\n')

        with codecs.open(self.args.output_dir + 'reference_unseen.txt', 'w', 'utf-8') as out:
            for example in test_examples:
                target = self.clean_text(example.target)
                out.write(target + '\n')

        with codecs.open(self.args.output_dir + 'all_results_unseen.csv', 'w', 'utf-8') as out:
            writer_ = csv.writer(out, delimiter=',')
            for i in range(len(pred)):
                writer_.writerow([i, test_examples[i].target, pred[i]])

        results = evaluate_nq(test_examples, pred, total_eval_loss, total_words)
        print(str(results))

    def write_train_info_to_tb(self, data):
        num_updates = data['num_updates']
        loss = data['loss']
        lr1 = data['lr1']
        lr2 = data['lr2']
        train_ppl = data['train_ppl']
        epoch = data['epoch']

        self.tb_writer.add_scalars("train_loss", {'loss': loss}, num_updates)
        self.tb_writer.add_scalars("train_ppl", {'train_ppl': train_ppl}, num_updates)
        self.tb_writer.add_scalars("lr1", {'lr1': lr1}, num_updates)
        self.tb_writer.add_scalars("lr2", {'lr2': lr2}, num_updates)
        self.tb_writer.close()

    def write_test_info_to_tb(self, data):
        eval_loss = data['eval_loss']
        acc = data['token_accuracy']
        f1 = data['f1_score']
        valid_ppl = data['valid_ppl']
        num_updates = data['num_updates']

        self.tb_writer.add_scalars("valid_loss", {'loss': eval_loss}, num_updates)
        self.tb_writer.add_scalars("valid_ppl", {'train_ppl': valid_ppl}, num_updates)
        self.tb_writer.add_scalars("token_accuracy", {'token_accuracy': acc}, num_updates)
        self.tb_writer.add_scalars("f1_score", {'f1_score': f1}, num_updates)
        self.tb_writer.close()

def main():
    qa = MultiBartQA()
    if qa.args.do_train:
        qa.train()
    elif qa.args.do_eval:
        results = qa.evaluate(save_file=True)
        print(str(results))
    elif qa.args.do_generate:
        qa.generate()
    else:
        print("Specify whether to train, eval or generate")
    
if __name__ == '__main__':
    main()
