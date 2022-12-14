import json
import csv
from tqdm import tqdm, trange
import argparse
import random
import math
import sys
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import pdb
import ast
import copy
import codecs
import sys
#csv.field_size_limit(sys.maxsize)

import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BartTokenizer, AdamW, get_linear_schedule_with_warmup
from transformers import BartConfig

from bart_decoder import MultiHeadBartForConditionalGeneration 

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from metrics import evaluate_nq

class GenerationInputExample(object):

    def __init__(self, guid, source, target, context=None):
        self.guid = guid
        self.source = source
        self.target = target
        self.context = context

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
                 source_ids,
                 source_mask,
                 source_len,
                 target_ids,
                 target_labels,
                 target_len,
                 doc_ids,
                 doc_mask,
                 doc_len):

        self.example_index = example_index
        
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.source_len = source_len

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

        self.generator = MultiHeadBartForConditionalGeneration.from_pretrained_multi(self.args, self.args.model_file_path)
        self.generator.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(self.args.model_name) # Need to add base to "tokenization_bart.py" when using transformers==2.11.0

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
            his = " ".join(his)
            tgt = samples.iloc[idx].tgt
            doc = samples.iloc[idx].doc
            examples.append(GenerationInputExample(
                guid=idx,
                source=his.strip(),
                target=tgt,
                context=doc
            ))
        return examples

    def convert_examples_to_features(self, examples):
        config = self.generator.model.config
        features = []
        index = 0

        for e in tqdm(examples, desc='Examples'):
            # Process source information

            his = e.source if e.source != "no content" else ""
            source = 'chat: ' + his

            source_tokens = self.tokenizer.tokenize(source)[:self.args.source_max_len-2]
            source_ids = [config.bos_token_id] + self.tokenizer.convert_tokens_to_ids(source_tokens) + [config.eos_token_id] # <s> ... </s>
            source_len = len(source_ids)
            source_mask = [1] * source_len

            padding_len = self.args.source_max_len - source_len
            source_ids += ([config.pad_token_id] * padding_len)
            source_mask += ([0] * padding_len)

            assert len(source_ids) == self.args.source_max_len
            assert len(source_mask) == self.args.source_max_len

            if self.args.experiment_type == 'doc_only':
                document = 'document: ' + e.context
            elif self.args.experiment_type == 'chat_document':
                document = 'chat: ' + his + ' document: ' + e.context
            # elif self.args.experiment_type == 'chat_wizard':
            #     context = e.context
            #     context = ast.literal_eval(context)
            #     all_docs = ''
            #     for doc in context:
            #         title = list(doc.keys())[0]
            #         passage = ' '.join(doc[title])
            #         all_docs = all_docs + ' title: ' + title + ' text: ' + passage
            #     document = 'chat: ' + e.source + ' document: ' + all_docs
            elif self.args.experiment_type == 'chat_wizard':
                document = 'chat: ' + e.source + 'document: ' + e.context
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
                print(e.source, e.context, e.target)
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
                                source_ids, 
                                source_mask, 
                                source_len, 
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
        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_source_len = torch.tensor([f.source_len for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_labels = torch.tensor([f.target_labels for f in train_features], dtype=torch.long)
        all_target_len = torch.tensor([f.target_len for f in train_features], dtype=torch.long)
        all_doc_ids = torch.tensor([f.doc_ids for f in train_features], dtype=torch.long)
        all_doc_mask = torch.tensor([f.doc_mask for f in train_features], dtype=torch.long)
        all_doc_len = torch.tensor([f.doc_len for f in train_features], dtype=torch.long)
        train_data = TensorDataset(
                                    all_source_ids,
                                    all_source_mask,
                                    all_source_len,
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
        all_source_ids = torch.tensor([f.source_ids for f in dev_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in dev_features], dtype=torch.long)
        all_source_len = torch.tensor([f.source_len for f in dev_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in dev_features], dtype=torch.long)
        all_target_labels = torch.tensor([f.target_labels for f in dev_features], dtype=torch.long)
        all_target_len = torch.tensor([f.target_len for f in dev_features], dtype=torch.long)
        all_doc_ids = torch.tensor([f.doc_ids for f in dev_features], dtype=torch.long)
        all_doc_mask = torch.tensor([f.doc_mask for f in dev_features], dtype=torch.long)
        all_doc_len = torch.tensor([f.doc_len for f in dev_features], dtype=torch.long)
        eval_data = TensorDataset(
                                    all_example_indices,
                                    all_source_ids,
                                    all_source_mask,
                                    all_source_len,
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

        batch_source_max_len = batch[2].max().item()
        batch_target_max_len = batch[5].max().item()
        batch_doc_max_len =  batch[8].max().item()
        batch_total_tokens = batch[5].sum().item()

        batch = tuple(t.to(self.device) for t in batch)
        source_ids, source_mask, _, target_ids, target_labels, _, doc_ids, doc_mask, _ = batch
        source_ids = source_ids[:, :batch_source_max_len]
        source_mask = source_mask[:, :batch_source_max_len]
        doc_ids = doc_ids[:, :batch_doc_max_len]
        doc_mask = doc_mask[:, :batch_doc_max_len]
        target_ids = target_ids[:, :batch_target_max_len]
        target_labels = target_labels[:, :batch_target_max_len].contiguous()
        
        return source_ids, source_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens

    def get_eval_batch_data(self, batch):

        example_indices = batch[0].tolist()
        batch_source_max_len = batch[3].max().item()
        batch_target_max_len = batch[6].max().item()
        batch_total_tokens = batch[6].sum().item()
        batch_doc_max_len = batch[9].max().item()

        batch = tuple(t.to(self.device) for t in batch)
        _, source_ids, source_mask, __, target_ids, target_labels, _, doc_ids, doc_mask, _ = batch
        source_ids = source_ids[:, :batch_source_max_len]
        source_mask = source_mask[:, :batch_source_max_len]
        doc_ids = doc_ids[:, :batch_doc_max_len]
        doc_mask = doc_mask[:, :batch_doc_max_len]
        target_ids = target_ids[:, :batch_target_max_len]
        target_labels = target_labels[:, :batch_target_max_len].contiguous()
        
        return example_indices, source_ids, source_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens

    def encode(self, source_ids, source_mask, doc_ids, doc_mask):

        # (B, N, L) -> (B*N, L) -> (B*N, L, D) -> (B, N*L, D) --> (B, N, L, D)
        # [(B, L1), (B, L2)] --> [(B, L1, D), (B, L2, D)]
        # (B, N, L) -> (B*N, L) -> (B*N, L, D) -> (B, N*L, D) --> Aggregate[(B, N*L, V) + (B, L, V)] --> (B, L, V)
        # (B, N, L) -> (B*N, L) -> (B, N*L)

        source_reps = self.generator.model.encoder(
                                        input_ids=source_ids,
                                        attention_mask=source_mask
                                    )
        source_reps = source_reps[0]

        doc_reps = self.generator.model.encoder(
                                        input_ids=doc_ids,
                                        attention_mask=doc_mask
                                    )
        doc_reps = doc_reps[0]

        return source_reps, doc_reps
    
    def train(self):

        self.init_seed()
        cached_features_devfile = os.path.join(
                self.args.data_dir,
                "cached_Bart_{}_DOHA_dataset_{}_hisLen_{}_task_{}_dev_srcLen{}_tgtLen{}".format(
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
                "cached_Bart_{}_DOHA_dataset_{}_hisLen_{}_task_{}_train_srcLen{}_tgtLen{}".format(
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
                source_ids, source_mask, target_ids, target_labels, doc_ids, doc_mask, batch_total_tokens = self.get_train_batch_data(batch)

                source_reps, doc_reps = self.encode(source_ids, source_mask, doc_ids, doc_mask)
                outputs = self.generator(input_ids=None,
                                         attention_mask=(source_mask, doc_mask),
                                         encoder_outputs=(source_reps, doc_reps),
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

                    if (num_updates+1) % 10 == 0:
                        train_stat_curr = {
                        'step': step,
                        'num_updates': num_updates,
                        'epoch': epoch,
                        'loss': loss.item(),
                        'train_ppl': math.exp(min(curr_loss/curr_total_words, 100))
                        }
                        print(str(train_stat_curr))
                        sys.stdout.flush()
                        curr_loss, curr_total_words = 0, 0

                    if num_updates % self.args.validation_timing == 0:
                        results = self.evaluate(dev_data, num_updates=num_updates)

                        results["steps"] = step
                        results["num_updates"] = num_updates
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

        for batch  in tqdm(eval_dataloader, desc="Generating"):
            example_indices, source_ids, source_mask, target_ids, \
                target_labels, doc_ids, doc_mask, batch_total_tokens = self.get_eval_batch_data(batch)
            with torch.no_grad():
                source_reps, doc_reps = self.encode(source_ids, source_mask, doc_ids, doc_mask)
                outputs = self.generator(input_ids=None,
                                         attention_mask=(source_mask, doc_mask),
                                         encoder_outputs=(source_reps, doc_reps),
                                         decoder_input_ids=target_ids,
                                         lm_labels=target_labels,
                                         labels=target_labels)
                loss = outputs[0]
                total_eval_loss += (loss.item()*batch_total_tokens)
                total_words += batch_total_tokens
                predicted_ids = self.generator.generate(
                                            input_ids=source_mask, 
                                            attention_mask=(source_mask, doc_mask),
                                            encoder_outputs=(source_reps, doc_reps),
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
                "cached_Bart_{}_DOHA_dataset_{}_hisLen_{}_task_{}_dev_srcLen{}_tgtLen{}".format(
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
                    "cached_Bart_{}_DOHA_dataset_{}_hisLen_{}_task_{}_test_srcLen{}_tgtLen{}".format(
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
            "cached_Bart_{}_DOHA_dataset_{}_hisLen_{}_task_{}_test_srcLen{}_tgtLen{}".format(
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
            "cached_Bart_{}_DOHA_dataset_{}_hisLen_{}_task_{}_test_srcLen{}_tgtLen{}".format(
                self.tokenizer.__class__.__name__,
                self.args.data_set,
                self.args.his_len,
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
