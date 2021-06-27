import datasets
from datasets import load_metric
from transformers import BertForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import torch
import numpy as np

from xtagger.utils.data_utils import xtagger_dataset_to_df, df_to_hf_dataset

import warnings
warnings.filterwarnings("ignore")

class BERTForTagging():
    def __init__(self, bert_config_path, device, tags, tokenizer, cuda=True, learning_rate = 2e-5,
                 train_batch_size=4, eval_batch_size=4, epochs=3, weight_decay=0.1, log_step = 10_000):
        self.device = device
        if self.device.type == "cpu" and cuda:
            print("Can't see cuda. Switching to cpu automatically.")

        self.config = bert_config_path
        self.tags = tags
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.tokenizer = tokenizer
        self.log_step = log_step

        self.build_model()


    def build_model(self):
        self.model = BertForTokenClassification.from_pretrained(self.config, num_labels=len(self.tags))
        self.model.to(self.device)

    def metrics(self, p):
        metric = load_metric("seqeval")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [self.tags[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.tags[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def fit(self, train, test):
        self.train = train
        self.test = test
        
        args = TrainingArguments(
            "tagging",
            evaluation_strategy = "epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            save_steps = self.log_step,
            logging_steps = self.log_step
        )

        data_collator = DataCollatorForTokenClassification(self.tokenizer)


        self.trainer = Trainer(
            self.model,
            args,
            train_dataset=self.train,
            eval_dataset=self.test,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.metrics
        )

        self.trainer.train()


    def evaluate(self):
        return self.trainer.evaluate()


    def predict(self,sentence):
        d = []
        for word in sentence.split():
            d.append((word,self.tags[0]))
        sample = xtagger_dataset_to_df([d], row_as_list=True)
        dataset_sample = df_to_hf_dataset(sample, self.tags, self.tokenizer, self.device)
        raw, _, _ = self.trainer.predict(dataset_sample)
        sentence_tags = []
        for sent in raw:
            words = []
            for word in sent:
                words.append(np.argmax(word))
            sentence_tags.append(words)
        predicted_tags = []
        for i in sentence_tags[0]:
            predicted_tags.append(self.tags[i])
        return_tup = []
        return predicted_tags, dataset_sample[0]["input_ids"]
        
    def load_model(self, save_name):
        self.model.load_state_dict(torch.load(save_name))
