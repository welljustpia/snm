from dataclasses import dataclass, field
from functools import partial
from typing import Tuple, List, Dict, Any
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

class NERWangchanBERTaInferenceModel(object):

    def __init__(
        self,
        model: AutoModelForTokenClassification,
        tokenizer: AutoTokenizer,
        idx_to_class: Dict[int, str],
        class_to_idx: Dict[str, int]
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.idx_to_class = idx_to_class
        self.class_to_idx = class_to_idx

        self.model.eval()

    def get_prediction_in_word_level(self, input_ids, word_ids, logits):
        # initialize
        predictions = []
        prev_word_id = None

        # predefined possible prediction index
        # this `possible_class_indices` variable
        # will be constantly updated through decoding
        # steps and store possible index for decoding
        # for example, if current time step is B-PERSON
        # the only possible next step are B-[other classes], O, and I-PERSON
        # the initial state would be O, B-[all classes]
        # start with I is impossible
        possible_class_indices = [self.class_to_idx["O"]]
        possible_class_indices += [
            self.class_to_idx[class_name]
            for class_name
            in self.class_to_idx.keys()
            if class_name.startswith("B-")]

        for token_id, word_id in enumerate(word_ids):
            # Skip special tokens ([START] and [END] tokens) and sub-word tokens
            if word_id == None or word_id == prev_word_id:
                continue
            if word_id != prev_word_id:
                # Only get predicted class from the first token in a word (we want prediction in word-level)
                prev_word_id = word_id

            # Get constrained prediction
            # argmax only logits that satisfy possible_class_indices
            filterd_logit = logits[token_id, possible_class_indices]
            pred_class_id = possible_class_indices[filterd_logit.argmax(0)]
            pred_class_name = self.idx_to_class[pred_class_id]
            predictions.append(pred_class_id)

            # Update possible_class_indices
            if pred_class_name == "O":
                # get next possible indices of "O"
                possible_class_indices = [self.class_to_idx["O"]]
                possible_class_indices += [
                    self.class_to_idx[class_name]
                    for class_name in self.class_to_idx.keys()
                    if class_name.startswith("B-")]
            elif pred_class_name.startswith("B-"):
                # get next possible indices of B-XXX
                possible_class_indices = [self.class_to_idx["O"]]
                possible_class_indices += [
                    self.class_to_idx[class_name]
                    for class_name in self.class_to_idx.keys()
                    if class_name.startswith(f"I-{pred_class_name[2:]}")]
            else:
                # get next possible indices of I-XXX
                possible_class_indices = [self.class_to_idx["O"]]
                possible_class_indices += [
                    self.class_to_idx[class_name]
                    for class_name in self.class_to_idx.keys()
                    if class_name.startswith("B-")]
                possible_class_indices += [
                    self.class_to_idx[class_name]
                    for class_name in self.class_to_idx.keys()
                    if class_name.startswith(f"I-{pred_class_name[2:]}")]

        # return constraint predictions
        return predictions

    def predict(self, words: List[str]) -> List[str]:
        # tokenize input words and move batch to GPU
        batch = self.tokenizer(words, return_tensors="pt", truncation=True, is_split_into_words=True, 
                              max_length = 510)
        batch = batch.to(self.model.device)

        # forward and get logits
        with torch.no_grad():
            logits = self.model(**batch).logits

        # get word-level predictions
        predictions = self.get_prediction_in_word_level(
            input_ids=batch["input_ids"][0],
            word_ids=batch.word_ids(batch_index=0),
            logits=logits[0]
        )

        return predictions