import pandas as pd
import json
import pickle
import numpy as np
import os
from copy import deepcopy


def load_trec():
    inv_label_dict = {"NUM": 0, "LOC": 1, "HUM": 2, "DESC": 3, "ENTY": 4, "ABBR": 5}
    train_sentences = []
    train_labels = []
    with open(f"{ROOT_DIR}/data/trec/train.txt", "r") as train_data:
        for line in train_data:
            train_label = line.split(" ")[0].split(":")[0]
            train_label = inv_label_dict[train_label]
            train_sentence = " ".join(line.split(" ")[1:]).strip()
            # basic cleaning
            train_sentence = (
                train_sentence.replace(" 's", "'s")
                .replace("`` ", '"')
                .replace(" ''", '"')
                .replace(" ?", "?")
                .replace(" ,", ",")
            )
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f"{ROOT_DIR}/data/trec/test.txt", "r") as test_data:
        for line in test_data:
            test_label = line.split(" ")[0].split(":")[0]
            test_label = inv_label_dict[test_label]
            test_sentence = " ".join(line.split(" ")[1:]).strip()
            test_sentence = (
                test_sentence.replace(" 's", "'s")
                .replace("`` ", '"')
                .replace(" ''", '"')
                .replace(" ?", "?")
                .replace(" ,", ",")
            )
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    return train_sentences, train_labels, test_sentences, test_labels


def load_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params["dataset"] == "trec":
        (
            orig_train_sentences,
            orig_train_labels,
            orig_test_sentences,
            orig_test_labels,
        ) = load_trec()
        params[
            "prompt_prefix"
        ] = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer Type: "
        params["label_dict"] = {
            0: ["Number"],
            1: ["Location"],
            2: ["Person"],
            3: ["Description"],
            4: ["Entity"],
            5: ["Ab"],
        }
        params["inv_label_dict"] = {
            "Number": 0,
            "Location": 1,
            "Person": 2,
            "Description": 3,
            "Entity": 4,
            "Ab": 5,
        }
        params["task_format"] = "classification"
        params["num_tokens_to_predict"] = 1
    else:
        raise NotImplementedError
    return (
        orig_train_sentences,
        orig_train_labels,
        orig_test_sentences,
        orig_test_labels,
    )
