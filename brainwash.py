# The code is built on top of the codebase of the paper "Calibrate Before Use: Improving Fewshot Performance of Language Models" https://github.com/tonyzhaozh/few-shot-learning

import argparse
from datetime import datetime
import numpy as np
import pickle
import random
from copy import deepcopy
from utils import (
    load_dataset,
    random_sampling,
    construct_prompt,
    retrieve_model,
    response_from_model,
)


def main(models, datasets, num_seeds, positions, all_shots):
    """
    Run experiment or load past results, print accuracy
    """
    default_params = {
        "conditioned_on_correct_classes": True,
    }
    current_date = datetime.now().strftime('%Y-%m-%d')

    # list of all experiment parameters to run
    all_params = []
    for model in models:
        for dataset in datasets:
            for position in positions:
                for num_shots in all_shots:
                    for seed in range(num_seeds):
                        p = deepcopy(default_params)
                        p["model"] = model
                        p["dataset"] = dataset
                        p["seed"] = seed
                        p["num_shots"] = num_shots
                        p['position'] = position
                        p[
                            "expr_name"
                        ] = f"{p['dataset']}_{p['model']}_subsample_seed{p['seed']}"
                        all_params.append(p)

    for param_index, params in enumerate(all_params):
        train_sentences, train_labels, test_sentences, test_labels = prepare_data(
            params
        )
        print(params)
        global llm_model
        global llm_tokenizer
        llm_model, llm_tokenizer = retrieve_model(params)
        all_member_list = []
        all_nonmember_list = []

        test_data = list(zip(test_sentences, test_labels))

        # create your prompt
        demo_sentences = []
        demo_labels = []

        # for example
        random_prepend = random.sample(test_data, params['num_shots'])
        for i in range(len(random_prepend)):
            demo_sentences.append(random_prepend[i][0])
            demo_labels.append(random_prepend[i][1])

        # based on the position
        if params['position'] == 'end':
            member_sentence = demo_sentences[-1]
            member_label = demo_labels[-1]
        elif params['position'] == 'begin':
            member_sentence = demo_sentences[0]
            member_label = demo_labels[0]

        # set nonmember_sentence and nonmember_label
        nonmember_sentences = SET_WITH_NO_OVERLAP
        nonmember_labels = SET_WITH_NO_OVERLAP


        required_for_mem = brainwash(
            params,
            demo_sentences,
            demo_labels,
            member_sentence,
            member_label,
        )
        if required_for_mem == None:
            continue

        required_for_nonmem = brainwash(
            params,
            demo_sentences,
            demo_labels,
            nonmember_sentences,
            nonmember_labels,
        )
        if required_for_nonmem == None:
            continue

        all_member_list.append(required_for_mem)
        all_nonmember_list.append(required_for_nonmem)
        with open(
            MEM_SAVE_PATH,
            "wb",
        ) as file:
            pickle.dump(all_member_list, file)
        with open(
            NONMEM_SAVE_PATH,
            "wb",
        ) as file:
            pickle.dump(all_nonmember_list, file)


def prepare_data(params):
    print("\nExperiment name:", params["expr_name"])
    (
        all_train_sentences,
        all_train_labels,
        all_test_sentences,
        all_test_labels,
    ) = load_dataset(params)

    np.random.seed(params["seed"])
    test_sentences, test_labels = random_sampling(
        all_test_sentences, all_test_labels, 500
    )

    train_sentences, train_labels = random_sampling(
        all_train_sentences, all_train_labels, 500
    )
    return train_sentences, train_labels, test_sentences, test_labels


def count_matching_items(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length for comparison.")

    matching_count = sum(1 for item1, item2 in zip(list1, list2) if item1 == item2)

    return matching_count / len(list1)


def prepare_input(
    params,
    train_sentences,
    train_labels,
    test_sentence,
    test_label,
    target_label,
    num_repeat,
):
    combined_sentences = deepcopy(train_sentences)
    combined_labels = deepcopy(train_labels)
    for repeat_times in range(num_repeat):
        combined_sentences.append(test_sentence)
        combined_labels.append(target_label)
    input_to_model = construct_prompt(
        params, combined_sentences, combined_labels, test_sentence
    )
    del combined_sentences
    del combined_labels
    return input_to_model


def brainwash(
    params, train_sentences, train_labels, test_sentence, test_label, target_label
):
    max_repeats = 10
    for num_repeat in range(max_repeats):
        # print(train_labels)
        input_to_model = prepare_input(
            params,
            train_sentences,
            train_labels,
            test_sentence,
            test_label,
            target_label,
            num_repeat,
        )

        return_idx = response_from_model(
            params, input_to_model, llm_model, llm_tokenizer
        )
        if return_idx == -1:
            return None
        if return_idx == target_label:
            return num_repeat
    return max_repeats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument(
        "--models",
        dest="models",
        action="store",
        required=True,
        help="name of model(s), e.g., GPT2-XL",
    )
    parser.add_argument(
        "--datasets",
        dest="datasets",
        action="store",
        required=True,
        help="name of dataset(s), e.g., agnews",
    )
    parser.add_argument(
        "--num_seeds",
        dest="num_seeds",
        action="store",
        required=True,
        help="num seeds for the training set",
        type=int,
    )
    parser.add_argument(
        "--all_shots",
        dest="all_shots",
        action="store",
        required=True,
        help="num training examples to use",
    )
    parser.add_argument(
        "--positions",
        dest="positions",
        action="store",
        required=True,
        help="the position of the target demo, e.g. begin or end.",
    )

    args = parser.parse_args()
    args = vars(args)
    print(args)

    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]

    args["models"] = convert_to_list(args["models"])
    args["datasets"] = convert_to_list(args["datasets"])
    args["positions"] = convert_to_list(args["positions"])
    args["all_shots"] = convert_to_list(args["all_shots"], is_int=True)

    main(**args)
