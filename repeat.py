# The code is built on top of the codebase of the paper "Calibrate Before Use: Improving Fewshot Performance of Language Models" https://github.com/tonyzhaozh/few-shot-learning

import argparse
from datetime import datetime
import numpy as np
import pickle
import random
from copy import deepcopy
from sentence_transformers import SentenceTransformer, util
from utils import (
    load_dataset,
    random_sampling,
    construct_prompt,
    retrieve_model,
    continue_generate,
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
                        # p['repeats'] = repeats
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
        global semantic_model
        llm_model, llm_tokenizer = retrieve_model(params)
        semantic_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
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


        required_for_mem = repeat(
            params,
            demo_sentences,
            demo_labels,
            member_sentence,
            member_label,
        )
        if required_for_mem == None:
            continue

        required_for_nonmem = repeat(
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


def repeat(params, train_sentences, train_labels, test_sentence, test_label):
    words = test_sentence.split()
    first_three_words = ' '.join(words[:3])
    input_to_model = construct_prompt_cut(
        params, train_sentences, train_labels, first_three_words
    )
    compare_sentence = construct_prompt(
        params, train_sentences, train_labels, ''
    )
    return_sentence = continue_generate(params, input_to_model, llm_model, llm_tokenizer)
    new_generate = return_sentence[len(compare_sentence[:-len(params["a_prefix"])]):].split('\n')[0]
    embedding_1= semantic_model.encode(test_sentence, convert_to_tensor=True)
    embedding_2 = semantic_model.encode(new_generate, convert_to_tensor=True)

    semantic_sim = util.pytorch_cos_sim(embedding_1, embedding_2).item()

    return semantic_sim
    



def construct_prompt_cut(params, train_sentences, train_labels, test_sentence):
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence
    return prompt


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
