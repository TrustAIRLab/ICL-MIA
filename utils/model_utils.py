import torch
from transformers import (
    GPT2Tokenizer,
    AutoModelForCausalLM,
)

def setup_gpt2(model_name):
    print("Setting up GPT-2 model")
    gpt2_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    gpt2_model.eval()
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    gpt2_tokenizer.padding_side = "left"
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
    return gpt2_model, gpt2_tokenizer


def retrieve_model(params):
    if params["model"] == 'gpt2-xl':
        return setup_gpt2(params["model"])
    else:
        raise NotImplementedError(f"Model {params['model']} not implemented")


def response_from_model(params, input_to_model, model, tokenizer):
    if params["model"] == "gpt2-xl":
        labels = []
        for i in range(len(params["label_dict"])):
            labels.append(
                tokenizer(params["label_dict"][i], truncation=True)["input_ids"][0][0]
            )

        input_ids = tokenizer(input_to_model, return_tensors="pt", padding=True)
        output = model(input_ids["input_ids"].cuda())
        logits = output.logits[:, -1, :]
        max_value, max_index = torch.max(logits[0, labels], dim=0)
        return max_index.item()
    else:
        raise NotImplementedError(f"Model {params['model']} not implemented")

def seen_before_from_model(params, input_to_model, model, tokenizer):
    if params["model"] == "gpt2-xl":
        labels = []
        answer_dict = {
            0: ["No"],
            1: ["Yes"],
        }
        for i in range(len(answer_dict)):
            labels.append(tokenizer(answer_dict[i], truncation=True)["input_ids"][0][0])

        input_ids = tokenizer(input_to_model, return_tensors="pt", padding=True)
        output = model(input_ids["input_ids"].cuda())
        logits = output.logits[:, -1, :]
        max_value, max_index = torch.max(logits[0, labels], dim=0)
        return max_index.item()
    else:
        raise NotImplementedError(f"Model {params['model']} not implemented")
    

def continue_generate(params, input_to_model, model, tokenizer):
    if params["model"] == "gpt2-xl":
        input_ids = tokenizer(input_to_model, return_tensors="pt", padding=True)
        output = model.generate(input_ids["input_ids"].cuda(),max_length=1000,min_new_tokens=20)
        new_sentence = tokenizer.batch_decode(output.detach().cpu().numpy(), skip_special_tokens=True)
        return new_sentence[0]
    else:
        raise NotImplementedError(f"Model {params['model']} not implemented")
