import argparse
import os
import random

from datasets import load_dataset, Dataset, DatasetDict
# from datasets.table import Table
from promptsource.templates import DatasetTemplates
from tqdm import tqdm
from transformers import T5Tokenizer, AutoTokenizer

train_file_name = "train"
valid_file_name = "val"
test_file_name = "test"
extension_list = ["json", "jsonl", "csv", "tsv","txt"]

DEBUG_TRAIN_TASK_LIST = ["super_glue/rte"]
DEBUG_TEST_TASK_LIST = ["super_glue/cb"]

TEST=True

TASK_TO_EXT_DICT = {
    "adversarial_qa/dbert":"json",
    # "web_questions":"jsonl",
    "wiqa": "jsonl",
    "glue/mrpc": "tsv",
    "glue/qqp": "tsv",
    "paws/labeled_final": "tsv",
    "ag_news": "csv",
    "cosmos_qa":"csv",
    "commonsense_qa":"jsonl",
    "common_gen": "json",
    "dbpedia_14": "csv",
    "dream": "jsonl",
    "hellaswag": "jsonl",
    "imdb": "json",
    "kilt_tasks/hotpotqa": "jsonl",
    "openbookqa/main": "jsonl",
    "super_glue/cb": "jsonl",
    "super_glue/wic": "jsonl",
    "super_glue/rte": "jsonl",
    "super_glue/boolq": "jsonl",
    # "super_glue/copa": "jsonl",
    # "wiki_qa": "tsv"
}

T0_TEST_TASK_LIST = [
    "super_glue/copa",
    # "story_cloze/2016",
    "hellaswag",
    "super_glue/cb",
    "super_glue/rte",
    "anli",
    "super_glue/wsc.fixed",
    "winogrande/winogrande_xl",
    "super_glue/wic"
]

T0_TEST_TASK_LIST_DEBUG = [
    "super_glue/copa",
    "super_glue/wic",
    "super_glue/wsc.fixed",
]


T0_TRAIN_TASK_LIST = [
    "glue/mrpc",
    "glue/qqp",
    "paws/labeled_final",
    "ag_news",
    "dbpedia_14",
    "dream",
    "kilt_tasks/hotpotqa",
    "trec",
    "cnn_dailymail/3.0.0",
    "gigaword",
    "multi_news",
    "samsum",
    "xsum",
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "wiki_qa",
    "common_gen",
    "wiki_bio",
    "adversarial_qa/dbidaf",
    "adversarial_qa/dbert",
    "adversarial_qa/droberta",
    "quoref",
    "ropes",
    "duorc/SelfRC",
    "duorc/ParaphraseRC",
    "wiki_hop/original",
    "sciq",
    "quarel",
    "qasc",
    "cosmos_qa",
    "wiqa",
    "social_i_qa",
    "quail",
    "quartz",
    "cos_e/v1.11",
    # "commonsense_qa"
]





TASK_TYPE_DICT = {
    "coreference_resolution": [
        "super_glue/wsc.fixed", "winogrande/winogrande_xl"
    ],
    "natural_language_inference":[
        "super_glue/cb", "super_glue/rte", "anli"
    ],
    "paraphrase_identification":[
        "glue/mrpc", "glue/qqp", "paws/labeled_final"
    ],
    "closed_book_qa":[
        "ai2_arc/ARC Challenge",
        "ai2_arc/ARC_Easy",
        "kilt_tasks/hotpotqa",
        "trivia_qa/unfiltered",
        "web_questions",
        "wiki_qa"
    ],
    "extractive_qa":[
        "adversarial_qa/dbidaf",
        "adversarial_qa/dbert",
        "adversarial_qa/droberta",
        "duorc/SelfRC",
        "duorc/ParaphraseRC",
        "ropes",
        "squad_v2",
        "super_glue/record",
        "quoref",
        "tydiqa"
        ],
    "multiple_choice_qa":[
        "commonsense_qa",
        "cos_e/v1.11",
        "cosmos_qa",
        "dream",
        "openbookqa/main",
        "qasc",
        "quail",
        "quarel",
        "quartz",
        "race/high",
        "race/middle",
        "sciq",
        "social_i_qa",
        "super_glue/boolq",
        "super_glue/multirc",
        "wiki_hop/original",
        "wiqa",
        "piqa"
        ],
    "sentiment": [
        "amazon_polarity", "app_reviews", "imdb", "rotten_tomatoes", "yelp_review_full"
    ],
    "sentence_completion": [
        "super_glue/copa", "story_cloze/2016", "hellaswag"
    ],
    "structure_to_text": [
        "common_gen", "wiki_bio"
    ],
    "summarization": [
        "cnn_dailymail/3.0.0", "gigaword", "multi_news", "samsum", "xsum"
    ],
    "topic_classification": [
        "ag_news", "dbpedia_14", "trec"
    ],
    "word_sense_disambiguation": [
        "super_glue/wic"
    ]
}

DOMAIN_TYPE_DICT = {
    "general": [
        "hellaswag",
        "super_glue/rte",
        "anli",
        "winogrande/winogrande_xl",
        "glue/mrpc",
        "dream",
        "trec",
        "gigaword",
        "adversarial_qa/dbidaf",
        "adversarial_qa/dbert",
        "adversarial_qa/droberta",
        "quail", #?
        "quartz",
        "cos_e_v1.11", # commensense_qa
        "wiqa"
    ],
    "wiki": [
        "super_glue/wic",
        "dbpedia_14",
        "kilt_tasks/hotpotqa", # ?
        "wiki_qa",
        "wiki_bio",
        "quoref",
        "ropes", #?
        "wiki_hop/original",
    ],
    "quora": [
        "glue/qqp",
        "paws/labeled_final", # ?
    ],
    "news": [
        "ag_news",
        "cnn_dailymail/3.0.0",
        "multi_news",
        "samsum",
        "xsum",
    ],
    "daily":[
        "social_i_qa",
        "cosmos_qa",
    ],
    "fiction":[
        "super_glue/wsc.fixed"
    ],
    "online_blogs":[
        "superglue/copa",
    ],
    "story":[
        "story_cloze/2016"
    ],
    "journal":[
        "super_glue/cb",
    ],
    "review":[
        "amazon_polarity",
        "yelp_review_full",
        "app_reviews",
    ],
    "caption":[
        "common_gen",
    ],
    "movie":[
        "duorc/SelfRC",
        "duorc/ParaphraseRC",
        "imdb",
        "rotten_tomatoes"
    ],
    "science_exam":[
        "sciq",
        "qasc"
    ],
    "arithmetic":[
        "quarel",
    ],
}





def _sample_train_data(task_name, train_split):
    train_number = len(train_split)
    prompt_number = len(DatasetTemplates(task_name).all_template_names)
    if train_number > 500000:
        sample_train_number = int(500000 / prompt_number)
        sample_train_index_list = random.sample(range(len(train_split)), k=sample_train_number)
        samples = train_split.select(sample_train_index_list)
        return samples
    else:
        return train_split


def get_dataset(data_dir, task_name, extension, for_training):
    def get_data(data_files, extension):
        assert extension in extension_list
        if extension == "jsonl":
            extension = "json"
        if extension == "tsv":
            raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t")
        elif extension == "txt":
            raw_datasets = load_dataset("txt", data_files=data_files)
        else:
            raw_datasets = load_dataset(extension, data_files=data_files)
        return raw_datasets
    if for_training:
        if task_name == "commonsense_qa":
            train_path = os.path.join(data_dir, task_name, "train_rand_split.jsonl")
            valid_path = os.path.join(data_dir, task_name, "dev_rand_split.jsonl")
            assert os.path.exists(train_path) & os.path.exists(valid_path)
            data_files = {"train": train_path, "valid": valid_path}
        elif task_name == "adversarial_qa":
            train_path = os.path.join(data_dir, task_name, "combined", "train.json")
            valid_path = os.path.join(data_dir, task_name, "combined", "dev.json")
            assert os.path.exists(train_path) & os.path.exists(valid_path)
            data_files = {"train": train_path, "valid": valid_path}
        elif task_name == "amazon_review_polarity":
            train_path = os.path.join(data_dir, task_name, "train.csv")
            valid_path = os.path.join(data_dir, task_name, "test.csv")
            assert os.path.exists(train_path) & os.path.exists(valid_path)
            data_files = {"train": train_path, "valid": valid_path}
        elif task_name == "hellaswag":
            train_path = os.path.join(data_dir, task_name, "train.jsonl")
            valid_path = os.path.join(data_dir, task_name, "val.jsonl")
            assert os.path.exists(train_path) & os.path.exists(valid_path)
            data_files = {"train": train_path, "valid": valid_path}
        elif task_name == "anli":
            train_path = os.path.join(data_dir, task_name, "anli_v0.1/R1", "train.jsonl")
            valid_path = os.path.join(data_dir, task_name, "anli_v0.1/R1", "dev.jsonl")
            assert os.path.exists(train_path) & os.path.exists(valid_path)
            data_files = {"train": train_path, "valid": valid_path}
        elif task_name == "imdb":
            train_path = os.path.join(data_dir, task_name, "train.json")
            valid_path = os.path.join(data_dir, task_name, "dev.json")
            assert os.path.exists(train_path) & os.path.exists(valid_path)
            data_files = {"train": train_path, "valid": valid_path}
        elif task_name == "web_questions":
            train_path = os.path.join(data_dir, task_name, "train.jsonl")
            valid_path = os.path.join(data_dir, task_name, "val.jsonl")
            assert os.path.exists(train_path) & os.path.exists(valid_path)
            data_files = {"train": train_path, "valid": valid_path}
        elif task_name == "PAQ": 
            train_path = os.path.join(data_dir, task_name, "PAQ.filtered.jsonl")
            valid_path = os.path.join(data_dir, task_name, "PAQ.filtered.jsonl")
            assert os.path.exists(train_path) & os.path.exists(valid_path)
            data_files = {"train": train_path, "valid": valid_path}                       
        else:
            train_path = os.path.join(data_dir, task_name, train_file_name + "." + extension)
            valid_path = os.path.join(data_dir, task_name, valid_file_name + "." + extension)
            assert os.path.exists(train_path) & os.path.exists(valid_path)
            data_files = {"train": train_path, "valid": valid_path}
        return get_data(data_files, extension)
    else:
        test_path = os.path.join(data_dir, task_name, test_file_name + "." + extension)
        assert os.path.exists(test_path)
        test_data_files = {"test": test_path}
        return get_data(test_data_files, extension)


def process_data_split(task_name, dataset, split_name):
    dataset_split = dataset[split_name]
    assert dataset_split is not None

    # Handling large datasets
    if split_name in ["train", "valid"]:
        dataset_split = _sample_train_data(task_name, dataset_split)

    # Handling special format
    dataset_split = _fix_dataset_format(task_name, dataset_split)

    # Apply prompted examples
    dataset_split = _apply_t0_prompts(task_name, dataset_split)
    return dataset_split



### TODO
def _fix_dataset_format(task_name, dataset_split):
    if task_name == "super_glue/wsc.fixed":
        span2_index = []
        span1_index = []
        span1_text = []
        span2_text = []
        for example in tqdm(dataset_split):
            span2_index.append(example["target"]["span2_index"])
            span1_index.append(example["target"]["span1_index"])
            span1_text.append(example["target"]["span1_text"])
            span2_text.append(example["target"]["span2_text"])
        dataset_split = dataset_split.add_column(name="span2_index", column=span2_index)
        dataset_split = dataset_split.add_column(name="span1_index", column=span1_index)
        dataset_split = dataset_split.add_column(name="span1_text", column=span1_text)
        dataset_split = dataset_split.add_column(name="span2_text", column=span2_text)
        return dataset_split
    elif task_name == "adversarial_qa":

        return dataset_split
    elif task_name == "ag_news":
        def change_label_ag(example):
            example['label'] = example['label'] - 1 
            return example
        # text_combined_ls = []
        # for example in tqdm(dataset_split):
        #     text_combined = example["title"] + " " + example["content"]
        #     text_combined_ls.append(text_combined)
        dataset_split = dataset_split.map(lambda example: {'text': example["title"] + " " + example["content"]})
        dataset_split = dataset_split.remove_columns(["title","content"])
        dataset_split = dataset_split.map(change_label_ag)
        return dataset_split
    elif task_name == "amazon_review_polarity":
        return dataset_split
    elif task_name == "anli":
        def transfer_label_anli(example):
            mapping_label = {"c":"2","e":"0","n":"1"}
            example["label"] = mapping_label[example["label"]]
            return example
        # dataset_split = dataset_split.map(transfer_label_anli)
        dataset_split = dataset_split.rename_column("context","premise")
        return dataset_split
    elif task_name == "commonsense_qa":
        def transfer_qa(example):
            ori_choices = example["question"]["choices"]
            choice = {"label":["A","B","C","D","E"],"text":[]}
            for label_text in ori_choices:
                choice["text"].append(label_text["text"])
            example["choices"] = choice
            return example
        dataset_split = dataset_split.map(transfer_qa)
        dataset_split = dataset_split.map(lambda example: {'question': example["question"]["stem"]})
        return dataset_split
    elif task_name == "common_gen":
        # sample
        dataset_split = dataset_split.map(lambda example: {'concepts': example["concept_set"].split("#")})
        dataset_split = dataset_split.map(lambda example: {'target': example["scene"][0]})
        return dataset_split
    elif task_name == 'dbpedia_14':
        def change_label_db(example):
            example['label'] = example['label'] - 1 
            return example
        dataset_split = dataset_split.map(change_label_db)
        return dataset_split
    elif task_name == "imdb":
        def label_imdb(example):
            mapping_dict = {"pos":1, "neg":0}
            example["label"] = mapping_dict[example["label"]]
            return example
        dataset_split = dataset_split.rename_column("src","text")
        dataset_split = dataset_split.rename_column("target","label")
        dataset_split = dataset_split.map(label_imdb)
        return dataset_split

    # elif task_name in ["openbookqa/main", "qasc"]:
    #     choices = []
    #     for example in tqdm(dataset_split):
    #         choices.append(example["question"]["choices"])
    #     dataset_split = dataset_split.add_column(name="choices", column=choices)
        return dataset_split

    elif task_name == "glue/mrpc":
        dataset_split = dataset_split.rename_column("#1 String","sentence1")
        dataset_split = dataset_split.rename_column("#2 String", "sentence2")
        if "Quality" in dataset_split.features:
            dataset_split = dataset_split.rename_column("Quality", "label")
        return dataset_split
    elif task_name == "glue/qqp":
        dataset_split = dataset_split.rename_column("is_duplicate","label")
        return dataset_split
    elif task_name == "openbookqa/main":
        def transfer_qa_ob(example):
            ori_choices = example["question"]["choices"]
            choice = {"label":["A","B","C","D"],"text":[]}
            for label_text in ori_choices:
                choice["text"].append(label_text["text"])
            example["choices"] = choice
            return example
        dataset_split = dataset_split.map(transfer_qa_ob)
        dataset_split = dataset_split.map(lambda example: {'question_stem': example["question"]["stem"]})
        return dataset_split
    elif task_name == "super_glue/cb":
        def label_cb(example):
            mapping_dict = {"entailment":0, "contradiction":1, "neutral":2}
            example["label"] = mapping_dict[example["label"]]
            return example
        dataset_split = dataset_split.map(label_cb)
        return dataset_split
    elif task_name == "super_glue/wic":
        def label_wic(example):
            mapping_dict = {False:0, True:1}
            example["label"] = mapping_dict[example["label"]]
            return example
        dataset_split = dataset_split.map(label_wic)
        return dataset_split
    elif task_name == "super_glue/rte":
        def label_rte(example):
            mapping_dict = {"not_entailment":1, "entailment":0}
            example["label"] = mapping_dict[example["label"]]
            return example
        dataset_split = dataset_split.map(label_rte)
        return dataset_split
    elif task_name == "wiki_qa":
        dataset_split = dataset_split.rename_column("Question", "question")
        dataset_split = dataset_split.rename_column("Label", "label")
        dataset_split = dataset_split.rename_column("DocumentTitle", "document_title")
        dataset_split = dataset_split.rename_column("Sentence", "answer")
        return dataset_split
    elif task_name == "wiqa":
        def transfer_qa_wiqa(example):
            ori_choices = example["question"]["choices"]
            choice = {"label":["A","B","C"],"text":[]}
            for label_text in ori_choices:
                choice["text"].append(label_text["text"])
            example["choices"] = choice
            return example
        dataset_split = dataset_split.map(lambda example: {'question_para_step': example["question"]["para_steps"]})
        dataset_split = dataset_split.map(lambda example: {'question_stem': example["question"]["stem"]})
        dataset_split = dataset_split.map(lambda example: {'answer_label_as_choice': example["question"]["answer_label_as_choice"]})
        dataset_split = dataset_split.map(lambda example: {'answer_label': example["question"]["answer_label"]})
        dataset_split = dataset_split.map(lambda example: {'metadata_question_type': example["metadata"]["question_type"]})
        dataset_split = dataset_split.map(transfer_qa_wiqa)
        return dataset_split
    elif task_name == "PAQ":
        dataset_split = dataset_split.map(lambda example: {'answers': example["answer"]})
        return dataset_split
    else:
        return dataset_split

def _apply_t0_prompts(task_name, dataset_split):
    def prompt_input_target(example):
        results = prompt.apply(example)
        if len(results) == 1 and len(results[0]) == 0: # some labels will lead to None, e.g., if two sentences are not similar to each other (glue/mrpc)
            example[f'prompt_input_{index}'] = None
            example[f'prompt_target_{index}'] = None
        else:
            example[f'prompt_input_{index}'] = results[0]
            example[f'prompt_target_{index}'] = results[1]
        return example
    prompts = DatasetTemplates(task_name)
    all_prompt_names = prompts.all_template_names

    print(f" [*] Task:{task_name} Number of Prompts: {len(all_prompt_names)}.")
    if task_name == "imdb":
        all_prompt_names = all_prompt_names[:2]
    if task_name == "PAQ":
        prompts = DatasetTemplates('web_questions')
        all_prompt_names = prompts.all_template_names
    for index, prompt_name in enumerate(all_prompt_names):
        print("The prompt_name is {}.".format(prompt_name))
        prompt = prompts[prompt_name]
        inputs = []
        targets = []
        dataset_split=dataset_split.map(prompt_input_target)
        # dataset_split=dataset_split.map(lambda example: {f'prompt_input_{index}': prompt.apply(example)[0]})
        # dataset_split=dataset_split.map(lambda example: {f'prompt_target_{index}': prompt.apply(example)[1]})
        # for example in tqdm(dataset_split):
        #     results = prompt.apply(example)
        #     if len(results) == 1 and len(results[0]) == 0:
        #         print(results)
        #         inputs.append(None)
        #         targets.append(None)
        #     else:
        #         inputs.append(results[0])
        #         targets.append(results[1])

        # if TEST:
        #     continue

        # dataset_split=dataset_split.add_column(name="prompt_input_"+str(index), column=inputs)
        # dataset_split=dataset_split.add_column(name="prompt_target_"+str(index), column=targets)
    return dataset_split





def preprocess_data(args, task_name, raw_datasets, tokenizer):
    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False
    def preprocess_function(examples):
        inputs = []
        targets = []
        prompt_number = len(DatasetTemplates(task_name).all_template_names)
        for index in range(prompt_number):
            input_name = "prompt_input_" + str(index)
            output_name = "prompt_target_" + str(index)
            inputs = inputs + examples[input_name]
            targets = targets + examples[output_name]
        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"]=[[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    keyword = list(raw_datasets.keys())[0]
    column_names = raw_datasets[keyword].column_names
    processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",)
    return processed_datasets

class MultiTaskData:
    def __init__(self,
                 args,
                 tokenizer,
                 train_task_list,
                 test_task_list):

        self.args = args
        self.data_dir = args.data_dir
        self.tokenizer=tokenizer
        self.train_task_list = train_task_list
        self.test_task_list = test_task_list

        train_task_dataset_dict = self._load_datasets(train_task_list, for_training=True)
        test_task_dataset_dict = self._load_datasets(test_task_list, for_training=False)

        self.train_task_dataset_dict = self._postprocess_data_split(train_task_dataset_dict, for_training=True)
        self.test_task_dataset_dict = self._postprocess_data_split(test_task_dataset_dict, for_training=False)

        self.train_dataset = {task_name: dataset["train"] for task_name, dataset in self.train_task_dataset_dict.items()}
        self.valid_dataset = {task_name: dataset["valid"] for task_name, dataset in self.train_task_dataset_dict.items()}
        self.test_dataset = {task_name: dataset["test"] for task_name, dataset in self.test_task_dataset_dict.items()}


    def _load_datasets(self, task_list, for_training):
        task_to_dataset_dict = {}
        for task_name in task_list:
            extension = TASK_TO_EXT_DICT[task_name]
            raw_datasets = get_dataset(self.data_dir, task_name, extension, for_training)
            task_to_dataset_dict[task_name] = raw_datasets
        return task_to_dataset_dict

    def _postprocess_data_split(self, task_dataset_dict, for_training):
        if for_training:
            train_task_to_dataset = {}
            for task_name, dataset in task_dataset_dict.items():
                train_split = process_data_split(task_name, dataset, split_name="train")
                valid_split = process_data_split(task_name, dataset, split_name="valid")
                new_dataset = DatasetDict({"train": train_split, "valid": valid_split})
                tokenized_dataset = preprocess_data(self.args, task_name, new_dataset, self.tokenizer)
                train_task_to_dataset[task_name] = tokenized_dataset
            return train_task_to_dataset
        else:
            test_task_to_dataset = {}
            for task_name, dataset in task_dataset_dict.items():
                test_split = process_data_split(task_name, dataset, split_name="test")
                new_dataset = DatasetDict({"test": test_split})
                tokenized_dataset = preprocess_data(self.args, task_name, new_dataset, self.tokenizer)
                test_task_to_dataset[task_name] = tokenized_dataset
            return test_task_to_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test Data Utilities.")
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--extension", type=str)
    parser.add_argument("--overwrite_cache", type=bool, default=True)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--pad_to_max_length",action="store_true")
    parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True)

    args = parser.parse_args()

    data_dir = "/home/linzongyu/multi_task/data"
    tokenizer = AutoTokenizer.from_pretrained("google/t5-small-lm-adapt")

    raw_datasets = get_dataset(data_dir, args.task_name, args.extension, for_training=True)
    # test_datasets = get_dataset(data_dir, args.task_name, args.extension, for_training=False)
    print(raw_datasets)
    if args.task_name == "web_questions":
        raw_datasets_sample = raw_datasets
    else:
        raw_datasets_sample = DatasetDict({"train": raw_datasets["train"].select(list(range(10))),
                                    "valid": raw_datasets["valid"].select(list(range(10)))})
    # test_datasets = DatasetDict({"test": test_datasets["test"].select(list(range(10)))})

    train_split = process_data_split(args.task_name, raw_datasets_sample, split_name="train")
    valid_split = process_data_split(args.task_name, raw_datasets_sample, split_name="valid")
    # test_split = process_data_split(args.task_name, test_datasets, split_name="test")

    new_dataset = DatasetDict({"train": train_split, "valid": valid_split})
    # new_test_dataset = DatasetDict({"test": test_split})

    # tokenized_dataset = preprocess_data(args, args.task_name, new_dataset, tokenizer)
    # test_tokenized_dataset = preprocess_data(args, args.task_name, new_test_dataset, tokenizer)