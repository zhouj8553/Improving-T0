# coding=utf-8
# Copyright 2020 BigScience Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""P3 (Public Pool of Prompts)"""

import os
import datasets
import json
import urllib
from collections import defaultdict
# import tensorflow as tf
import torch

_CITATION = """@misc{sanh2021multitask,
      title={Multitask Prompted Training Enables Zero-Shot Task Generalization},
      author={Victor Sanh and Albert Webson and Colin Raffel and Stephen H. Bach and Lintang Sutawika and Zaid Alyafeai and Antoine Chaffin and Arnaud Stiegler and Teven Le Scao and Arun Raja and Manan Dey and M Saiful Bari and Canwen Xu and Urmish Thakker and Shanya Sharma Sharma and Eliza Szczechla and Taewoon Kim and Gunjan Chhablani and Nihal Nayak and Debajyoti Datta and Jonathan Chang and Mike Tian-Jian Jiang and Han Wang and Matteo Manica and Sheng Shen and Zheng Xin Yong and Harshit Pandey and Rachel Bawden and Thomas Wang and Trishala Neeraj and Jos Rozen and Abheesht Sharma and Andrea Santilli and Thibault Fevry and Jason Alan Fries and Ryan Teehan and Stella Biderman and Leo Gao and Tali Bers and Thomas Wolf and Alexander M. Rush},
      year={2021},
      eprint={2110.08207},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}"""

_DESCRIPTION = """\
P3 (Public Pool of Prompts) is a collection of prompted English datasets covering a diverse set of NLP tasks. A prompt is the combination of an input template and a target template. The templates are functions mapping a data example into natural language for the input and target sequences. For example, in the case of an NLI dataset, the data example would include fields for *Premise, Hypothesis, Label*. An input template would be *If {Premise} is true, is it also true that {Hypothesis}?*, whereas a target template can be defined with the label choices *Choices[label]*. Here *Choices* is prompt-specific metadata that consists of the options *yes, maybe, no* corresponding to *label* being entailment (0), neutral (1) or contradiction (2).

Prompts are collected using [Promptsource](https://github.com/bigscience-workshop/promptsource), an interface to interactively write prompts on datasets, and collect prompt-specific metadata such as evaluation metrics. As of October 13th, there are 2'000 prompts collected for 270+ data(sub)sets. The collection of prompts of P3 is publicly available on [Promptsource](https://github.com/bigscience-workshop/promptsource).

To train [T0*](https://huggingface.co/bigscience/T0pp), we used a subset of the prompts available in Promptsource (see details [here](https://huggingface.co/bigscience/T0pp#training-data)). However, some of the prompts use `random.choice`, a method that selects uniformly at random an option in a list of valid possibilities. For reproducibility purposes, we release the collection of prompted examples used to train T0*. **The data available here are the materialized version of the prompted datasets used in [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207) which represent only a subset of the datasets for which there is at least one prompt in Promptsource.**
"""

_LICENSE = "Apache License 2.0"

_HOMEPAGE = "https://github.com/bigscience-workshop/promptsource"

# _HUB_PATH = "https://huggingface.co/datasets/bigscience/P3/raw/main"


logger = datasets.logging.get_logger(__name__)

"""
def load_cached_task(features_dict, tfrecord):
    # Use `FixedLenSequenceFeature` for sequences with variable length.
    def _feature_config(shape, dtype):
        if dtype in ("int32", "bool"):
            # int32 and bool are stored as int64 in the tf.train.Example protobuf.
            dtype = "int64"
        if shape and shape[0] is None:
            return tf.io.FixedLenSequenceFeature(
                shape[1:], dtype, allow_missing=True
            )
        return tf.io.FixedLenFeature(shape, dtype)

    feature_description = {
        feat: _feature_config(**desc) for feat, desc in features_dict.items()
    }

    ds = tf.data.TFRecordDataset(tf.io.gfile.glob([tfrecord])) # TODO -> handle multiple shards
    ds = ds.map(
        lambda pb: tf.io.parse_single_example(pb, feature_description),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Cast features back to the types from the info JSON since some features
    # must be cast for storage (e.g., int32 is stored as int64).
    ds = ds.map(
        lambda x: {k: tf.cast(v, features_dict[k]["dtype"]) for k, v in x.items()},
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return ds
"""
"""
def read_from_url(url):
    try:
        content = urllib.request.urlopen(url, timeout=10.0)
        logger.info(f"Downloaded {url}")
    except urllib.error.URLError as e:
        raise ConnectionError(e)
    return content.read().decode("utf-8")
"""

# def find_task_splits_and_features_dict():
#     """Get the task available (list was pre-computed by `print_data_split_sizes.py`), and get the features for each task."""
#     task_splits_and_features = defaultdict(dict)
#     targets_max_tokens_dict = defaultdict(dict)
#     file_path = os.path.join(_RAW_DATA_PATH, "data_split_sizes.csv")
#     task_to_split_dict = {}
#     with open(file_path, "r") as f:
#         for idx, line in enumerate(f):
#             if idx == 0:
#                 continue
#             line = line.strip()
#             line_splits = line.split("|")
#             task_to_split_dict[line_splits[0]] = json.loads(line_splits[1])

#     for task_name, split_sizes in task_to_split_dict.items():
#         for split_name in split_sizes.keys():
#             ## TODO: change the path
#             split_file_path=f"{_DATA_PATH}/{task_name}/info.{split_name}.json"
#             split_info = json.loads(open(split_file_path, "r").read())
#             features_dict = split_info["features"]
#             assert split_info["num_shards"] == 1 # TODO -> handle multiple shards
#             if not task_splits_and_features[task_name]:
#                 task_splits_and_features[task_name] = {
#                     "splits": [],
#                     "features_dict": features_dict,
#                 }
#             task_splits_and_features[task_name]["splits"].append(split_name)
#             assert features_dict == task_splits_and_features[task_name]["features_dict"]
#         info_file_path=f"{_DATA_PATH}/{task_name}/stats.train.json"
#         split_info = json.loads(open(info_file_path, "r").read())
#         targets_max_tokens = split_info["targets_max_tokens"]
#         targets_max_tokens_dict[task_name] = int(targets_max_tokens)
#     return task_splits_and_features, targets_max_tokens_dict


# _TASK_TARGET_MAX_TOKENS_DICT={'adversarial_qa_dbert_answer_the_following_q': 385, 'adversarial_qa_dbert_based_on': 385, 'adversarial_qa_dbert_generate_question': 59, 'adversarial_qa_dbert_question_context_answer': 385, 'adversarial_qa_dbert_tell_what_it_is': 385, 'adversarial_qa_dbidaf_answer_the_following_q': 179, 'adversarial_qa_dbidaf_based_on': 179, 'adversarial_qa_dbidaf_generate_question': 58, 'adversarial_qa_dbidaf_question_context_answer': 179, 'adversarial_qa_dbidaf_tell_what_it_is': 179, 'adversarial_qa_droberta_answer_the_following_q': 212, 'adversarial_qa_droberta_based_on': 212, 'adversarial_qa_droberta_generate_question': 48, 'adversarial_qa_droberta_question_context_answer': 212, 'adversarial_qa_droberta_tell_what_it_is': 212, 'ag_news_classify': 3, 'ag_news_classify_question_first': 3, 'ag_news_classify_with_choices': 3, 'ag_news_classify_with_choices_question_first': 3, 'ag_news_recommend': 4, 'ag_news_which_section': 3, 'ag_news_which_section_choices': 3, 'ai2_arc_ARC_Challenge_heres_a_problem': 1, 'ai2_arc_ARC_Challenge_i_am_hesitating': 41, 'ai2_arc_ARC_Challenge_multiple_choice': 41, 'ai2_arc_ARC_Challenge_pick_false_options': 130, 'ai2_arc_ARC_Challenge_pick_the_most_correct_option': 1, 'ai2_arc_ARC_Challenge_qa_options': 41, 'ai2_arc_ARC_Easy_heres_a_problem': 1, 'ai2_arc_ARC_Easy_i_am_hesitating': 51, 'ai2_arc_ARC_Easy_multiple_choice': 51, 'ai2_arc_ARC_Easy_pick_false_options': 122, 'ai2_arc_ARC_Easy_pick_the_most_correct_option': 1, 'ai2_arc_ARC_Easy_qa_options': 51, 'amazon_polarity_Is_this_product_review_positive': 1, 'amazon_polarity_Is_this_review': 2, 'amazon_polarity_Is_this_review_negative': 1, 'amazon_polarity_User_recommend_this_product': 1, 'amazon_polarity_convey_negative_or_positive_sentiment': 2, 'amazon_polarity_flattering_or_not': 4, 'amazon_polarity_negative_or_positive_tone': 2, 'amazon_polarity_user_satisfied': 5, 'amazon_polarity_would_you_buy': 1, 'anli_GPT_3_style_r1': 3, 'anli_GPT_3_style_r1_score_eval': 3, 'anli_GPT_3_style_r2': 3, 'anli_GPT_3_style_r2_score_eval': 3, 'anli_GPT_3_style_r3': 3, 'anli_GPT_3_style_r3_score_eval': 3, 'anli_MNLI_crowdsource_r1': 5, 'anli_MNLI_crowdsource_r1_score_eval': 5, 'anli_MNLI_crowdsource_r2': 5, 'anli_MNLI_crowdsource_r2_score_eval': 5, 'anli_MNLI_crowdsource_r3': 5, 'anli_MNLI_crowdsource_r3_score_eval': 5, 'anli_always_sometimes_never_r1': 1, 'anli_always_sometimes_never_r1_score_eval': 1, 'anli_always_sometimes_never_r2': 1, 'anli_always_sometimes_never_r2_score_eval': 1, 'anli_always_sometimes_never_r3': 1, 'anli_always_sometimes_never_r3_score_eval': 1, 'anli_based_on_the_previous_passage_r1': 1, 'anli_based_on_the_previous_passage_r1_score_eval': 1, 'anli_based_on_the_previous_passage_r2': 1, 'anli_based_on_the_previous_passage_r2_score_eval': 1, 'anli_based_on_the_previous_passage_r3': 1, 'anli_based_on_the_previous_passage_r3_score_eval': 1, 'anli_can_we_infer_r1': 1, 'anli_can_we_infer_r1_score_eval': 1, 'anli_can_we_infer_r2': 1, 'anli_can_we_infer_r2_score_eval': 1, 'anli_can_we_infer_r3': 1, 'anli_can_we_infer_r3_score_eval': 1, 'anli_claim_true_false_inconclusive_r1': 5, 'anli_claim_true_false_inconclusive_r1_score_eval': 5, 'anli_claim_true_false_inconclusive_r2': 5, 'anli_claim_true_false_inconclusive_r2_score_eval': 5, 'anli_claim_true_false_inconclusive_r3': 5, 'anli_claim_true_false_inconclusive_r3_score_eval': 5, 'anli_consider_always_sometimes_never_r1': 1, 'anli_consider_always_sometimes_never_r1_score_eval': 1, 'anli_consider_always_sometimes_never_r2': 1, 'anli_consider_always_sometimes_never_r2_score_eval': 1, 'anli_consider_always_sometimes_never_r3': 1, 'anli_consider_always_sometimes_never_r3_score_eval': 1, 'anli_does_it_follow_that_r1': 1, 'anli_does_it_follow_that_r1_score_eval': 1, 'anli_does_it_follow_that_r2': 1, 'anli_does_it_follow_that_r2_score_eval': 1, 'anli_does_it_follow_that_r3': 1, 'anli_does_it_follow_that_r3_score_eval': 1, 'anli_does_this_imply_r1': 1, 'anli_does_this_imply_r1_score_eval': 1, 'anli_does_this_imply_r2': 1, 'anli_does_this_imply_r2_score_eval': 1, 'anli_does_this_imply_r3': 1, 'anli_does_this_imply_r3_score_eval': 1, 'anli_guaranteed_possible_impossible_r1': 4, 'anli_guaranteed_possible_impossible_r1_score_eval': 4, 'anli_guaranteed_possible_impossible_r2': 4, 'anli_guaranteed_possible_impossible_r2_score_eval': 4, 'anli_guaranteed_possible_impossible_r3': 4, 'anli_guaranteed_possible_impossible_r3_score_eval': 4, 'anli_guaranteed_true_r1': 1, 'anli_guaranteed_true_r1_score_eval': 1, 'anli_guaranteed_true_r2': 1, 'anli_guaranteed_true_r2_score_eval': 1, 'anli_guaranteed_true_r3': 1, 'anli_guaranteed_true_r3_score_eval': 1, 'anli_justified_in_saying_r1': 1, 'anli_justified_in_saying_r1_score_eval': 1, 'anli_justified_in_saying_r2': 1, 'anli_justified_in_saying_r2_score_eval': 1, 'anli_justified_in_saying_r3': 1, 'anli_justified_in_saying_r3_score_eval': 1, 'anli_must_be_true_r1': 1, 'anli_must_be_true_r1_score_eval': 1, 'anli_must_be_true_r2': 1, 'anli_must_be_true_r2_score_eval': 1, 'anli_must_be_true_r3': 1, 'anli_must_be_true_r3_score_eval': 1, 'anli_should_assume_r1': 1, 'anli_should_assume_r1_score_eval': 1, 'anli_should_assume_r2': 1, 'anli_should_assume_r2_score_eval': 1, 'anli_should_assume_r3': 1, 'anli_should_assume_r3_score_eval': 1, 'anli_take_the_following_as_truth_r1': 5, 'anli_take_the_following_as_truth_r1_score_eval': 5, 'anli_take_the_following_as_truth_r2': 5, 'anli_take_the_following_as_truth_r2_score_eval': 5, 'anli_take_the_following_as_truth_r3': 5, 'anli_take_the_following_as_truth_r3_score_eval': 5, 'app_reviews_categorize_rating_using_review': 3, 'app_reviews_convert_to_rating': 1, 'app_reviews_convert_to_star_rating': 2, 'app_reviews_generate_review': 678, 'cnn_dailymail_3.0.0_2_or_3_sentences': 848, 'cnn_dailymail_3.0.0_generate_story': 1185, 'cnn_dailymail_3.0.0_news_card_view': 848, 'cnn_dailymail_3.0.0_news_stock': 848, 'cnn_dailymail_3.0.0_news_summary': 848, 'cnn_dailymail_3.0.0_spice_up_story': 1185, 'cnn_dailymail_3.0.0_sum_in_brief': 848, 'cnn_dailymail_3.0.0_tldr_summary': 848, 'cnn_dailymail_3.0.0_write_an_outline': 848, 'common_gen_Example_prompt': 35, 'common_gen_Given_concepts_type_1': 35, 'common_gen_Given_concepts_type_2': 35, 'common_gen_Put_together': 35, 'common_gen_choice_in_concept_centric_sentence_generation': 35, 'common_gen_random_task_template_prompt': 35, 'common_gen_sentence_to_concepts': 18, 'common_gen_topic_to_sentence': 35, 'common_gen_topics_from_the_sentence': 18, 'cos_e_v1.11_aligned_with_common_sense': 68, 'cos_e_v1.11_description_question_option_id': 1, 'cos_e_v1.11_description_question_option_text': 9, 'cos_e_v1.11_explain_why_human': 68, 'cos_e_v1.11_generate_explanation_given_text': 68, 'cos_e_v1.11_i_think': 68, 'cos_e_v1.11_question_description_option_id': 1, 'cos_e_v1.11_question_description_option_text': 9, 'cos_e_v1.11_question_option_description_id': 1, 'cos_e_v1.11_question_option_description_text': 9, 'cos_e_v1.11_rationale': 68, 'cosmos_qa_context_answer_to_question': 46, 'cosmos_qa_context_description_question_answer_id': 1, 'cosmos_qa_context_description_question_answer_text': 53, 'cosmos_qa_context_description_question_text': 53, 'cosmos_qa_context_question_description_answer_id': 1, 'cosmos_qa_context_question_description_answer_text': 53, 'cosmos_qa_context_question_description_text': 53, 'cosmos_qa_description_context_question_answer_id': 1, 'cosmos_qa_description_context_question_answer_text': 53, 'cosmos_qa_description_context_question_text': 53, 'cosmos_qa_no_prompt_id': 1, 'cosmos_qa_no_prompt_text': 53, 'cosmos_qa_only_question_answer': 53, 'dbpedia_14_given_a_choice_of_categories_': 4, 'dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to': 4, 'dbpedia_14_given_list_what_category_does_the_paragraph_belong_to': 4, 'dbpedia_14_pick_one_category_for_the_following_text': 4, 'dream_answer_to_dialogue': 647, 'dream_baseline': 24, 'dream_generate_first_utterance': 192, 'dream_generate_last_utterance': 349, 'dream_read_the_following_conversation_and_answer_the_question': 24, 'duorc_ParaphraseRC_answer_question': 42, 'duorc_ParaphraseRC_build_story_around_qa': 946, 'duorc_ParaphraseRC_decide_worth_it': 42, 'duorc_ParaphraseRC_extract_answer': 42, 'duorc_ParaphraseRC_generate_question': 66, 'duorc_ParaphraseRC_generate_question_by_answer': 66, 'duorc_ParaphraseRC_movie_director': 42, 'duorc_ParaphraseRC_question_answering': 42, 'duorc_ParaphraseRC_title_generation': 21, 'duorc_SelfRC_answer_question': 35, 'duorc_SelfRC_build_story_around_qa': 667, 'duorc_SelfRC_decide_worth_it': 35, 'duorc_SelfRC_extract_answer': 35, 'duorc_SelfRC_generate_question': 66, 'duorc_SelfRC_generate_question_by_answer': 66, 'duorc_SelfRC_movie_director': 35, 'duorc_SelfRC_question_answering': 35, 'duorc_SelfRC_title_generation': 23, 'gigaword_TLDR': 97, 'gigaword_first_sentence_title': 97, 'gigaword_generate_summary_for_this': 97, 'gigaword_in_a_nutshell': 97, 'gigaword_make_a_title': 97, 'gigaword_reverse_writing': 280, 'gigaword_write_a_title_for_this_sentence': 97, 'gigaword_write_an_article': 280, 'gigaword_write_its_sentence': 97, \
# 'glue_mrpc_equivalent': 2, 'glue_mrpc_generate_paraphrase': 66, 'glue_mrpc_generate_sentence': 66, 'glue_mrpc_paraphrase': 1, 'glue_mrpc_replace': 1, 'glue_mrpc_same_thing': 1, 'glue_mrpc_want_to_know': 1, 'glue_qqp_answer': 1, 'glue_qqp_duplicate': 1, 'glue_qqp_duplicate_or_not': 3, 'glue_qqp_meaning': 0, 'glue_qqp_quora': 1, 'glue_qqp_same_thing': 1, 'hellaswag_Appropriate_continuation_Yes_or_No': 1, 'hellaswag_Open_ended_completion': 102, 'hellaswag_Open_ended_start': 139, 'hellaswag_Predict_ending_with_hint': 102, 'hellaswag_Predict_ending_with_hint_score_eval': 127, 'hellaswag_Randomized_prompts_template': 102, 'hellaswag_Randomized_prompts_template_score_eval': 127, 'hellaswag_Reversed_appropriate_continuation_Yes_or_No': 1, 'hellaswag_Topic_of_the_context': 10, 'hellaswag_Topic_without_the_ending_answer': 10, 'hellaswag_complete_first_then': 102, 'hellaswag_complete_first_then_score_eval': 127, 'hellaswag_how_ends': 3, 'hellaswag_if_begins_how_continues': 3, 'hellaswag_if_begins_how_continues_score_eval': 3, 'imdb_Movie_Expressed_Sentiment': 1, 'imdb_Movie_Expressed_Sentiment_2': 1, 'imdb_Negation_template_for_positive_and_negative': 3, 'imdb_Reviewer_Enjoyment': 7, 'imdb_Reviewer_Enjoyment_Yes_No': 1, 'imdb_Reviewer_Expressed_Sentiment': 1, 'imdb_Reviewer_Opinion_bad_good_choices': 1, 'imdb_Reviewer_Sentiment_Feeling': 1, 'imdb_Sentiment_with_choices_': 1, 'imdb_Text_Expressed_Sentiment': 1, 'imdb_Writer_Expressed_Sentiment': 1, 'kilt_tasks_hotpotqa_combining_facts': 187, 'kilt_tasks_hotpotqa_complex_question': 187, 'kilt_tasks_hotpotqa_final_exam': 187, 'kilt_tasks_hotpotqa_formulate': 187, 'kilt_tasks_hotpotqa_straighforward_qa': 187, 'multi_news_distill': 604, 'multi_news_expand_reverse_task_': 1082, 'multi_news_summarize': 604, 'multi_news_summary_scenario': 604, 'multi_news_synthesize': 604, 'multi_news_what_are_the_key_points': 604, 'openbookqa_main_choices': 27, 'openbookqa_main_choose_an_answer_with_options': 27, 'openbookqa_main_only_options': 27, 'openbookqa_main_pick_answer_with_options': 27, 'openbookqa_main_pick_using_id': 1, 'openbookqa_main_which_correct': 27, 'openbookqa_main_which_correct_inverse': 27, 'paws_labeled_final_Concatenation': 1, 'paws_labeled_final_Concatenation_no_label': 1, 'paws_labeled_final_Meaning': 1, 'paws_labeled_final_Meaning_no_label': 1, 'paws_labeled_final_PAWS_ANLI_GPT3': 3, 'paws_labeled_final_PAWS_ANLI_GPT3_no_label': 1, 'paws_labeled_final_Rewrite': 1, 'paws_labeled_final_Rewrite_no_label': 1, 'paws_labeled_final_context_question': 1, 'paws_labeled_final_context_question_no_label': 1, 'paws_labeled_final_paraphrase_task': 75, 'paws_labeled_final_task_description_no_label': 1, 'piqa_Correct_the_solution': 511, 'piqa_Correct_the_solution_if_false_from_sol_1': 519, 'piqa_Correct_the_solution_if_false_from_sol_2': 519, 'piqa_Does_this_solution_make_sense_sol1': 1, 'piqa_Does_this_solution_make_sense_sol2': 1, 'piqa_choose_the_most_appropriate_solution': 2, 'piqa_finish_sentence_with_correct_choice': 511, 'piqa_no_prompt_needed': 511, 'piqa_pick_correct_choice_index': 1, 'piqa_pick_correct_choice_with_choice_given_before_goal': 511, 'piqa_what_is_the_correct_ending': 511, 'qasc_is_correct_1': 1, 'qasc_is_correct_2': 1, 'qasc_qa_with_combined_facts_1': 16, 'qasc_qa_with_separated_facts_1': 16, 'qasc_qa_with_separated_facts_2': 16, 'qasc_qa_with_separated_facts_3': 16, 'qasc_qa_with_separated_facts_4': 16, 'qasc_qa_with_separated_facts_5': 16, 'quail_context_description_question_answer_id': 1, 'quail_context_description_question_answer_text': 44, 'quail_context_description_question_text': 44, 'quail_context_question_answer_description_id': 1, 'quail_context_question_answer_description_text': 44, 'quail_context_question_description_answer_id': 1, 'quail_context_question_description_answer_text': 44, 'quail_context_question_description_text': 44, 'quail_description_context_question_answer_id': 1, 'quail_description_context_question_answer_text': 44, 'quail_description_context_question_text': 44, 'quail_no_prompt_id': 1, 'quail_no_prompt_text': 44, 'quarel_choose_between': 10, 'quarel_do_not_use': 10, 'quarel_heres_a_story': 10, 'quarel_logic_test': 10, 'quarel_testing_students': 10, 'quartz_answer_question_based_on': 19, 'quartz_answer_question_below': 19, 'quartz_given_the_fact_answer_the_q': 19, 'quartz_having_read_above_passage': 19, 'quartz_paragraph_question_plain_concat': 19, 'quartz_read_passage_below_choose': 19, 'quartz_use_info_from_paragraph_question': 19, 'quartz_use_info_from_question_paragraph': 19, 'quoref_Answer_Friend_Question': 44, 'quoref_Answer_Question_Given_Context': 44, 'quoref_Answer_Test': 44, 'quoref_Context_Contains_Answer': 44, 'quoref_Find_Answer': 44, 'quoref_Found_Context_Online': 44, 'quoref_Given_Context_Answer_Question': 44, 'quoref_Guess_Answer': 44, 'quoref_Guess_Title_For_Context': 36, 'quoref_Read_And_Extract_': 44, 'quoref_What_Is_The_Answer': 44, 'race_high_Is_this_the_right_answer': 1, 'race_high_Read_the_article_and_answer_the_question_no_option_': 133, 'race_high_Select_the_best_answer': 1, 'race_high_Select_the_best_answer_generate_span_': 133, 'race_high_Select_the_best_answer_no_instructions_': 1, 'race_high_Taking_a_test': 1, 'race_high_Write_a_multi_choice_question_for_the_following_article': 392, 'race_high_Write_a_multi_choice_question_options_given_': 88, 'race_middle_Is_this_the_right_answer': 1, 'race_middle_Read_the_article_and_answer_the_question_no_option_': 34, 'race_middle_Select_the_best_answer': 1, 'race_middle_Select_the_best_answer_generate_span_': 34, 'race_middle_Select_the_best_answer_no_instructions_': 1, 'race_middle_Taking_a_test': 1, 'race_middle_Write_a_multi_choice_question_for_the_following_article': 136, 'race_middle_Write_a_multi_choice_question_options_given_': 89, 'ropes_background_new_situation_answer': 20, 'ropes_background_situation_middle': 20, 'ropes_given_background_situation': 20, 'ropes_new_situation_background_answer': 20, 'ropes_plain_background_situation': 20, 'ropes_plain_bottom_hint': 20, 'ropes_plain_no_background': 20, 'ropes_prompt_beginning': 20, 'ropes_prompt_bottom_hint_beginning': 20, 'ropes_prompt_bottom_no_hint': 20, 'ropes_prompt_mix': 20, 'ropes_read_background_situation': 20, 'rotten_tomatoes_Movie_Expressed_Sentiment': 1, 'rotten_tomatoes_Movie_Expressed_Sentiment_2': 1, 'rotten_tomatoes_Reviewer_Enjoyment': 6, 'rotten_tomatoes_Reviewer_Enjoyment_Yes_No': 1, 'rotten_tomatoes_Reviewer_Expressed_Sentiment': 1, 'rotten_tomatoes_Reviewer_Opinion_bad_good_choices': 1, 'rotten_tomatoes_Reviewer_Sentiment_Feeling': 1, 'rotten_tomatoes_Sentiment_with_choices_': 1, 'rotten_tomatoes_Text_Expressed_Sentiment': 1, 'rotten_tomatoes_Writer_Expressed_Sentiment': 1, 'samsum_Generate_a_summary_for_this_dialogue': 93, 'samsum_Given_the_above_dialogue_write_a_summary': 93, 'samsum_Sum_up_the_following_dialogue': 93, 'samsum_Summarize_': 93, 'samsum_Summarize_this_dialogue_': 93, 'samsum_To_sum_up_this_dialog': 93, 'samsum_Write_a_dialogue_that_match_this_summary': 659, 'sciq_Direct_Question': 27, 'sciq_Direct_Question_Closed_Book_': 27, 'sciq_Multiple_Choice': 27, 'sciq_Multiple_Choice_Closed_Book_': 27, 'sciq_Multiple_Choice_Question_First': 27, 'social_i_qa_Check_if_a_random_answer_is_valid_or_not': 1, 'social_i_qa_Generate_answer': 31, 'social_i_qa_Generate_the_question_from_the_answer': 22, 'social_i_qa_I_was_wondering': 31, 'social_i_qa_Show_choices_and_generate_answer': 31, 'social_i_qa_Show_choices_and_generate_index': 1, 'squad_v2_Jeopardy_with_Context': 66, 'squad_v2_Jeopardy_without_Context': 66, 'squad_v2_Questions_with_Context': 75, 'squad_v2_Questions_with_Context_Without_Prompt_Keywords': 75, 'squad_v2_Questions_with_Context_Without_Prompt_Keywords_unanswerable': 75, 'squad_v2_Questions_with_Context_unanswerable': 75, 'squad_v2_Topic_Prediction_Context': 17, 'squad_v2_Topic_Prediction_Context_with_randomized_prompt_options': 17, 'squad_v2_Topic_Prediction_Context_with_randomized_prompt_options_placed_in_the_end': 17, 'squad_v2_Topic_Prediction_Question_and_Answer_Pair': 26, 'squad_v2_Trivia': 75, 'squad_v2_Unanwerable_question': 1, 'super_glue_boolq_GPT_3_Style': 1, 'super_glue_boolq_I_wonder_': 1, 'super_glue_boolq_after_reading': 3, 'super_glue_boolq_based_on_the_following_passage': 1, 'super_glue_boolq_based_on_the_previous_passage': 1, 'super_glue_boolq_could_you_tell_me_': 1, 'super_glue_boolq_exam': 1, 'super_glue_boolq_exercise': 3, 'super_glue_boolq_valid_binary': 3, 'super_glue_boolq_yes_no_question': 1, \
# 'super_glue_cb_GPT_3_style': 3, 'super_glue_cb_GPT_3_style_score_eval': 3, 'super_glue_cb_MNLI_crowdsource': 5, 'super_glue_cb_MNLI_crowdsource_score_eval': 5, 'super_glue_cb_always_sometimes_never': 1, 'super_glue_cb_always_sometimes_never_score_eval': 1, 'super_glue_cb_based_on_the_previous_passage': 1, 'super_glue_cb_based_on_the_previous_passage_score_eval': 1, 'super_glue_cb_can_we_infer': 1, 'super_glue_cb_can_we_infer_score_eval': 1, 'super_glue_cb_claim_true_false_inconclusive': 5, 'super_glue_cb_claim_true_false_inconclusive_score_eval': 5, 'super_glue_cb_consider_always_sometimes_never': 1, 'super_glue_cb_consider_always_sometimes_never_score_eval': 1, 'super_glue_cb_does_it_follow_that': 1, 'super_glue_cb_does_it_follow_that_score_eval': 1, 'super_glue_cb_does_this_imply': 1, 'super_glue_cb_does_this_imply_score_eval': 1, 'super_glue_cb_guaranteed_possible_impossible': 4, 'super_glue_cb_guaranteed_possible_impossible_score_eval': 4, 'super_glue_cb_guaranteed_true': 1, 'super_glue_cb_guaranteed_true_score_eval': 1, 'super_glue_cb_justified_in_saying': 1, 'super_glue_cb_justified_in_saying_score_eval': 1, 'super_glue_cb_must_be_true': 1, 'super_glue_cb_must_be_true_score_eval': 1, 'super_glue_cb_should_assume': 1, 'super_glue_cb_should_assume_score_eval': 1, 'super_glue_cb_take_the_following_as_truth': 5, 'super_glue_cb_take_the_following_as_truth_score_eval': 5, 'super_glue_copa_C1_or_C2_premise_so_because_': 16, 'super_glue_copa_C1_or_C2_premise_so_because__score_eval': 16, 'super_glue_copa__As_a_result_C1_or_C2_': 13, 'super_glue_copa__As_a_result_C1_or_C2__score_eval': 13, 'super_glue_copa__What_could_happen_next_C1_or_C2_': 13, 'super_glue_copa__What_could_happen_next_C1_or_C2__score_eval': 13, 'super_glue_copa__which_may_be_caused_by': 16, 'super_glue_copa__which_may_be_caused_by_score_eval': 16, 'super_glue_copa__why_C1_or_C2': 16, 'super_glue_copa__why_C1_or_C2_score_eval': 16, 'super_glue_copa_best_option': 16, 'super_glue_copa_best_option_score_eval': 16, 'super_glue_copa_cause_effect': 16, 'super_glue_copa_cause_effect_score_eval': 16, 'super_glue_copa_choose': 16, 'super_glue_copa_choose_score_eval': 16, 'super_glue_copa_exercise': 16, 'super_glue_copa_exercise_score_eval': 16, 'super_glue_copa_i_am_hesitating': 16, 'super_glue_copa_i_am_hesitating_score_eval': 16, 'super_glue_copa_more_likely': 16, 'super_glue_copa_more_likely_score_eval': 16, 'super_glue_copa_plausible_alternatives': 16, 'super_glue_copa_plausible_alternatives_score_eval': 16, 'super_glue_multirc_I_was_going_to_say_': 1, 'super_glue_multirc_Would_it_be_good_to_answer_': 1, 'super_glue_multirc_confirm': 1, 'super_glue_multirc_correct': 1, 'super_glue_multirc_decide_valid': 1, 'super_glue_multirc_found_this_answer': 1, 'super_glue_multirc_grading': 1, 'super_glue_multirc_is_a_correct_answer_': 1, 'super_glue_multirc_is_the_correct_answer_': 1, 'super_glue_multirc_paragraph_question_is_it_': 1, 'super_glue_record_Add_sentence_after_after_continuation_choices_': 144, 'super_glue_record_Add_sentence_after_continuation_choices_': 144, 'super_glue_record_Can_you_figure_out_': 18, 'super_glue_record_GPT_3_style_continuation_choices_': 146, 'super_glue_record_GPT_3_style_summary_only_continuation_choices_': 146, 'super_glue_record_GPT_3_style_with_labels_continuation_choices_': 143, 'super_glue_record_GPT_3_style_with_labels_without_hyphens_continuation_choices_': 141, 'super_glue_record_GPT_3_style_without_hyphens_continuation_choices_': 144, 'super_glue_record_In_the_question_above_the_placeholder_stands_for': 18, 'super_glue_record_New_highlight_continuation_choices_': 146, 'super_glue_record_News_article_continuation_choices_': 144, 'super_glue_record_Summary_first_continuation_choices_': 144, 'super_glue_record_What_could_the_placeholder_be_': 18, 'super_glue_record_Which_one_is_the_placeholder_': 18, 'super_glue_record_choose_between': 18, 'super_glue_record_corrupted': 18, 'super_glue_record_exercise': 18, 'super_glue_record_pick_one_option': 18, 'super_glue_record_the_placeholder_refers_to_': 18, 'super_glue_record_trying_to_decide': 18, 'super_glue_rte_GPT_3_style': 3, 'super_glue_rte_GPT_3_style_score_eval': 3, 'super_glue_rte_MNLI_crowdsource': 1, 'super_glue_rte_MNLI_crowdsource_score_eval': 1, 'super_glue_rte_based_on_the_previous_passage': 1, 'super_glue_rte_based_on_the_previous_passage_score_eval': 1, 'super_glue_rte_can_we_infer': 1, 'super_glue_rte_can_we_infer_score_eval': 1, 'super_glue_rte_does_it_follow_that': 1, 'super_glue_rte_does_it_follow_that_score_eval': 1, 'super_glue_rte_does_this_imply': 1, 'super_glue_rte_does_this_imply_score_eval': 1, 'super_glue_rte_guaranteed_true': 1, 'super_glue_rte_guaranteed_true_score_eval': 1, 'super_glue_rte_justified_in_saying': 1, 'super_glue_rte_justified_in_saying_score_eval': 1, 'super_glue_rte_must_be_true': 1, 'super_glue_rte_must_be_true_score_eval': 1, 'super_glue_rte_should_assume': 1, 'super_glue_rte_should_assume_score_eval': 1, 'super_glue_wic_GPT_3_prompt': 1, 'super_glue_wic_GPT_3_prompt_score_eval': 1, 'super_glue_wic_GPT_3_prompt_with_label': 1, 'super_glue_wic_GPT_3_prompt_with_label_score_eval': 1, 'super_glue_wic_affirmation_true_or_false': 3, 'super_glue_wic_affirmation_true_or_false_score_eval': 3, 'super_glue_wic_grammar_homework': 1, 'super_glue_wic_grammar_homework_score_eval': 1, 'super_glue_wic_polysemous': 1, 'super_glue_wic_polysemous_score_eval': 1, 'super_glue_wic_question_context': 1, 'super_glue_wic_question_context_meaning': 1, 'super_glue_wic_question_context_meaning_score_eval': 1, 'super_glue_wic_question_context_meaning_with_label': 1, 'super_glue_wic_question_context_meaning_with_label_score_eval': 1, 'super_glue_wic_question_context_score_eval': 1, 'super_glue_wic_same_sense': 1, 'super_glue_wic_same_sense_score_eval': 1, 'super_glue_wic_similar_sense': 1, 'super_glue_wic_similar_sense_score_eval': 1, 'super_glue_wsc.fixed_GPT_3_Style': 1, 'super_glue_wsc.fixed_GPT_3_Style_score_eval': 1, 'super_glue_wsc.fixed_I_think_they_mean': 1, 'super_glue_wsc.fixed_I_think_they_mean_score_eval': 1, 'super_glue_wsc.fixed_Who_or_what_is_are': 1, 'super_glue_wsc.fixed_Who_or_what_is_are_score_eval': 1, 'super_glue_wsc.fixed_by_p_they_mean': 1, 'super_glue_wsc.fixed_by_p_they_mean_score_eval': 1, 'super_glue_wsc.fixed_does_p_stand_for': 1, 'super_glue_wsc.fixed_does_p_stand_for_score_eval': 1, 'super_glue_wsc.fixed_does_the_pronoun_refer_to': 1, 'super_glue_wsc.fixed_does_the_pronoun_refer_to_score_eval': 1, 'super_glue_wsc.fixed_in_other_words': 3, 'super_glue_wsc.fixed_in_other_words_score_eval': 3, 'super_glue_wsc.fixed_p_is_are_r': 3, 'super_glue_wsc.fixed_p_is_are_r_score_eval': 3, 'super_glue_wsc.fixed_replaced_with': 1, 'super_glue_wsc.fixed_replaced_with_score_eval': 1, 'super_glue_wsc.fixed_the_pronoun_refers_to': 3, 'super_glue_wsc.fixed_the_pronoun_refers_to_score_eval': 3, 'trec_fine_grained_ABBR': 5, 'trec_fine_grained_ABBR_context_first': 5, 'trec_fine_grained_DESC': 3, 'trec_fine_grained_DESC_context_first': 3, 'trec_fine_grained_ENTY': 5, 'trec_fine_grained_HUM': 1, 'trec_fine_grained_HUM_context_first': 1, 'trec_fine_grained_LOC': 2, 'trec_fine_grained_LOC_context_first': 2, 'trec_fine_grained_NUM': 3, 'trec_fine_grained_NUM_context_first': 3, 'trec_fine_grained_open': 5, 'trec_fine_grained_open_context_first': 5, 'trec_pick_the_best_descriptor': 4, 'trec_trec1': 4, 'trec_trec2': 4, 'trec_what_category_best_describe': 4, 'trec_which_category_best_describes': 4, 'trivia_qa_unfiltered_first_person_context': 158, 'trivia_qa_unfiltered_formal_description': 158, 'trivia_qa_unfiltered_guess_question': 276, 'trivia_qa_unfiltered_question_answer': 158, 'trivia_qa_unfiltered_question_with_instruction': 158, 'web_questions_get_the_answer': 109, 'web_questions_potential_correct_answer': 109, 'web_questions_question_answer': 109, 'web_questions_short_general_knowledge_q': 109, 'web_questions_whats_the_answer': 109, 'wiki_bio_comprehension': 1837, 'wiki_bio_guess_person': 88, 'wiki_bio_key_content': 1837, 'wiki_bio_what_content': 423, 'wiki_bio_who': 1157, 'wiki_hop_original_choose_best_object_affirmative_1': 25, 'wiki_hop_original_choose_best_object_affirmative_2': 25, 'wiki_hop_original_choose_best_object_affirmative_3': 25, 'wiki_hop_original_choose_best_object_interrogative_1': 25, 'wiki_hop_original_choose_best_object_interrogative_2': 25, 'wiki_hop_original_explain_relation': 8, 'wiki_hop_original_generate_object': 25, 'wiki_hop_original_generate_subject': 29, 'wiki_hop_original_generate_subject_and_object': 39, 'wiki_qa_Decide_good_answer': 1, 'wiki_qa_Direct_Answer_to_Question': 374, 'wiki_qa_Generate_Question_from_Topic': 24, 'wiki_qa_Is_This_True_': 1, 'wiki_qa_Jeopardy_style': 26, 'wiki_qa_Topic_Prediction_Answer_Only': 15, 'wiki_qa_Topic_Prediction_Question_Only': 15, 'wiki_qa_Topic_Prediction_Question_and_Answer_Pair': 15, 'wiki_qa_automatic_system': 1, 'wiki_qa_exercise': 3, 'wiki_qa_found_on_google': 1, \
# 'winogrande_winogrande_debiased_Replace': 8, 'winogrande_winogrande_debiased_Replace_score_eval': 8, 'winogrande_winogrande_debiased_does_underscore_refer_to': 8, 'winogrande_winogrande_debiased_does_underscore_refer_to_score_eval': 8, 'winogrande_winogrande_debiased_fill_in_the_blank': 8, 'winogrande_winogrande_debiased_fill_in_the_blank_score_eval': 8, 'winogrande_winogrande_debiased_stand_for': 8, 'winogrande_winogrande_debiased_stand_for_score_eval': 8, 'winogrande_winogrande_debiased_underscore_refer_to': 8, 'winogrande_winogrande_debiased_underscore_refer_to_score_eval': 8, 'winogrande_winogrande_xl_Replace': 9, 'winogrande_winogrande_xl_Replace_score_eval': 9, 'winogrande_winogrande_xl_does_underscore_refer_to': 9, 'winogrande_winogrande_xl_does_underscore_refer_to_score_eval': 9, 'winogrande_winogrande_xl_fill_in_the_blank': 9, 'winogrande_winogrande_xl_fill_in_the_blank_score_eval': 9, 'winogrande_winogrande_xl_stand_for': 9, 'winogrande_winogrande_xl_stand_for_score_eval': 9, 'winogrande_winogrande_xl_underscore_refer_to': 9, 'winogrande_winogrande_xl_underscore_refer_to_score_eval': 9, 'wiqa_does_the_supposed_perturbation_have_an_effect': 1, 'wiqa_effect_with_label_answer': 1, 'wiqa_effect_with_string_answer': 2, 'wiqa_what_is_the_final_step_of_the_following_process': 43, 'wiqa_what_is_the_missing_first_step': 27, 'wiqa_what_might_be_the_first_step_of_the_process': 27, 'wiqa_what_might_be_the_last_step_of_the_process': 43, 'wiqa_which_of_the_following_is_the_supposed_perturbation': 9, 'xsum_DOC_boils_down_to_simple_idea_that': 177, 'xsum_DOC_given_above_write_one_sentence': 177, 'xsum_DOC_how_would_you_rephrase_few_words': 177, 'xsum_DOC_tldr': 177, 'xsum_DOC_write_summary_of_above': 177, 'xsum_article_DOC_summary': 177, 'xsum_college_roommate_asked_DOC_so_I_recap': 177, 'xsum_read_below_DOC_write_abstract': 177, 'xsum_summarize_DOC': 177, 'xsum_summarize_this_DOC_summary': 177, 'yelp_review_full_based_on_that': 2, 'yelp_review_full_format_rating': 2, 'yelp_review_full_format_score': 1, 'yelp_review_full_format_star': 2, 'yelp_review_full_on_a_scale': 1, 'yelp_review_full_so_i_would': 2, 'yelp_review_full_this_place': 2}

# _TASK_SPLITS_AND_FEATURES_DICT, _TASK_TARGET_MAX_TOKENS_DICT = find_task_splits_and_features_dict()
# with open('p3_config.json','w') as f:
#     json.dump(_TASK_SPLITS_AND_FEATURES_DICT,f)

with open('p3_config.json') as f:
    _TASK_SPLITS_AND_FEATURES_DICT = json.load(f)

P3_TASK_LIST = list(_TASK_SPLITS_AND_FEATURES_DICT.keys())
# _URLs = {
#     task_name: {
#         split_name: {
#             "tfrecord": f"{_DATA_PATH}/{task_name}/{split_name}.tfrecord-00000-of-00001", # TODO -> handle multiple shards
#         }
#         for split_name in splits_and_features_dict["splits"]
#     }
#     for task_name, splits_and_features_dict in _TASK_SPLITS_AND_FEATURES_DICT.items()
# }

datasets_without_validation = [
    "ag_news", "dbpedia_14", "trec", "amazon_polarity", "imdb", "yelp_review_full", "wiki_bio",
    "web_questions"]

# large_t0_tasks = [
#     "gigaword", "amazon_polarity", "wiki_bio", "dbpedia_14", "yelp_review_full",
#     "ag_news", "app_reviews", "cnn_dailymail/3.0.0", "common_gen", "duorc/ParaphraseRC",
#     "duorc/SelfRC", "glue/qqp", 
#     # summarization 接近5w也砍掉
#     "paws/labeled_final","xsum",
#     # "kilt_tasks/hotpotqa", "multi_news", 
# ]
# large_t0_tasks_prompt_count = {}
# for large_task in large_t0_tasks:
#     for task_name in P3_TASK_LIST:
#         if task_name.startswith(large_task):
#             if large_task not in large_t0_tasks_prompt_count:
#                 large_t0_tasks_prompt_count[large_task] = 1
#             else:
#                 large_t0_tasks_prompt_count[large_task] += 1

# large_t0_task_dict = {}
# for cur_task in P3_TASK_LIST:
#     for large_task_prefix in large_t0_tasks_prompt_count.keys():
#         if cur_task.startswith(large_task_prefix):
#             large_t0_task_dict[cur_task] = int(_MAX_DATASET_SIZE / large_t0_tasks_prompt_count[large_task_prefix])

DEBUG_TRAIN_TASK_NAME = ["ropes"]
DEBUG_TRAIN_TASK_LIST=[]
for task_name in DEBUG_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    DEBUG_TRAIN_TASK_LIST = DEBUG_TRAIN_TASK_LIST + sub_list


T0_TRAIN_TASK_NAME = [
    "cos_e/v1.11",
    "wiki_bio",
    "cnn_dailymail/3.0.0",
    "gigaword",
    "wiki_hop/original",
    "glue/mrpc",
    "glue/qqp",
    "amazon_polarity",
    "paws/labeled_final",
    "dbpedia_14",
    "dream",
    "kilt_tasks/hotpotqa",
    "trec",
    "multi_news",
    "samsum",
    "xsum",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "wiki_qa",
    "common_gen",
    "adversarial_qa/dbidaf",
    "adversarial_qa/dbert",
    "adversarial_qa/droberta",
    "quoref",
    "ropes",
    "duorc/SelfRC",
    "duorc/ParaphraseRC",
    "sciq",
    "quarel",
    "qasc",
    "cosmos_qa",
    "wiqa",
    "social_i_qa",
    "quail",
    "quartz",
    "ag_news",
    "app_reviews",
]
T0_TRAIN_TASK_LIST=[]

for task_name in T0_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_TRAIN_TASK_LIST = T0_TRAIN_TASK_LIST + sub_list

REGISTERED_DATA_LIST={}

zj_para_ls=['adversarial_qa_dbert_paraphrase', 'adversarial_qa_dbidaf_paraphrase', 'adversarial_qa_droberta_paraphrase', 'ag_news_paraphrase', 'amazon_polarity_paraphrase', 'app_reviews_paraphrase', 'cnn_dailymail_3.0.0_paraphrase', 'common_gen_paraphrase', 'cosmos_qa_paraphrase', 'dbpedia_14_paraphrase', 'dream_paraphrase', 'duorc_ParaphraseRC_paraphrase', 'duorc_SelfRC_paraphrase', 'gigaword_paraphrase', 'imdb_paraphrase', 'multi_news_paraphrase', 'qasc_paraphrase', 'quail_paraphrase', 'quartz_paraphrase', 'quoref_paraphrase', 'ropes_paraphrase', 'rotten_tomatoes_paraphrase', 'samsum_paraphrase', 'social_i_qa_paraphrase', 'wiki_bio_paraphrase', 'wiki_hop_original_paraphrase', 'wiqa_paraphrase', 'xsum_paraphrase', 'yelp_review_full_paraphrase']
taxonomy_total_5w=['adversarial_qa_dbert-cosmos_qa_context_description_question_text', 'adversarial_qa_dbert-cosmos_qa_description_context_question_text', 'adversarial_qa_dbert-quail_context_description_question_text', 'adversarial_qa_dbert-quail_context_question_description_text', 'adversarial_qa_dbert-quail_description_context_question_text', 'adversarial_qa_dbert-quoref_Answer_Friend_Question', 'adversarial_qa_dbert-quoref_Answer_Test', 'adversarial_qa_dbert-quoref_Context_Contains_Answer', 'adversarial_qa_dbert-quoref_Find_Answer', 'adversarial_qa_dbert-quoref_Found_Context_Online', 'adversarial_qa_dbert-quoref_Given_Context_Answer_Question', 'adversarial_qa_dbert-quoref_Guess_Answer', 'adversarial_qa_dbert-quoref_Read_And_Extract_', 'adversarial_qa_dbert-quoref_What_Is_The_Answer', 'adversarial_qa_dbert-ropes_plain_no_background', 'adversarial_qa_dbert-ropes_prompt_bottom_no_hint', 'adversarial_qa_dbert-social_i_qa_Generate_answer', 'adversarial_qa_dbert-social_i_qa_I_was_wondering', 'adversarial_qa_dbert_answer_the_following_q', 'adversarial_qa_dbert_based_on', 'adversarial_qa_dbert_generate_question', 'adversarial_qa_dbert_question_context_answer', 'adversarial_qa_dbert_tell_what_it_is', 'adversarial_qa_dbidaf-cosmos_qa_context_description_question_text', 'adversarial_qa_dbidaf-cosmos_qa_description_context_question_text', 'adversarial_qa_dbidaf-quail_context_description_question_text', 'adversarial_qa_dbidaf-quail_context_question_description_text', 'adversarial_qa_dbidaf-quail_description_context_question_text', 'adversarial_qa_dbidaf-quoref_Answer_Friend_Question', 'adversarial_qa_dbidaf-quoref_Answer_Test', 'adversarial_qa_dbidaf-quoref_Context_Contains_Answer', 'adversarial_qa_dbidaf-quoref_Find_Answer', 'adversarial_qa_dbidaf-quoref_Found_Context_Online', 'adversarial_qa_dbidaf-quoref_Given_Context_Answer_Question', 'adversarial_qa_dbidaf-quoref_Guess_Answer', 'adversarial_qa_dbidaf-quoref_Read_And_Extract_', 'adversarial_qa_dbidaf-quoref_What_Is_The_Answer', 'adversarial_qa_dbidaf-ropes_plain_no_background', 'adversarial_qa_dbidaf-ropes_prompt_bottom_no_hint', 'adversarial_qa_dbidaf-social_i_qa_Generate_answer', 'adversarial_qa_dbidaf-social_i_qa_I_was_wondering', 'adversarial_qa_dbidaf_answer_the_following_q', 'adversarial_qa_dbidaf_based_on', 'adversarial_qa_dbidaf_generate_question', 'adversarial_qa_dbidaf_question_context_answer', 'adversarial_qa_dbidaf_tell_what_it_is', 'adversarial_qa_droberta-cosmos_qa_context_description_question_text', 'adversarial_qa_droberta-cosmos_qa_description_context_question_text', 'adversarial_qa_droberta-quail_context_description_question_text', 'adversarial_qa_droberta-quail_context_question_description_text', 'adversarial_qa_droberta-quail_description_context_question_text', 'adversarial_qa_droberta-quoref_Answer_Friend_Question', 'adversarial_qa_droberta-quoref_Answer_Test', 'adversarial_qa_droberta-quoref_Context_Contains_Answer', 'adversarial_qa_droberta-quoref_Find_Answer', 'adversarial_qa_droberta-quoref_Found_Context_Online', 'adversarial_qa_droberta-quoref_Given_Context_Answer_Question', 'adversarial_qa_droberta-quoref_Guess_Answer', 'adversarial_qa_droberta-quoref_Read_And_Extract_', 'adversarial_qa_droberta-quoref_What_Is_The_Answer', 'adversarial_qa_droberta-ropes_plain_no_background', 'adversarial_qa_droberta-ropes_prompt_bottom_no_hint', 'adversarial_qa_droberta-social_i_qa_Generate_answer', 'adversarial_qa_droberta-social_i_qa_I_was_wondering', 'adversarial_qa_droberta_answer_the_following_q', 'adversarial_qa_droberta_based_on', 'adversarial_qa_droberta_generate_question', 'adversarial_qa_droberta_question_context_answer', 'adversarial_qa_droberta_tell_what_it_is', 'ag_news-dbpedia_14_given_list_what_category_does_the_paragraph_belong_to', 'ag_news-trec_fine_grained_open', 'ag_news-trec_fine_grained_open_context_first', 'ag_news_classify', 'ag_news_classify_question_first', 'ag_news_classify_with_choices', 'ag_news_classify_with_choices_question_first', 'ag_news_recommend', 'ag_news_which_section', 'ag_news_which_section_choices', 'amazon_polarity-imdb_Movie_Expressed_Sentiment', 'amazon_polarity-imdb_Movie_Expressed_Sentiment_2', 'amazon_polarity-imdb_Negation_template_for_positive_and_negative', 'amazon_polarity-imdb_Reviewer_Enjoyment', 'amazon_polarity-imdb_Reviewer_Enjoyment_Yes_No', 'amazon_polarity-imdb_Reviewer_Expressed_Sentiment', 'amazon_polarity-imdb_Reviewer_Opinion_bad_good_choices', 'amazon_polarity-imdb_Reviewer_Sentiment_Feeling', 'amazon_polarity-imdb_Sentiment_with_choices_', 'amazon_polarity-imdb_Text_Expressed_Sentiment', 'amazon_polarity-imdb_Writer_Expressed_Sentiment', 'amazon_polarity_Is_this_product_review_positive', 'amazon_polarity_Is_this_review', 'amazon_polarity_Is_this_review_negative', 'amazon_polarity_User_recommend_this_product', 'amazon_polarity_convey_negative_or_positive_sentiment', 'amazon_polarity_flattering_or_not', 'amazon_polarity_negative_or_positive_tone', 'amazon_polarity_user_satisfied', 'amazon_polarity_would_you_buy', 'app_reviews_categorize_rating_using_review', 'app_reviews_convert_to_rating', 'app_reviews_convert_to_star_rating', 'app_reviews_generate_review', 'cnn_dailymail_3.0.0-common_gen_Example_prompt', 'cnn_dailymail_3.0.0-common_gen_Given_concepts_type_1', 'cnn_dailymail_3.0.0-common_gen_Given_concepts_type_2', 'cnn_dailymail_3.0.0-common_gen_Put_together', 'cnn_dailymail_3.0.0-common_gen_random_task_template_prompt', 'cnn_dailymail_3.0.0-common_gen_sentence_to_concepts', 'cnn_dailymail_3.0.0-common_gen_topics_from_the_sentence', 'cnn_dailymail_3.0.0-duorc_SelfRC_title_generation', 'cnn_dailymail_3.0.0-gigaword_TLDR', 'cnn_dailymail_3.0.0-gigaword_first_sentence_title', 'cnn_dailymail_3.0.0-gigaword_generate_summary_for_this', 'cnn_dailymail_3.0.0-gigaword_in_a_nutshell', 'cnn_dailymail_3.0.0-gigaword_make_a_title', 'cnn_dailymail_3.0.0-gigaword_reverse_writing', 'cnn_dailymail_3.0.0-gigaword_write_a_title_for_this_sentence', 'cnn_dailymail_3.0.0-gigaword_write_an_article', 'cnn_dailymail_3.0.0-gigaword_write_its_sentence', 'cnn_dailymail_3.0.0-quoref_Guess_Title_For_Context', 'cnn_dailymail_3.0.0-samsum_Generate_a_summary_for_this_dialogue', 'cnn_dailymail_3.0.0-samsum_Given_the_above_dialogue_write_a_summary', 'cnn_dailymail_3.0.0-samsum_Sum_up_the_following_dialogue', 'cnn_dailymail_3.0.0-samsum_Summarize_', 'cnn_dailymail_3.0.0-samsum_Summarize_this_dialogue_', 'cnn_dailymail_3.0.0-samsum_To_sum_up_this_dialog', 'cnn_dailymail_3.0.0-samsum_Write_a_dialogue_that_match_this_summary', 'cnn_dailymail_3.0.0-wiki_qa_Topic_Prediction_Answer_Only', 'cnn_dailymail_3.0.0-wiki_qa_Topic_Prediction_Question_Only', 'cnn_dailymail_3.0.0-xsum_DOC_boils_down_to_simple_idea_that', 'cnn_dailymail_3.0.0-xsum_DOC_given_above_write_one_sentence', 'cnn_dailymail_3.0.0-xsum_DOC_how_would_you_rephrase_few_words', 'cnn_dailymail_3.0.0-xsum_DOC_tldr', 'cnn_dailymail_3.0.0-xsum_DOC_write_summary_of_above', 'cnn_dailymail_3.0.0-xsum_article_DOC_summary', 'cnn_dailymail_3.0.0-xsum_college_roommate_asked_DOC_so_I_recap', 'cnn_dailymail_3.0.0-xsum_read_below_DOC_write_abstract', 'cnn_dailymail_3.0.0-xsum_summarize_DOC', 'cnn_dailymail_3.0.0-xsum_summarize_this_DOC_summary', 'cnn_dailymail_3.0.0_2_or_3_sentences', 'cnn_dailymail_3.0.0_generate_story', 'cnn_dailymail_3.0.0_news_card_view', 'cnn_dailymail_3.0.0_news_stock', 'cnn_dailymail_3.0.0_news_summary', 'cnn_dailymail_3.0.0_spice_up_story', 'cnn_dailymail_3.0.0_sum_in_brief', 'cnn_dailymail_3.0.0_tldr_summary', 'cnn_dailymail_3.0.0_write_an_outline', 'common_gen-cnn_dailymail_3.0.0_2_or_3_sentences', 'common_gen-cnn_dailymail_3.0.0_generate_story', 'common_gen-cnn_dailymail_3.0.0_news_card_view', 'common_gen-cnn_dailymail_3.0.0_news_stock', 'common_gen-cnn_dailymail_3.0.0_news_summary', 'common_gen-cnn_dailymail_3.0.0_spice_up_story', 'common_gen-cnn_dailymail_3.0.0_sum_in_brief', 'common_gen-cnn_dailymail_3.0.0_tldr_summary', 'common_gen-cnn_dailymail_3.0.0_write_an_outline', 'common_gen-duorc_SelfRC_title_generation', 'common_gen-gigaword_TLDR', 'common_gen-gigaword_first_sentence_title', 'common_gen-gigaword_generate_summary_for_this', 'common_gen-gigaword_in_a_nutshell', 'common_gen-gigaword_make_a_title', 'common_gen-gigaword_reverse_writing', 'common_gen-gigaword_write_a_title_for_this_sentence', 'common_gen-gigaword_write_an_article', 'common_gen-gigaword_write_its_sentence', 'common_gen-quoref_Guess_Title_For_Context', 'common_gen-samsum_Generate_a_summary_for_this_dialogue', 'common_gen-samsum_Given_the_above_dialogue_write_a_summary', 'common_gen-samsum_Sum_up_the_following_dialogue', 'common_gen-samsum_Summarize_', 'common_gen-samsum_Summarize_this_dialogue_', 'common_gen-samsum_To_sum_up_this_dialog', 'common_gen-samsum_Write_a_dialogue_that_match_this_summary', 'common_gen-wiki_qa_Topic_Prediction_Answer_Only', 'common_gen-wiki_qa_Topic_Prediction_Question_Only', 'common_gen-xsum_DOC_boils_down_to_simple_idea_that', 'common_gen-xsum_DOC_given_above_write_one_sentence', 'common_gen-xsum_DOC_how_would_you_rephrase_few_words', 'common_gen-xsum_DOC_tldr', 'common_gen-xsum_DOC_write_summary_of_above', 'common_gen-xsum_article_DOC_summary', 'common_gen-xsum_college_roommate_asked_DOC_so_I_recap', 'common_gen-xsum_read_below_DOC_write_abstract', 'common_gen-xsum_summarize_DOC', 'common_gen-xsum_summarize_this_DOC_summary', 'common_gen_Example_prompt', 'common_gen_Given_concepts_type_1', 'common_gen_Given_concepts_type_2', 'common_gen_Put_together', 'common_gen_choice_in_concept_centric_sentence_generation', 'common_gen_random_task_template_prompt', 'common_gen_sentence_to_concepts', 'common_gen_topic_to_sentence', 'common_gen_topics_from_the_sentence', 'cos_e_v1.11_aligned_with_common_sense', 'cos_e_v1.11_description_question_option_id', 'cos_e_v1.11_description_question_option_text', 'cos_e_v1.11_explain_why_human', 'cos_e_v1.11_generate_explanation_given_text', 'cos_e_v1.11_i_think', 'cos_e_v1.11_question_description_option_id',\
 'cos_e_v1.11_question_description_option_text', 'cos_e_v1.11_question_option_description_id', 'cos_e_v1.11_question_option_description_text', 'cos_e_v1.11_rationale', 'cosmos_qa-adversarial_qa_dbidaf_answer_the_following_q', 'cosmos_qa-adversarial_qa_dbidaf_based_on', 'cosmos_qa-adversarial_qa_dbidaf_question_context_answer', 'cosmos_qa-adversarial_qa_dbidaf_tell_what_it_is', 'cosmos_qa-dream_baseline', 'cosmos_qa-dream_read_the_following_conversation_and_answer_the_question', 'cosmos_qa-duorc_SelfRC_generate_question_by_answer', 'cosmos_qa-kilt_tasks_hotpotqa_combining_facts', 'cosmos_qa-kilt_tasks_hotpotqa_complex_question', 'cosmos_qa-kilt_tasks_hotpotqa_final_exam', 'cosmos_qa-kilt_tasks_hotpotqa_formulate', 'cosmos_qa-kilt_tasks_hotpotqa_straighforward_qa', 'cosmos_qa-qasc_qa_with_combined_facts_1', 'cosmos_qa-quail_context_description_question_text', 'cosmos_qa-quail_context_question_answer_description_text', 'cosmos_qa-quail_context_question_description_answer_text', 'cosmos_qa-quail_context_question_description_text', 'cosmos_qa-quail_description_context_question_answer_text', 'cosmos_qa-quail_description_context_question_text', 'cosmos_qa-quail_no_prompt_text', 'cosmos_qa-quartz_answer_question_based_on', 'cosmos_qa-quartz_answer_question_below', 'cosmos_qa-quartz_given_the_fact_answer_the_q', 'cosmos_qa-quartz_having_read_above_passage', 'cosmos_qa-quartz_paragraph_question_plain_concat', 'cosmos_qa-quartz_read_passage_below_choose', 'cosmos_qa-quartz_use_info_from_paragraph_question', 'cosmos_qa-quartz_use_info_from_question_paragraph', 'cosmos_qa-quoref_Answer_Friend_Question', 'cosmos_qa-quoref_Answer_Test', 'cosmos_qa-quoref_Context_Contains_Answer', 'cosmos_qa-quoref_Find_Answer', 'cosmos_qa-quoref_Found_Context_Online', 'cosmos_qa-quoref_Given_Context_Answer_Question', 'cosmos_qa-quoref_Guess_Answer', 'cosmos_qa-quoref_Read_And_Extract_', 'cosmos_qa-quoref_What_Is_The_Answer', 'cosmos_qa-ropes_plain_no_background', 'cosmos_qa-ropes_prompt_bottom_no_hint', 'cosmos_qa-sciq_Direct_Question_Closed_Book_', 'cosmos_qa-sciq_Multiple_Choice', 'cosmos_qa-social_i_qa_Generate_answer', 'cosmos_qa-social_i_qa_Generate_the_question_from_the_answer', 'cosmos_qa-social_i_qa_I_was_wondering', 'cosmos_qa-social_i_qa_Show_choices_and_generate_answer', 'cosmos_qa-wiki_qa_Direct_Answer_to_Question', 'cosmos_qa-wiki_qa_Generate_Question_from_Topic', 'cosmos_qa-wiki_qa_Jeopardy_style', 'cosmos_qa-wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'cosmos_qa_context_answer_to_question', 'cosmos_qa_context_description_question_answer_id', 'cosmos_qa_context_description_question_answer_text', 'cosmos_qa_context_description_question_text', 'cosmos_qa_context_question_description_answer_id', 'cosmos_qa_context_question_description_answer_text', 'cosmos_qa_context_question_description_text', 'cosmos_qa_description_context_question_answer_id', 'cosmos_qa_description_context_question_answer_text', 'cosmos_qa_description_context_question_text', 'cosmos_qa_no_prompt_id', 'cosmos_qa_no_prompt_text', 'cosmos_qa_only_question_answer', 'dbpedia_14-trec_fine_grained_open', 'dbpedia_14-trec_fine_grained_open_context_first', 'dbpedia_14-trec_pick_the_best_descriptor', 'dbpedia_14-trec_trec1', 'dbpedia_14-trec_trec2', 'dbpedia_14-trec_what_category_best_describe', 'dbpedia_14-trec_which_category_best_describes', 'dbpedia_14_given_a_choice_of_categories_', 'dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to', 'dbpedia_14_given_list_what_category_does_the_paragraph_belong_to', 'dbpedia_14_pick_one_category_for_the_following_text', 'dream-app_reviews_generate_review', 'dream-cosmos_qa_context_description_question_answer_text', 'dream-cosmos_qa_context_question_description_answer_text', 'dream-cosmos_qa_description_context_question_answer_text', 'dream-cosmos_qa_no_prompt_text', 'dream-duorc_SelfRC_build_story_around_qa', 'dream-qasc_qa_with_combined_facts_1', 'dream-quail_context_question_answer_description_text', 'dream-quail_context_question_description_answer_text', 'dream-quail_description_context_question_answer_text', 'dream-quail_no_prompt_text', 'dream-quartz_answer_question_based_on', 'dream-quartz_answer_question_below', 'dream-quartz_given_the_fact_answer_the_q', 'dream-quartz_having_read_above_passage', 'dream-quartz_paragraph_question_plain_concat', 'dream-quartz_read_passage_below_choose', 'dream-quartz_use_info_from_paragraph_question', 'dream-quartz_use_info_from_question_paragraph', 'dream-sciq_Multiple_Choice', 'dream-social_i_qa_Show_choices_and_generate_answer', 'dream-wiqa_what_is_the_final_step_of_the_following_process', 'dream_answer_to_dialogue', 'dream_baseline', 'dream_generate_first_utterance', 'dream_generate_last_utterance', 'dream_read_the_following_conversation_and_answer_the_question', 'duorc_ParaphraseRC-cnn_dailymail_3.0.0_2_or_3_sentences', 'duorc_ParaphraseRC-cnn_dailymail_3.0.0_news_card_view', 'duorc_ParaphraseRC-cnn_dailymail_3.0.0_news_stock', 'duorc_ParaphraseRC-cnn_dailymail_3.0.0_news_summary', 'duorc_ParaphraseRC-cnn_dailymail_3.0.0_sum_in_brief', 'duorc_ParaphraseRC-cnn_dailymail_3.0.0_tldr_summary', 'duorc_ParaphraseRC-cnn_dailymail_3.0.0_write_an_outline', 'duorc_ParaphraseRC-common_gen_sentence_to_concepts', 'duorc_ParaphraseRC-common_gen_topics_from_the_sentence', 'duorc_ParaphraseRC-cosmos_qa_context_answer_to_question', 'duorc_ParaphraseRC-dream_answer_to_dialogue', 'duorc_ParaphraseRC-gigaword_TLDR', 'duorc_ParaphraseRC-gigaword_first_sentence_title', 'duorc_ParaphraseRC-gigaword_generate_summary_for_this', 'duorc_ParaphraseRC-gigaword_in_a_nutshell', 'duorc_ParaphraseRC-gigaword_make_a_title', 'duorc_ParaphraseRC-gigaword_write_a_title_for_this_sentence', 'duorc_ParaphraseRC-gigaword_write_its_sentence', 'duorc_ParaphraseRC-quoref_Guess_Title_For_Context', 'duorc_ParaphraseRC-ropes_background_new_situation_answer', 'duorc_ParaphraseRC-ropes_new_situation_background_answer', 'duorc_ParaphraseRC-ropes_plain_background_situation', 'duorc_ParaphraseRC-ropes_plain_bottom_hint', 'duorc_ParaphraseRC-ropes_prompt_beginning', 'duorc_ParaphraseRC-ropes_prompt_bottom_hint_beginning', 'duorc_ParaphraseRC-ropes_prompt_mix', 'duorc_ParaphraseRC-ropes_read_background_situation', 'duorc_ParaphraseRC-samsum_Generate_a_summary_for_this_dialogue', 'duorc_ParaphraseRC-samsum_Given_the_above_dialogue_write_a_summary', 'duorc_ParaphraseRC-samsum_Sum_up_the_following_dialogue', 'duorc_ParaphraseRC-samsum_Summarize_', 'duorc_ParaphraseRC-samsum_Summarize_this_dialogue_', 'duorc_ParaphraseRC-samsum_To_sum_up_this_dialog', 'duorc_ParaphraseRC-social_i_qa_Generate_the_question_from_the_answer', 'duorc_ParaphraseRC-wiki_qa_Generate_Question_from_Topic', 'duorc_ParaphraseRC-wiki_qa_Jeopardy_style', 'duorc_ParaphraseRC-wiki_qa_Topic_Prediction_Answer_Only', 'duorc_ParaphraseRC-wiki_qa_Topic_Prediction_Question_Only', 'duorc_ParaphraseRC-wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'duorc_ParaphraseRC-xsum_DOC_boils_down_to_simple_idea_that', 'duorc_ParaphraseRC-xsum_DOC_given_above_write_one_sentence', 'duorc_ParaphraseRC-xsum_DOC_how_would_you_rephrase_few_words', 'duorc_ParaphraseRC-xsum_DOC_tldr', 'duorc_ParaphraseRC-xsum_DOC_write_summary_of_above', 'duorc_ParaphraseRC-xsum_article_DOC_summary', 'duorc_ParaphraseRC-xsum_college_roommate_asked_DOC_so_I_recap', 'duorc_ParaphraseRC-xsum_read_below_DOC_write_abstract', 'duorc_ParaphraseRC-xsum_summarize_DOC', 'duorc_ParaphraseRC-xsum_summarize_this_DOC_summary', 'duorc_ParaphraseRC_answer_question', 'duorc_ParaphraseRC_build_story_around_qa', 'duorc_ParaphraseRC_decide_worth_it', 'duorc_ParaphraseRC_extract_answer', 'duorc_ParaphraseRC_generate_question', 'duorc_ParaphraseRC_generate_question_by_answer', 'duorc_ParaphraseRC_movie_director', 'duorc_ParaphraseRC_question_answering', 'duorc_ParaphraseRC_title_generation', 'duorc_SelfRC-cnn_dailymail_3.0.0_2_or_3_sentences', 'duorc_SelfRC-cnn_dailymail_3.0.0_news_card_view', 'duorc_SelfRC-cnn_dailymail_3.0.0_news_stock', 'duorc_SelfRC-cnn_dailymail_3.0.0_news_summary', 'duorc_SelfRC-cnn_dailymail_3.0.0_sum_in_brief', 'duorc_SelfRC-cnn_dailymail_3.0.0_tldr_summary', 'duorc_SelfRC-cnn_dailymail_3.0.0_write_an_outline', 'duorc_SelfRC-common_gen_sentence_to_concepts', 'duorc_SelfRC-common_gen_topics_from_the_sentence', 'duorc_SelfRC-cosmos_qa_context_answer_to_question', 'duorc_SelfRC-dream_answer_to_dialogue', 'duorc_SelfRC-gigaword_TLDR', 'duorc_SelfRC-gigaword_first_sentence_title', 'duorc_SelfRC-gigaword_generate_summary_for_this', 'duorc_SelfRC-gigaword_in_a_nutshell', 'duorc_SelfRC-gigaword_make_a_title', 'duorc_SelfRC-gigaword_write_a_title_for_this_sentence', 'duorc_SelfRC-gigaword_write_its_sentence', 'duorc_SelfRC-quoref_Guess_Title_For_Context', 'duorc_SelfRC-ropes_background_new_situation_answer', 'duorc_SelfRC-ropes_new_situation_background_answer', 'duorc_SelfRC-ropes_plain_background_situation', 'duorc_SelfRC-ropes_plain_bottom_hint', 'duorc_SelfRC-ropes_prompt_beginning', 'duorc_SelfRC-ropes_prompt_bottom_hint_beginning', 'duorc_SelfRC-ropes_prompt_mix', 'duorc_SelfRC-ropes_read_background_situation', 'duorc_SelfRC-samsum_Generate_a_summary_for_this_dialogue', 'duorc_SelfRC-samsum_Given_the_above_dialogue_write_a_summary', 'duorc_SelfRC-samsum_Sum_up_the_following_dialogue', 'duorc_SelfRC-samsum_Summarize_', 'duorc_SelfRC-samsum_Summarize_this_dialogue_', 'duorc_SelfRC-samsum_To_sum_up_this_dialog', 'duorc_SelfRC-social_i_qa_Generate_the_question_from_the_answer', 'duorc_SelfRC-wiki_qa_Generate_Question_from_Topic', 'duorc_SelfRC-wiki_qa_Jeopardy_style', 'duorc_SelfRC-wiki_qa_Topic_Prediction_Answer_Only', 'duorc_SelfRC-wiki_qa_Topic_Prediction_Question_Only', 'duorc_SelfRC-wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'duorc_SelfRC-xsum_DOC_boils_down_to_simple_idea_that', 'duorc_SelfRC-xsum_DOC_given_above_write_one_sentence', 'duorc_SelfRC-xsum_DOC_how_would_you_rephrase_few_words', 'duorc_SelfRC-xsum_DOC_tldr', 'duorc_SelfRC-xsum_DOC_write_summary_of_above', \
 'duorc_SelfRC-xsum_article_DOC_summary', 'duorc_SelfRC-xsum_college_roommate_asked_DOC_so_I_recap', 'duorc_SelfRC-xsum_read_below_DOC_write_abstract', 'duorc_SelfRC-xsum_summarize_DOC', 'duorc_SelfRC-xsum_summarize_this_DOC_summary', 'duorc_SelfRC_answer_question', 'duorc_SelfRC_build_story_around_qa', 'duorc_SelfRC_decide_worth_it', 'duorc_SelfRC_extract_answer', 'duorc_SelfRC_generate_question', 'duorc_SelfRC_generate_question_by_answer', 'duorc_SelfRC_movie_director', 'duorc_SelfRC_question_answering', 'duorc_SelfRC_title_generation', 'gigaword-cnn_dailymail_3.0.0_2_or_3_sentences', 'gigaword-cnn_dailymail_3.0.0_generate_story', 'gigaword-cnn_dailymail_3.0.0_news_card_view', 'gigaword-cnn_dailymail_3.0.0_news_stock', 'gigaword-cnn_dailymail_3.0.0_news_summary', 'gigaword-cnn_dailymail_3.0.0_spice_up_story', 'gigaword-cnn_dailymail_3.0.0_sum_in_brief', 'gigaword-cnn_dailymail_3.0.0_tldr_summary', 'gigaword-cnn_dailymail_3.0.0_write_an_outline', 'gigaword-common_gen_Example_prompt', 'gigaword-common_gen_Given_concepts_type_1', 'gigaword-common_gen_Given_concepts_type_2', 'gigaword-common_gen_Put_together', 'gigaword-common_gen_random_task_template_prompt', 'gigaword-common_gen_sentence_to_concepts', 'gigaword-common_gen_topics_from_the_sentence', 'gigaword-duorc_SelfRC_title_generation', 'gigaword-quoref_Guess_Title_For_Context', 'gigaword-samsum_Generate_a_summary_for_this_dialogue', 'gigaword-samsum_Given_the_above_dialogue_write_a_summary', 'gigaword-samsum_Sum_up_the_following_dialogue', 'gigaword-samsum_Summarize_', 'gigaword-samsum_Summarize_this_dialogue_', 'gigaword-samsum_To_sum_up_this_dialog', 'gigaword-samsum_Write_a_dialogue_that_match_this_summary', 'gigaword-wiki_qa_Topic_Prediction_Answer_Only', 'gigaword-wiki_qa_Topic_Prediction_Question_Only', 'gigaword-xsum_DOC_boils_down_to_simple_idea_that', 'gigaword-xsum_DOC_given_above_write_one_sentence', 'gigaword-xsum_DOC_how_would_you_rephrase_few_words', 'gigaword-xsum_DOC_tldr', 'gigaword-xsum_DOC_write_summary_of_above', 'gigaword-xsum_article_DOC_summary', 'gigaword-xsum_college_roommate_asked_DOC_so_I_recap', 'gigaword-xsum_read_below_DOC_write_abstract', 'gigaword-xsum_summarize_DOC', 'gigaword-xsum_summarize_this_DOC_summary', 'gigaword_TLDR', 'gigaword_first_sentence_title', 'gigaword_generate_summary_for_this', 'gigaword_in_a_nutshell', 'gigaword_make_a_title', 'gigaword_reverse_writing', 'gigaword_write_a_title_for_this_sentence', 'gigaword_write_an_article', 'gigaword_write_its_sentence', 'glue_mrpc-glue_qqp_answer', 'glue_mrpc-glue_qqp_duplicate', 'glue_mrpc-glue_qqp_duplicate_or_not', 'glue_mrpc-glue_qqp_meaning', 'glue_mrpc-glue_qqp_quora', 'glue_mrpc-glue_qqp_same_thing', 'glue_mrpc-paws_labeled_final_paraphrase_task', 'glue_mrpc_equivalent', 'glue_mrpc_generate_paraphrase', 'glue_mrpc_generate_sentence', 'glue_mrpc_paraphrase', 'glue_mrpc_replace', 'glue_mrpc_same_thing', 'glue_mrpc_want_to_know', 'glue_qqp-glue_mrpc_equivalent', 'glue_qqp-glue_mrpc_paraphrase', 'glue_qqp-glue_mrpc_replace', 'glue_qqp-glue_mrpc_same_thing', 'glue_qqp-glue_mrpc_want_to_know', 'glue_qqp_answer', 'glue_qqp_duplicate', 'glue_qqp_duplicate_or_not', 'glue_qqp_meaning', 'glue_qqp_quora', 'glue_qqp_same_thing', 'imdb-amazon_polarity_User_recommend_this_product', 'imdb_Movie_Expressed_Sentiment', 'imdb_Movie_Expressed_Sentiment_2', 'imdb_Negation_template_for_positive_and_negative', 'imdb_Reviewer_Enjoyment', 'imdb_Reviewer_Enjoyment_Yes_No', 'imdb_Reviewer_Expressed_Sentiment', 'imdb_Reviewer_Opinion_bad_good_choices', 'imdb_Reviewer_Sentiment_Feeling', 'imdb_Sentiment_with_choices_', 'imdb_Text_Expressed_Sentiment', 'imdb_Writer_Expressed_Sentiment', 'kilt_tasks_hotpotqa-cosmos_qa_only_question_answer', 'kilt_tasks_hotpotqa-sciq_Direct_Question_Closed_Book_', 'kilt_tasks_hotpotqa-wiki_qa_Direct_Answer_to_Question', 'kilt_tasks_hotpotqa_combining_facts', 'kilt_tasks_hotpotqa_complex_question', 'kilt_tasks_hotpotqa_final_exam', 'kilt_tasks_hotpotqa_formulate', 'kilt_tasks_hotpotqa_straighforward_qa', 'multi_news_distill', 'multi_news_expand_reverse_task_', 'multi_news_summarize', 'multi_news_summary_scenario', 'multi_news_synthesize', 'multi_news_what_are_the_key_points', 'paws_labeled_final-glue_mrpc_generate_paraphrase', 'paws_labeled_final-glue_mrpc_generate_sentence', 'paws_labeled_final_Concatenation', 'paws_labeled_final_Concatenation_no_label', 'paws_labeled_final_Meaning', 'paws_labeled_final_Meaning_no_label', 'paws_labeled_final_PAWS_ANLI_GPT3', 'paws_labeled_final_PAWS_ANLI_GPT3_no_label', 'paws_labeled_final_Rewrite', 'paws_labeled_final_Rewrite_no_label', 'paws_labeled_final_context_question', 'paws_labeled_final_context_question_no_label', 'paws_labeled_final_paraphrase_task', 'paws_labeled_final_task_description_no_label', 'qasc-cosmos_qa_context_description_question_answer_text', 'qasc-cosmos_qa_context_question_description_answer_text', 'qasc-cosmos_qa_description_context_question_answer_text', 'qasc-cosmos_qa_no_prompt_text', 'qasc-dream_baseline', 'qasc-dream_read_the_following_conversation_and_answer_the_question', 'qasc-quail_context_question_answer_description_text', 'qasc-quail_context_question_description_answer_text', 'qasc-quail_description_context_question_answer_text', 'qasc-quail_no_prompt_text', 'qasc-quartz_answer_question_based_on', 'qasc-quartz_answer_question_below', 'qasc-quartz_given_the_fact_answer_the_q', 'qasc-quartz_having_read_above_passage', 'qasc-quartz_paragraph_question_plain_concat', 'qasc-quartz_read_passage_below_choose', 'qasc-quartz_use_info_from_paragraph_question', 'qasc-quartz_use_info_from_question_paragraph', 'qasc-sciq_Multiple_Choice', 'qasc-social_i_qa_Show_choices_and_generate_answer', 'qasc_is_correct_1', 'qasc_is_correct_2', 'qasc_qa_with_combined_facts_1', 'qasc_qa_with_separated_facts_1', 'qasc_qa_with_separated_facts_2', 'qasc_qa_with_separated_facts_3', 'qasc_qa_with_separated_facts_4', 'qasc_qa_with_separated_facts_5', 'quail-adversarial_qa_dbidaf_answer_the_following_q', 'quail-adversarial_qa_dbidaf_based_on', 'quail-adversarial_qa_dbidaf_question_context_answer', 'quail-adversarial_qa_dbidaf_tell_what_it_is', 'quail-cosmos_qa_context_description_question_answer_text', 'quail-cosmos_qa_context_description_question_text', 'quail-cosmos_qa_context_question_description_answer_text', 'quail-cosmos_qa_description_context_question_answer_text', 'quail-cosmos_qa_description_context_question_text', 'quail-cosmos_qa_no_prompt_text', 'quail-dream_baseline', 'quail-dream_read_the_following_conversation_and_answer_the_question', 'quail-qasc_qa_with_combined_facts_1', 'quail-quartz_answer_question_based_on', 'quail-quartz_answer_question_below', 'quail-quartz_given_the_fact_answer_the_q', 'quail-quartz_having_read_above_passage', 'quail-quartz_paragraph_question_plain_concat', 'quail-quartz_read_passage_below_choose', 'quail-quartz_use_info_from_paragraph_question', 'quail-quartz_use_info_from_question_paragraph', 'quail-quoref_Answer_Friend_Question', 'quail-quoref_Answer_Test', 'quail-quoref_Context_Contains_Answer', 'quail-quoref_Find_Answer', 'quail-quoref_Found_Context_Online', 'quail-quoref_Given_Context_Answer_Question', 'quail-quoref_Guess_Answer', 'quail-quoref_Read_And_Extract_', 'quail-quoref_What_Is_The_Answer', 'quail-ropes_plain_no_background', 'quail-ropes_prompt_bottom_no_hint', 'quail-sciq_Multiple_Choice', 'quail-social_i_qa_Generate_answer', 'quail-social_i_qa_I_was_wondering', 'quail-social_i_qa_Show_choices_and_generate_answer', 'quail_context_description_question_answer_id', 'quail_context_description_question_answer_text', 'quail_context_description_question_text', 'quail_context_question_answer_description_id', 'quail_context_question_answer_description_text', 'quail_context_question_description_answer_id', 'quail_context_question_description_answer_text', 'quail_context_question_description_text', 'quail_description_context_question_answer_id', 'quail_description_context_question_answer_text', 'quail_description_context_question_text', 'quail_no_prompt_id', 'quail_no_prompt_text', 'quarel-cos_e_v1.11_description_question_option_text', 'quarel-cos_e_v1.11_question_description_option_text', 'quarel-cos_e_v1.11_question_option_description_text', 'quarel-sciq_Multiple_Choice_Closed_Book_', 'quarel-sciq_Multiple_Choice_Question_First', 'quarel_choose_between', 'quarel_do_not_use', 'quarel_heres_a_story', 'quarel_logic_test', 'quarel_testing_students', 'quartz-cosmos_qa_context_description_question_answer_text', 'quartz-cosmos_qa_context_question_description_answer_text', 'quartz-cosmos_qa_description_context_question_answer_text', 'quartz-cosmos_qa_no_prompt_text', 'quartz-dream_baseline', 'quartz-dream_read_the_following_conversation_and_answer_the_question', 'quartz-qasc_qa_with_combined_facts_1', 'quartz-quail_context_question_answer_description_text', 'quartz-quail_context_question_description_answer_text', 'quartz-quail_description_context_question_answer_text', 'quartz-quail_no_prompt_text', 'quartz-sciq_Multiple_Choice', 'quartz-social_i_qa_Show_choices_and_generate_answer', 'quartz_answer_question_based_on', 'quartz_answer_question_below', 'quartz_given_the_fact_answer_the_q', 'quartz_having_read_above_passage', 'quartz_paragraph_question_plain_concat', 'quartz_read_passage_below_choose', 'quartz_use_info_from_paragraph_question', 'quartz_use_info_from_question_paragraph', 'quoref-adversarial_qa_dbidaf_answer_the_following_q', 'quoref-adversarial_qa_dbidaf_based_on', 'quoref-adversarial_qa_dbidaf_question_context_answer', 'quoref-adversarial_qa_dbidaf_tell_what_it_is', 'quoref-cnn_dailymail_3.0.0_2_or_3_sentences', 'quoref-cnn_dailymail_3.0.0_news_card_view', 'quoref-cnn_dailymail_3.0.0_news_stock', 'quoref-cnn_dailymail_3.0.0_news_summary', 'quoref-cnn_dailymail_3.0.0_sum_in_brief', 'quoref-cnn_dailymail_3.0.0_tldr_summary', 'quoref-cnn_dailymail_3.0.0_write_an_outline', 'quoref-common_gen_sentence_to_concepts', 'quoref-common_gen_topics_from_the_sentence', \
 'quoref-cosmos_qa_context_description_question_text', 'quoref-cosmos_qa_description_context_question_text', 'quoref-duorc_SelfRC_title_generation', 'quoref-gigaword_TLDR', 'quoref-gigaword_first_sentence_title', 'quoref-gigaword_generate_summary_for_this', 'quoref-gigaword_in_a_nutshell', 'quoref-gigaword_make_a_title', 'quoref-gigaword_write_a_title_for_this_sentence', 'quoref-gigaword_write_its_sentence', 'quoref-quail_context_description_question_text', 'quoref-quail_context_question_description_text', 'quoref-quail_description_context_question_text', 'quoref-ropes_plain_no_background', 'quoref-ropes_prompt_bottom_no_hint', 'quoref-samsum_Generate_a_summary_for_this_dialogue', 'quoref-samsum_Given_the_above_dialogue_write_a_summary', 'quoref-samsum_Sum_up_the_following_dialogue', 'quoref-samsum_Summarize_', 'quoref-samsum_Summarize_this_dialogue_', 'quoref-samsum_To_sum_up_this_dialog', 'quoref-social_i_qa_Generate_answer', 'quoref-social_i_qa_I_was_wondering', 'quoref-wiki_qa_Topic_Prediction_Answer_Only', 'quoref-wiki_qa_Topic_Prediction_Question_Only', 'quoref-xsum_DOC_boils_down_to_simple_idea_that', 'quoref-xsum_DOC_given_above_write_one_sentence', 'quoref-xsum_DOC_how_would_you_rephrase_few_words', 'quoref-xsum_DOC_tldr', 'quoref-xsum_DOC_write_summary_of_above', 'quoref-xsum_article_DOC_summary', 'quoref-xsum_college_roommate_asked_DOC_so_I_recap', 'quoref-xsum_read_below_DOC_write_abstract', 'quoref-xsum_summarize_DOC', 'quoref-xsum_summarize_this_DOC_summary', 'quoref_Answer_Friend_Question', 'quoref_Answer_Question_Given_Context', 'quoref_Answer_Test', 'quoref_Context_Contains_Answer', 'quoref_Find_Answer', 'quoref_Found_Context_Online', 'quoref_Given_Context_Answer_Question', 'quoref_Guess_Answer', 'quoref_Guess_Title_For_Context', 'quoref_Read_And_Extract_', 'quoref_What_Is_The_Answer', 'ropes-adversarial_qa_dbidaf_answer_the_following_q', 'ropes-adversarial_qa_dbidaf_based_on', 'ropes-adversarial_qa_dbidaf_question_context_answer', 'ropes-adversarial_qa_dbidaf_tell_what_it_is', 'ropes-cosmos_qa_context_description_question_text', 'ropes-cosmos_qa_description_context_question_text', 'ropes-duorc_SelfRC_answer_question', 'ropes-duorc_SelfRC_decide_worth_it', 'ropes-duorc_SelfRC_movie_director', 'ropes-duorc_SelfRC_question_answering', 'ropes-quail_context_description_question_text', 'ropes-quail_context_question_description_text', 'ropes-quail_description_context_question_text', 'ropes-quoref_Answer_Friend_Question', 'ropes-quoref_Answer_Test', 'ropes-quoref_Context_Contains_Answer', 'ropes-quoref_Find_Answer', 'ropes-quoref_Found_Context_Online', 'ropes-quoref_Given_Context_Answer_Question', 'ropes-quoref_Guess_Answer', 'ropes-quoref_Read_And_Extract_', 'ropes-quoref_What_Is_The_Answer', 'ropes-social_i_qa_Generate_answer', 'ropes-social_i_qa_I_was_wondering', 'ropes_background_new_situation_answer', 'ropes_background_situation_middle', 'ropes_given_background_situation', 'ropes_new_situation_background_answer', 'ropes_plain_background_situation', 'ropes_plain_bottom_hint', 'ropes_plain_no_background', 'ropes_prompt_beginning', 'ropes_prompt_bottom_hint_beginning', 'ropes_prompt_bottom_no_hint', 'ropes_prompt_mix', 'ropes_read_background_situation', 'rotten_tomatoes-amazon_polarity_User_recommend_this_product', 'rotten_tomatoes_Movie_Expressed_Sentiment', 'rotten_tomatoes_Movie_Expressed_Sentiment_2', 'rotten_tomatoes_Reviewer_Enjoyment', 'rotten_tomatoes_Reviewer_Enjoyment_Yes_No', 'rotten_tomatoes_Reviewer_Expressed_Sentiment', 'rotten_tomatoes_Reviewer_Opinion_bad_good_choices', 'rotten_tomatoes_Reviewer_Sentiment_Feeling', 'rotten_tomatoes_Sentiment_with_choices_', 'rotten_tomatoes_Text_Expressed_Sentiment', 'rotten_tomatoes_Writer_Expressed_Sentiment', 'samsum-cnn_dailymail_3.0.0_2_or_3_sentences', 'samsum-cnn_dailymail_3.0.0_generate_story', 'samsum-cnn_dailymail_3.0.0_news_card_view', 'samsum-cnn_dailymail_3.0.0_news_stock', 'samsum-cnn_dailymail_3.0.0_news_summary', 'samsum-cnn_dailymail_3.0.0_spice_up_story', 'samsum-cnn_dailymail_3.0.0_sum_in_brief', 'samsum-cnn_dailymail_3.0.0_tldr_summary', 'samsum-cnn_dailymail_3.0.0_write_an_outline', 'samsum-common_gen_Example_prompt', 'samsum-common_gen_Given_concepts_type_1', 'samsum-common_gen_Given_concepts_type_2', 'samsum-common_gen_Put_together', 'samsum-common_gen_random_task_template_prompt', 'samsum-common_gen_sentence_to_concepts', 'samsum-common_gen_topics_from_the_sentence', 'samsum-duorc_SelfRC_title_generation', 'samsum-gigaword_TLDR', 'samsum-gigaword_first_sentence_title', 'samsum-gigaword_generate_summary_for_this', 'samsum-gigaword_in_a_nutshell', 'samsum-gigaword_make_a_title', 'samsum-gigaword_reverse_writing', 'samsum-gigaword_write_a_title_for_this_sentence', 'samsum-gigaword_write_an_article', 'samsum-gigaword_write_its_sentence', 'samsum-quoref_Guess_Title_For_Context', 'samsum-wiki_qa_Topic_Prediction_Answer_Only', 'samsum-wiki_qa_Topic_Prediction_Question_Only', 'samsum-xsum_DOC_boils_down_to_simple_idea_that', 'samsum-xsum_DOC_given_above_write_one_sentence', 'samsum-xsum_DOC_how_would_you_rephrase_few_words', 'samsum-xsum_DOC_tldr', 'samsum-xsum_DOC_write_summary_of_above', 'samsum-xsum_article_DOC_summary', 'samsum-xsum_college_roommate_asked_DOC_so_I_recap', 'samsum-xsum_read_below_DOC_write_abstract', 'samsum-xsum_summarize_DOC', 'samsum-xsum_summarize_this_DOC_summary', 'samsum_Generate_a_summary_for_this_dialogue', 'samsum_Given_the_above_dialogue_write_a_summary', 'samsum_Sum_up_the_following_dialogue', 'samsum_Summarize_', 'samsum_Summarize_this_dialogue_', 'samsum_To_sum_up_this_dialog', 'samsum_Write_a_dialogue_that_match_this_summary', 'sciq-cos_e_v1.11_description_question_option_text', 'sciq-cos_e_v1.11_question_description_option_text', 'sciq-cos_e_v1.11_question_option_description_text', 'sciq-cosmos_qa_context_description_question_answer_text', 'sciq-cosmos_qa_context_question_description_answer_text', 'sciq-cosmos_qa_description_context_question_answer_text', 'sciq-cosmos_qa_no_prompt_text', 'sciq-cosmos_qa_only_question_answer', 'sciq-dream_baseline', 'sciq-dream_read_the_following_conversation_and_answer_the_question', 'sciq-kilt_tasks_hotpotqa_combining_facts', 'sciq-kilt_tasks_hotpotqa_complex_question', 'sciq-kilt_tasks_hotpotqa_final_exam', 'sciq-kilt_tasks_hotpotqa_formulate', 'sciq-kilt_tasks_hotpotqa_straighforward_qa', 'sciq-qasc_qa_with_combined_facts_1', 'sciq-quail_context_question_answer_description_text', 'sciq-quail_context_question_description_answer_text', 'sciq-quail_description_context_question_answer_text', 'sciq-quail_no_prompt_text', 'sciq-quarel_choose_between', 'sciq-quarel_do_not_use', 'sciq-quarel_heres_a_story', 'sciq-quarel_logic_test', 'sciq-quarel_testing_students', 'sciq-quartz_answer_question_based_on', 'sciq-quartz_answer_question_below', 'sciq-quartz_given_the_fact_answer_the_q', 'sciq-quartz_having_read_above_passage', 'sciq-quartz_paragraph_question_plain_concat', 'sciq-quartz_read_passage_below_choose', 'sciq-quartz_use_info_from_paragraph_question', 'sciq-quartz_use_info_from_question_paragraph', 'sciq-social_i_qa_Show_choices_and_generate_answer', 'sciq-wiki_qa_Direct_Answer_to_Question', 'sciq_Direct_Question', 'sciq_Direct_Question_Closed_Book_', 'sciq_Multiple_Choice', 'sciq_Multiple_Choice_Closed_Book_', 'sciq_Multiple_Choice_Question_First', 'social_i_qa-adversarial_qa_dbidaf_answer_the_following_q', 'social_i_qa-adversarial_qa_dbidaf_based_on', 'social_i_qa-adversarial_qa_dbidaf_question_context_answer', 'social_i_qa-adversarial_qa_dbidaf_tell_what_it_is', 'social_i_qa-cosmos_qa_context_answer_to_question', 'social_i_qa-cosmos_qa_context_description_question_answer_text', 'social_i_qa-cosmos_qa_context_description_question_text', 'social_i_qa-cosmos_qa_context_question_description_answer_text', 'social_i_qa-cosmos_qa_description_context_question_answer_text', 'social_i_qa-cosmos_qa_description_context_question_text', 'social_i_qa-cosmos_qa_no_prompt_text', 'social_i_qa-dream_baseline', 'social_i_qa-dream_read_the_following_conversation_and_answer_the_question', 'social_i_qa-duorc_SelfRC_generate_question_by_answer', 'social_i_qa-qasc_qa_with_combined_facts_1', 'social_i_qa-quail_context_description_question_text', 'social_i_qa-quail_context_question_answer_description_text', 'social_i_qa-quail_context_question_description_answer_text', 'social_i_qa-quail_context_question_description_text', 'social_i_qa-quail_description_context_question_answer_text', 'social_i_qa-quail_description_context_question_text', 'social_i_qa-quail_no_prompt_text', 'social_i_qa-quartz_answer_question_based_on', 'social_i_qa-quartz_answer_question_below', 'social_i_qa-quartz_given_the_fact_answer_the_q', 'social_i_qa-quartz_having_read_above_passage', 'social_i_qa-quartz_paragraph_question_plain_concat', 'social_i_qa-quartz_read_passage_below_choose', 'social_i_qa-quartz_use_info_from_paragraph_question', 'social_i_qa-quartz_use_info_from_question_paragraph', 'social_i_qa-quoref_Answer_Friend_Question', 'social_i_qa-quoref_Answer_Test', 'social_i_qa-quoref_Context_Contains_Answer', 'social_i_qa-quoref_Find_Answer', 'social_i_qa-quoref_Found_Context_Online', 'social_i_qa-quoref_Given_Context_Answer_Question', 'social_i_qa-quoref_Guess_Answer', 'social_i_qa-quoref_Read_And_Extract_', 'social_i_qa-quoref_What_Is_The_Answer', 'social_i_qa-sciq_Multiple_Choice', 'social_i_qa-wiki_qa_Generate_Question_from_Topic', 'social_i_qa-wiki_qa_Jeopardy_style', 'social_i_qa-wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'social_i_qa_Check_if_a_random_answer_is_valid_or_not', 'social_i_qa_Generate_answer', 'social_i_qa_Generate_the_question_from_the_answer', 'social_i_qa_I_was_wondering', 'social_i_qa_Show_choices_and_generate_answer', 'social_i_qa_Show_choices_and_generate_index', 'trec-dbpedia_14_given_a_choice_of_categories_', 'trec-dbpedia_14_given_list_what_category_does_the_paragraph_belong_to', 'trec-dbpedia_14_pick_one_category_for_the_following_text', 'trec_fine_grained_ABBR', \
 'trec_fine_grained_ABBR_context_first', 'trec_fine_grained_DESC', 'trec_fine_grained_DESC_context_first', 'trec_fine_grained_ENTY', 'trec_fine_grained_HUM', 'trec_fine_grained_HUM_context_first', 'trec_fine_grained_LOC', 'trec_fine_grained_LOC_context_first', 'trec_fine_grained_NUM', 'trec_fine_grained_NUM_context_first', 'trec_fine_grained_open', 'trec_fine_grained_open_context_first', 'trec_pick_the_best_descriptor', 'trec_trec1', 'trec_trec2', 'trec_what_category_best_describe', 'trec_which_category_best_describes', 'wiki_bio_comprehension', 'wiki_bio_guess_person', 'wiki_bio_key_content', 'wiki_bio_what_content', 'wiki_bio_who', 'wiki_hop_original_choose_best_object_affirmative_1', 'wiki_hop_original_choose_best_object_affirmative_2', 'wiki_hop_original_choose_best_object_affirmative_3', 'wiki_hop_original_choose_best_object_interrogative_1', 'wiki_hop_original_choose_best_object_interrogative_2', 'wiki_hop_original_explain_relation', 'wiki_hop_original_generate_object', 'wiki_hop_original_generate_subject', 'wiki_hop_original_generate_subject_and_object', 'wiki_qa-cnn_dailymail_3.0.0_2_or_3_sentences', 'wiki_qa-cnn_dailymail_3.0.0_news_card_view', 'wiki_qa-cnn_dailymail_3.0.0_news_stock', 'wiki_qa-cnn_dailymail_3.0.0_news_summary', 'wiki_qa-cnn_dailymail_3.0.0_sum_in_brief', 'wiki_qa-cnn_dailymail_3.0.0_tldr_summary', 'wiki_qa-cnn_dailymail_3.0.0_write_an_outline', 'wiki_qa-common_gen_sentence_to_concepts', 'wiki_qa-common_gen_topics_from_the_sentence', 'wiki_qa-cosmos_qa_context_answer_to_question', 'wiki_qa-cosmos_qa_only_question_answer', 'wiki_qa-duorc_SelfRC_generate_question_by_answer', 'wiki_qa-duorc_SelfRC_title_generation', 'wiki_qa-gigaword_TLDR', 'wiki_qa-gigaword_first_sentence_title', 'wiki_qa-gigaword_generate_summary_for_this', 'wiki_qa-gigaword_in_a_nutshell', 'wiki_qa-gigaword_make_a_title', 'wiki_qa-gigaword_write_a_title_for_this_sentence', 'wiki_qa-gigaword_write_its_sentence', 'wiki_qa-kilt_tasks_hotpotqa_combining_facts', 'wiki_qa-kilt_tasks_hotpotqa_complex_question', 'wiki_qa-kilt_tasks_hotpotqa_final_exam', 'wiki_qa-kilt_tasks_hotpotqa_formulate', 'wiki_qa-kilt_tasks_hotpotqa_straighforward_qa', 'wiki_qa-quoref_Guess_Title_For_Context', 'wiki_qa-samsum_Generate_a_summary_for_this_dialogue', 'wiki_qa-samsum_Given_the_above_dialogue_write_a_summary', 'wiki_qa-samsum_Sum_up_the_following_dialogue', 'wiki_qa-samsum_Summarize_', 'wiki_qa-samsum_Summarize_this_dialogue_', 'wiki_qa-samsum_To_sum_up_this_dialog', 'wiki_qa-sciq_Direct_Question_Closed_Book_', 'wiki_qa-social_i_qa_Generate_the_question_from_the_answer', 'wiki_qa-xsum_DOC_boils_down_to_simple_idea_that', 'wiki_qa-xsum_DOC_given_above_write_one_sentence', 'wiki_qa-xsum_DOC_how_would_you_rephrase_few_words', 'wiki_qa-xsum_DOC_tldr', 'wiki_qa-xsum_DOC_write_summary_of_above', 'wiki_qa-xsum_article_DOC_summary', 'wiki_qa-xsum_college_roommate_asked_DOC_so_I_recap', 'wiki_qa-xsum_read_below_DOC_write_abstract', 'wiki_qa-xsum_summarize_DOC', 'wiki_qa-xsum_summarize_this_DOC_summary', 'wiki_qa_Decide_good_answer', 'wiki_qa_Direct_Answer_to_Question', 'wiki_qa_Generate_Question_from_Topic', 'wiki_qa_Is_This_True_', 'wiki_qa_Jeopardy_style', 'wiki_qa_Topic_Prediction_Answer_Only', 'wiki_qa_Topic_Prediction_Question_Only', 'wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'wiki_qa_automatic_system', 'wiki_qa_exercise', 'wiki_qa_found_on_google', 'wiqa-dream_generate_first_utterance', 'wiqa-dream_generate_last_utterance', 'wiqa_does_the_supposed_perturbation_have_an_effect', 'wiqa_effect_with_label_answer', 'wiqa_effect_with_string_answer', 'wiqa_what_is_the_final_step_of_the_following_process', 'wiqa_what_is_the_missing_first_step', 'wiqa_what_might_be_the_first_step_of_the_process', 'wiqa_what_might_be_the_last_step_of_the_process', 'wiqa_which_of_the_following_is_the_supposed_perturbation', 'xsum-cnn_dailymail_3.0.0_2_or_3_sentences', 'xsum-cnn_dailymail_3.0.0_news_card_view', 'xsum-cnn_dailymail_3.0.0_news_stock', 'xsum-cnn_dailymail_3.0.0_news_summary', 'xsum-cnn_dailymail_3.0.0_sum_in_brief', 'xsum-cnn_dailymail_3.0.0_tldr_summary', 'xsum-cnn_dailymail_3.0.0_write_an_outline', 'xsum-common_gen_sentence_to_concepts', 'xsum-common_gen_topics_from_the_sentence', 'xsum-duorc_SelfRC_title_generation', 'xsum-gigaword_TLDR', 'xsum-gigaword_first_sentence_title', 'xsum-gigaword_generate_summary_for_this', 'xsum-gigaword_in_a_nutshell', 'xsum-gigaword_make_a_title', 'xsum-gigaword_write_a_title_for_this_sentence', 'xsum-gigaword_write_its_sentence', 'xsum-quoref_Guess_Title_For_Context', 'xsum-samsum_Generate_a_summary_for_this_dialogue', 'xsum-samsum_Given_the_above_dialogue_write_a_summary', 'xsum-samsum_Sum_up_the_following_dialogue', 'xsum-samsum_Summarize_', 'xsum-samsum_Summarize_this_dialogue_', 'xsum-samsum_To_sum_up_this_dialog', 'xsum-wiki_qa_Topic_Prediction_Answer_Only', 'xsum-wiki_qa_Topic_Prediction_Question_Only', 'xsum_DOC_boils_down_to_simple_idea_that', 'xsum_DOC_given_above_write_one_sentence', 'xsum_DOC_how_would_you_rephrase_few_words', 'xsum_DOC_tldr', 'xsum_DOC_write_summary_of_above', 'xsum_article_DOC_summary', 'xsum_college_roommate_asked_DOC_so_I_recap', 'xsum_read_below_DOC_write_abstract', 'xsum_summarize_DOC', 'xsum_summarize_this_DOC_summary', 'yelp_review_full_based_on_that', 'yelp_review_full_format_rating', 'yelp_review_full_format_score', 'yelp_review_full_format_star', 'yelp_review_full_on_a_scale', 'yelp_review_full_so_i_would', 'yelp_review_full_this_place']
ori_t0_ls=['glue_mrpc_replace', 'qasc_qa_with_separated_facts_5', 'adversarial_qa_dbert_generate_question', 'trec_trec1', 'quoref_Answer_Question_Given_Context', 'gigaword_write_a_title_for_this_sentence', 'quoref_Given_Context_Answer_Question', 'amazon_polarity_Is_this_product_review_positive', 'wiqa_effect_with_string_answer', 'adversarial_qa_dbidaf_based_on', 'cosmos_qa_context_description_question_answer_text', 'dream_baseline', 'ropes_background_situation_middle', 'paws_labeled_final_Rewrite', 'duorc_ParaphraseRC_generate_question', 'duorc_ParaphraseRC_title_generation', 'ropes_prompt_beginning', 'quartz_read_passage_below_choose', 'yelp_review_full_based_on_that', 'paws_labeled_final_Meaning', 'rotten_tomatoes_Reviewer_Expressed_Sentiment', 'gigaword_reverse_writing', 'social_i_qa_Generate_answer', 'adversarial_qa_dbert_question_context_answer', 'duorc_SelfRC_question_answering', 'imdb_Reviewer_Sentiment_Feeling', 'app_reviews_convert_to_rating', 'imdb_Reviewer_Enjoyment', 'ropes_plain_bottom_hint', 'social_i_qa_Show_choices_and_generate_answer', 'cosmos_qa_description_context_question_answer_text', 'gigaword_TLDR', 'wiki_qa_automatic_system', 'paws_labeled_final_paraphrase_task', 'yelp_review_full_format_star', 'duorc_ParaphraseRC_decide_worth_it', 'quoref_Context_Contains_Answer', 'sciq_Direct_Question', 'wiqa_what_is_the_missing_first_step', 'imdb_Writer_Expressed_Sentiment', 'cos_e_v1.11_explain_why_human', 'multi_news_synthesize', 'multi_news_summary_scenario', 'adversarial_qa_dbidaf_answer_the_following_q', 'quail_description_context_question_answer_text', 'duorc_SelfRC_title_generation', 'amazon_polarity_convey_negative_or_positive_sentiment', 'trec_fine_grained_ABBR_context_first', 'cos_e_v1.11_i_think', 'xsum_summarize_DOC', 'wiqa_does_the_supposed_perturbation_have_an_effect', 'quartz_having_read_above_passage', 'trec_trec2', 'quoref_Answer_Friend_Question', 'rotten_tomatoes_Movie_Expressed_Sentiment_2', 'trec_fine_grained_DESC', 'wiki_qa_Is_This_True_', 'cnn_dailymail_3.0.0_news_summary', 'trec_fine_grained_HUM_context_first', 'qasc_is_correct_2', 'trec_fine_grained_open', 'ag_news_classify_with_choices_question_first', 'cosmos_qa_context_question_description_text', 'kilt_tasks_hotpotqa_complex_question', 'common_gen_Given_concepts_type_1', 'quoref_Find_Answer', 'trec_fine_grained_LOC', 'trec_fine_grained_LOC_context_first', 'quarel_testing_students', 'paws_labeled_final_Concatenation_no_label', 'quoref_Guess_Answer', 'dream_generate_last_utterance', 'cosmos_qa_description_context_question_answer_id', 'ropes_background_new_situation_answer', 'glue_qqp_same_thing', 'quail_context_description_question_text', 'imdb_Reviewer_Expressed_Sentiment', 'multi_news_summarize', 'yelp_review_full_on_a_scale', 'imdb_Reviewer_Opinion_bad_good_choices', 'wiqa_what_is_the_final_step_of_the_following_process', 'yelp_review_full_so_i_would', 'xsum_summarize_this_DOC_summary', 'qasc_qa_with_separated_facts_4', 'quoref_Found_Context_Online', 'rotten_tomatoes_Movie_Expressed_Sentiment', 'ag_news_which_section_choices', 'wiqa_which_of_the_following_is_the_supposed_perturbation', 'duorc_ParaphraseRC_build_story_around_qa', 'wiki_hop_original_choose_best_object_affirmative_1', 'adversarial_qa_dbert_tell_what_it_is', 'yelp_review_full_this_place', 'glue_qqp_meaning', 'glue_qqp_quora', 'cosmos_qa_context_question_description_answer_text', 'samsum_Sum_up_the_following_dialogue', 'glue_qqp_duplicate', 'dream_read_the_following_conversation_and_answer_the_question', 'glue_mrpc_equivalent', 'paws_labeled_final_Rewrite_no_label', 'adversarial_qa_dbidaf_tell_what_it_is', 'ropes_read_background_situation', 'qasc_is_correct_1', 'xsum_DOC_how_would_you_rephrase_few_words', 'duorc_ParaphraseRC_answer_question', 'gigaword_make_a_title', 'trec_fine_grained_ENTY', 'rotten_tomatoes_Reviewer_Enjoyment', 'wiki_qa_Jeopardy_style', 'paws_labeled_final_context_question', 'duorc_SelfRC_generate_question', 'trec_pick_the_best_descriptor', 'quoref_Read_And_Extract_', 'dbpedia_14_given_a_choice_of_categories_', 'cos_e_v1.11_question_description_option_id', 'quartz_answer_question_below', 'wiqa_what_might_be_the_last_step_of_the_process', 'quail_context_description_question_answer_id', 'paws_labeled_final_task_description_no_label', 'amazon_polarity_User_recommend_this_product', 'gigaword_write_an_article', 'wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'cos_e_v1.11_generate_explanation_given_text', 'wiki_qa_Topic_Prediction_Question_Only', 'amazon_polarity_Is_this_review_negative', 'sciq_Multiple_Choice_Closed_Book_', 'qasc_qa_with_separated_facts_1', 'glue_mrpc_same_thing', 'wiki_hop_original_choose_best_object_interrogative_2', 'duorc_SelfRC_generate_question_by_answer', 'duorc_ParaphraseRC_generate_question_by_answer', 'cosmos_qa_description_context_question_text', 'wiki_qa_exercise', 'paws_labeled_final_context_question_no_label', 'wiki_hop_original_generate_subject', 'dream_answer_to_dialogue', 'qasc_qa_with_separated_facts_2', 'paws_labeled_final_PAWS_ANLI_GPT3_no_label', 'multi_news_what_are_the_key_points', 'wiki_hop_original_choose_best_object_affirmative_2', 'cos_e_v1.11_aligned_with_common_sense', 'amazon_polarity_negative_or_positive_tone', 'imdb_Negation_template_for_positive_and_negative', 'cosmos_qa_no_prompt_id', 'wiqa_effect_with_label_answer', 'multi_news_expand_reverse_task_', 'rotten_tomatoes_Writer_Expressed_Sentiment', 'qasc_qa_with_combined_facts_1', 'quartz_given_the_fact_answer_the_q', 'wiki_hop_original_explain_relation', 'trec_fine_grained_HUM', 'quail_no_prompt_id', 'amazon_polarity_user_satisfied', 'kilt_tasks_hotpotqa_final_exam', 'qasc_qa_with_separated_facts_3', 'adversarial_qa_droberta_answer_the_following_q', 'duorc_SelfRC_build_story_around_qa', 'common_gen_Example_prompt', 'quarel_logic_test', 'quartz_use_info_from_question_paragraph', 'common_gen_choice_in_concept_centric_sentence_generation', 'yelp_review_full_format_score', 'rotten_tomatoes_Text_Expressed_Sentiment', 'multi_news_distill', 'quail_context_question_answer_description_text', 'xsum_college_roommate_asked_DOC_so_I_recap', 'cnn_dailymail_3.0.0_generate_story', 'ropes_prompt_bottom_no_hint', 'common_gen_topic_to_sentence', 'kilt_tasks_hotpotqa_straighforward_qa', 'glue_mrpc_generate_sentence', 'cosmos_qa_context_question_description_answer_id', 'imdb_Reviewer_Enjoyment_Yes_No', 'samsum_Summarize_', 'wiki_hop_original_choose_best_object_affirmative_3', 'duorc_SelfRC_extract_answer', 'amazon_polarity_would_you_buy', 'glue_mrpc_want_to_know', 'imdb_Movie_Expressed_Sentiment', 'cos_e_v1.11_question_option_description_text', 'ag_news_classify_with_choices', 'wiki_hop_original_choose_best_object_interrogative_1', 'amazon_polarity_Is_this_review', 'social_i_qa_Generate_the_question_from_the_answer', 'adversarial_qa_dbidaf_generate_question', 'wiki_bio_guess_person', 'ropes_plain_no_background', 'ag_news_classify_question_first', 'amazon_polarity_flattering_or_not', 'cosmos_qa_only_question_answer', 'glue_mrpc_generate_paraphrase', 'kilt_tasks_hotpotqa_formulate', 'quoref_Guess_Title_For_Context', 'cosmos_qa_context_description_question_text', 'ropes_prompt_bottom_hint_beginning', 'trec_fine_grained_DESC_context_first', 'quarel_choose_between', 'quail_context_description_question_answer_text', 'gigaword_first_sentence_title', 'adversarial_qa_dbidaf_question_context_answer', 'social_i_qa_Check_if_a_random_answer_is_valid_or_not', 'wiki_qa_found_on_google', 'cnn_dailymail_3.0.0_2_or_3_sentences', 'dbpedia_14_given_list_what_category_does_the_paragraph_belong_to', 'glue_qqp_duplicate_or_not', 'dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to', 'cnn_dailymail_3.0.0_news_card_view', 'ropes_new_situation_background_answer', 'common_gen_Given_concepts_type_2', 'cosmos_qa_context_description_question_answer_id', 'common_gen_topics_from_the_sentence', 'duorc_ParaphraseRC_movie_director', 'cnn_dailymail_3.0.0_write_an_outline', 'app_reviews_generate_review', 'wiki_qa_Topic_Prediction_Answer_Only', 'quarel_do_not_use', 'rotten_tomatoes_Reviewer_Opinion_bad_good_choices', 'sciq_Multiple_Choice_Question_First', 'cos_e_v1.11_description_question_option_text', 'quail_description_context_question_text', 'cosmos_qa_no_prompt_text', 'wiki_qa_Generate_Question_from_Topic', 'xsum_article_DOC_summary', 'ropes_prompt_mix', 'kilt_tasks_hotpotqa_combining_facts', 'quail_context_question_description_answer_text', 'samsum_To_sum_up_this_dialog', 'quail_context_question_answer_description_id', 'ag_news_classify', 'xsum_DOC_given_above_write_one_sentence', 'duorc_ParaphraseRC_extract_answer', 'samsum_Generate_a_summary_for_this_dialogue', 'cos_e_v1.11_question_option_description_id', 'wiki_hop_original_generate_object', 'gigaword_write_its_sentence', 'quail_description_context_question_answer_id', 'cnn_dailymail_3.0.0_news_stock', 'xsum_DOC_write_summary_of_above', 'imdb_Movie_Expressed_Sentiment_2', 'trec_which_category_best_describes', 'cos_e_v1.11_question_description_option_text', 'samsum_Write_a_dialogue_that_match_this_summary', 'common_gen_sentence_to_concepts', 'ropes_plain_background_situation', 'dream_generate_first_utterance', 'quail_no_prompt_text', 'cos_e_v1.11_rationale', 'social_i_qa_I_was_wondering', 'samsum_Summarize_this_dialogue_', 'wiki_bio_key_content', 'wiqa_what_might_be_the_first_step_of_the_process', 'gigaword_generate_summary_for_this', 'adversarial_qa_droberta_based_on', 'gigaword_in_a_nutshell', 'duorc_ParaphraseRC_question_answering', 'ropes_given_background_situation', 'adversarial_qa_droberta_question_context_answer', 'imdb_Text_Expressed_Sentiment', 'social_i_qa_Show_choices_and_generate_index', 'xsum_DOC_boils_down_to_simple_idea_that', 'quail_context_question_description_answer_id', 'wiki_qa_Direct_Answer_to_Question', 'quoref_Answer_Test', 'app_reviews_categorize_rating_using_review', 'wiki_bio_who', 'adversarial_qa_dbert_based_on', 'wiki_hop_original_generate_subject_and_object', 'cnn_dailymail_3.0.0_spice_up_story',\
 'paws_labeled_final_Concatenation', 'quartz_answer_question_based_on', 'quartz_use_info_from_paragraph_question', 'samsum_Given_the_above_dialogue_write_a_summary', 'trec_fine_grained_NUM_context_first', 'rotten_tomatoes_Sentiment_with_choices_', 'quartz_paragraph_question_plain_concat', 'sciq_Multiple_Choice', 'xsum_DOC_tldr', 'quarel_heres_a_story', 'cnn_dailymail_3.0.0_sum_in_brief', 'app_reviews_convert_to_star_rating', 'wiki_bio_comprehension', 'ag_news_which_section', 'trec_fine_grained_NUM', 'imdb_Sentiment_with_choices_', 'dbpedia_14_pick_one_category_for_the_following_text', 'wiki_bio_what_content', 'trec_fine_grained_ABBR', 'common_gen_Put_together', 'adversarial_qa_droberta_tell_what_it_is', 'glue_qqp_answer', 'quail_context_question_description_text', 'glue_mrpc_paraphrase', 'duorc_SelfRC_movie_director', 'duorc_SelfRC_decide_worth_it', 'xsum_read_below_DOC_write_abstract', 'trec_what_category_best_describe', 'cos_e_v1.11_description_question_option_id', 'quoref_What_Is_The_Answer', 'cosmos_qa_context_answer_to_question', 'yelp_review_full_format_rating', 'cnn_dailymail_3.0.0_tldr_summary', 'ag_news_recommend', 'adversarial_qa_dbert_answer_the_following_q', 'duorc_SelfRC_answer_question', 'sciq_Direct_Question_Closed_Book_', 'common_gen_random_task_template_prompt', 'adversarial_qa_droberta_generate_question', 'wiki_qa_Decide_good_answer', 'paws_labeled_final_PAWS_ANLI_GPT3', 'trec_fine_grained_open_context_first', 'rotten_tomatoes_Reviewer_Sentiment_Feeling', 'rotten_tomatoes_Reviewer_Enjoyment_Yes_No', 'paws_labeled_final_Meaning_no_label']

taxonomy_origin_5w=[t for t in taxonomy_total_5w if t not in ori_t0_ls]
REGISTERED_DATA_LIST['zj_para_ls']=zj_para_ls
REGISTERED_DATA_LIST['taxonomy_origin_5w']=taxonomy_origin_5w
for task_name in T0_TRAIN_TASK_NAME:
    new_task_name = task_name.replace("/","_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(new_task_name)]
    REGISTERED_DATA_LIST[task_name]=sub_list

BIG_BENCH_TEST_TASK_LIST = [
    "code_line_description",
    "conceptual_combinations",
    "hindu_knowledge",
    "known_unknowns",
    "language_identification",
    "logic_grid_puzzle",
    "logical_deduction",
    "misconceptions",
    "movie_dialog_same_or_different",
    "novel_concepts",
    "strategyqa",
    "formal_fallacies_syllogisms_negation",
    "vitaminc_fact_verification",
    "winowhy"
]

T0_TEST_TASK_NAME = [
    "super_glue/wsc.fixed",
    "super_glue/wic",
    "super_glue/copa",
    # "story_cloze/2016",
    "super_glue/cb",
    "super_glue/rte",
    "hellaswag",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "winogrande/winogrande_xl",
]
T0_TEST_TASK_LIST=[]
for task_name in T0_TEST_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_TEST_TASK_LIST = T0_TEST_TASK_LIST + sub_list


T0_plus_TRAIN_TASK_NAME = [
    "glue/mrpc",
    "glue/qqp",
    "paws/labeled_final",
    "ai2_arc/ARC_Challenge"
    "ai2_arc/ARC_Easy",
    "kilt_tasks/hotpotqa",
    "trivia_qa/unfiltered",
    "web_questions",
    "wiki_qa",
    "adversarial_qa/dbidaf",
    "adversarial_qa/dbert",
    "adversarial_qa/droberta",
    "duorc/SelfRC",
    "duorc/ParaphraseRC",
    "ropes",
    "squad_v2",
    "quoref",
    "tydiqa",
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
    "wiki_hop/original",
    "wiqa",
    "piqa",
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "hellaswag",
    "common_gen",
    "wiki_bio",
    "cnn_dailymail/3.0.0",
    "gigaword",
    "multi_news",
    "samsum",
    "xsum",
    "ag_news",
    "dbpedia_14",
    "trec"
]

T0_PLUS_TRAIN_TASK_LIST=[]
for task_name in T0_plus_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_PLUS_TRAIN_TASK_LIST = T0_PLUS_TRAIN_TASK_LIST + sub_list

T0_PLUS_PLUS_TRAIN_TASK_NAME = [
    "super_glue/wsc.fixed",
    "super_glue/record",
    "super_glue/boolq",
    "super_glue/multirc",
    "super_glue/copa",
    "super_glue/wic",
    "glue/mrpc",
    "glue/qqp",
    "paws/labeled_final",
    "ai2_arc/ARC_Challenge"
    "ai2_arc/ARC_Easy",
    "kilt_tasks/hotpotqa",
    "trivia_qa/unfiltered",
    "web_questions",
    "wiki_qa",
    "adversarial_qa/dbidaf",
    "adversarial_qa/dbert",
    "adversarial_qa/droberta",
    "duorc/SelfRC",
    "duorc/ParaphraseRC",
    "ropes",
    "squad_v2",
    "quoref",
    "tydiqa",
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
    "wiki_hop/original",
    "wiqa",
    "piqa",
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "hellaswag",
    "common_gen",
    "wiki_bio",
    "cnn_dailymail/3.0.0",
    "gigaword",
    "multi_news",
    "samsum",
    "xsum",
    "ag_news",
    "dbpedia_14",
    "trec"
]

T0_PLUS_PLUS_TRAIN_TASK_LIST=[]
for task_name in T0_PLUS_PLUS_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_PLUS_PLUS_TRAIN_TASK_LIST = T0_PLUS_PLUS_TRAIN_TASK_LIST + sub_list



class P3Config(datasets.BuilderConfig):
    """BuilderConfig for P3."""

    def __init__(self, splits, features_dict, score_eval, **kwargs):
        """BuilderConfig for P3.

        Args:
          splits: `List[str]`, the lists of splits which are available for this task
          features_dict: `dict`, the dict of features for this task
          score_eval: `bool`, whether this is task formulated as a rank classification problem
          **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 0.1 initial commit
        super(P3Config, self).__init__(version=datasets.Version("0.1.0"), **kwargs)
        self.splits = splits
        self.features_dict = features_dict
        self.score_eval = score_eval


# class P3(datasets.GeneratorBasedBuilder):
#     """Subset of P3 used in `Multitask Prompted Training Enables Zero-Shot Task Generalization`"""

#     BUILDER_CONFIGS = [
#         P3Config(
#             name=task_name,
#             splits=splits_and_features_dict["splits"],
#             features_dict=splits_and_features_dict["features_dict"],
#             score_eval=task_name.endswith("score_eval")
#         )
#         for task_name, splits_and_features_dict in _TASK_SPLITS_AND_FEATURES_DICT.items()
#     ]

#     def _info(self):
#         # All features available are: 'inputs', 'inputs_pretokenized', 'targets',
#         # 'targets_pretokenized', 'idx', 'is_correct', 'weight', and 'answer_choices'
#         _FEAT_MAPPING = {
#             "answer_choices": datasets.Sequence(datasets.Value("string")),
#             "inputs": datasets.Sequence(datasets.Value("int32")),
#             "inputs_pretokenized": datasets.Value("string"),
#             "targets": datasets.Sequence(datasets.Value("int32")),
#             "targets_pretokenized": datasets.Value("string"),
#             "idx": datasets.Sequence(datasets.Value("int32")),
#             "weight": datasets.Value("float32"),
#             "is_correct": datasets.Value("bool"),
#         }

#         features = {}
#         for feat_name in self.config.features_dict.keys():
#             features[feat_name] = _FEAT_MAPPING[feat_name]

#         return datasets.DatasetInfo(
#             description=_DESCRIPTION,
#             features=datasets.Features(features),
#             supervised_keys=None,
#             homepage=_HOMEPAGE,
#             citation=_CITATION,
#             license=_LICENSE,
#         )

#     def _split_generators(self, dl_manager):

#         # data_dir = dl_manager.download_and_extract(_URLs)
#         data_dir = _URLs
#         split_generators = []
#         task_name = self.config.name
#         if "train" in self.config.splits:
#             split_name = "train"
#             split_generators.append(
#                 datasets.SplitGenerator(
#                     name=datasets.Split.TRAIN,
#                     gen_kwargs={
#                         "tfrecord": data_dir[task_name][split_name]["tfrecord"],
#                     }
#                 )
#             )
#         if "validation" in self.config.splits:
#             split_name = "validation"
#             split_generators.append(
#                 datasets.SplitGenerator(
#                     name=datasets.Split.VALIDATION,
#                     gen_kwargs={
#                         "tfrecord": data_dir[task_name][split_name]["tfrecord"],
#                     }
#                 )
#             )
#         if "test" in self.config.splits:
#             split_name = "test"
#             split_generators.append(
#                 datasets.SplitGenerator(
#                     name=datasets.Split.TEST,
#                     gen_kwargs={
#                         "tfrecord": data_dir[task_name][split_name]["tfrecord"],
#                     }
#                 )
#             )
#         # Handle splits that are not train, validation or test
#         special_splits = set(self.config.splits) - set(["train", "validation", "test"])
#         for special_split_name in special_splits:
#             split_generators.append(
#                 datasets.SplitGenerator(
#                     name=datasets.Split(special_split_name),
#                     gen_kwargs={
#                         "tfrecord": data_dir[task_name][special_split_name]["tfrecord"],
#                     }
#                 )
#             )
#         return split_generators


#     def _generate_examples(self, tfrecord):
#         """This function returns the examples in the raw (text) form."""
#         _FEAT_MAPPING_FUNCTIONS = {
#             "answer_choices": lambda x: [choice.decode("utf-8") for choice in x],
#             "inputs": lambda x: x.tolist(),
#             "inputs_pretokenized": lambda x: x.decode("utf-8"),
#             "targets": lambda x: x.tolist(),
#             "targets_pretokenized": lambda x: x.decode("utf-8"),
#             "idx": lambda x: x.tolist(),
#             "weight": lambda x: float(x),
#             "is_correct": lambda x: x,
#         }

#         key = 0
#         features_dict = self.config.features_dict
#         ds = load_cached_task(features_dict, tfrecord)

#         for ex in ds.as_numpy_iterator():
#             ex_dict = {}
#             for feat_name, feat_value in ex.items():
#                 ex_dict[feat_name] = _FEAT_MAPPING_FUNCTIONS[feat_name](feat_value)
#             yield key, ex_dict
#             key += 1

if __name__ == "__main__":
    print("1234")
