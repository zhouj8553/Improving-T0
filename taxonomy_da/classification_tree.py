import os
from collections import defaultdict

# 先实现两个树木的分支
def tree():
    return defaultdict(tree)
clf_tree_fine = tree()
clf_tree_coarse = tree()

#### single-sent
### discriminative
## opposite label
sin_sent_disc_fix_oppo_ls_1 = [
    "imdb_Text_Expressed_Sentiment","imdb_Movie_Expressed_Sentiment",
    "imdb_Movie_Expressed_Sentiment_2","imdb_Reviewer_Enjoyment",
    "imdb_Reviewer_Enjoyment_Yes_No","imdb_Reviewer_Expressed_Sentiment",
    "imdb_Reviewer_Opinion_bad_good_choices","imdb_Reviewer_Sentiment_Feeling",
    "imdb_Writer_Expressed_Sentiment","amazon_polarity_User_recommend_this_product",
    "imdb_Negation_template_for_positive_and_negative",
    "imdb_Sentiment_with_choices_"
]
sin_sent_disc_fix_oppo_ls_2 = [
    "imdb","amazon_polarity","rotten_tomatoes",
]
## parellel label
sin_sent_disc_fix_ls_parel_1 = [
    "dbpedia_14_given_list_what_category_does_the_paragraph_belong_to",
    # "dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to",
    "trec_fine_grained_open","trec_fine_grained_open_context_first",
    "ag_news_classify","ag_news_classify_question_first",
    "ag_news_classify_with_choices","ag_news_classify_with_choices_question_first",
    "ag_news_recommend","ag_news_which_section",
    "ag_news_which_section_choices"
]
sin_sent_disc_fix_ls_parel_2 = [
    "dbpedia_14","trec","ag_news"
]
## rating label
sin_sent_disc_fix_ls_rate_1 = [
    "yelp_review_full_based_on_that","yelp_review_full_format_rating",
    "yelp_review_full_format_score","yelp_review_full_format_star",
    "yelp_review_full_on_a_scale","yelp_review_full_so_i_would",
    "yelp_review_full_this_place","app_reviews_convert_to_rating",
    "app_reviews_convert_to_star_rating",
]
sin_sent_disc_fix_ls_rate_2 = [
    "app_reviews","yelp_review_full",
]
### generative
## creative
# QA
sin_sent_gen_create_qa_1 = [
    "kilt_tasks_hotpotqa_combining_facts","kilt_tasks_hotpotqa_complex_question",
    "kilt_tasks_hotpotqa_final_exam","kilt_tasks_hotpotqa_formulate",
    "kilt_tasks_hotpotqa_straighforward_qa","wiki_qa_Direct_Answer_to_Question",
    "cosmos_qa_only_question_answer","sciq_Direct_Question_Closed_Book_",
]
sin_sent_gen_create_qa_2 = [
    "kilt_tasks/hotpotqa","wiki_qa","cosmos_qa","sciq"
]
# sent -> context ? common_gen
sin_sent_gen_create_s2c_1 = [
    "cnn_dailymail_3.0.0_generate_story","cnn_dailymail_3.0.0_spice_up_story",
    "gigaword_reverse_writing","gigaword_write_an_article",
    "multi_news_expand_reverse_task_","samsum_Write_a_dialogue_that_match_this_summary",
    "common_gen_Example_prompt","common_gen_Given_concepts_type_2",
    "common_gen_Given_concepts_type_1","common_gen_Put_together",
    # "common_gen_choice_in_concept_centric_sentence_generation",
    "common_gen_random_task_template_prompt",
    # "common_gen_topic_to_sentence",
]

sin_sent_gen_create_s2c_2 = [
    "cnn_dailymail/3.0.0","gigaword","multi_news","samsum",
    "common_gen",
]
# context -> sent ? question generation
sin_sent_gen_create_c2s_1 = [
    # "adversarial_qa_dbidaf_generate_question","duorc_SelfRC_generate_question",
    "dream_generate_first_utterance","dream_generate_last_utterance",
    "wiqa_what_is_the_final_step_of_the_following_process",
    # "wiqa_what_is_the_missing_first_step",
    # "wiqa_what_might_be_the_first_step_of_the_process",
    # "wiqa_what_might_be_the_last_step_of_the_process",
]

sin_sent_gen_create_c2s_2 = [
    "adversarial_qa/dbidaf","adversarial_qa/dbert","adversarial_qa/droberta",
    "duorc/SelfRC","duorc/ParaphraseRC","dream","wiqa",
]

## extractive
# context -> sent
sin_sent_gen_extract_c2s_1 = [
    "common_gen_sentence_to_concepts","common_gen_topics_from_the_sentence",
    "cnn_dailymail_3.0.0_2_or_3_sentences","cnn_dailymail_3.0.0_news_card_view",
    "cnn_dailymail_3.0.0_news_stock","cnn_dailymail_3.0.0_news_summary",
    "cnn_dailymail_3.0.0_sum_in_brief","cnn_dailymail_3.0.0_tldr_summary",
    "cnn_dailymail_3.0.0_write_an_outline","gigaword_TLDR",
    "gigaword_first_sentence_title","gigaword_generate_summary_for_this",
    "gigaword_in_a_nutshell","gigaword_make_a_title",
    "gigaword_write_a_title_for_this_sentence","gigaword_write_its_sentence",
    "multi_news_distill","multi_news_summarize",
    "multi_news_summary_scenario","multi_news_synthesize",
    "multi_news_what_are_the_key_points","samsum_Generate_a_summary_for_this_dialogue",
    "samsum_Given_the_above_dialogue_write_a_summary","samsum_Sum_up_the_following_dialogue",
    "samsum_Summarize_this_dialogue_","samsum_Summarize_",
    "samsum_To_sum_up_this_dialog","xsum_DOC_boils_down_to_simple_idea_that",
    "xsum_DOC_given_above_write_one_sentence","xsum_DOC_how_would_you_rephrase_few_words",
    "xsum_DOC_tldr","xsum_DOC_write_summary_of_above",
    "xsum_article_DOC_summary","xsum_college_roommate_asked_DOC_so_I_recap",
    "xsum_read_below_DOC_write_abstract","xsum_summarize_DOC",
    "xsum_summarize_this_DOC_summary",
    # "wiki_qa_Topic_Prediction_Answer_Only","wiki_qa_Topic_Prediction_Question_Only",
    "quoref_Guess_Title_For_Context",
    "duorc_SelfRC_title_generation",
]

sin_sent_gen_extract_c2s_2 = [
    "common_gen","cnn_dailymail/3.0.0","gigaword","multi_news",
    "samsum","xsum","wiki_qa","quoref","duorc/SelfRC","duorc/ParaphraseRC"
]

# sent -> sent
sin_sent_gen_extract_s2s_1 = [
    "glue_mrpc_generate_paraphrase","glue_mrpc_generate_sentence",
    "paws_labeled_final_paraphrase_task",
]
sin_sent_gen_extract_s2s_2 = [
    "glue/mrpc","paws/labeled_final",
]

#### double
### discriminative

## fixed
# oppo
# 0315 fix
double_sent_disc_fix_oppo_1 = [
    # "amazon_polarity_flattering_or_not", 
    # "amazon_polarity_user_satisfied",
    # "amazon_polarity_would_you_buy", "amazon_polarity_Is_this_product_review_positive",
    # "amazon_polarity_Is_this_review", "amazon_polarity_Is_this_review_negative",
    # "amazon_polarity_convey_negative_or_positive_sentiment", "amazon_polarity_negative_or_positive_tone",
    # "wiki_qa_Is_This_True_", 
    # "wiki_qa_automatic_system", "wiki_qa_exercise","wiki_qa_found_on_google", 
    "glue_mrpc_equivalent", "glue_mrpc_paraphrase",
    "glue_mrpc_replace", "glue_mrpc_same_thing", "glue_mrpc_want_to_know",
    "glue_qqp_answer", 
    "glue_qqp_duplicate","glue_qqp_duplicate_or_not","glue_qqp_meaning",
    "glue_qqp_quora", "glue_qqp_same_thing",    
]
double_sent_disc_fix_oppo_2 = [
    "amazon_polarity", "glue/mrpc", "glue/qqp", "wiki_qa"
]
# parallel
double_sent_disc_fix_para_1 = [
    "dbpedia_14_given_a_choice_of_categories_", "dbpedia_14_pick_one_category_for_the_following_text",
    "trec_trec1", "trec_trec2", "trec_pick_the_best_descriptor",
    "trec_what_category_best_describe", "trec_which_category_best_describes"
]

double_sent_disc_fix_para_2 = [
    "dbpedia_14", "trec",
]
## flexible

# self
double_sent_disc_flex_self_1 = [
    "quarel_choose_between", "quarel_do_not_use",
    "quarel_logic_test", "quarel_testing_students",
    "quarel_heres_a_story", "sciq_Multiple_Choice_Closed_Book_",
    "sciq_Multiple_Choice_Question_First","cos_e_v1.11_description_question_option_text",
    "cos_e_v1.11_question_description_option_text","cos_e_v1.11_question_option_description_text",
]
double_sent_disc_flex_self_2 = [
    "quarel", "sciq","cos_e/v1.11",
]
### generative

## extractive
# QA
double_sent_gen_extract_qa_1 = [
    "adversarial_qa_dbidaf_answer_the_following_q", "adversarial_qa_dbidaf_based_on",
    "adversarial_qa_dbidaf_question_context_answer",
    "adversarial_qa_dbidaf_tell_what_it_is",
    "quoref_Answer_Friend_Question", "quoref_Answer_Test",
    "quoref_Context_Contains_Answer", "quoref_Find_Answer",
    "quoref_Found_Context_Online", "quoref_Given_Context_Answer_Question",
    "quoref_Guess_Answer", "quoref_Read_And_Extract_", "quoref_What_Is_The_Answer",
    "ropes_plain_no_background", "ropes_prompt_bottom_no_hint", 
    "quail_context_description_question_text", "quail_context_question_description_text",
    "quail_description_context_question_text",
    "social_i_qa_Generate_answer", "social_i_qa_I_was_wondering",
    "cosmos_qa_context_description_question_text", "cosmos_qa_description_context_question_text",
    "cosmos_qa_description_context_question_text"
]
double_sent_gen_extract_qa_2 = [
    "social_i_qa", "quail", "quoref", "ropes", "adversarial_qa/dbidaf","adversarial_qa/dbert","adversarial_qa/droberta",
]
## creative
double_sent_gen_create_ssc_1 =[
    "app_reviews_generate_review", 
    "duorc_SelfRC_build_story_around_qa",
    "dream_answer_to_dialogue", 
]
double_sent_gen_create_ssc_2 = [
    "app_reviews", "duorc/SelfRC","duorc/ParaphraseRC","dream"
]

double_sent_gen_create_ccs_1 = [
    "wiki_qa_Topic_Prediction_Question_and_Answer_Pair",
    "wiki_qa_Generate_Question_from_Topic", "wiki_qa_Jeopardy_style",
    "social_i_qa_Generate_the_question_from_the_answer", "cosmos_qa_context_answer_to_question",
    "duorc_SelfRC_generate_question_by_answer",
]
double_sent_gen_create_ccs_2 = [
    "duorc/SelfRC","wiki_qa","social_i_qa","cosmos_qa",
]

triple_sent_disc_flex_self_1 = [
    "sciq_Multiple_Choice", 
    "quail_context_question_answer_description_text",
    "quail_context_question_description_answer_text",
    "quail_description_context_question_answer_text", "quail_no_prompt_text",
    "dream_baseline", "dream_read_the_following_conversation_and_answer_the_question",
    "social_i_qa_Show_choices_and_generate_answer",
    "cosmos_qa_context_description_question_answer_text",
    "cosmos_qa_context_question_description_answer_text",
    "cosmos_qa_description_context_question_answer_text",
    "cosmos_qa_no_prompt_text",
    "qasc_qa_with_combined_facts_1",
    # "qasc_qa_with_separated_facts_1", "qasc_qa_with_separated_facts_2",
    # "qasc_qa_with_separated_facts_3", "qasc_qa_with_separated_facts_4",
    # "qasc_qa_with_separated_facts_5",
    # "wiki_hop_original_choose_best_object_affirmative_1",
    # "wiki_hop_original_choose_best_object_affirmative_2",
    # "wiki_hop_original_choose_best_object_affirmative_3",
    # "wiki_hop_original_choose_best_object_interrogative_1",
    # "wiki_hop_original_choose_best_object_interrogative_2",
    "quartz_answer_question_based_on",
    "quartz_answer_question_below", "quartz_given_the_fact_answer_the_q",
    "quartz_having_read_above_passage", "quartz_paragraph_question_plain_concat",
    "quartz_read_passage_below_choose", "quartz_use_info_from_paragraph_question",
    "quartz_use_info_from_question_paragraph"
]
triple_sent_disc_flex_self_2 = [
    "sciq", "quail", "dream", "cosmos_qa", 
    "qasc", "quartz", "social_i_qa"
    # "wiki_hop/original",
]

triple_sent_gen_extract_ccqa_1 = [
    "ropes_background_new_situation_answer",
    "ropes_prompt_beginning", "ropes_read_background_situation",
    "ropes_new_situation_background_answer", "ropes_plain_background_situation",
    "ropes_plain_bottom_hint", "ropes_prompt_beginning",
    "ropes_prompt_bottom_hint_beginning", "ropes_prompt_mix",
    "ropes_read_background_situation",
    "duorc_SelfRC_answer_question", "duorc_SelfRC_decide_worth_it",
    "duorc_SelfRC_movie_director", "duorc_SelfRC_question_answering",
]

triple_sent_gen_extract_ccqa_2 = [
    "ropes", "duorc/SelfRC",
]
clf_tree_fine['triple_sent']['generative']['extract']["ccqa"] = triple_sent_gen_extract_ccqa_1
clf_tree_coarse['triple_sent']['generative']['extract']["ccqa"] = triple_sent_gen_extract_ccqa_2

clf_tree_fine['triple_sent']['discrim']['flexible']["self"] = triple_sent_disc_flex_self_1
clf_tree_coarse['triple_sent']['discrim']['flexible']["self"] = triple_sent_disc_flex_self_2

clf_tree_fine['double_sent']['generative']['creative']["ccs"] = double_sent_gen_create_ccs_1
clf_tree_coarse['double_sent']['generative']['creative']["ccs"] = double_sent_gen_create_ccs_2

clf_tree_fine['double_sent']['generative']['creative']["ssc"] = double_sent_gen_create_ssc_1
clf_tree_coarse['double_sent']['generative']['creative']["ssc"] = double_sent_gen_create_ssc_2

clf_tree_fine['double_sent']['generative']['extractive']["qa"] = double_sent_gen_extract_qa_1
clf_tree_coarse['double_sent']['generative']['extractive']["qa"] = double_sent_gen_extract_qa_2

clf_tree_fine['double_sent']['discrim']['fixed']["parallel"] = double_sent_disc_fix_para_1
clf_tree_coarse['double_sent']['discrim']['fixed']["parallel"] = double_sent_disc_fix_para_2

clf_tree_fine['double_sent']['discrim']['fixed']["oppo"] = double_sent_disc_fix_oppo_1
clf_tree_coarse['double_sent']['discrim']['fixed']["oppo"] = double_sent_disc_fix_oppo_2

clf_tree_fine['double_sent']['discrim']['flexible']["self"] = double_sent_disc_flex_self_1
clf_tree_coarse['double_sent']['discrim']['flexible']["self"] = double_sent_disc_flex_self_2


clf_tree_fine['single_sent']['discrim']['fixed']["oppo"] = sin_sent_disc_fix_oppo_ls_1
clf_tree_coarse['single_sent']['discrim']['fixed']["oppo"] = sin_sent_disc_fix_oppo_ls_2

clf_tree_fine['single_sent']['discrim']['fixed']["parellel"] = sin_sent_disc_fix_ls_parel_1
clf_tree_coarse['single_sent']['discrim']['fixed']["parellel"] = sin_sent_disc_fix_ls_parel_2

clf_tree_fine['single_sent']['discrim']['fixed']["rating"] = sin_sent_disc_fix_ls_rate_1
clf_tree_coarse['single_sent']['discrim']['fixed']["rating"] = sin_sent_disc_fix_ls_rate_2

clf_tree_fine['single_sent']['generative']['creative']["qa"] = sin_sent_gen_create_qa_1
clf_tree_coarse['single_sent']['generative']['creative']["qa"] = sin_sent_gen_create_qa_2

clf_tree_fine['single_sent']['generative']['creative']["s2c"] = sin_sent_gen_create_s2c_1
clf_tree_coarse['single_sent']['generative']['creative']["s2c"] = sin_sent_gen_create_s2c_2

clf_tree_fine['single_sent']['generative']['creative']["c2s"] = sin_sent_gen_create_c2s_1
clf_tree_coarse['single_sent']['generative']['creative']["c2s"] = sin_sent_gen_create_c2s_2

clf_tree_fine['single_sent']['generative']['extractive']["c2s"] = sin_sent_gen_extract_c2s_1
clf_tree_coarse['single_sent']['generative']['extractive']["c2s"] = sin_sent_gen_extract_c2s_2

clf_tree_fine['single_sent']['generative']['extractive']["s2s"] = sin_sent_gen_extract_s2s_1
clf_tree_coarse['single_sent']['generative']['extractive']["s2s"] = sin_sent_gen_extract_s2s_2

print(clf_tree_coarse)
print(clf_tree_fine)

