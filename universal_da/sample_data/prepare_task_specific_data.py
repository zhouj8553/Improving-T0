template_list = {
    ("glue", "mrpc"):['want to know','paraphrase','equivalent','replace','same thing'],
    ("glue","qqp"):['quora','duplicate or not','same thing','answer','meaning','duplicate'],
    ('paws','labeled_final'):['task_description-no-label','Meaning','context-question-no-label','Rewrite-no-label',
                'context-question','Concatenation','Concatenation-no-label','Meaning-no-label',
                'PAWS-ANLI GPT3','Rewrite','PAWS-ANLI GPT3-no-label'],
    ('wiki_qa',""):['Is This True?','automatic_system','found_on_google','exercise','Decide_good_answer'],
    # ('cos_e','v1.11'):['question_description_option_text','question_description_option_id','question_option_description_text',
    #                                 'description_question_option_id','description_question_option_text','question_option_description_id'],
    ('cosmos_qa',""):['description_context_question_answer_text','description_context_question_answer_id',
                                    'context_description_question_answer_text','no_prompt_id','no_prompt_text','context_description_question_answer_id',
                                    'context_question_description_answer_id','context_question_description_answer_text'],
    ('dream',""):['baseline','read_the_following_conversation_and_answer_the_question'],
    # ('qasc',""):['qa_with_separated_facts_1','qa_with_separated_facts_3','qa_with_separated_facts_4','qa_with_separated_facts_5',
    #                 'qa_with_combined_facts_1','qa_with_separated_facts_2']+['is_correct_1','is_correct_2'],
    ('quail',""):['context_question_answer_description_id','context_question_answer_description_text','description_context_question_answer_id',
                    'context_question_description_answer_text','context_question_description_answer_id','no_prompt_id',
                    'context_description_question_answer_id','no_prompt_text','context_description_question_answer_text','description_context_question_answer_text'],
    ('quarel',""):['do_not_use','logic_test','heres_a_story','choose_between','testing_students'],
    ('quartz',""):['use_info_from_question_paragraph','paragraph_question_plain_concat','use_info_from_paragraph_question','answer_question_based_on',
				'answer_question_below','read_passage_below_choose','having_read_above_passage','given_the_fact_answer_the_q'],
	('sciq',""):['Direct Question'],
	('social_i_qa',""):['Show choices and generate answer','Show choices and generate index'],
	('wiki_hop','original'):['choose_best_object_interrogative_1','choose_best_object_affirmative_1','choose_best_object_affirmative_3',
				'choose_best_object_affirmative_2','choose_best_object_interrogative_2'],
	('amazon_polarity',""):['Is_this_review','User_recommend_this_product','Is_this_product_review_positive','Is_this_review_negative',
				'convey_negative_or_positive_sentiment','negative_or_positive_tone','user_satisfied','would_you_buy','flattering_or_not'],
	('imdb',""):['Movie Expressed Sentiment 2','Reviewer Opinion bad good choices','Sentiment with choices ','Reviewer Sentiment Feeling',
				'Writer Expressed Sentiment','Movie Expressed Sentiment','Text Expressed Sentiment',
				'Reviewer Enjoyment Yes No','Reviewer Expressed Sentiment','Reviewer Enjoyment'],
	('rotten_tomatoes',""):['Reviewer Opinion bad good choices','Text Expressed Sentiment','Sentiment with choices ','Reviewer Enjoyment Yes No','Reviewer Enjoyment','Movie Expressed Sentiment',
				'Writer Expressed Sentiment','Movie Expressed Sentiment 2','Reviewer Expressed Sentiment','Reviewer Sentiment Feeling'],
	('yelp_review_full',""):['so_i_would', 'based_on_that', 'format_star', 'this_place', 'format_score', 'on_a_scale', 'format_rating'],
	('ag_news',""):['classify_question_first', 'classify_with_choices_question_first', 'recommend', 'which_section_choices', 'which_section', 'classify_with_choices', 'classify'],
	('dbpedia_14',""):['given_list_what_category_does_the_paragraph_belong_to', 'pick_one_category_for_the_following_text', 'given_a_choice_of_categories '],
	('trec',""):['what_category_best_describe','pick_the_best_descriptor','fine_grained_open_context_first','which_category_best_describes','trec1','trec2']
}

import os
import datasets
import pdb 
# file_names=['wiqa_which_of_the_following_is_the_supposed_perturbation','wiqa_what_might_be_the_last_step_of_the_process','wiqa_what_might_be_the_first_step_of_the_process','wiqa_what_is_the_missing_first_step','wiqa_what_is_the_final_step_of_the_following_process','wiqa_effect_with_string_answer','wiqa_effect_with_label_answer','wiqa_does_the_supposed_perturbation_have_an_effect']

# task_name='qasc'

multi_data_dir='/share/zongyu/zhoujing/huggingface_datasets/P3'
# multi_data_dir='/sharefs/english/yanan/huggingface_datasets0210/P3'
file_names=os.listdir(multi_data_dir)
# from universal_da.dataset_config import default_T0_tasks

def clean(s): # convert the prompt name into the names in P3
    new_s=''
    for c in s:
        if c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            if len(new_s)==0 or new_s[-1]!='_': new_s+='_'
        else: new_s+=c
    return new_s

task_names=['glue/mrpc', 'glue/qqp', 'paws/labeled_final', 'wiki_qa', 'cos_e/v1.11', 'cosmos_qa', 'dream', 'quail', 'quarel', 'quartz', 'sciq', 'social_i_qa', 'wiki_hop/original', 'amazon_polarity', 'imdb', 'yelp_review_full', 'ag_news', 'dbpedia_14', 'trec']
for task_name in task_names:
    train_examples=[]
    valid_examples=[]
    task_template_name=(task_name.split('/')[0],'') if len(task_name.split('/'))==1 else (task_name.split('/')[0],task_name.split('/')[1])
    for prompt_name in template_list[task_template_name]:
    # for file in file_names:
        file='_'.join(task_name.split('/')+[clean(prompt_name)])
        # if file.startswith(task_name.replace('/','_'))==False: continue
        print(file)
        examples=datasets.load_from_disk(os.path.join(multi_data_dir,file))
        # pdb.set_trace()
        new_train=datasets.Dataset.from_dict({'inputs_pretokenized':examples['train']['inputs_pretokenized'],'targets_pretokenized':examples['train']['targets_pretokenized']})
        if 'validation' in examples:
            new_valid=datasets.Dataset.from_dict({'inputs_pretokenized':examples['validation']['inputs_pretokenized'],'targets_pretokenized':examples['validation']['targets_pretokenized']})
        elif 'test' in examples:
            new_valid=datasets.Dataset.from_dict({'inputs_pretokenized':examples['test']['inputs_pretokenized'],'targets_pretokenized':examples['test']['targets_pretokenized']})
        else:
            new_valid=datasets.Dataset.from_dict({'inputs_pretokenized':examples['train'].select([0,1,2,3,4,5,6,7,8,9])['inputs_pretokenized'],'targets_pretokenized':examples['train'].select([0,1,2,3,4,5,6,7,8,9])['targets_pretokenized']})
        train_examples.append(new_train)
        valid_examples.append(new_valid)
    train_data=datasets.concatenate_datasets(train_examples)
    valid_data=datasets.concatenate_datasets(valid_examples)
    th=500000
    if len(train_data)>th: 
        train_idxs=list(range(len(train_data)))
        select_idxs=np.random.choice(train_idxs,th,False).tolist()
    # else: select_idxs=np.random.choice(train_idxs,th,True).tolist()
        train_examples.append(tmp_train_data.select(select_idxs))
    data=datasets.DatasetDict({'train':train_data,'validation':valid_data})
    import pdb
    pdb.set_trace()
    data.save_to_disk('../zj_spec_single_train_datasets/{}'.format(task_name))
    # data.save_to_disk('/sharefs/english/yanan/zj_spec_single_train_datasets/{}'.format(task_name))
# python -m universal_da.upsample_data.prepare_task_specific_data