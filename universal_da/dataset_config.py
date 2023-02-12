default_T0_tasks={
    'paraphrase':['glue/mrpc','glue/qqp','paws/labeled_final'],
    'qa_closed_book':['kilt_tasks/hotpotqa','wiki_qa'],
    'qa_extractive':['adversarial_qa/dbidaf','adversarial_qa/dbert','adversarial_qa/droberta','duorc/SelfRC','duorc/ParaphraseRC','ropes','quoref'],
    'qa_multiple_choice':['cos_e/v1.11','cosmos_qa','dream','qasc','quail','quarel','quartz','sciq','social_i_qa','wiki_hop/original','wiqa'],
    'sentiment':['amazon_polarity','app_reviews','imdb','rotten_tomatoes','yelp_review_full'],
    'stucture_to_text':['common_gen','wiki_bio'],
    'summarization':['cnn_dailymail/3.0.0','gigaword','multi_news','samsum','xsum'],
    'topic_classification':['ag_news','dbpedia_14','trec'],
}
default_T0_test_tasks={
    'sentence_completion':['super_glue/copa','hellaswag'],
    'natural_language_inference':['anli/r1','anli/r2','anli/r3','super_glue/cb','super_glue/rte'],
    'coreference_resolution':['super_glue/wsc.fixed','winogrande/winogrande_xl'],
    'word_sense_disambiguation':['super_glue/wic'],
}

cls_tasks={
    'paraphrase':['glue/mrpc','glue/qqp','paws/labeled_final'],
    'sentiment':['amazon_polarity','app_reviews','imdb','rotten_tomatoes','yelp_review_full'],
    'topic_classification':['ag_news','dbpedia_14','trec'],
}
qa_tasks={
    'qa_closed_book':['kilt_tasks/hotpotqa','wiki_qa'],
    'qa_extractive':['adversarial_qa/dbidaf','adversarial_qa/dbert','adversarial_qa/droberta','coqa','duorc/SelfRC','duorc/ParaphraseRC','ropes','quoref'],
    'qa_multiple_choice':['cos_e/v1.11','cosmos_qa','dream','qasc','quail','quarel','quartz','sciq','social_i_qa','wiki_hop/original','wiqa'],
}
gen_tasks={
    'stucture_to_text':['common_gen','wiki_bio'],
    'summarization':['cnn_dailymail/3.0.0','gigaword','multi_news','samsum','xsum'],
}

task_type_map={'glue/mrpc': 'paraphrase', 'glue/qqp': 'paraphrase', 'paws/labeled_final': 'paraphrase', 
    'kilt_tasks/hotpotqa': 'qa_closed_book', 'wiki_qa': 'qa_closed_book', 'adversarial_qa/dbidaf': 'qa_extractive', 
    'adversarial_qa/dbert': 'qa_extractive', 'adversarial_qa/droberta': 'qa_extractive', 'coqa': 'qa_extractive', 
    'duorc/SelfRC': 'qa_extractive', 'duorc/ParaphraseRC': 'qa_extractive', 'ropes': 'qa_extractive', 'quoref': 'qa_extractive', 
    'cos_e/v1.11': 'qa_multiple_choice', 'cosmos_qa': 'qa_multiple_choice', 'dream': 'qa_multiple_choice', 
    'qasc': 'qa_multiple_choice', 'quail': 'qa_multiple_choice', 'quarel': 'qa_multiple_choice', 
    'quartz': 'qa_multiple_choice', 'sciq': 'qa_multiple_choice', 'social_i_qa': 'qa_multiple_choice', 
    'wiki_hop/original': 'qa_multiple_choice', 'wiqa': 'qa_multiple_choice', 'amazon_polarity': 'sentiment', 
    'app_reviews': 'sentiment', 'imdb': 'sentiment', 'rotten_tomatoes': 'sentiment', 'yelp_review_full': 'sentiment', 
    'common_gen': 'stucture_to_text', 'wiki_bio': 'stucture_to_text', 'cnn_dailymail/3.0.0': 'summarization', 
    'gigaword': 'summarization', 'multi_news': 'summarization', 'samsum': 'summarization', 'xsum': 'summarization', 
    'ag_news': 'topic_classification', 'dbpedia_14': 'topic_classification', 'trec': 'topic_classification',
    'copa': 'sentence_completion', 'hellaswag': 'sentence_completion', 'story_cloze': 'sentence_completion', 
    'anli/r1': 'natural_language_inference', 'anli/r2': 'natural_language_inference', 'anli/r3': 'natural_language_inference', 
    'cb': 'natural_language_inference', 'rte': 'natural_language_inference', 'wsc': 'coreference_resolution', 
    'winogrande/winogrande_xl': 'coreference_resolution', 'wic': 'word_sense_disambiguation'}

registered_task_groups={
    'default_T0_tasks':default_T0_tasks,
    'default_T0_test':default_T0_test_tasks,
    'cls':cls_tasks,
    'gen':gen_tasks,
    'qa':qa_tasks,
}

