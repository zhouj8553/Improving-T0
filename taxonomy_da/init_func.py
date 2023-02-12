from classification_tree import *
from promptsource.templates import DatasetTemplates
from collections import Counter, defaultdict

MAX_SIZE = 200000
def _sample_train_data(task_name, train_split,rng):
    train_number = len(train_split)
    prompt_number = len(DatasetTemplates(task_name).all_template_names)
    if train_number > MAX_SIZE:
        sample_train_number = int(MAX_SIZE / prompt_number)
        sample_train_index_list = rng.sample(range(len(train_split)), k=sample_train_number)
        samples = train_split.select(sample_train_index_list)
        return samples
    else:
        return train_split

def search_branch(query, tree_fine=clf_tree_fine):
    # 输入task_prompt_name 得到对应分支
    # print(tree_fine)
    for c1,t1 in tree_fine.items():
        for c2,t2 in t1.items():
            for c3,t3 in t2.items():
                for c4,tns in t3.items():
                    if query in tns:
                        tax = "_".join([c1,c2,c3,c4]) # record 当前分支
                        break
    return tax



def search_tree(query, tree_fine=clf_tree_fine, tree_coarse=clf_tree_coarse,c1_=None):
    # 输入task_name和树，得到对应的所在的分支
    # print(tree_fine)
    tax2candls = {}
    tax2branch = {}
    keys = []
    for c1,t1 in tree_coarse.items():
        for c2,t2 in t1.items():
            for c3,t3 in t2.items():
                for c4,tns in t3.items():
                    if query in tns:
                        tax = "_".join([c1,c2,c3,c4]) # record 当前分支
                        keys.append((c1,c2,c3,c4))
                        # tns.remove(query)
                        tax2candls[tax] = tns
    if not c1_: # for debug
        for (t1,t2,t3,t4) in keys:
            branch = tree_fine[t1][t2][t3][t4]
            tax = "_".join([t1,t2,t3,t4])
            tax2branch[tax] = branch
    else:
        for (t1,t2,t3,t4) in keys:
            if t1 == c1_:
                branch = tree_fine[t1][t2][t3][t4]
                tax = "_".join([t1,t2,t3,t4])
                tax2branch[tax] = branch
    return tax2branch, tax2candls

# for test
# search_tree("ag_news", clf_tree_fine, clf_tree_coarse)
# search_tree("kilt_tasks/hotpotqa", clf_tree_fine, clf_tree_coarse)
# search_tree("wiki_qa", clf_tree_fine, clf_tree_coarse, c1_="double_sent")

def balance_counter(count_dict,balance_num=0):
    if balance_num:
        balanced_count_dict = {i:balance_num for i,v in count_dict.items()}
    else:
        balance_num = min([v for i,v in count_dict.items()])
        balanced_count_dict = {i:balance_num for i,v in count_dict.items()}
    return balanced_count_dict

def balance_dataset(p3_list, train_ls, test=True, balance_num=0):
    train_ls = [t.replace("/","_") for t in train_ls]
    old_p3_ls = [p for p in p3_list if not "-" in p]
    if test:
        test_p3_ls = [p for p in old_p3_ls if p.endswith("test_prompt")]
    new_p3_ls = [p for p in p3_list if "-" in p]
    print(len(new_p3_ls))
    new_p3_ls = sorted(new_p3_ls)
    cross_tn = set([p[:p.find("-")] for p in new_p3_ls])
    print(f"{len(cross_tn)} out of 38 datasets are crossed!")
    chosen_p3_list = []
    for t in cross_tn:
        cand_p3 = [p for p in new_p3_ls if p.startswith(t)]
        tn2tpn_tup_ls = [p.split("-") for p in cand_p3]
        count_ls = []
        cand_p3_dict = defaultdict(list)
        for tn, tpn in tn2tpn_tup_ls:
            for tt in train_ls:
                if tt in tpn:
                    count_ls.append(tt)
                    cand_p3_dict[tt].append(tn + "-" + tpn)
        a = Counter(count_ls)
        a = balance_counter(a,balance_num)
        for cand_tn, tn_tpns in cand_p3_dict.items():
            sample_num = a[cand_tn]
            chosen_tpns = tn_tpns[:sample_num]
            chosen_p3_list += chosen_tpns
    chosen_p3_list += old_p3_ls
    print(f"Finally we choose {len(chosen_p3_list)} out of {len(p3_list)}")
    return chosen_p3_list

def func1000(source_data):
    revised_data = {}
    revised_data['premise'] = source_data['title']
    revised_data['hypothesis'] = source_data['content']
    return revised_data

def func1001(source_data):
    revised_data = {}
    revised_data['premise'] = source_data['package']
    revised_data['hypothesis'] = source_data['review']
    return revised_data

ag_news_prompts = DatasetTemplates("ag_news")
ag_news_pn = ag_news_prompts.all_template_names[0]
ag_news_prompt = ag_news_prompts[ag_news_pn]


# revise source prompt to unified schema
tpn2branch_revise = {}
# triple_sent_generative_extract_ccqa
def rep_map(jinja_,rep_dict):
    for i,v in rep_dict.items():
        jinja_ = jinja_.replace(i,v)
    return jinja_

# 记录一个 tpn -> branch, revise_dict
# 用处1: 修正prompt字段
# 用处2: 修正raw data字段
# 准则：尽可能取消掉nested schema
# raw data修改的时候如果遇到本身没有的字段，如果是no_answer则判断为有，即让if条件一律成立
def rep_map(template_,rep_dict):
    # 修改jinja 字段
    # 
    jinja_ = template_.jinja
    if "answer_choices" in jinja_:
        pre_choices = template_.answer_choices
        choices_ls_ = '{{answers.join(" ||| ")}}'
        # choices_ls = pre_choices.split(" ||| ")
        # choices_ls_ = ["{{choice"+str(i+1)+"}}" for i in range(len(choices_ls))]
        template_.answer_choices = " ||| ".join(choices_ls_)
    for i,v in rep_dict.items():
        jinja_ = jinja_.replace(i,v)
    template_ = jinja_
    return template_
def revise_raw_data(example, map_dict):
    for src,tgt in map_dict.items():
        if src in example: # check if in key_ls or not
            example[tgt] = example[src]
        else: # 分类讨论
            if src == "no_answer":
                example[tgt] = True
            elif "[" in src:
                sch = src[:src.find("[")]
                idx = int(src[src.find("[")+1:src.find("]")])
                example[tgt] = example[sch][idx]
    return example
# triple_sent_gen_extract_ccqa_1
def f1(example):
    if example['answers']['text']:
        example['context1'] = example['background']
        example['context2'] = example['situation']
        example['answers'] = example['answers']['text']
        example['no_answer'] = False
    else:
        example['context1'] = example['background']
        example['context2'] = example['situation']
        example['answers'] = example['answers']['text']
        example['no_answer'] = True
    return example
dt1 = {
    "jinja":{
        "background":"context1", 
        "situation":"context2",
        "answers.text":"answers", # List
        "question":"question",
        "no_answer":"no_answer",
    },
    "data": f1
}
def f2(example):
    example['context1'] = example['title']
    example['context2'] = example['plot']
    return example
dt2 = {
    "jinja":{
    "{{title}}":"{{context1}}", 
    "{{plot}}":"{{context2}}",
    "answers":"answers", # List
    "question":"question",
    "no_answer":"no_answer",
    },
    "data": f2
}
# triple_sent_discrim_flexible_self_1
# [context, question, answers(List), correct_answer_id]
def f3(example):
    example['context'] = example['support']
    example['answers'] = [example['distractor1'],example['distractor2'],example['distractor3'],example['correct_answer']]
    example['correct_answer_id'] = 3
    return example
dt3 = {
    "jinja":{
        "support":"context", 
        "answer_choices[3]": "answers[correct_answer_id]",
        "answer_choices":"answers",
        "question":"question",
    },
    "data": f3
}
def f4(example):
    return example
dt4 = {
    "jinja":{
        "context":"context", 
        "question":"question",
        "answer_choices":"answers"
    },
    "data": f4
}

def f5(example):
    example['context'] = "\n\n".join(example['dialogue'])
    choices = example['choice']
    example['answers'] = [choices[0],choices[1],choices[2],example['answer']]
    example['correct_answer_id'] = 3
    return example
dt5 = {
    "jinja":{
        'dialogue | join("\\n\\n")':"context", 
        "question":"question",
        "answer_choices":"answers",
        "{{answer}}": "{{answers[correct_answer_id]}}"
    },
    "data": f5
}
def f6(example):
    example['answers'] = [example['answerA'],example['answerB'],example['answerC']]
    example['correct_answer_id'] = int(example['label']) - 1
    return example
dt6 = {
    "jinja":{
        '{{context}}':'{{context}}', 
        "question":"question",
        "answer_choices[label | int - 1]":"answers[correct_answer_id]",
        "answer_choices":"answers",
    },
    "data": f6
}

def f7(example):
    example['answers'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']]
    example['correct_answer_id'] = int(example['label'])
    return example

dt7 = {
    "jinja":{
        'context':"context", 
        "question":"question",
        "answer_choices": "answers",
        "answers[label]":"answers[correct_answer_id]",
    },
    "data": f7
}
def f8(example):
    example['context'] = example['combinedfact']
    correct_id = example['choices']['label'].index(example['answerKey'])
    correct_answer = example['choices']['text'][correct_id]
    answers_without_correct = example['choices']['text']
    answers_without_correct.remove(correct_answer)
    example['answers'] = [correct_answer] + answers_without_correct
    example['correct_answer_id'] = 0
    return example
dt8 = {
    "jinja":{
        'combinedfact':'context', 
        'question':'question',
        '{% for choice in choices.label %} {% if choice == answerKey %}{{ answer_choices[loop.index - 1] }}{% endif %}{% endfor %}' : '{{answers[correct_answer_id]}}',
        'answer_choices[label]':'answers[correct_answer_id]',
        'answer_choices':'answers'
    },
    "data": f8
}

def f9(example):
    example['context'] = example['para']
    example['answers'] = example['choices']['text']
    example['correct_answer_id'] = example['choices']['label'].index(example['answerKey'])
    return example
dt9 = {
    "jinja":{
        'answer_choices[choices.label.index(answerKey)]':'answers[correct_answer_id]',
        'para':'context',
        'answer_choices':'answers',
    },
    "data": f9
}

for tpn in triple_sent_gen_extract_ccqa_1:
    if tpn.startswith("ropes"):
        tpn2branch_revise[tpn] = ("triple_sent_gen_extract_ccqa",dt1)
    elif tpn.startswith("duorc"):
        tpn2branch_revise[tpn] = ("triple_sent_gen_extract_ccqa",dt2)

for tpn in triple_sent_disc_flex_self_1:
    if tpn.startswith("sciq"):
        tpn2branch_revise[tpn] = ("triple_sent_discrim_flexible_self",dt3)
    elif tpn.startswith("quail"):
        tpn2branch_revise[tpn] = ("triple_sent_discrim_flexible_self",dt4)
    elif tpn.startswith("dream"):
        tpn2branch_revise[tpn] = ("triple_sent_discrim_flexible_self",dt5)
    elif tpn.startswith("social_i_qa"):
        tpn2branch_revise[tpn] = ("triple_sent_discrim_flexible_self",dt6)
    elif tpn.startswith("cosmos_qa"):
        tpn2branch_revise[tpn] = ("triple_sent_discrim_flexible_self",dt7)
    elif tpn.startswith("qasc"):
        tpn2branch_revise[tpn] = ("triple_sent_discrim_flexible_self",dt8)
    elif tpn.startswith("quartz"):
        tpn2branch_revise[tpn] = ("triple_sent_discrim_flexible_self",dt9)
# double_sent_gen_create_ccs_1
# ['context1','context2','sent'] + ['no_answer']
def f10(example):
    example['context1'] = example['context']
    example['context2'] = [example['answerA'],example['answerB'],example['answerC']][int(example['label'])-1]
    example['sent'] = example['question']
    example['no_answer'] = False
    return example
dt10 = {
    "jinja":{
        'context':'context1',
        '{"1": answerA, "2": answerB, "3": answerC}[label]':'context2',
        '{{question}}':'{{sent}}',
    },
    "data": f10
}
def f105(example):
    if example['label']:
        example['context1'] = example['question']
        example['context2'] = example['answer']
        example['sent'] = example['document_title']
        example['no_answer'] = False
    else:
        example['context1'] = example['question']
        example['context2'] = example['answer']
        example['sent'] = example['document_title']
        example['no_answer'] = True
    return example
dt105 = {
    "jinja":{
        '{% if label == 1 %}':'{% if no_answer == false %}',
        '{{question}}':'{{context1}}',
        '{{answer}}':'{{context2}}',
        '{{document_title}}':'{{sent}}'
    },
    "data": f105
}
def f11(example):
    example['context1'] = example['context']
    example['context2'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']][int(example['label'])]
    example['sent'] = example['question']
    example['no_answer'] = False
    return example
dt11 = {
    "jinja":{
        '{{context}}':'{{context1}}',
        '{% if label == 0 %}\n{{answer0}}\n{% elif label == 1 %}\n{{answer1}}\n{% elif label == 2 %}\n{{answer2}}\n{% elif label == 3 %}\n{{answer3}}\n{% endif %}':'{{context2}}',
        '{{question}}':'{{sent}}',
    },
    "data": f11
}

def f12(example):
    if not example['no_answer']:
        example['context1'] = example['plot']
        example['context2'] = example['answers'][0] #improve?
        example['sent'] = example['question']
    else:
        example['context1'] = example['plot']
        example['context2'] = "no answer" #improve?
        example['sent'] = example['question']
    return example
dt12 = {
    "jinja":{
        '{{plot}}':'{{context1}}',
        'answers|choice':'context2',
        '{{question}}':'{{sent}}',
    },
    "data": f12
}
# double_sent_generative_creative_ssc
def f13(example):
    example['sent1'] = example['star']
    example['sent2'] = example['package_name']
    example['context'] = example['review']
    return example
dt13 = {
    "jinja":{
        '{{star}}':'{{sent1}}',
        'package_name':'sent2',
        '{{review}}':'{{context}}',
    },
    "data": f13
}
def f14(example):
    if not example['no_answer']:
        example['sent1'] = example['question']
        example['sent2'] = example['answers'][0]
        example['context'] = example['plot']
        example['no_answer'] = False
    else:
        example['sent1'] = example['question']
        example['sent2'] = "no answer"
        example['context'] = example['plot']
        example['no_answer'] = True
    return example
dt14 = {
    "jinja":{
        '{{ question }}':'{{ sent1 }}',
        'answers|choice':'sent2',
        '{{ plot }}':'{{ context }}',
    },
    "data": f14
}
def f15(example):
    example['sent1'] = example['question']
    example['sent2'] = example['answer']
    example['context'] = "\n\n".join(example['dialogue'])
    example['no_answer'] = False
    return example
dt15 = {
    "jinja":{
        '{{question}}':'{{sent1}}',
        '{{answer}}':'{{sent2}}',
        'dialogue | join("\\n\\n")':'context',
    },
    "data": f15
}
# double_sent_gen_extract_qa_1
# ["context","question","answers"(list)]
# TODO: multiple-answer
def f16(example):
    example['answers'] = example['answers']['text'] # improve
    example['no_answer'] = False
    return example
dt16 = {
    "jinja":{
        'answers.text':'answers',
        'metadata.split != "test"':'no_answer == false'
    },
    "data": f16
}
def f17(example):
    example['answers'] = example['answers']['text'] # improve
    example['no_answer'] = False
    return example
dt17 = {
    "jinja":{
        'answers.text':'answers',
    },
    "data": f17
}
def f18(example):
    if example['answers']['text']:
        example['context'] = example['situation']
        example['answers'] = example['answers']['text'] # improve
        example['no_answer'] = False
    else:
        example['context'] = example['situation']
        example['answers'] = example['answers']['text'] # improve
        example['no_answer'] = True
    return example
dt18 = {
    "jinja":{
        'answers.text':'answers',
    },
    "data": f18
}
def f19(example):
    example['answers'] = [example['answers'][example['correct_answer_id']]] # improve
    example['no_answer'] = False
    return example
dt19 = {
    "jinja":{
        'answer_choices[correct_answer_id]':'answers | choice',
    },
    "data": f19
}

def f20(example):
    example['answers'] = list([[example['answerA'],example['answerB'],example['answerC']][int(example['label'])-1]]) # improve
    example['no_answer'] = False
    return example
dt20 = {
    "jinja":{
        'answer_choices[label | int - 1]':'answers | choice',
    },
    "data": f20
}
def f21(example):
    example['answers'] = list([[example['answer0'],example['answer1'],example['answer2'],example['answer3']][int(example['label'])]]) # improve
    example['no_answer'] = False
    return example
dt21 = {
    "jinja":{
        'answer_choices[label]':'answers | choice',
    },
    "data": f21
}
def f22(example):
    answers = [example['answerA'],example['answerB'],example['answerC']]
    example['answers'] = answers[int(example['label'])] # improve
    example['no_answer'] = False
    return example
dt22 = {
    "jinja":{
        'answer_choices[label]':'answers | choice',
    },
    "data": f22
}
for tpn in double_sent_gen_create_ccs_1:
    if tpn.startswith("social_i_qa"):
        tpn2branch_revise[tpn] = ("double_sent_generative_creative_ccs",dt10)
    elif tpn.startswith("cosmos_qa"):
        tpn2branch_revise[tpn] = ("double_sent_generative_creative_ccs",dt11)
    elif tpn.startswith("duorc"):
        tpn2branch_revise[tpn] = ("double_sent_generative_creative_ccs",dt12)
    elif tpn.startswith("wiki_qa"):
        tpn2branch_revise[tpn] = ("double_sent_generative_creative_ccs",dt105)
for tpn in double_sent_gen_create_ssc_1:
    if tpn.startswith("app_reviews"):
        tpn2branch_revise[tpn] = ("double_sent_generative_creative_ssc",dt13)
    elif tpn.startswith("duorc"):
        tpn2branch_revise[tpn] = ("double_sent_generative_creative_ssc",dt14)
    elif tpn.startswith("dream"):
        tpn2branch_revise[tpn] = ("double_sent_generative_creative_ssc",dt15)

for tpn in double_sent_gen_extract_qa_1:
    if tpn.startswith("adversarial_qa"):
        tpn2branch_revise[tpn] = ("double_sent_generative_extractive_ssc",dt16)
    elif tpn.startswith("quoref"):
        tpn2branch_revise[tpn] = ("double_sent_generative_extractive_ssc",dt17)
    elif tpn.startswith("ropes"):
        tpn2branch_revise[tpn] = ("double_sent_generative_extractive_ssc",dt18)
    elif tpn.startswith("quail"):
        tpn2branch_revise[tpn] = ("double_sent_generative_extractive_ssc",dt19)
    elif tpn.startswith("social_i_qa"):
        tpn2branch_revise[tpn] = ("double_sent_generative_extractive_ssc",dt20)
    elif tpn.startswith("cosmos_qa"):
        tpn2branch_revise[tpn] = ("double_sent_generative_extractive_ssc",dt21)

# double_sent_discrim_fixed_parallel
# ["text","choices['label']"]
db_choices = ["Company", "Educational Institution", "Artist", "Athlete", "Office Holder", 
"Mean Of Transportation", "Building", "Natural Place", "Village", "Animal", "Plant", "Album", "Film", "Written Work"]
trec_choices = ["Description","Entity","Abbreviation","Person","Quantity","Location"]
def f23(example):
    example['text'] = example['title'] + " - " + example['content']
    example['choices'] = db_choices
    example['label'] = example['label']
    return example
dt23 = {
    "jinja":{
        '{{title}} - {{content}}':'{{text}}',
        '{{"company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work"}}':'{{choices}}',
        'answer_choices[label]':'choices[label]',
    },
    "data": f23
}
def f24(example):
    example['choices'] = trec_choices
    example['label'] = example['label-coarse']
    return example
dt24 = {
    "jinja":{
        'answer_choices':'choices',
        'label_coarse':'label',
    },
    "data": f24
}
for tpn in double_sent_disc_fix_para_1:
    if tpn.startswith("dbpedia_14"):
        tpn2branch_revise[tpn] = ("double_sent_discrim_fixed_parallel",dt23)
    elif tpn.startswith("trec"):
        tpn2branch_revise[tpn] = ("double_sent_discrim_fixed_parallel",dt24)

# double_sent_discrim_fixed_oppo
# ["context","text","label"] + "choices"
amazon_choices = ["Negative","Positive"]
def f25(example):
    example['choices'] = amazon_choices
    example['label'] = example['label']
    example['text'] = example['content'] 
    example['context'] = example['title']
    return example
dt25 = {
    "jinja":{
        'answer_choices':'choices',
        'title':'context',
        'content':'text',
    },
    "data": f25
}
wiki_qa_choices = ["No","Yes"]
def f26(example):
    example['choices'] = wiki_qa_choices
    example['label'] = example['label']
    example['text'] = example['answer'] 
    example['context'] = example['question']
    return example
dt26 = {
    "jinja":{
        'answer_choices':'choices',
        '{{question}}':'{{context}}',
        '{{answer}}':'{{text}}',
    },
    "data": f26
}
para_choices = ['not equivalent','equivalent']
def f27(example):
    example['choices'] = para_choices
    example['label'] = example['label']
    example['text'] = example['sentence2'] 
    example['context'] = example['sentence1']
    return example
dt27 = {
    "jinja":{
        'answer_choices':'choices',
        '{{sentence1}}':'{{context}}',
        '{{sentence2}}':'{{text}}',
    },
    "data": f27
}
def f28(example):
    example['choices'] = para_choices
    example['label'] = example['label']
    example['text'] = example['question2'] 
    example['context'] = example['question1']
    return example
dt28 = {
    "jinja":{
        'answer_choices':'choices',
        '{{question1}}':'{{context}}',
        '{{question2}}':'{{text}}',
    },
    "data": f28
}
for tpn in double_sent_disc_fix_oppo_1:
    if tpn.startswith("amazon_polarity"):
        tpn2branch_revise[tpn] = ("double_sent_discrim_fixed_oppo",dt25)
    elif tpn.startswith("wiki_qa"):
        tpn2branch_revise[tpn] = ("double_sent_discrim_fixed_oppo",dt26)
    elif tpn.startswith("glue_mrpc"):
        tpn2branch_revise[tpn] = ("double_sent_discrim_fixed_oppo",dt27)
    elif tpn.startswith("glue_qqp"):
        tpn2branch_revise[tpn] = ("double_sent_discrim_fixed_oppo",dt28)

# double_sent_discrim_flexible_self
# ["question","choices","answer"]
def f29(example):
    example['choices'] = [example['world_literals']['world1'][0],example['world_literals']['world2'][0]]
    example['question'] = example['question']
    example['answer'] = example['choices'][example['answer_index']]
    return example
dt29 = {
    "jinja":{
        'answer_choices[answer_index]':'answer',
        'answer_choices':'choices',
    },
    "data": f29
}

def f30(example):
    example['choices'] = [example['correct_answer'],example['distractor1'],example['distractor2'],example['distractor3']] # 0315 move correct answer to the front to avoid no answer in the choices of some prompts
    example['question'] = example['question']
    example['answer'] = example['correct_answer']
    return example
dt30 = {
    "jinja":{
        'answer_choices[3]':'answer',
        'answer_choices':'choices',
    },
    "data": f30
}
def f31(example):
    return example
dt31 = {
    "jinja":{
        'answer_choices':'choices',
    },
    "data": f31
}
for tpn in double_sent_disc_flex_self_1:
    if tpn.startswith("quarel"):
        tpn2branch_revise[tpn] = ("double_sent_discrim_flexible_self",dt29)
    elif tpn.startswith("sciq"):
        tpn2branch_revise[tpn] = ("double_sent_discrim_flexible_self",dt30)
    elif tpn.startswith("cos_e_v1.11"):
        tpn2branch_revise[tpn] = ("double_sent_discrim_flexible_self",dt31)
# single_sent_discrim_fixed_oppo
# ['text','choices[label]']
imdb_choices = ["Negative","Positive"]
def f32(example):
    example['choices'] = imdb_choices
    return example
dt32 = {
    "jinja":{
        'answer_choices':'choices',
    },
    "data": f32
}
amazon_choices = ["Negative","Positive"]
def f33(example):
    example['choices'] = amazon_choices
    example['text'] = example['content']
    return example
dt33 = {
    "jinja":{
        'answer_choices':'choices',
        '{{content}}':'{{text}}',
    },
    "data": f33
}

def f34(example):
    example['choices'] = db_choices
    example['text'] = example['content']
    return example
dt34 = {
    "jinja":{
        '{{"company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work"}}':'{{choices}}',
        'answer_choices':'choices',
        '{{content}}':'{{text}}',
    },
    "data": f34
}

def f35(example):
    example['choices'] = trec_choices
    example['label'] = example['label-coarse'] # improve
    return example
dt35 = {
    "jinja":{
        'answer_choices':'choices',
        'label_fine':'label'
    },
    "data": f35
}
ag_choices = ["World politics","Sports","Business","Science and technology"]
def f36(example):
    example['choices'] = ag_choices
    return example
dt36 = {
    "jinja":{
        'answer_choices':'choices',
    },
    "data": f36
}


# single_sent_generative_creative_qa
# ["question","answer"] + "no_answer"?
def f37(example):
    example['question'] = example['input']
    example['answer'] = example['output'][0]['answer']
    return example
dt37 = {
    "jinja":{
        'output | map(attribute="answer") | list | choice':'answer',
        'output':'answer',
        'input':'question'
    },
    "data": f37
}
def f38(example):
    if example['label']:
        example['answer'] = example['answer']
    else:
        example['answer'] = ""
    return example
dt38 = {
    "jinja":{
        'label == 1':'answer',
    },
    "data": f38
}
def f39(example):
    example['answer'] = [example['answer0'],example['answer1'],example['answer2'],example['answer3']][int(example['label'])]
    return example
dt39 = {
    "jinja":{
        'answer_choices[label]':'answer',
    },
    "data": f39
}
def f40(example):
    example['answer'] = example['correct_answer']
    return example
dt40 = {
    "jinja":{
        'answer_choices[3]':'answer',
    },
    "data": f40
}
for tpn in sin_sent_disc_fix_oppo_ls_1:
    if tpn.startswith("imdb"):
        tpn2branch_revise[tpn] = ("single_sent_discrim_fixed_oppo",dt32)
    elif tpn.startswith("amazon_polarity"):
        tpn2branch_revise[tpn] = ("single_sent_discrim_fixed_oppo",dt33)
    elif tpn.startswith("rotten_tomatoes"):
        tpn2branch_revise[tpn] = ("single_sent_discrim_fixed_oppo",dt32)
for tpn in sin_sent_disc_fix_ls_parel_1:
    if tpn.startswith("dbpedia_14"):
        tpn2branch_revise[tpn] = ("single_sent_discrim_fixed_parellel",dt34)
    elif tpn.startswith("trec"):
        tpn2branch_revise[tpn] = ("single_sent_discrim_fixed_parellel",dt35)
    elif tpn.startswith("ag_news"):
        tpn2branch_revise[tpn] = ("single_sent_discrim_fixed_parellel",dt36)

for tpn in sin_sent_gen_create_qa_1:
    if tpn.startswith("kilt_tasks"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_qa",dt37)
    elif tpn.startswith("wiki_qa"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_qa",dt38)
    elif tpn.startswith("cosmos_qa"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_qa",dt39)
    elif tpn.startswith("sciq"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_qa",dt40)

# sin_sent_gen_create_s2c_1
# ['sent','context']
def f41(example):
    example['context'] = example['article']
    example['sent'] = example['highlights']
    return example
dt41 = {
    "jinja":{
        'highlights':'sent',
        'article':'context'
    },
    "data": f41
}
def f42(example):
    example['context'] = example['document']
    example['sent'] = example['summary']
    return example
dt42 = {
    "jinja":{
        'summary':'sent',
        'document':'context'
    },
    "data": f42
}
def f43(example):
    example['context'] = example['document']
    example['sent'] = example['summary']
    return example
dt43 = {
    "jinja":{
        'summary':'sent',
        'document':'context'
    },
    "data": f43
} # ?
def f44(example):
    example['context'] = example['dialogue']
    example['sent'] = example['summary']
    return example
dt44 = {
    "jinja":{
        '{{summary}}':'{{sent}}',
        '{{dialogue}}':'{{context}}'
    },
    "data": f44
}
def f45(example):
    example['context'] = example['target']
    example['sent'] = ", ".join(example['concepts'])
    return example
dt45 = {
    "jinja":{
        'concepts | join(", ")':'sent',
        'target':'context'
    },
    "data": f45
}

# single_sent_generative_creative_c2s
def f46(example):
    example['context'] = example['context']
    example['sent'] = example['question']
    return example
dt46 = {
    "jinja":{
        '{{question}}':'{{sent}}',
    },
    "data": f46
}

def f47(example):
    example['context'] = example['plot']
    example['sent'] = example['question']
    return example
dt47 = {
    "jinja":{
        '{{ question }}':'{{ sent }}',
        '{{ plot }}':'{{ context }}',
    },
    "data": f47
}

def f48(example):
    example['context'] = "\n\n".join(example['dialogue'][1:])
    example['sent'] = example['dialogue'][0]
    return example
dt48 = {
    "jinja":{
        'dialogue[1:] | join("\\n\\n")':'context',
        'dialogue[0]':'sent',
    },
    "data": f48
}

def f49(example):
    example['context'] = "\n\n".join(example['dialogue'][:-1])
    example['sent'] = example['dialogue'][-1]
    return example
dt49 = {
    "jinja":{
        'dialogue[:-1] | join("\\n\\n")':'context',
        'dialogue[-1]':'sent',
    },
    "data": f49
}

def f50(example):
    if example['question_para_step'][-1]:
        example['context'] = "\n- ".join(example['question_para_step'][:-1])
        example['sent'] = example['question_para_step'][-1]
    else:
        example['question_para_step'][-1] = "finished"
        example['context'] = "\n- ".join(example['question_para_step'][:-1])
        example['sent'] = example['question_para_step'][-1]
    return example
dt50 = {
    "jinja":{
        '{% set process_list = question_para_step[:-1] if question_para_step[-1] == "" else question_para_step %}':'',
        'process_list[:-1] | join("\\n- ")':'context',
        'process_list | last':'sent'
    },
    "data": f50
}

for tpn in sin_sent_gen_create_s2c_1:
    if tpn.startswith("cnn"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_s2c",dt41)
    elif tpn.startswith("gigaword"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_s2c",dt42)
    # elif tpn.startswith("multi_news"):
    #     tpn2branch_revise[tpn] = ("single_sent_generative_creative_s2c",dt43)
    elif tpn.startswith("samsum"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_s2c",dt44)
    elif tpn.startswith("common_gen"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_s2c",dt45)

for tpn in sin_sent_gen_create_c2s_1:
    if tpn.startswith("adversarial"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_c2s",dt46)
    elif tpn.startswith("duorc"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_c2s",dt47)
    elif tpn.startswith("dream") and "first" in tpn:
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_c2s",dt48)
    elif tpn.startswith("dream") and "last" in tpn:
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_c2s",dt49)
    elif tpn.startswith("wiqa"):
        tpn2branch_revise[tpn] = ("single_sent_generative_creative_c2s",dt50)

# single_sent_generative_extractive_c2s
# ['context','sent']
def f51(example):
    example['context'] = example['target']
    example['sent'] = ", ".join(example['concepts'])
    return example
dt51 = {
    "jinja":{
        'concepts | join(", ")':'sent',
        'target':'context'
    },
    "data": f51
}
def f52(example):
    example['context'] = example['article']
    example['sent'] = example['highlights']
    return example
dt52 = {
    "jinja":{
        '{{article}}':'{{context}}',
        '{{highlights}}':'{{sent}}'
    },
    "data": f52
}
def f53(example):
    example['context'] = example['document']
    example['sent'] = example['summary']
    return example
dt53 = {
    "jinja":{
        '{{document}}':'{{context}}',
        '{{summary}}':'{{sent}}'
    },
    "data": f53
}
def f54(example):
    example['context'] = example['document']
    example['sent'] = example['summary']
    return example
dt54 = {
    "jinja":{
        '{{document}}':'{{context}}',
        '{{summary}}':'{{sent}}'
    },
    "data": f54
}
def f55(example):
    example['context'] = example['dialogue']
    example['sent'] = example['summary']
    return example
dt55 = {
    "jinja":{
        '{{dialogue}}':'{{context}}',
        '{{summary}}':'{{sent}}'
    },
    "data": f55
}
def print_test():
    print("Yesyesyesyes!")
    return 0
def f56(example):
    example['context'] = example['answer']
    example['sent'] = example['document_title']
    # if example['label']:
    #     example['context'] = example['answer']
    #     example['sent'] = example['document_title']
    # else:
    #     example['context'] = example['answer'] # 0315 revise, wrong answer also makes sense
    #     example['sent'] = example['document_title']        
    return example
dt56 = {
    "jinja":{
        'label == 1':'context',
        '{{answer}}':'{{context}}',
        '{{document_title}}':'{{sent}}'
    },
    "data": f56
}
def f57(example):
    example['context'] = example['context']
    example['sent'] = example['title']
    return example
dt57 = {
    "jinja":{
        '{{title}}':'{{sent}}'
    },
    "data": f57
}
def f58(example):
    example['context'] = example['plot']
    example['sent'] = example['title']
    return example
dt58 = {
    "jinja":{
        '{{plot}}':'{{context}}',
        '{{title}}':'{{sent}}'
    },
    "data": f58
}
for tpn in sin_sent_gen_extract_c2s_1:
    if tpn.startswith("common_gen"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_c2s",dt51)
    elif tpn.startswith("cnn"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_c2s",dt52)
    elif tpn.startswith("gigaword"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_c2s",dt53)
    # elif tpn.startswith("multi_news"):
    #     tpn2branch_revise[tpn] = ("single_sent_generative_extractive_c2s",dt49)
    elif tpn.startswith("samsum"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_c2s",dt55)
    elif tpn.startswith("xsum"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_c2s",dt53)
    elif tpn.startswith("wiki_qa"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_c2s",dt56)
    elif tpn.startswith("quoref"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_c2s",dt57)
    elif tpn.startswith("duorc"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_c2s",dt58)

# single_sent_generative_extractive_s2s
# ['sent1','sent2']
def f59(example):
    if example['label']:
        example['sent1'] = example['sentence1']
        example['sent2'] = example['sentence2']
    else:
        example['sent1'] = example['sentence1']
        example['sent2'] = ""
    return example
dt59 = {
    "jinja":{
        'label == 1':'sent2',
        '{{sentence1}}':'{{sent1}}',
        '{{sentence2}}':'{{sent2}}'
    },
    "data": f59
}
for tpn in sin_sent_gen_extract_s2s_1:
    if tpn.startswith("glue_mrpc"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_s2s",dt59)
    elif tpn.startswith("paws"):
        tpn2branch_revise[tpn] = ("single_sent_generative_extractive_s2s",dt59)

# rating先跳过

# revise source data to target template/prompt
pair2func = {}

def func0(source_data):
    return source_data
def func1(src): # 改prompt
    
    src['content'] = src['text']
    src['answer_choice'] = ag_news_prompt.answer_choices
    return src
def func2(src): # 改prompt
    
    src['label_fine'] = src['label']
    src['answer_choice_'] = ag_news_prompt.answer_choices
    return src
def func3(src):
    
    src['label_fine'] = src['label']
    src['text'] = src['content']
    src['answer_choice_'] = 1
    return src
def func4(src):
    
    src['text'] = src['content']
    src['answer_choice_'] = 1
    return src
def func5(src):
    
    src['context'] = "\n".join(src['dialogue'])
    src['question'] = src['question']
    return src
def func6(src):
    
    src['plot'] = "\n".join(src['dialogue'])
    src['question'] = src['question']
    return src
def func7(src):
    
    src['question_para_step'] = src['dialogue']
    return src
def func8(src):
    
    src['content'] = src['text']
    src['label'] = src['label-coarse']
    src['answer_choice_'] = 1
    return src
def func9(src):
    
    src['text'] = src['text']
    src['label'] = src['label-coarse']
    src['answer_choice_'] = 1
    return src
def func10(src):
    
    src['summary'] = src['highlights']
    src['document'] = src['article']
    return src
def func11(src): # cnn->multi_news ?
    
    src['summary'] = src['highlights']
    src['document'] = src['article']
    return src
def func12(src): # cnn->samsum
    
    src['summary'] = src['highlights']
    src['dialogue'] = src['article']
    return src
def func13(src): # cnn->common_gen ?? 牵强
    
    src['concepts'] = src['highlights'].split(",") # no exception
    src['target'] = src['article']
    return src
def func14(src): # cnn->xsum
    
    src['summary'] = src['highlights']
    src['document'] = src['article']
    return src
def func15(src): # cnn->wiki_qa
    
    src['document_title'] = src['highlights']
    src['question'] = src['article']
    return src
def func16(src): # cnn->duorc_SelfRC
    
    src['title'] = src['highlights']
    src['plot'] = src['article']
    return src
def func17(src): # cnn->quoref
    
    src['title'] = src['highlights']
    src['context'] = src['article']
    return src
def func18(src): # giga->cnn
    
    src['highlights'] = src['summary']
    src['article'] = src['document']
    return src
def func19(src): # giga->quoref
    
    src['title'] = src['summary']
    src['context'] = src['document']
    return src
def func20(src): # giga->multi_news
    
    src['summary'] = src['summary']
    src['document'] = src['document']
    return src
def func21(src): # giga->samsum
    
    src['summary'] = src['summary']
    src['dialogue'] = src['document']
    return src
def func22(src): # giga->common_gen
    
    src['concepts'] = src['summary'].split(",")
    src['target'] = src['document']
    return src
def func23(src): # giga->xsum
    
    src['summary'] = src['summary']
    src['document'] = src['document']
    return src
def func24(src): # giga->duorc_SelfRC
    
    src['title'] = src['summary']
    src['plot'] = src['document']
    return src
def func25(src): # giga->wiki_qa
    
    src['document_title'] = src['summary']
    src['question'] = src['document']
    return src
def func26(src): # giga->wiki_qa
    
    src['document_title'] = src['summary']
    src['question'] = src['document']
    return src
def func27(src): 
    
    src['text'] = src['content']
    src['label'] = src['label']
    src['answer_choice_'] = 1
    return src
def func28(src):
    
    src['text'] = src['review']
    src['label'] = src['star']
    src['answer_choice_'] = 1
    return src
def func29(src):
    
    src['content'] = src['text']
    src['label'] = src['label']
    src['answer_choice_'] = 1
    return src
def func30(src):
    
    src['question'] = src['question']
    src['plot'] = src['context']
    return src
def func31(src): # ? transfer adversarial_qa/dbidaf to prompt dream_generate_first_utterance
    
    src['dialogue'] = [src['question'], src['context']] # 没有answer?
    return src
def func32(src): # 
    
    src['question_para_step'] = [src['question'], src['context']] # 没有answer?
    return src
def func33(src): # 
    
    src['context'] = "\n".join(src['question_para_step'])
    src['question'] = src['question_stem']
    return src
def func34(src): # 
    
    src['plot'] = "\n".join(src['question_para_step'])
    src['question'] = src['question_stem'] 
    return src
def func35(src): # 
    
    src['dialogue'] = src['question_para_step']
    src['question'] = src['question_stem'] 
    return src
def func36(src): # kilt->wiki_qa huggingface datasets issue!
    src['question'] = src['input']
    if len(src['output'])==0:
        src['answer'] = "no"
        src['label'] = 0
    else:
        src['answers'] = src['output'][0]['answer']
        src['answer'] = src['answers']
        src['label'] = 1  # 1
    return src
def func37(src): 
    src['question'] = src['sentence1']
    src['answer'] = src['sentence2']
    src['label'] = src['label']
    return src
def func38(src): 
    src['question1'] = src['sentence1']
    src['question2'] = src['sentence2']
    src['label'] = src['label']
    return src
def func39(src): 
    src['question'] = src['question1']
    src['answer'] = src['question2']
    src['label'] = src['label']
    return src
def func40(src): 
    src['sentence1'] = src['question1']
    src['sentence2'] = src['question2']
    src['label'] = src['label']
    return src
def func41(src): 
    src['text'] = src['content']
    src['label-coarse'] = src['label']
    src['answer_choice_'] = 1
    return src
def func42(src): 
    src['plot'] = src['dialogue']
    src['answers'] = [src['answer']]
    src['question'] = src['question']
    src['no_answer'] = False
    return src
def func43(src): 
    src['sentence1'] = src['content']
    src['sentence2'] = [src['title']]
    src['label'] = 1
    return src
def func44(src): 
    src['sentence1'] = src['question']
    src['sentence2'] = src['answer']
    src['label'] = src['label']
    return src
def func45(src): 
    src['question1'] = src['question']
    src['question2'] = src['answer']
    src['label'] = src['label']
    return src
def func46(src): 
    src['context'] = src['context']
    src['question'] = src['question']
    # src['answers'] = {}
    # src['answers']['text'] = src['answers']['text']
    src['metadata'] = {}
    src['metadata']['split'] = "train"
    return src
def func47(src): 
    src['situation'] = src['context']
    src['question'] = src['question']
    # src['answers'] = {}
    # src['answers']['text'] = src['answers']['text']
    return src
def func48(src): 
    src['context'] = src['context']
    src['question'] = src['question']
    src['answers'] = src['answers']['text'] # only one
    src['correct_answer_id'] = 0
    return src
def func49(src): 
    src['context'] = src['context']
    src['question'] = src['question']
    src['label'] = 0 # only one
    src['answer0'] = src['answers']['text'][0]
    src['answer1'] = 0
    src['answer2'] = 0
    src['answer3'] = 0
    return src
def func50(src): 
    src['context'] = src['background']
    src['question'] = src['question']
    src['metadata'] = {}
    src['metadata']['split'] = "train"
    return src
def func51(src): 
    src['context'] = src['background']
    src['question'] = src['question']
    return src
def func52(src): 
    src['context'] = src['background']
    src['question'] = src['question']
    src['answers'] = src['answers']['text']
    src['correct_answer_id'] = 0
    return src
def func53(src): 
    src['context'] = src['background']
    src['question'] = src['question']
    src['label'] = 0
    src['answer0'] = src['answers']['text'][0]
    src['answer1'] = 0
    src['answer2'] = 0
    src['answer3'] = 0
    return src
def func54(src): 
    src['question'] = src['question']
    src['dialogue'] = src['plot']
    if src['no_answer']:
        src['answer'] = "no answer"
    else:
        src['answer'] = src['answers'][0]
    return src
def func55(src): 
    src['question'] = src['question']
    src['world_literals'] = {}
    src['world_literals']['world1'] = [src['correct_answer']]
    src['world_literals']['world2'] = [src['distractor1']]
    src['answer_index'] = 0
    return src
def func56(src): 
    src['question'] = src['question']
    answers = [src['world_literals']['world1'],src['world_literals']['world2']]
    src['distractor1'] = src['world_literals']['world1']
    src['distractor2'] = src['world_literals']['world2']
    src['distractor3'] = "no answer"
    src['correct_answer'] = answers[src['answer_index']]
    return src
def func57(src): 
    src['question'] = src['question']
    answers = [src['world_literals']['world1'],src['world_literals']['world2']]
    src['distractor1'] = src['world_literals']['world1']
    src['distractor2'] = src['world_literals']['world2']
    src['distractor3'] = "no answer"
    src['correct_answer'] = answers[src['answer_index']]
    return src
def func58(src): 
    src['question'] = src['question']
    src['context'] = src['context']
    src['metadata'] = {}
    src['metadata']['split'] = "train"
    answers = [src['answerA'],src['answerB'],src['answerC']]
    src['answers'] = {}
    src['answers']['text'] = [answers[int(src['label'])-1]]
    return src
def func59(src): 
    src['question'] = src['question']
    src['context'] = src['context']
    answers = [src['answerA'],src['answerB'],src['answerC']]
    src['answers'] = {}
    src['answers']['text'] = [answers[int(src['label'])-1]]
    return src
def func60(src): 
    src['question'] = src['question']
    src['situation'] = src['context']
    answers = [src['answerA'],src['answerB'],src['answerC']]
    src['answers'] = {}
    src['answers']['text'] = [answers[int(src['label'])-1]]
    return src
def func61(src): 
    src['question'] = src['question']
    src['context'] = src['context']
    answers = [src['answerA'],src['answerB'],src['answerC']]
    src['answers'] = answers
    src['correct_answer_id'] = int(src['label']) - 1
    return src
def func62(src): 
    src['question'] = src['question']
    src['context'] = src['context']
    answers = [src['answerA'],src['answerB'],src['answerC'],"no answer"]
    src['answer0'] = answers[0]
    src['answer1'] = answers[1]
    src['answer2'] = answers[2]
    src['answer3'] = answers[3]
    src['label'] = int(src['label']) - 1
    return src
def func63(src): 
    src['question'] = src['question']
    src['context'] = src['context']
    src['metadata'] = {}
    src['metadata']['split'] = "train"
    hh = [src['answers'][src['correct_answer_id']]]
    src['answers'] = {}
    src['answers']['text'] = hh
    return src
def func64(src): 
    src['question'] = src['question']
    src['context'] = src['context']
    answers = src['answers']
    src['answer0'] = answers[0]
    src['answer1'] = answers[1]
    src['answer2'] = answers[2]
    src['answer3'] = answers[3]
    src['label'] = src['correct_answer_id']
    return src
def func65(src): # cos_e -> quarel
    src['question'] = src['question']
    choices_without_correct = src['choices']
    choices_without_correct.remove(src['answer'])
    src['world_literals'] = {}
    src['world_literals']['world1'] = [src['answer']]
    src['world_literals']['world2'] = [choices_without_correct[0]]
    src['answer_index'] = 0
    return src
def func66(src): 
    src['question'] = src['question']
    choices_without_correct = src['choices']
    choices_without_correct.remove(src['answer'])
    src['distractor1'] = choices_without_correct[0]
    src['distractor2'] = choices_without_correct[1]
    src['distractor3'] = choices_without_correct[2]
    src['correct_answer'] = src['answer']
    return src

def fc1(src):
    if not src['label']:
        src['choice1'] = "no answer"
        src['choice2'] = src['sentence2']
        src['premise'] = src['sentence1']
        src['question'] = "effect"
        src['label'] == -1
        return src
    else:
        src['choice1'] = "no answer" # random sampling is better?
        src['choice2'] = src['sentence2']
        src['premise'] = src['sentence1'] # 改变了prompt本身含义了？
        src['question'] = "effect"
        src['label'] = 1
        return src
def fc2(src):
    if not src['label']:
        src['choice1'] = src['question1']
        src['choice2'] = src['question2']
        src['premise'] = src['question1']
        src['question'] = "effect"
        src['label'] == -1
        return src
    else:
        src['choice1'] = src['question1']
        src['choice2'] = src['question2']
        src['premise'] = src['question1'] # 改变了prompt本身含义了？
        src['question'] = "effect"
        src['label'] = 1
        return src
def fc3(src):
    src['choice1'] = src["world_literals"]["world1"][0]
    src['choice2'] = src["world_literals"]["world2"][0]
    src['premise'] = src['question'] # 改变了prompt本身含义了？
    src['question'] = "effect"
    src['label'] = src["answer_index"]
    return src
def fc4(src):
    src['choice1'] = src["distractor1"]
    src['choice2'] = src["correct_answer"]
    src['premise'] = src['support'] + "\n" + src['question'] # 改变了prompt本身含义了？
    src['question'] = "effect"
    src['label'] = 1
    return src
def fc5(src):
    src['choice1'] = "no_answer"
    src['choice2'] = src["answer"]
    src['premise'] = src['question']
    src['question'] = "effect"
    src['label'] = 1
    return src
def fc6(src):
    src['premise'] = src["sentence1"]
    src['hypothesis'] = src["sentence2"]
    src['label'] = 1 - src['label']
    return src
def fc7(src):
    src['premise'] = src["question1"]
    src['hypothesis'] = src["question2"]
    src['label'] = 1 - src['label']
    return src
def fc8(src):
    src['premise'] = src["content"]
    src['hypothesis'] = "positive"
    src['label'] = 1 - src['label']
    return src
def fc9(src):
    src['premise'] = src["question"]
    src['hypothesis'] = src["answer"]
    src['label'] = 1 - src['label']
    return src
def fc10(src):
    src['premise'] = src["text"]
    src['hypothesis'] = "Topic: " + src["label-coarse"]
    src['label'] = 0
    return src
def fc11(src):
    src['premise'] = src["content"]
    src['hypothesis'] = "Topic: " + src["label"]
    src['label'] = 0
    return src
def fc12(src):
    src['sentence'] = src['question'] + "\nAnswer:_"
    src['option1'] = src['world_literals']['world1'][0]
    src['option2'] = src['world_literals']['world2'][0]
    src['answer'] = src['answer_index']
    return src
def fc13(src):
    src['sentence'] = src['support'] + "\n" + src['question'] + "\nAnswer:_"
    src['option1'] = src['correct_answer']
    src['option2'] = src['distractor1']
    src['answer'] = 1
    return src
def fc14(src):
    src['sentence'] = src['support'] + "\n" + src['question'] + "\nAnswer:_"
    src['option1'] = src['correct_answer']
    src['option2'] = src['distractor1']
    src['answer'] = 1
    return src
# test func

wino_prompts = ['winogrande_winogrande_xl_Replace_score_eval',
 'winogrande_winogrande_xl_does_underscore_refer_to_score_eval',
 'winogrande_winogrande_xl_fill_in_the_blank_score_eval', 'winogrande_winogrande_xl_stand_for_score_eval', 'winogrande_winogrande_xl_underscore_refer_to_score_eval']
for pn in wino_prompts:
    pair2func[("quarel",pn)] = fc12
    pair2func[("sciq",pn)] = fc13

rte_prompts = ['super_glue_cb_GPT_3_style_r3_score_eval', 'super_glue_cb_MNLI_crowdsource_r3_score_eval', 'super_glue_cb_always_sometimes_never_r3_score_eval', 'super_glue_cb_based_on_the_previous_passage_r3_score_eval', 'super_glue_cb_can_we_infer_r3_score_eval',
 'super_glue_cb_claim_true_false_inconclusive_r3_score_eval', 'super_glue_cb_consider_always_sometimes_never_r3_score_eval', 'super_glue_cb_does_it_follow_that_r3_score_eval', 'super_glue_cb_does_this_imply_r3_score_eval',
 'super_glue_cb_guaranteed_true_r3_score_eval', 'super_glue_cb_guaranteed_possible_impossible_r3_score_eval', 'super_glue_cb_justified_in_saying_r3_score_eval', 'super_glue_cb_must_be_true_r3_score_eval', 'super_glue_cb_should_assume_r3_score_eval', 'super_glue_cb_take_the_following_as_truth_r3_score_eval']
for pn in rte_prompts:
    pair2func[("glue/mrpc",pn)] = fc6
    pair2func[("glue/qqp",pn)] = fc7
    pair2func[("paws/labeled_final",pn)] = fc6
    pair2func[("amazon_polarity",pn)] = fc8
    pair2func[("wiki_qa",pn)] = fc9
    # pair2func[("trec",pn)] = fc10
    # pair2func[("dbpedia_14",pn)] = fc11

copa_prompts = ['super_glue_copa_C1_or_C2_premise_so_because__score_eval', 'super_glue_copa_best_option_score_eval', 'super_glue_copa_cause_effect_score_eval', 
'super_glue_copa_choose_score_eval', 'super_glue_copa_exercise_score_eval', 'super_glue_copa_i_am_hesitating_score_eval',
'super_glue_copa_more_likely_score_eval', 'super_glue_copa_plausible_alternatives_score_eval', 'super_glue_copa__As_a_result_C1_or_C2__score_eval', 'super_glue_copa__What_could_happen_next_C1_or_C2__score_eval', 'super_glue_copa__which_may_be_caused_by_score_eval', 'super_glue_copa__why_C1_or_C2_score_eval']
for pn in copa_prompts:
    pair2func[("glue/mrpc",pn)] = fc1
    pair2func[("glue/qqp",pn)] = fc2
    pair2func[("paws/labeled_final",pn)] = fc1
    pair2func[("quarel",pn)] = fc3
    pair2func[("sciq",pn)] = fc4
    pair2func[("wiki_qa",pn)] = fc5




# for test (apply test prompts on related tasks)
test2relatedtns = {}
sc_cr_relatedtns = ["glue/mrpc", "glue/qqp", "paws/labeled_final", "trec",
 "dbpedia_14", "app_reviews", "quarel", "sciq", "wiki_qa"]
nli_relatedtns = ["glue/mrpc", "glue/qqp", "paws/labeled_final", "amazon_polarity", 
"app_reviews", "wiki_qa", "trec", "dbpedia_14", "app_reviews"]
test2relatedtns["super_glue/copa"] = sc_cr_relatedtns
test2relatedtns["winogrande/winogrande_xl"] = sc_cr_relatedtns

test2relatedtns["super_glue/cb"] = nli_relatedtns
test2relatedtns["super_glue/rte"] = nli_relatedtns
test2relatedtns["anli"] = nli_relatedtns


# for train
for tn in double_sent_gen_extract_qa_1:
    if pn.startswith("adversarial_qa"):
        pair2func[("quail",pn)] = func63
        pair2func[("social_i_qa",pn)] = func58
    elif pn.startswith("quail"):
        pair2func[("ropes",pn)] = func52
        pair2func[("quoref",pn)] = func48
    elif pn.startswith("cosmos_qa"):
        pair2func[("ropes",pn)] = func53
        pair2func[("quoref",pn)] = func49
        pair2func[("social_i_qa",pn)] = func62
    elif pn.startswith("quoref"):   
        pair2func[("social_i_qa",pn)] = func59
        pair2func[("ropes",pn)] = func51



pair2func[("cos_e/v1.11","sciq_Multiple_Choice_Closed_Book_")] = func66
pair2func[("cos_e/v1.11","sciq_Multiple_Choice_Question_First")] = func66
pair2func[("cos_e/v1.11","quarel_heres_a_story")] = func65
pair2func[("cos_e/v1.11","quarel_choose_between")] = func65
pair2func[("cos_e/v1.11","quarel_do_not_use")] = func65
pair2func[("cos_e/v1.11","quarel_logic_test")] = func65
pair2func[("cos_e/v1.11","quarel_testing_students")] = func65

pair2func[("quail","cosmos_qa_description_context_question_text")] = func64
pair2func[("quail","adversarial_qa_dbidaf_answer_the_following_q")] = func63
# ign 5
pair2func[("social_i_qa","cosmos_qa_description_context_question_text")] = func62
pair2func[("social_i_qa","cosmos_qa_context_description_question_text")] = func62
pair2func[("social_i_qa","quail_context_question_description_text")] = func61

pair2func[("social_i_qa","ropes_prompt_bottom_no_hint")] = func60
pair2func[("social_i_qa","ropes_plain_no_background")] = func60

pair2func[("social_i_qa","quoref_Answer_Test")] = func59
pair2func[("social_i_qa","quoref_Answer_Friend_Question")] = func59

pair2func[("social_i_qa","adversarial_qa_dbidaf_based_on")] = func58
pair2func[("social_i_qa","adversarial_qa_dbidaf_answer_the_following_q")] = func58

pair2func[("qasc","sciq_Multiple_Choice_Question_First")] = func57
pair2func[("qasc","sciq_Multiple_Choice_Closed_Book_")] = func57

pair2func[("quarel","sciq_Multiple_Choice_Question_First")] = func56
pair2func[("quarel","sciq_Multiple_Choice_Closed_Book_")] = func56

pair2func[("sciq","quarel_do_not_use")] = func55
pair2func[("sciq","quarel_heres_a_story")] = func55
pair2func[("sciq","quarel_choose_between")] = func55
pair2func[("sciq","quarel_testing_students")] = func55
pair2func[("sciq","quarel_logic_test")] = func55

pair2func[("duorc/SelfRC","dream_answer_to_dialogue")] = func54
pair2func[("ropes","cosmos_qa_context_description_question_text")] = func53
pair2func[("ropes","quail_context_description_question_text")] = func52
pair2func[("ropes","quoref_Answer_Test")] = func51
pair2func[("ropes","quoref_Answer_Friend_Question")] = func51
pair2func[("ropes","adversarial_qa_dbidaf_based_on")] = func50
pair2func[("ropes","adversarial_qa_dbidaf_answer_the_following_q")] = func50

pair2func[("quoref","cosmos_qa_description_context_question_text")] = func49
pair2func[("quoref","cosmos_qa_context_description_question_text")] = func49
pair2func[("quoref","quail_context_question_description_text")] = func48
pair2func[("quoref","quail_context_description_question_text")] = func48
pair2func[("quoref","ropes_prompt_bottom_no_hint")] = func47
pair2func[("quoref","ropes_plain_no_background")] = func47
pair2func[("quoref","adversarial_qa_dbidaf_based_on")] = func46
pair2func[("quoref","adversarial_qa_dbidaf_answer_the_following_q")] = func46

pair2func[("wiki_qa","glue_qqp_answer")] = func45
pair2func[("wiki_qa","glue_mrpc_paraphrase")] = func44
pair2func[("wiki_qa","glue_mrpc_equivalent")] = func44
# pair2func[("app_reviews","duorc_SelfRC_build_story_around_qa")] = func44
amazon2gluemrpc = ["glue_mrpc_equivalent", "glue_mrpc_paraphrase",
    "glue_mrpc_replace", "glue_mrpc_same_thing", "glue_mrpc_want_to_know",]
for pn in amazon2gluemrpc:
    pair2func[("amazon_polarity",pn)] = func43
pair2func[("dream","duorc_SelfRC_build_story_around_qa")] = func42
# pair2func[("dream","app_reviews_generate_review")] = func42

# trec -> dbpedia 不太行
pair2func[("dbpedia_14","trec_trec2")] = func41
pair2func[("dbpedia_14","trec_trec1")] = func41
pair2func[("dbpedia_14","trec_what_category_best_describe")] = func41
pair2func[("dbpedia_14","trec_which_category_best_describes")] = func41
pair2func[("dbpedia_14","trec_pick_the_best_descriptor")] = func41

# TODO: paws
for pn in double_sent_disc_fix_oppo_1:
    if pn.startswith("glue_mrpc"):
        pair2func[("glue/qqp",pn)] = func40
        pair2func[("wiki_qa",pn)] = func44
    elif pn.startswith("wiki_qa"):
        pair2func[("glue/mrpc",pn)] = func37
        pair2func[("glue/qqp",pn)] = func39
    elif pn.startswith("glue_qqp"):
        pair2func[("glue/mrpc",pn)] = func38
        pair2func[("wiki_qa",pn)] = func45
        

pair2func[("glue/qqp","wiki_qa_Is_This_True_")] = func39
pair2func[("glue/qqp","glue_mrpc_equivalent")] = func40

pair2func[("glue/mrpc","glue_qqp_answer")] = func38
pair2func[("glue/mrpc","wiki_qa_Is_This_True_")] = func37
# double above

# pair2func[("glue/mrpc","amazon_polarity_flattering_or_not")] = func37
pair2func[("kilt_tasks/hotpotqa","wiki_qa_Direct_Answer_to_Question")] = func36
pair2func[("wiqa","dream_generate_last_utterance")] = func35
pair2func[("wiqa","dream_generate_first_utterance")] = func35
pair2func[("wiqa","duorc_SelfRC_generate_question")] = func34
pair2func[("wiqa","adversarial_qa_dbidaf_generate_question")] = func33
adv_pns = ['duorc_SelfRC_generate_question', 'dream_generate_first_utterance', 'dream_generate_last_utterance', 'wiqa_what_is_the_final_step_of_the_following_process', 'wiqa_what_is_the_missing_first_step', 'wiqa_what_might_be_the_first_step_of_the_process', 'wiqa_what_might_be_the_last_step_of_the_process']
for pn in adv_pns:
    if pn.startswith("duorc"):
        pair2func[("adversarial_qa/dbidaf",pn)] = func30
        pair2func[("adversarial_qa/dbert",pn)] = func30
        pair2func[("adversarial_qa/droberta",pn)] = func30
    elif pn.startswith("dream"):
        pair2func[("adversarial_qa/dbidaf",pn)] = func31
        pair2func[("adversarial_qa/dbert",pn)] = func31
        pair2func[("adversarial_qa/droberta",pn)] = func31
    elif pn.startswith("wiqa"):
        pair2func[("adversarial_qa/dbidaf",pn)] = func32
        pair2func[("adversarial_qa/dbert",pn)] = func32
        pair2func[("adversarial_qa/droberta",pn)] = func32          
rotten_pns = ['imdb_Text_Expressed_Sentiment', 'imdb_Movie_Expressed_Sentiment', 'imdb_Movie_Expressed_Sentiment_2', 
'imdb_Reviewer_Enjoyment', 'imdb_Reviewer_Enjoyment_Yes_No', 'imdb_Reviewer_Expressed_Sentiment', 
'imdb_Reviewer_Opinion_bad_good_choices', 'imdb_Reviewer_Sentiment_Feeling', 'imdb_Writer_Expressed_Sentiment', 'amazon_polarity_User_recommend_this_product', 'amazon_polarity_User_recommend_this_product', 'imdb_Negation_template_for_positive_and_negative', 'imdb_Sentiment_with_choices_']
for pn in rotten_pns:
    pair2func[("rotten_tomatoes",pn)] = func0
pair2func[("app_reviews","amazon_polarity_User_recommend_this_product")] = func28

app_review_pns = ['yelp_review_full_based_on_that', 'yelp_review_full_format_rating', 'yelp_review_full_format_score', 'yelp_review_full_format_star', 
'yelp_review_full_on_a_scale', 'yelp_review_full_so_i_would', 'yelp_review_full_this_place']
for pn in app_review_pns:
    pair2func[("app_reviews",pn)] = func28
amazon_polarity_pns = ['imdb_Text_Expressed_Sentiment', 'imdb_Movie_Expressed_Sentiment', 'imdb_Movie_Expressed_Sentiment_2', 'imdb_Reviewer_Enjoyment', 'imdb_Reviewer_Enjoyment_Yes_No', 
'imdb_Reviewer_Expressed_Sentiment', 'imdb_Reviewer_Opinion_bad_good_choices', 'imdb_Reviewer_Sentiment_Feeling', 
'imdb_Writer_Expressed_Sentiment', 'imdb_Negation_template_for_positive_and_negative', 'imdb_Sentiment_with_choices_']
for pn in amazon_polarity_pns:
    pair2func[("amazon_polarity",pn)] = func27

cnn_related_pns = [
    'gigaword_reverse_writing', 'gigaword_write_an_article', 'multi_news_expand_reverse_task_', 'samsum_Write_a_dialogue_that_match_this_summary', 
    'common_gen_Example_prompt', 'common_gen_Given_concepts_type_2', 
    'common_gen_Given_concepts_type_1', 'common_gen_Put_together', 'common_gen_choice_in_concept_centric_sentence_generation', 
    'common_gen_random_task_template_prompt', 'common_gen_topic_to_sentence', 'common_gen_sentence_to_concepts', 
    'common_gen_topics_from_the_sentence', 'gigaword_TLDR', 'gigaword_first_sentence_title', 
    'gigaword_generate_summary_for_this', 'gigaword_in_a_nutshell', 'gigaword_make_a_title', 'gigaword_write_a_title_for_this_sentence', 'gigaword_write_its_sentence', 
    'multi_news_distill', 'multi_news_summarize', 
    'multi_news_summary_scenario', 'multi_news_synthesize', 'multi_news_what_are_the_key_points', 'samsum_Generate_a_summary_for_this_dialogue', 
    'samsum_Given_the_above_dialogue_write_a_summary', 'samsum_Sum_up_the_following_dialogue', 
    'samsum_Summarize_this_dialogue_', 'samsum_Summarize_', 
    'samsum_To_sum_up_this_dialog', 'xsum_DOC_boils_down_to_simple_idea_that', 'xsum_DOC_given_above_write_one_sentence', 
    'xsum_DOC_how_would_you_rephrase_few_words', 'xsum_DOC_tldr', 'xsum_DOC_write_summary_of_above', 'xsum_article_DOC_summary', 
    'xsum_college_roommate_asked_DOC_so_I_recap', 'xsum_read_below_DOC_write_abstract', 'xsum_summarize_DOC', 'xsum_summarize_this_DOC_summary', 'wiki_qa_Topic_Prediction_Answer_Only', 
    'wiki_qa_Topic_Prediction_Question_Only', 'quoref_Guess_Title_For_Context', 'duorc_SelfRC_title_generation',
]
gigaword_related_pns = ['cnn_dailymail_3.0.0_generate_story', 'cnn_dailymail_3.0.0_spice_up_story', 'multi_news_expand_reverse_task_', 'samsum_Write_a_dialogue_that_match_this_summary', 'common_gen_Example_prompt', 'common_gen_Given_concepts_type_2', 'common_gen_Given_concepts_type_1', 'common_gen_Put_together', 
'common_gen_choice_in_concept_centric_sentence_generation', 'common_gen_random_task_template_prompt', 
'common_gen_topic_to_sentence', 'common_gen_sentence_to_concepts', 'common_gen_topics_from_the_sentence', 'cnn_dailymail_3.0.0_2_or_3_sentences', 'cnn_dailymail_3.0.0_news_card_view', 'cnn_dailymail_3.0.0_news_stock', 
'cnn_dailymail_3.0.0_news_summary', 'cnn_dailymail_3.0.0_sum_in_brief', 'cnn_dailymail_3.0.0_tldr_summary', 'cnn_dailymail_3.0.0_write_an_outline', 'multi_news_distill', 
'multi_news_summarize', 'multi_news_summary_scenario', 
'multi_news_synthesize', 'multi_news_what_are_the_key_points', 
'samsum_Generate_a_summary_for_this_dialogue', 
'samsum_Given_the_above_dialogue_write_a_summary', 
'samsum_Sum_up_the_following_dialogue', 'samsum_Summarize_this_dialogue_', 'samsum_Summarize_', 'samsum_To_sum_up_this_dialog', 
'xsum_DOC_boils_down_to_simple_idea_that', 'xsum_DOC_given_above_write_one_sentence', 'xsum_DOC_how_would_you_rephrase_few_words', 'xsum_DOC_tldr', 'xsum_DOC_write_summary_of_above', 
'xsum_article_DOC_summary', 'xsum_college_roommate_asked_DOC_so_I_recap', 'xsum_read_below_DOC_write_abstract', 'xsum_summarize_DOC', 'xsum_summarize_this_DOC_summary', 'wiki_qa_Topic_Prediction_Answer_Only', 'wiki_qa_Topic_Prediction_Question_Only', 
'quoref_Guess_Title_For_Context', 'duorc_SelfRC_title_generation']
multi_related_pns = ['cnn_dailymail_3.0.0_generate_story', 'cnn_dailymail_3.0.0_spice_up_story', 'gigaword_reverse_writing', 
'gigaword_write_an_article', 'samsum_Write_a_dialogue_that_match_this_summary', 'common_gen_Example_prompt', 'common_gen_Given_concepts_type_2', 'common_gen_Given_concepts_type_1', 'common_gen_Put_together', 
'common_gen_choice_in_concept_centric_sentence_generation', 'common_gen_random_task_template_prompt', 'common_gen_topic_to_sentence', 'common_gen_sentence_to_concepts', 'common_gen_topics_from_the_sentence', 
'cnn_dailymail_3.0.0_2_or_3_sentences', 'cnn_dailymail_3.0.0_news_card_view', 'cnn_dailymail_3.0.0_news_stock', 'cnn_dailymail_3.0.0_news_summary', 'cnn_dailymail_3.0.0_sum_in_brief', 'cnn_dailymail_3.0.0_tldr_summary', 'cnn_dailymail_3.0.0_write_an_outline', 'gigaword_TLDR', 'gigaword_first_sentence_title', 'gigaword_generate_summary_for_this', 'gigaword_in_a_nutshell', 
'gigaword_make_a_title', 'gigaword_write_a_title_for_this_sentence', 'gigaword_write_its_sentence', 'samsum_Generate_a_summary_for_this_dialogue', 'samsum_Given_the_above_dialogue_write_a_summary', 'samsum_Sum_up_the_following_dialogue', 
'samsum_Summarize_this_dialogue_', 'samsum_Summarize_', 'samsum_To_sum_up_this_dialog', 'xsum_DOC_boils_down_to_simple_idea_that', 'xsum_DOC_given_above_write_one_sentence', 'xsum_DOC_how_would_you_rephrase_few_words', 'xsum_DOC_tldr', 'xsum_DOC_write_summary_of_above', 'xsum_article_DOC_summary', 'xsum_college_roommate_asked_DOC_so_I_recap', 'xsum_read_below_DOC_write_abstract',
 'xsum_summarize_DOC', 'xsum_summarize_this_DOC_summary', 'wiki_qa_Topic_Prediction_Answer_Only', 'wiki_qa_Topic_Prediction_Question_Only', 'quoref_Guess_Title_For_Context', 'duorc_SelfRC_title_generation']

for pn in gigaword_related_pns:
    if pn.startswith("cnn"):
        pair2func[("gigaword",pn)] = func18
    elif pn.startswith("xsum"):
        pair2func[("gigaword",pn)] = func23
    elif pn.startswith("multi_news"):
        pair2func[("gigaword",pn)] = func20
    elif pn.startswith("samsum"):
        pair2func[("gigaword",pn)] = func21
    elif pn.startswith("common_gen"):
        pair2func[("gigaword",pn)] = func22
    elif pn.startswith("wiki_qa"):
        pair2func[("gigaword",pn)] = func25
    elif pn.startswith("duorc"):
        pair2func[("gigaword",pn)] = func24
    elif pn.startswith("quoref"):
        pair2func[("gigaword",pn)] = func19
for pn in cnn_related_pns:
    if pn.startswith("gigaword"):
        pair2func[("cnn_dailymail/3.0.0",pn)] = func10
    elif pn.startswith("xsum"):
        pair2func[("cnn_dailymail/3.0.0",pn)] = func14
    elif pn.startswith("multi_news"):
        pair2func[("cnn_dailymail/3.0.0",pn)] = func11
    elif pn.startswith("samsum"):
        pair2func[("cnn_dailymail/3.0.0",pn)] = func12
    elif pn.startswith("common_gen"):
        pair2func[("cnn_dailymail/3.0.0",pn)] = func13
    elif pn.startswith("wiki_qa"):
        pair2func[("cnn_dailymail/3.0.0",pn)] = func15
    elif pn.startswith("duorc"):
        pair2func[("cnn_dailymail/3.0.0",pn)] = func16
    elif pn.startswith("quoref"):
        pair2func[("cnn_dailymail/3.0.0",pn)] = func17
    
# ignore - multi_news, samsum, xsum, wiki-qa repeat
# pair2func[("cnn_dailymail/3.0.0","gigaword_write_an_article")] = func10
# pair2func[("cnn_dailymail/3.0.0","gigaword_reverse_writing")] = func10

pair2func[("trec","ag_news_which_section_choices")] = func9
pair2func[("trec","ag_news_which_section")] = func9
pair2func[("trec","ag_news_recommend")] = func9
pair2func[("trec","ag_news_classify_with_choices_question_first")] = func9
pair2func[("trec","ag_news_classify_with_choices")] = func9
pair2func[("trec","ag_news_classify_question_first")] = func9
pair2func[("trec","ag_news_classify")] = func9
pair2func[("trec","dbpedia_14_given_list_what_category_does_the_paragraph_belong_to")] = func8
pair2func[("dream","wiqa_what_might_be_the_last_step_of_the_process")] = func7
pair2func[("dream","wiqa_what_might_be_the_first_step_of_the_process")] = func7
pair2func[("dream","wiqa_what_is_the_missing_first_step")] = func7
pair2func[("dream","wiqa_what_is_the_final_step_of_the_following_process")] = func7
pair2func[("dream","duorc_SelfRC_generate_question")] = func6
pair2func[("dream","adversarial_qa_dbidaf_generate_question")] = func5
pair2func[("dbpedia_14","ag_news_which_section_choices")] = func4
pair2func[("dbpedia_14","ag_news_which_section")] = func4
pair2func[("dbpedia_14","ag_news_recommend")] = func4
pair2func[("dbpedia_14","ag_news_classify_with_choices_question_first")] = func4
pair2func[("dbpedia_14","ag_news_classify_question_first")] = func4
pair2func[("dbpedia_14","ag_news_classify")] = func4
pair2func[("dbpedia_14","trec_fine_grained_open_context_first")] = func3
pair2func[("dbpedia_14","trec_fine_grained_open")] = func3
pair2func[("dbpedia_14","trec_fine_grained_open")] = func3
pair2func[("ag_news","trec_fine_grained_open_context_first")] = func2
pair2func[("ag_news","trec_fine_grained_open")] = func2
pair2func[("glue/mrpc","paws_labeled_final_paraphrase_task")] = func0
pair2func[("paws/labeled_final","glue_mrpc_generate_paraphrase")] = func0
pair2func[("paws/labeled_final","glue_mrpc_generate_sentence")] = func0
pair2func[("ag_news","dbpedia_14_given_list_what_category_does_the_paragraph_belong_to")] = func1
# pair2func[("amazon_polarity","super_glue_rte_GPT_3_style")] = func1
# pair2func[("app_reviews","super_glue_rte_GPT_3_style")] = func2
