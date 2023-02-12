import os
import random
import logging
from universal_da import T0_data_utils

def new_unified_example():
	return {'para1':None,'para2':None, \
		'para1_meta':{'similar_sentence':[],'paraphrase_label':[]}, \
		'questions':[],'answers':[], \
		'questions_meta':{'similar_sentence':[],'paraphrase_label':[]}, \
		'attributes':{'title':{'answer':None}, \
			'sentiment_2':{'answer':None,'candidates':[0,1]},
			'sentiment_5':{'answer':None,'candidates':[1,2,3,4,5]},
			'topic':{'answer':None},
			'topic1':{
				'answer':None,
				'candidates':['World politics','Sports','Business','Science and technology']
			},
			'topic2':{
				'answer':None,'candidates':['company', 'educational institution', 'artist', 'athlete', 'office holder', 'mean of transportation','building', 'natural place', 'village', 'animal', 'plant', 'album', 'film', 'written work']
			},
			'keywords':{'answer':[],'tag':[]},
			'hints':[],
			},
		}

def new_unified_subexample():
	return {'para1':None,'para2':None, \
		'para1_meta':{'similar_sentence':[],'paraphrase_label':[]}, \
		'question':None,'answer':None,'answer_label':None, 'choices':[], \
		'questions_meta':{'similar_sentence':[],'paraphrase_label':[]}, \
		'attributes':{'title':{'answer':None}, \
			'sentiment_2':{'answer':None,'candidates':[0,1]},
			'sentiment_5':{'answer':None,'candidates':[1,2,3,4,5]},
			'topic':{'answer':None},
			'topic1':{
				'answer':None,
				'candidates':['World politics','Sports','Business','Science and technology']
			},
			'topic2':{
				'answer':None,'candidates':['company', 'educational institution', 'artist', 'athlete', 'office holder', 'mean of transportation','building', 'natural place', 'village', 'animal', 'plant', 'album', 'film or written work']
			},
			'keywords':{'answer':[],'tag':[]},
			'hints':[],
			},
		}

def convert_example_to_subexample(e):
	question=None;answer=None;answer_label=None;all_answers=[];
	questions_meta_similar_sentence=[]
	questions_meta_paraphrase_label=[]
	if len(e['questions'])!=0:
		# [{'correct':['Tsinghua University'],'wrong':['Peking University']},{'correct':['Yes'],'wrong':['No']}]
		question_idx=random.choice(range(len(e['questions'])))
		question=e['questions'][question_idx]
		try: # to be fixed
			questions_meta_similar_sentence=e['questions_meta']['similar_sentence'][question_idx]
			questions_meta_paraphrase_label=e['questions_meta']['paraphrase_label'][question_idx]
		except:
			pass
		if len(e['answers'])!=0:
			answers=e['answers'][question_idx]
			if len(answers['correct'])==0: answers['correct'].append('no answer')
			choosen_answer=random.choice(answers['correct'])
			choices=[choosen_answer]+answers['wrong']
			if len(choices)!=0:
				answer_idx=random.choice(range(len(choices)))
				answer_idx=0
				answer=choices[answer_idx]
				answer_label=True if answer_idx==0 else False
				random.shuffle(choices)
	return {'para1':e['para1'],'para2':e['para2'], \
		'para1_meta':e['para1_meta'], \
		'question':question,'answer':answer,'answer_label':answer_label,'choices':all_answers, \
		'questions_meta':{'similar_sentence':questions_meta_similar_sentence,'paraphrase_label':questions_meta_paraphrase_label}, \
		'attributes':e['attributes']
		}

def convert_examples_to_subexamples(examples,convert_type='rand'):
	new_examples=[]
	for e in examples:
		new_e=convert_example_to_subexample(e)
		new_examples.append(new_e)
	return new_examples

def update_example(example,key):
	(key_name,key_value)=key
	example[key_name]=key_value
	return example 


def update_example_with_emap(uni_e,orig_e,emap):
	# map origional example "orig_e" to unified example "uni_e"
	# print(uni_e)
	def get_name_map(orig_name):
		if isinstance(orig_name,tuple):
			orig_name,list_num=orig_name[0],orig_name[1]
			list_num=int(list_num.split('list_')[1])
		else: list_num=0
		if isinstance(orig_name,list):
			new_value=orig_e[orig_name[0]]
			for name in orig_name[1:]:
				new_value = new_value[name]
		else:
			new_value=orig_e[orig_name]
		# if isinstance(uni_name,tuple):
		# 	uni_name,list_num=uni_name[0],uni_name[1]
		# 	list_num=int(list_num.split('list_')[1])
		if list_num==1: new_value=[new_value]
		elif list_num==2: new_value=[[new_value]]
		elif list_num==-1: 
			if new_value[0].endswith('.') or new_value[0].endswith('?'):
				new_value=' '.join(new_value)
			else:
				new_value='. '.join(new_value)
		return new_value
	for (orig_name,uni_name) in emap:
		# print(orig_name,uni_name)
		if isinstance(orig_name,dict):
			answers_name=orig_name['answers']
			choices_name=orig_name['choices']
			answers=get_name_map(answers_name)
			choices=get_name_map(choices_name)
		else:
			new_value=get_name_map(orig_name)
		if 'answers' in uni_name:
			# import pdb 
			# pdb.set_trace()
			if uni_name=='answers': uni_e['answers'].append({'correct':new_value,'wrong':[]})
			elif uni_name=='answers_choices': uni_e['answers'].append({'correct':answers,'wrong':[x for x in choices if x not in answers]})
		elif isinstance(uni_name,list):
			t=uni_e[uni_name[0]]
			for name in uni_name[1:-1]:
				t = t[name]
			if t[uni_name[-1]] is None: t[uni_name[-1]]=new_value
			else: t[uni_name[-1]].append(new_value)
		else:
			if uni_name=='para':
				length=len(new_value.split())
				# if length>256: uni_name='para3'
				if length>128: uni_name='para2'
				else: uni_name='para1'
			# print(uni_name)
			if uni_e[uni_name] is None or isinstance(uni_e[uni_name],list)==False: uni_e[uni_name]=new_value
			else: uni_e[uni_name].append(new_value)
	# postprocess
	if uni_e['para1'] is not None and len(uni_e['para1'])==0: uni_e['para1']=None
	if uni_e['para2'] is not None and len(uni_e['para2'])==0: uni_e['para2']=None
	# if uni_e['para3'] is not None and len(uni_e['para3'])==0: uni_e['para3']=None
	return uni_e

def update_example_with_additional_constraints(uni_e,orig_e,task_name):
	if task_name.startswith('duorc'):
		if orig_e['no_answer']==True:
			uni_e['answers'][0]={'correct':['no answer'],'wrong':[]}
	elif task_name=='ropes':
		uni_e['attributes']['hints'].append(orig_e['background'])
	elif task_name=='cosmos_qa':
		choices=[orig_e['answer0'],orig_e['answer1'],orig_e['answer2'],orig_e['answer3']]
		answers=[choices[orig_e['label']]]
		uni_e['answers'].append({'correct':answers,'wrong':[x for x in choices if x not in answers]})
	elif task_name=='dream':
		uni_e['para1']='\n\n'.join(orig_e['dialogue'])
	elif task_name=='qasc':
		choices=orig_e['choices']['text']
		answers=[choices[orig_e['choices']['label'].index(orig_e['answerKey'])]]
		uni_e['answers'].append({'correct':answers,'wrong':[x for x in choices if x not in answers]})
		uni_e['attributes']['hints']=[orig_e['fact1'],orig_e['fact2']]
	elif task_name=='quail':
		choices=orig_e['answers']
		answers=[choices[orig_e['correct_answer_id']]]
		uni_e['answers'].append({'correct':answers,'wrong':[x for x in choices if x not in answers]})
	elif task_name=='quarel':
		choices=[orig_e['world_literals']['world1'][0],orig_e['world_literals']['world2'][0]]
		answers=[choices[orig_e['answer_index']]]
		uni_e['answers'].append({'correct':answers,'wrong':[x for x in choices if x not in answers]})
	elif task_name=='quartz':
		choices=orig_e['choices']['text']
		answers=[choices[orig_e['choices']['label'].index(orig_e['answerKey'])]]
		uni_e['answers'].append({'correct':answers,'wrong':[x for x in choices if x not in answers]})
		if '_____' in orig_e['question']:
			question=orig_e['question'].replace('_____',' or '.join(choices)).rstrip('.?!')+'?'
		else:
			question=orig_e['question'].rstrip('.?!')+' {}'.format(' or '.join(choices))+'?'
		uni_e['questions'].append(question)
	elif task_name=='sciq':
		uni_e['answers'].append({
			'correct': [orig_e['correct_answer']],
			'wrong': [orig_e['distractor1'],orig_e['distractor2'],orig_e['distractor3']]
		})
	elif task_name=='social_i_qa':
		choices=[orig_e['answerA'],orig_e['answerB'],orig_e['answerC']]
		answers=[choices[int(orig_e['label'])-1]]
		uni_e['answers'].append({'correct':answers,'wrong':[x for x in choices if x not in answers]})
	elif task_name=='wiki_hop/original':
		choices=orig_e['candidates']
		uni_e['para2']='\n\n'.join(orig_e['supports'])
		question_split=orig_e['question'].split()
		question="What object entity has the relation of '{}' with the subject '{}'?".format(question_split[0].replace("_", " "),' '.join(question_split[1:]))
		answers=[orig_e['answer']]
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':answers,'wrong':[x for x in choices if x not in answers]})

		question="What is the relationship between '{}' and '{}'?".format(' '.join(question_split[1:]),orig_e['answer'])
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[question_split[0].replace("_", " ")],'wrong':[]})

		question="What entity does '{}' has the relation '{}' with?".format(' '.join(question_split[1:]),question_split[0].replace("_", " "))
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[orig_e['answer']],'wrong':[x for x in choices if x not in answers]})

		question="what entity has the relation '{}' with '{}'".format(question_split[0].replace("_", " "),orig_e['answer'])
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[' '.join(question_split[1:])],'wrong':[]})

		question="the object entity that exhibits the relation '{}' with the subject '{}'".format(question_split[0].replace("_", " "),' '.join(question_split[1:]))
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[orig_e['answer']],'wrong':[x for x in choices if x not in answers]})
	
		question="the entity with which \'{}\' exhibits the relationship of \'{}\'".format(' '.join(question_split[1:]),question_split[0].replace("_", " "))
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[orig_e['answer']],'wrong':[x for x in choices if x not in answers]})
	
		question="choose the subject and object entities that have the relation of \'{}\'.".format(question_split[0].replace("_", " "))
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':["{} , {}".format(' '.join(question_split[1:]),orig_e['answer'])],'wrong':[]})
	
		question="choose the best answer for the entity that related to \'{}\' with the relationship of \'{}\'.".format(" ".join(question_split[1:]),question_split[0].replace("_", " "))
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[orig_e['answer']],'wrong':[x for x in choices if x not in answers]})

		question="\'{}\' is related to which object entity through the relation of \'{}\'?".format(" ".join(question_split[1:]),question_split[0].replace("_", " "))
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[orig_e['answer']],'wrong':[x for x in choices if x not in answers]})

	elif task_name=='wiqa':
		uni_e['para1']='\n\n'.join(orig_e['question_para_step'])
		question='{}\n\nHow does the supposed perturbation influence the second effect mentioned.'.format(orig_e['question_stem'])
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[orig_e['answer_label'].replace("_"," ")],'wrong':[x for x in ['more','less','no effect'] if x!=orig_e['answer_label'].replace("_"," ")]})
		
		question='{}'.format(orig_e['question_stem'])
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[orig_e['answer_label'].replace("_"," ")],'wrong':[x for x in ['more','less','no effect'] if x!=orig_e['answer_label'].replace("_"," ")]})
	
		question="{}\n\nWhich of the following is the supposed perturbation?".format(orig_e['question_stem'])
		choices={"EXOGENOUS_EFFECT": "indirectly impacting a step of the process", "OUTOFPARA_DISTRACTOR": "not impacting any step of the process", "INPARA_EFFECT": "directly impacting a step of the process"}
		correct_answer=choices[orig_e['metadata_question_type']]
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[correct_answer],'wrong':[x for (y,x) in choices.items() if x!=correct_answer]})
	
		question='Perturbation hypothesis:\n{}\n\nDoes the supposed perturbation have an effect (direct or indirect) on the process?'.format(orig_e['question_stem'])
		choices={"EXOGENOUS_EFFECT": "yes", "OUTOFPARA_DISTRACTOR": "no", "INPARA_EFFECT": "yes"}
		correct_answer=choices[orig_e['metadata_question_type']]
		uni_e['questions'].append(question)
		uni_e['answers'].append({'correct':[correct_answer],'wrong':[x for (y,x) in choices.items() if x!=correct_answer]})
	elif task_name=='yelp_review_full':
		uni_e['attributes']['sentiment_5']['answer']=orig_e['label']+1
	elif task_name=='common_gen':
		uni_e['attributes']['keywords']['answer']=orig_e['concepts']
	elif task_name=='wiki_bio':
		keywords=[];tags=[]
		for keyword,tag in zip(orig_e['input_text']['table']['content'],orig_e['input_text']['table']['column_header']):
			if 'tag'!='article_title':
				keywords.append(keyword)
				tags.append(tag)
			else:
				uni_e['attributes']['title']['answer']=answer
		uni_e['attributes']['keywords']['answer']=keywords
		uni_e['attributes']['keywords']['tag']=tags
	elif task_name=='multi_news':
		uni_e['para1']=orig_e['summary'][2:]
	elif task_name=='ag_news':
		# choices=['World politics','Sports','Business','Science and technology']
		choices=uni_e['attributes']['topic1']['candidates']
		answer=choices[int(orig_e['label'])]
		uni_e['attributes']['topic1']={'answer':answer,'candidates':choices}
	elif task_name=='dbpedia_14':
		# choices=['company', 'educational institution', 'artist', 'athlete', 'office holder', 'mean of transportation','building', 'natural place', 'village', 'animal', 'plant', 'album', 'film or written work']
		choices=uni_e['attributes']['topic2']['candidates']
		answer=choices[int(orig_e['label'])]
		uni_e['attributes']['topic2']={'answer':answer,'candidates':choices}
	return uni_e

def get_emap(task_name):
	if task_name=='glue/mrpc' or task_name=='paws/labeled_final':
		return [('sentence1','para1'),('sentence2',['para1_meta','similar_sentence']),('label',['para1_meta','paraphrase_label'])]
	elif task_name=='glue/qqp':
		return [('question1','questions'),(('question2','list_1'),['questions_meta','similar_sentence']),(('label','list_1'),['questions_meta','paraphrase_label'])]
	elif task_name=='kilt_tasks/hotpotqa':
		return [('input','questions'),((['output',0,'answer'],'list_1'),'answers')]
	elif task_name=='wiki_qa':
		return [('question','questions'),('document_title',['attributes','title','answer'])]
	elif task_name=='adversarial_qa/dbidaf' or task_name=='adversarial_qa/dbert' or task_name=='adversarial_qa/droberta':
		return [('context','para'),('question','questions'),(['answers','text'],'answers'),('title',['attributes','title','answer'])]
	elif task_name=='duorc/SelfRC' or task_name=='duorc/ParaphraseRC':
		return [('plot','para'),('question','questions'),('answers','answers'),('title',['attributes','title','answer'])]
	elif task_name=='ropes':
		return [('situation','para'),('question','questions'),(['answers','text'],'answers')]
	elif task_name=='quoref':
		return [('context','para'),('question','questions'),(['answers','text'],'answers'),('title',['attributes','title','answer'])]
	elif task_name=='cos_e/v1.11':
		return [('question','questions'),({'answers':('answer','list_1'),'choices':'choices'},'answers_choices')]
	elif task_name=='cosmos_qa':
		return [('context','para'),('question','questions')]
	elif task_name=='dream':
		return [('question','questions'),({'answers':('answer','list_1'),'choices':'choice'},'answers_choices')]
	elif task_name=='qasc':
		return [('combinedfact','para'),('question','questions')]
	elif task_name=='quail':
		return [('context','para'),('question','questions'),(['metadata','title'],['attributes','title','answer']),('domain',['attributes','topic','answer'])]
	elif task_name=='quarel':
		return [('question','questions')]
	elif task_name=='quartz':
		return [('para','para')]
	elif task_name=='sciq':
		return [('question','questions'),('support','para')]
	elif task_name=='social_i_qa':
		return [('context','para'),('question','questions')]
	elif task_name=='wiki_hop/original':
		# return [(('supports','list_-1'),'para')]
		return []
	elif task_name=='wiqa':
		return []
		# return [(('question_para_step','list_-1'),'para'),('question_stem','questions'),({'answers':('answer_label','list_1'),'choices':['choices','text']},'answers_choices')]
	elif task_name=='amazon_polarity':
		return [('content','para'),('title',['attributes','title','answer']),('label',['attributes','sentiment_2','answer'])]
	elif task_name=='app_reviews':
		return [('review','para'),('star',['attributes','sentiment_5','answer']),('package_name',['attributes','title','answer'])]
	elif task_name=='imdb':
		return [('text','para'),('label',['attributes','sentiment_2','answer'])]
	elif task_name=='rotten_tomatoes':
		return [('text','para'),('label',['attributes','sentiment_2','answer'])]
	elif task_name=='yelp_review_full':
		return [('text','para')]
	elif task_name=='common_gen':
		return [('target','para')]
	elif task_name=='wiki_bio':
		return [('target_text','para')]		
	elif task_name=='cnn_dailymail/3.0.0':
		return [('article','para2'),('highlights','para1')]
	elif task_name=='gigaword':
		return [('document','para2'),('summary',['attributes','title','answer'])]
		# return [('document','para2'),('summary','para1')]
	elif task_name=='multi_news':
		return [('document','para2')]
	elif task_name=='samsum':
		return [('dialogue','para2'),('summary','para1')]
	elif task_name=='xsum':
		return [('document','para2'),('summary','para1')]
	elif task_name=='ag_news':
		return [('text','para')]
	elif task_name=='dbpedia_14':
		return [('content','para'),('title',['attributes','title','answer'])]
	elif task_name=='trec':
		return [('text','questions')]

def build_wiki_uniformed_dataset(train_examples):
	uniformed_dataset={}
	for orig_e in train_examples:
		uni_e=new_unified_example()
		if orig_e['question_id'] not in uniformed_dataset:
			uni_e['questions']=[orig_e['question']]
			uni_e['attributes']['title']['answer']=orig_e['document_title']
			uni_e['answers']=[{'correct':[],'wrong':[]}]
			if orig_e['label']==0: uni_e['answers'][0]['wrong'].append(orig_e['answer'])
			else: uni_e['answers'][0]['correct'].append(orig_e['answer'])
			uniformed_dataset[orig_e['question_id']]=uni_e
		else:
			if orig_e['label']==0:
				uniformed_dataset[orig_e['question_id']]['answers'][0]['wrong'].append(orig_e['answer'])
			else:
				uniformed_dataset[orig_e['question_id']]['answers'][0]['correct'].append(orig_e['answer'])
	uniformed_dataset=[x for _,x in uniformed_dataset.items()]
	return uniformed_dataset

def get_uniformed_dataset(args,task_name):
	uniformed_dataset=[]
	if args.debug:
		args.max_train_num_per_dataset=5
	train_examples,valid_examples=T0_data_utils.load_datasets(data_dir=args.multi_cache_dir,task_name=task_name,train_num=args.max_train_num_per_dataset,valid_num=args.max_valid_num_per_dataset,k_fold=args.k_fold,fold_id=args.fold_id)
	# import pdb 
	# pdb.set_trace()
	if task_name=='wiki_qa':
		uniformed_dataset=build_wiki_uniformed_dataset(train_examples)
	else:
		for orig_e in train_examples:
			uni_e=new_unified_example()
			emap=get_emap(task_name)
			uni_e=update_example_with_emap(uni_e,orig_e,emap)
			uni_e=update_example_with_additional_constraints(uni_e,orig_e,task_name)
			uniformed_dataset.append(uni_e)
	return uniformed_dataset


if __name__ == '__main__':
	# test the file
	from universal_da.dataset_config import default_T0_tasks
	from arguments import get_args
	args=get_args()
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
	)
	task_names=[]
	for task_type,names in default_T0_tasks.items():
		task_names+=names
	task_names=['wiki_qa']
	for task_name in task_names:
		print(task_name)
		nd=get_uniformed_dataset(args,task_name)
		import pdb 
		pdb.set_trace()
		new_sub_examples=convert_examples_to_subexamples(nd)
		print(new_sub_examples[0])
		print(new_sub_examples[0]['question'])
		print(new_sub_examples[1]['question'])
		print(new_sub_examples[2]['question'])
		print(new_sub_examples[3]['question'])
		print(new_sub_examples[4]['question'])
		import pdb 
		pdb.set_trace()

'''
python -m universal_da.simple_cross.convert_dataset_uniform --multi-cache-dir ../../huggingface_datasets --debug
'''