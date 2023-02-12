import os
import re
import math
import string
import torch
import numpy as np
from scipy.special import softmax
from SwissArmyTransformer.model import T5Model
from universal_da.simple_cross import convert_dataset_uniform
from universal_da.simple_cross.uniform_prompts import uniformed_prompt_templates

def distinct_metric(seq,n_gram=2):
    if len(seq)<=2: return 1
    n_gram_list=[]
    n_gram_list=[seq[i:i+n_gram] for i in range(len(seq)-n_gram)]
    score=len(set(n_gram_list))/len(n_gram_list)
    return score

def abandon_low_distinct_answer(answer_with_logits,th=0.1):
    # if some answer appears like 'a a a a a ',we will delete it
    for i in range(len(answer_with_logits)):
        if distinct_metric(answer_with_logits[i][0])<th:
            answer_with_logits[i]=(answer_with_logits[i][0],-math.inf)
    return answer_with_logits

def abandon_prompts(answer_with_logits,abandon_ids,only_logits=False):
    # set the logits of abandon_ids to -inf
    # import pdb 
    # pdb.set_trace()
    for i in range(len(answer_with_logits)):
        if only_logits==False:
            if i in abandon_ids:
                # print(i,len(answer_with_logits))
                answer_with_logits[i]=(answer_with_logits[i][0],-math.inf)
        else:
            if i in abandon_ids:
                for j in range(len(answer_with_logits[i])):
                    answer_with_logits[i][j]=-math.inf
    return answer_with_logits

def abandon_meaningless_sentences(answer_with_logits,th=0.5):
    # if punctuations and '<unk>' makes up more than 50% of the sentence, then it is meaningless.
    # print(answer_with_logits)
    for i in range(len(answer_with_logits)):
        answer=answer_with_logits[i][0].replace('<pad>','')
        all_chars=''.join(answer.split())
        tmp_sentence=''.join(all_chars.split('<unk>'))
        tmp_sentence=tmp_sentence.translate(tmp_sentence.maketrans('', '', string.punctuation))
        # print(len(tmp_sentence),len(all_chars))
        if len(all_chars)==0 or len(tmp_sentence)/len(all_chars)<th:
            answer_with_logits[i]=(answer,-math.inf)
        else:
            answer_with_logits[i]=(answer,answer_with_logits[i][1])
    return answer_with_logits

def abandon_invalid_questions(answer_with_logits):
    for i in range(len(answer_with_logits)):
        if answer_with_logits[i][0]=='No Valid Question':
            answer_with_logits[i]=('No Valid Question',-math.inf)
            # import pdb 
            # pdb.set_trace()
    return answer_with_logits

def ensemble_logits(logits): # only supports for classification tasks
    #logits=[[0.1,0.9],[0.2,0.8],[0.5,0.5]]
    logits=np.array([softmax(la) for la in logits if la[0]!=-math.inf])
    if len(logits)==0: return (None,-math.inf)
    logits=logits.mean(axis=0)
    ans=np.argmax(logits)
    logit=logits[ans]
    return (ans,logit)

def choose_most_probable_answer(answer_with_logits=None, logits=None):
    # if none of the answers are probable, return None
    if answer_with_logits is not None:
        if isinstance(answer_with_logits,tuple): return answer_with_logits
        answer=None;max_logits=-math.inf
        for (ans,logit) in answer_with_logits:
            if logit>max_logits: max_logits=logit;answer=ans.replace('<pad>','')
        return (answer,max_logits)
    else: # final_check, this is designed for classification tasks, but not used in the final version, we replace this with ensemble_logits
        answer=-1;max_logits=-math.inf;total_max_logits=[-math.inf]*len(logits[0])
        for la in logits:
            la=softmax(la)
            print(la)
            ans=np.argmax(la)
            print(ans)
            if la[ans]>total_max_logits[ans]:
                total_max_logits[ans]=la[ans]
                if la[ans]>max_logits:
                    answer=ans
                    max_logits=la[ans]
        return (answer,max_logits)

def filter_summary(args,task_name,uniformed_dataset,crossdataset):
    ###############################################################################
    # filter examples of the crossdataset and then merge it with the uniformed_dataset with the most probable answer.
    ###############################################################################
    if crossdataset is None:
        for (i,uni_e) in enumerate(uniformed_dataset):
            uniformed_dataset[i]['para1']=(uni_e['para1'],100)
            uniformed_dataset[i]['para2']=(uni_e['para2'],100)
        return uniformed_dataset
    for i,(uni_e, e) in enumerate(zip(uniformed_dataset,crossdataset)):
        abandon_para1_ids=range(12,len(uniformed_prompt_templates['paragraph3_to_paragraph1']))
        answer1_with_logits=e['para1']
        if isinstance(answer1_with_logits,list):
            answer1_with_logits=abandon_prompts(answer1_with_logits,abandon_para1_ids)
            answer1_with_logits=abandon_meaningless_sentences(answer1_with_logits)
        answer2_with_logits=e['para2']
        if isinstance(answer2_with_logits,list):
            answer2_with_logits=abandon_meaningless_sentences(answer2_with_logits)
        uniformed_dataset[i]['para1']=choose_most_probable_answer(answer1_with_logits)
        uniformed_dataset[i]['para2']=choose_most_probable_answer(answer2_with_logits)
    return uniformed_dataset


def filter_qa_orig(args,task_name,uniformed_dataset,crossdataset):
    ###############################################################################
    # filter examples of the crossdataset and then merge it with the uniformed_dataset with the most probable answer.
    ###############################################################################
    if crossdataset is None:
        for i in range(len(uniformed_dataset)):
            uniformed_dataset[i]['questions']=[(uniformed_dataset[i]['questions'][j],100) for j in range(len(uniformed_dataset[i]['questions']))]
            for j in range(len(uniformed_dataset[i]['answers'])):
                uniformed_dataset[i]['answers'][j]['correct']=[(x,100) for x in uniformed_dataset[i]['answers'][j]['correct']]
                uniformed_dataset[i]['answers'][j]['wrong']=[(x,-100) for x in uniformed_dataset[i]['answers'][j]['wrong']]
        return uniformed_dataset
    for i,(uni_e, e) in enumerate(zip(uniformed_dataset,crossdataset)):
        # import pdb 
        # pdb.set_trace()
        question_answer_with_logits=abandon_meaningless_sentences(e['question'])
        question_answer_with_logits=abandon_low_distinct_answer(question_answer_with_logits)
        has_question_flag=False
        if len(uniformed_dataset[i]['questions'])!=0:
            uniformed_dataset[i]['questions']=[(uniformed_dataset[i]['questions'][0],100)]
            uniformed_dataset[i]['answers'][0]['correct']=[(x,100) for x in uniformed_dataset[i]['answers'][0]['correct']]
            uniformed_dataset[i]['answers'][0]['wrong']=[(x,-100) for x in uniformed_dataset[i]['answers'][0]['wrong']]
            has_question_flag=True
        for question,answer_with_logits in zip(question_answer_with_logits,e['choices']):
            if has_question_flag==False and question[1]==-math.inf: continue
            if has_question_flag==True and len(uniformed_dataset[i]['answers'][0]['correct'])!=0:
                    wrong_answers=[]
                    # import pdb 
                    # pdb.set_trace()
                    for (tmp_ans,y) in answer_with_logits:
                        if uniformed_dataset[i]['answers'][0]['correct'][0][0].strip(string.punctuation+' ').lower()==tmp_ans.strip(string.punctuation+' ').lower():
                            continue
                        wrong_answers.append((tmp_ans,y))
                    if len(wrong_answers)==0:
                        print(uniformed_dataset[i]['answers'][0]['correct'],e['choices'])
                    uniformed_dataset[i]['answers'][0]['wrong']+=wrong_answers
            else:
                answer_with_logits=abandon_meaningless_sentences(answer_with_logits)
                answer_with_logits=abandon_low_distinct_answer(answer_with_logits,0.2)
                (correct_answer,logit)=choose_most_probable_answer(answer_with_logits)
                if correct_answer is None:
                    uniformed_dataset[i]['questions'].append(question)
                    uniformed_dataset[i]['answers'].append({'correct':[(None,-math.inf)],'wrong':[]})
                    continue
                correct_answers=[];wrong_answers=[]
                for (answer,logit) in answer_with_logits:
                    if answer.strip(string.punctuation+' ').lower()==correct_answer.strip(string.punctuation+' ').lower():
                        correct_answers.append((answer,logit))
                    else:
                        if logit!=-math.inf: 
                            wrong_answers.append((answer.strip(string.punctuation+' '),logit))
                if has_question_flag==True:
                    print('enter into has_question_flag, update uniformed_dataset[i]')
                    if len(correct_answers)==0:
                        import pdb 
                        pdb.set_trace()
                    # print(correct_answers,wrong_answers)
                    uniformed_dataset[i]['answers'][0]={'correct':correct_answers,'wrong':wrong_answers}
                else:
                    uniformed_dataset[i]['questions'].append(question)
                    uniformed_dataset[i]['answers'].append({'correct':correct_answers,'wrong':wrong_answers})
        # print(uniformed_dataset[i]['answers'])
        try:
            if len(uniformed_dataset[i]['answers'][0]['correct'])==0:
            #  or len(uniformed_dataset[i]['answers'][0]['wrong'])==0:
                print('Length could not be 0')
                print(uniformed_dataset[i]['answers'])
                import pdb 
                pdb.set_trace()
        except:
            import pdb 
            pdb.set_trace()
    return uniformed_dataset

def filter_qa(args,task_name,uniformed_dataset,crossdataset):
    ###############################################################################
    # filter examples of the crossdataset and then merge it with the uniformed_dataset with the most probable answer.
    ###############################################################################
    if crossdataset is None:
        for i in range(len(uniformed_dataset)):
            uniformed_dataset[i]['questions']=[(uniformed_dataset[i]['questions'][j],100) for j in range(len(uniformed_dataset[i]['questions']))]
            for j in range(len(uniformed_dataset[i]['answers'])):
                uniformed_dataset[i]['answers'][j]['correct']=[(x,100) for x in uniformed_dataset[i]['answers'][j]['correct']]
                uniformed_dataset[i]['answers'][j]['wrong']=[(x,-100) for x in uniformed_dataset[i]['answers'][j]['wrong']]
        return uniformed_dataset
    # import pdb 
    # pdb.set_trace()
    for i,(uni_e, e) in enumerate(zip(uniformed_dataset,crossdataset)):
        if len(e['question'])>0 and isinstance(e['question'][0],tuple):
            question_answer_with_logits=abandon_invalid_questions(e['question'])
            question_answer_with_logits=abandon_meaningless_sentences(question_answer_with_logits)
            question_answer_with_logits=abandon_low_distinct_answer(question_answer_with_logits)
        else: # use the original questions
            question_answer_with_logits=[(uniformed_dataset[i]['questions'][0],100)]
        has_question_flag=False
        # import pdb 
        # pdb.set_trace()
        if len(uniformed_dataset[i]['questions'])!=0:
            uniformed_dataset[i]['questions']=[(uniformed_dataset[i]['questions'][0],100)]
            uniformed_dataset[i]['answers'][0]['correct']=[(x,100) for x in uniformed_dataset[i]['answers'][0]['correct']]
            uniformed_dataset[i]['answers'][0]['wrong']=[(x,-100) for x in uniformed_dataset[i]['answers'][0]['wrong']]
            has_question_flag=True
        for question,answer_with_logits in zip(question_answer_with_logits,e['choices']):
            if has_question_flag==False and question[1]==-math.inf: 
                # print(e)
                # import pdb 
                # pdb.set_trace()
                continue
            if has_question_flag==True and len(uniformed_dataset[i]['answers'][0]['correct'])!=0:
                wrong_answers=[];lower_wrong_answers=[]
                answer_with_logits=abandon_meaningless_sentences(answer_with_logits)
                answer_with_logits=abandon_low_distinct_answer(answer_with_logits,0.2)
                pred_logits=[y for (x,y) in answer_with_logits]
                pred_scores=softmax(pred_logits)
                for (j,(tmp_ans,logit)) in enumerate(answer_with_logits):
                    if uniformed_dataset[i]['answers'][0]['correct'][0][0].strip(string.punctuation+' ').lower()==tmp_ans.strip(string.punctuation+' ').lower():
                        continue
                    if logit!=-math.inf and pred_scores[j]<1.0/(len(answer_with_logits)+1):
                        if tmp_ans.strip(string.punctuation+' ').lower() in lower_wrong_answers: continue
                        wrong_answers.append((tmp_ans,logit))
                        lower_wrong_answers.append(tmp_ans.strip(string.punctuation+' ').lower())
                if len(wrong_answers)==0:
                    print(uniformed_dataset[i]['answers'][0]['correct'],e['choices'])
                uniformed_dataset[i]['answers'][0]['wrong']+=wrong_answers
            elif 'model_answer' in e:
                # import pdb 
                # pdb.set_trace()
                model_answer=e['model_answer'][0]
                answer_with_logits=abandon_meaningless_sentences(answer_with_logits)
                answer_with_logits=abandon_low_distinct_answer(answer_with_logits,0.2)
                pred_logits=[y for (x,y) in answer_with_logits]
                pred_scores=softmax(pred_logits)
                correct_answer,_=choose_most_probable_answer(answer_with_logits)
                if correct_answer!=model_answer: continue # if the pred score disagrees with the correct answer, just ignore it
                if pred_scores[0]<1.0/len(answer_with_logits): 
                    # print(pred_scores)
                    # import pdb 
                    # pdb.set_trace()
                    continue # if it is predicted as wrong label, just ignore it.
                correct_answer=model_answer
                correct_answers=[(correct_answer,pred_logits[0])];wrong_answers=[];lower_wrong_answers=[]
                assert(len(pred_scores)==len(answer_with_logits))
                for ((answer,logit),pred_score) in zip(answer_with_logits[1:],pred_scores[1:]):
                    if answer.strip(string.punctuation+' ').lower()==correct_answer.strip(string.punctuation+' ').lower():
                        continue
                        # correct_answers.append((answer,logit))
                    else:
                        if logit!=-math.inf and pred_score<1.0/(len(answer_with_logits)+1): # if the answer is valid and the pred_score is negative enough, add as the wrong choices 
                            # print(answer,logit,pred_score)
                            if answer.strip(string.punctuation+' ').lower() in lower_wrong_answers: continue
                            wrong_answers.append((answer.strip(string.punctuation+' '),logit))
                            lower_wrong_answers.append(answer.strip(string.punctuation+' ').lower())
                uniformed_dataset[i]['questions'].append(question)
                uniformed_dataset[i]['answers'].append({'correct':correct_answers,'wrong':wrong_answers})     
            else:
                raise NotImplementedError
                answer_with_logits=abandon_meaningless_sentences(answer_with_logits)
                answer_with_logits=abandon_low_distinct_answer(answer_with_logits,0.2)
                (correct_answer,logit)=choose_most_probable_answer(answer_with_logits)
                if correct_answer is None:
                    uniformed_dataset[i]['questions'].append(question)
                    uniformed_dataset[i]['answers'].append({'correct':[(None,-math.inf)],'wrong':[]})
                    continue
                correct_answers=[];wrong_answers=[]
                for (answer,logit) in answer_with_logits:
                    if answer.strip(string.punctuation+' ').lower()==correct_answer.strip(string.punctuation+' ').lower():
                        correct_answers.append((answer,logit))
                    else:
                        if logit!=-math.inf: 
                            wrong_answers.append((answer.strip(string.punctuation+' '),logit))
                if has_question_flag==True:
                    print('enter into has_question_flag, update uniformed_dataset[i]')
                    if len(correct_answers)==0:
                        import pdb 
                        pdb.set_trace()
                    # print(correct_answers,wrong_answers)
                    uniformed_dataset[i]['answers'][0]={'correct':correct_answers,'wrong':wrong_answers}
                else:
                    uniformed_dataset[i]['questions'].append(question)
                    uniformed_dataset[i]['answers'].append({'correct':correct_answers,'wrong':wrong_answers})
        # print(uniformed_dataset[i]['questions'],uniformed_dataset[i]['answers'])
        # import pdb 
        # pdb.set_trace()
        # try:
        #     if len(uniformed_dataset[i]['questions'])==0 and len(uniformed_dataset[i]['answers'][0]['correct'])==0:
        #     #  or len(uniformed_dataset[i]['answers'][0]['wrong'])==0:
        #         print('Length could not be 0')
        #         print(uniformed_dataset[i]['answers'])
        #         import pdb 
        #         pdb.set_trace()
        # except:
        #     import pdb 
        #     pdb.set_trace()
    # import pdb 
    # pdb.set_trace()
    return uniformed_dataset

def filter_paraphrase(args,task_name,uniformed_dataset,crossdataset):
    ###############################################################################
    # filter examples of the crossdataset and then merge it with the uniformed_dataset with the most probable answer.
    ###############################################################################
    # TODO: it's better to be labeled with prompts within "paraphrase_tf"
    if crossdataset is None:
        for (i,uni_e) in enumerate(uniformed_dataset):
            uniformed_dataset[i]['para1_meta']['similar_sentence'][0]=(uni_e['para1_meta']['similar_sentence'][0],100)
            uniformed_dataset[i]['para1_meta']['paraphrase_label'][0]=(uni_e['para1_meta']['paraphrase_label'][0],100)
        return uniformed_dataset

    for i,(uni_e, e) in enumerate(zip(uniformed_dataset,crossdataset)):
        answer1_with_logits=e['para1_meta']['similar_sentence']
        answer2_with_logits=e['para1_meta']['paraphrase_label']
        answer1_with_logits=abandon_meaningless_sentences(answer1_with_logits)
        for j,(answer,logit) in enumerate(answer1_with_logits):
            if logit==-math.inf: continue
            uniformed_dataset[i]['para1_meta']['similar_sentence'].append((answer,logit))
            logits=ensemble_logits(answer2_with_logits[j])
            uniformed_dataset[i]['para1_meta']['paraphrase_label'].append(logits)
    return uniformed_dataset

def filter_title_keys(args,task_name,uniformed_dataset,crossdataset):
    ###############################################################################
    # filter examples of the crossdataset and then merge it with the uniformed_dataset with the most probable answer.
    ###############################################################################
    if crossdataset is None:
        for (i,uni_e) in enumerate(uniformed_dataset):
            uniformed_dataset[i]['attributes']['title']['answer']=(uni_e['attributes']['title']['answer'],100)
            uniformed_dataset[i]['attributes']['keywords']['answer']=[(x,100) for x in uni_e['attributes']['keywords']['answer']]
        return uniformed_dataset
    for i,(uni_e, e) in enumerate(zip(uniformed_dataset,crossdataset)):
        answer1_with_logits=e['attributes']['title']['answer']
        uniformed_dataset[i]['attributes']['title']['answer']=choose_most_probable_answer(answer1_with_logits)
        # answer2_with_logits=e['attributes']['keywords']['answer']
        # uniformed_dataset[i]['attributes']['keywords']['answer']=choose_most_probable_answer(answer2_with_logits)
        # import pdb 
        # pdb.set_trace()
        key_words=dict()
        if isinstance(e['attributes']['keywords']['answer'], tuple):
            key_words=[(x,100) for x in e['attributes']['keywords']['answer'][0]]
        else:
            for (keywords,logit) in e['attributes']['keywords']['answer']:
                words=keywords.split(',')
                for word in words:
                    word=word.strip()
                    if word not in key_words: key_words[word]=logit
                    else: key_words[word]+=logit
            key_words=sorted(key_words.items(),key=lambda d:d[1], reverse=True)
        # import pdb 
        # pdb.set_trace()
        # print(key_words)
        uniformed_dataset[i]['attributes']['keywords']['answer']=key_words[:4]
    # import pdb
    # pdb.set_trace()
    return uniformed_dataset

def filter_sentiment(args,task_name,uniformed_dataset,crossdataset):
    ###############################################################################
    # filter examples of the crossdataset and then merge it with the uniformed_dataset with the most probable answer.
    ###############################################################################
    if crossdataset is None:
        for (i,uni_e) in enumerate(uniformed_dataset):
            uniformed_dataset[i]['attributes']['sentiment_2']['answer']=(uni_e['attributes']['sentiment_2']['answer'],100)
            uniformed_dataset[i]['attributes']['sentiment_5']['answer']=(uni_e['attributes']['sentiment_5']['answer'],100)
        return uniformed_dataset
    for i,(uni_e, e) in enumerate(zip(uniformed_dataset,crossdataset)):
        answer1_with_logits=e['attributes']['sentiment_2']['answer']
        abandon_sentiment_ids=[11]
        answer1_with_logits=abandon_prompts(answer1_with_logits,abandon_sentiment_ids,only_logits=True)
        answer2_with_logits=e['attributes']['sentiment_5']['answer']
        # uniformed_dataset[i]['attributes']['sentiment_2']['answer']=choose_most_probable_answer(logits=answer1_with_logits)
        # uniformed_dataset[i]['attributes']['sentiment_5']['answer']=choose_most_probable_answer(logits=answer2_with_logits)
        uniformed_dataset[i]['attributes']['sentiment_2']['answer']=ensemble_logits(logits=answer1_with_logits)
        sentiment_5_ans,sentiment_5_logit=ensemble_logits(logits=answer2_with_logits)
        uniformed_dataset[i]['attributes']['sentiment_5']['answer']=(sentiment_5_ans+1,sentiment_5_logit)
    return uniformed_dataset

def filter_domain(args,task_name,uniformed_dataset,crossdataset):
    ###############################################################################
    # filter examples of the crossdataset and then merge it with the uniformed_dataset with the most probable answer.
    ###############################################################################
    if crossdataset is None:
        for (i,uni_e) in enumerate(uniformed_dataset):
            uniformed_dataset[i]['attributes']['topic1']['answer']=(uni_e['attributes']['topic1']['answer'],100)
            uniformed_dataset[i]['attributes']['topic2']['answer']=(uni_e['attributes']['topic2']['answer'],100)
        return uniformed_dataset
    for i,(uni_e, e) in enumerate(zip(uniformed_dataset,crossdataset)):
        answer1_with_logits=e['attributes']['topic1']['answer']
        if isinstance(answer1_with_logits,str):
            uniformed_dataset[i]['attributes']['topic1']['answer']=(answer1_with_logits,100)
        else:  
            # label,logit=choose_most_probable_answer(logits=answer1_with_logits)
            label,logit=ensemble_logits(logits=answer1_with_logits)
            uniformed_dataset[i]['attributes']['topic1']['answer']=(uniformed_dataset[i]['attributes']["topic1"]["candidates"][label],logit)
        
        answer2_with_logits=e['attributes']['topic2']['answer']
        if isinstance(answer2_with_logits,str):
            uniformed_dataset[i]['attributes']['topic2']['answer']=(answer2_with_logits,100)
        else:
            # label,logit=choose_most_probable_answer(logits=answer2_with_logits)
            label,logit=ensemble_logits(logits=answer2_with_logits)
            # print(uniformed_dataset[i]['attributes']["topic2"]["candidates"])
            uniformed_dataset[i]['attributes']['topic2']['answer']=(uniformed_dataset[i]['attributes']["topic2"]["candidates"][label],logit)
    return uniformed_dataset


def build_uniformed_dataset_from_crossdatasets(args,task_name):
    target_datas=['summary','qa','paraphrase','title_keys','sentiment','domain']
    filter_funcs={'summary':filter_summary,'qa':filter_qa,'paraphrase':filter_paraphrase,'title_keys':filter_title_keys,
        'sentiment':filter_sentiment,'domain':filter_domain}
    # uniformed_dataset=convert_dataset_uniform.get_uniformed_dataset(args,task_name)
    uniformed_dataset=torch.load(os.path.join(args.multi_cache_dir,'jing_crossed_dataset_unfiltered/uniformed_dataset',task_name,'fold{}of{}.pt'.format(args.fold_id,args.k_fold)))
    if task_name in ['glue/qqp','kilt_tasks/hotpotqa','wiki_qa','cos_e/v1.11','quarel','sciq','trec']:
        return uniformed_dataset
    for target_data in target_datas:
        print(target_data)
        filter_func=filter_funcs[target_data]
        # import pdb
        # pdb.set_trace()
        crossdataset_path=os.path.join(args.multi_cache_dir,'jing_crossed_dataset_unfiltered/{}'.format(target_data),task_name,'fold{}of{}.pt'.format(args.fold_id,args.k_fold))
        if os.path.exists(crossdataset_path)==False: 
            crossdataset=None
        else:
            crossdataset=torch.load(crossdataset_path)
            assert len(uniformed_dataset)==len(crossdataset)
        uniformed_dataset=filter_func(args,task_name,uniformed_dataset,crossdataset)
    # import pdb 
    # pdb.set_trace()
    print(uniformed_dataset[0])
    print(uniformed_dataset[1])
    print(uniformed_dataset[2])
    print(uniformed_dataset[3])
    print(uniformed_dataset[4])
    return uniformed_dataset

def build_uniformed_dataset_from_crossdatasets_0527(args,task_name):
    target_datas=['summary','qa','paraphrase','title_keys','sentiment','domain']
    filter_funcs={'summary':filter_summary,'qa':filter_qa,'paraphrase':filter_paraphrase,'title_keys':filter_title_keys,
        'sentiment':filter_sentiment,'domain':filter_domain}
    # uniformed_dataset=convert_dataset_uniform.get_uniformed_dataset(args,task_name)
    uniformed_dataset=torch.load(os.path.join(args.multi_cache_dir,'jing_crossed_dataset_unfiltered/uniformed_dataset',task_name,'fold{}of{}.pt'.format(args.fold_id,args.k_fold)))
    if task_name in ['glue/qqp','kilt_tasks/hotpotqa','wiki_qa','cos_e/v1.11','quarel','sciq','trec']:
        return uniformed_dataset
    for target_data in target_datas:
        print(target_data)
        filter_func=filter_funcs[target_data]
        # import pdb
        # pdb.set_trace()
        if target_data=='qa':
            crossdataset_path=os.path.join(args.multi_cache_dir,'jing_crossed_dataset_unfiltered/{}'.format('qa_0527_logits'),task_name,'fold{}of{}.pt'.format(args.fold_id,args.k_fold))
            # if os.path.exists(crossdataset_path)==False:
            #     crossdataset_path=os.path.join(args.multi_cache_dir,'jing_crossed_dataset_unfiltered/{}'.format(target_data),task_name,'fold{}of{}.pt'.format(args.fold_id,args.k_fold))
        else:
            crossdataset_path=os.path.join(args.multi_cache_dir,'jing_crossed_dataset_unfiltered/{}'.format(target_data),task_name,'fold{}of{}.pt'.format(args.fold_id,args.k_fold))
        if os.path.exists(crossdataset_path)==False: 
            crossdataset=None
        else:
            crossdataset=torch.load(crossdataset_path)
            assert len(uniformed_dataset)==len(crossdataset)
        uniformed_dataset=filter_func(args,task_name,uniformed_dataset,crossdataset)
    # import pdb 
    # pdb.set_trace()
    print(uniformed_dataset[0])
    print(uniformed_dataset[1])
    print(uniformed_dataset[2])
    print(uniformed_dataset[3])
    print(uniformed_dataset[4])
    return uniformed_dataset

if __name__ == '__main__': # only used for testing the functions in this file
    from universal_da.dataset_config import default_T0_tasks
    from SwissArmyTransformer.training.deepspeed_training import initialize_distributed, set_random_seed
    # from arguments import get_args
    # args,_=get_args()
    import argparse
    from arguments import get_args
    py_parser = argparse.ArgumentParser(add_help=False)
    T5Model.add_model_specific_args(py_parser)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    # import pdb 
    # pdb.set_trace()
    torch.backends.cudnn.enabled = False
    task_names=sum([names for (_,names) in default_T0_tasks.items()],[])
    set_random_seed(args.seed)
    for (i,task_name) in enumerate(task_names):
        # uniformed_dataset=build_uniformed_dataset_from_crossdatasets_0527(args,task_name)
        # if task_name.startswith('duorc')==False: continue
        # if task_name!='wiqa': continue
        # if task_name!='multi_news': continue
        # if task_name!='dbpedia_14': continue
        # if task_name!='quartz': continue
        # if task_name!='ropes': continue
        # if task_name!='qasc': continue
        # if task_name!='dream': continue
        # if task_name!='app_reviews': continue
        print(i,task_name)
        # uniformed_dataset=build_uniformed_dataset_from_crossdatasets(args,task_name)
        uniformed_dataset=build_uniformed_dataset_from_crossdatasets_0527(args,task_name)
        print("#####################################################################")
        print("#####################################################################")
        print(uniformed_dataset[0]['para1'])
        print(uniformed_dataset[0]['questions'],uniformed_dataset[0]['answers'])
        print('title',uniformed_dataset[0]['attributes']['title'])
        print('keywords',uniformed_dataset[0]['attributes']['keywords'])
        print('sentiment_2',uniformed_dataset[0]['attributes']['sentiment_2'])
        print('sentiment_5',uniformed_dataset[0]['attributes']['sentiment_5'])
        print('topic1',uniformed_dataset[0]['attributes']['topic1'])
        print('topic2',uniformed_dataset[0]['attributes']['topic2'])
        # save_path=os.path.join(args.multi_cache_dir,'jing_crossed_dataset_unfiltered/uniformed_dataset',task_name)
        # if os.path.exists(os.path.join(save_path,'fold{}of2.pt'.format(args.fold_id)))==False:
        #     raise NotImplementedError
        #     uniformed_dataset=convert_dataset_uniform.get_uniformed_dataset(args,task_name)
        #     if os.path.exists(save_path)==False: os.makedirs(save_path)
        #     torch.save(uniformed_dataset,os.path.join(save_path,'fold{}of2.pt'.format(args.fold_id)))
        # else:
        #     uniformed_dataset=torch.load(os.path.join(save_path,'fold{}of2.pt'.format(args.fold_id)))
        # crossdataset_path=os.path.join(args.multi_cache_dir,'jing_crossed_dataset_unfiltered/qa_0527_logits',task_name,'fold{}of2.pt'.format(args.fold_id))
        # # if os.path.exists(crossdataset_path)==False: continue
        # # crossdataset=torch.load(crossdataset_path)
        # crossdataset=None
        # new_uniformed_dataset=filter_qa(args,task_name,uniformed_dataset,crossdataset)
        import pdb 
        pdb.set_trace()
        # crossdataset=torch.load(crossdataset_path)
    #     # # new_uniformed_dataset=filter_summary(args,task_name,uniformed_dataset,crossdataset)
    #     # # crossdataset=torch.load(os.path.join(args.multi_cache_dir,'jing_crossed_dataset_unfiltered/sentiment',task_name,'fold1of2.pt'))
    #     # # new_uniformed_dataset=filter_sentiment(args,task_name,uniformed_dataset,crossdataset)
    #     # new_uniformed_dataset=filter_domain(args,task_name,uniformed_dataset,crossdataset)
    #     # import pdb 
    #     # pdb.set_trace()

'''
python -m universal_da.simple_cross.filter_cross_data \
--load ../../huggingface_models/t5-large-lm-adapt \
--multi-cache-dir ../../huggingface_datasets \
--eval-batch-size 128 \
--multi-src-seq-length 1024 \
--multi-tgt-seq-length 256 \
--test-ckpt /share/zongyu/zhoujing/T0-Multi-Task_zj-adapt/checkpoints_genmodel/mt-t5-lm-large-T0_adam_0.0001_default-T0-tasks_trisoall_validisorand_1/1500 \
--deepspeed \
--deepspeed_config "config/t5_lm/config_t5_lm_large.json" \
--save test \
--no-deepspeed-load \
--no-load-optim \
--no-load-lr-scheduler \
--no-load-rng \
--no-load-iteration \
--bert-mask-ratio 0.15 \
--avg-block-length 3 \
--experiment-name test \
--model-parallel-size 1 \
--t5-model \
--vocab-size 32128 \
--num-layers 24 \
--hidden-size 1024 \
--inner-hidden-size 2816 \
--num-attention-heads 16 \
--hidden-size-per-attention-head 64 \
--relative-attention-num-buckets 32 \
--no-share-embeddings \
--gated-gelu-mlp \
--layernorm-epsilon 1e-6 \
--init-method-std 1.0 \
--seq-length 512 \
--shuffle \
--tokenizer-type hf_T5Tokenizer \
--tokenizer-model-type "/share/zongyu/huggingface_models/t5-large-lm-adapt" \
--k_fold 2 \
--fold_id 1 \
--debug
'''