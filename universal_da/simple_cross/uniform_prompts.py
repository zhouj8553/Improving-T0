import copy
from promptsource.templates import Template
uniformed_prompt_templates={
    'generate_paraphrased_sentence':{
        'generate_paraphrase':Template(name='generate_paraphrase',jinja="{% if para1_meta['paraphrase_label'][0] == 1 %}\nParaphrase the following sentence: {{para1}}\n|||\n{{para1_meta['similar_sentence'][0]}}\n{% endif %}",reference=''),
        'generate_sentence':Template(name='generate_sentence',jinja="{% if para1_meta['paraphrase_label'][0] == 1 %}\nGenerate a sentence that means the same thing as this one: {{para1}}\n|||\n{{para1_meta['similar_sentence'][0]}}\n{% endif %}",reference=''),
        'paraphrase-task':Template(name='paraphrase-task',jinja='{% if para1_meta["paraphrase_label"][0] == 1 %}\nParaphrase the sentence: {{para1}} \n||| \n{{para1_meta["similar_sentence"][0]}} \n{% endif %}',answer_choices=None,reference=''),
    },
    'paragraph_question_tf':{
        'want to know':Template(name='want to know',jinja="I want to know whether the following two sentences mean the same thing.\n{{para1}}\n{{para1_meta['similar_sentence'][0]}}\nDo they?\n|||\n{{answer_choices[para1_meta['paraphrase_label'][0]]}}",answer_choices='no ||| yes',reference=''),
        'paraphrase':Template(name='paraphrase',jinja="Does the sentence\n{{para1}}\nparaphrase (that is, mean the same thing as) this sentence?\n{{para1_meta['similar_sentence'][0]}}\n|||\n{{answer_choices[para1_meta['paraphrase_label'][0]]}}",answer_choices='no ||| yes',reference=''),
        'equivalent':Template(name='equivalent',jinja="Are the following two sentences \"{{\"equivalent\"}}\" or \"{{\"not equivalent\"}}\"?\n{{para1}}\n{{para1_meta['similar_sentence'][0]}}\n|||\n{{answer_choices[para1_meta['paraphrase_label'][0]]}}",answer_choices='not equivalent ||| equivalent',reference=''),
        'replace': Template(name='replace',jinja="Can I replace the sentence\n{{para1}}\nwith the sentence\n{{para1_meta['similar_sentence'][0]}}\nand have it mean the same thing?\n|||\n{{answer_choices[para1_meta['paraphrase_label'][0]]}}", answer_choices='no ||| yes',reference=''),
        'same thing':Template(name='same thing',jinja="Do the following two sentences mean the same thing?\n{{para1}}\n{{para1_meta['similar_sentence'][0]}}\n|||\n{{answer_choices[para1_meta['paraphrase_label'][0]]}}",answer_choices='no ||| yes',reference=''),
        'task_description-no-label':Template(name='task_description-no-label',jinja='Determine if the following two sentences paraphrase each other or not.\nSent 1: {{para1}}\nSent 2: {{para1_meta["similar_sentence"][0]}}\n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices='No ||| Yes',reference=''),
        'Meaning':Template(name='Meaning',jinja='Sentence 1: {{para1}}\nSentence 2: {{para1_meta["similar_sentence"][0]}}\nQuestion: Do Sentence 1 and Sentence 2 express the same meaning? Yes or No? \n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}', answer_choices="No ||| Yes",reference=''),
        'context-question-no-label':Template(name='context-question-no-label',jinja='{{para1}}\nIs that a paraphrase of the following sentence?\n{{para1_meta["similar_sentence"][0]}}?\n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices='No ||| Yes',reference=''),
        'Rewrite-no-label':Template(name='Rewrite-no-label',jinja='Sentence 1: {{para1}}\nSentence 2: {{para1_meta["similar_sentence"][0]}}\nQuestion: Can we rewrite Sentence 1 to Sentence 2? \n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices="No ||| Yes",reference=''),
        'context-question':Template(name='context-question',jinja='{{para1}}\nIs that a paraphrase of the following sentence?\n{{para1_meta["similar_sentence"][0]}}?\nYes or No.\n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices='No ||| Yes',reference=''),
        'Concatenation':Template(name='Concatenation',jinja='Sentence 1: {{para1}}\nSentence 2: {{para1_meta["similar_sentence"][0]}}\nQuestion: Does Sentence 1 paraphrase Sentence 2? Yes or No? \n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices='No ||| Yes',reference=''),
        'Concatenation-no-label':Template(name='Concatenation-no-label',jinja='Sentence 1: {{para1}}\nSentence 2: {{para1_meta["similar_sentence"][0]}}\nQuestion: Does Sentence 1 paraphrase Sentence 2? \n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices='No ||| Yes',reference=''),
        'Meaning-no-label':Template(name='Meaning-no-label',jinja='Sentence 1: {{para1}}\nSentence 2: {{para1_meta["similar_sentence"][0]}}\nQuestion: Do Sentence 1 and Sentence 2 express the same meaning? \n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices='No ||| Yes',reference=''),
        'PAWS-ANLI GPT3':Template(name='PAWS-ANLI GPT3',jinja='{{para1}} Question: {{para1_meta["similar_sentence"][0]}} True or False? \n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices='False ||| True',reference=''),
        'Rewrite':Template(name='Rewrite',jinja='Sentence 1: {{para1}}\nSentence 2: {{para1_meta["similar_sentence"][0]}}\nQuestion: Can we rewrite Sentence 1 to Sentence 2? Yes or No? \n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices='No ||| Yes',reference=''),
        'PAWS-ANLI GPT3-no-label':Template(name='PAWS-ANLI GPT3-no-label',jinja='{{para1}} Question: {{para1_meta["similar_sentence"][0]}} Paraphrase or not?\n||| \n{{answer_choices[para1_meta["paraphrase_label"][0]]}}',answer_choices='No ||| Yes',reference=''),
    },
    'question_paraphrase_tf':{
        'quora':Template(name='quora',jinja="I'm an administrator on the website Quora. There are two posts, one that asks \"{{question}}\" and another that asks \"{{questions_meta['similar_sentence'][0]}}\". I can merge questions if they are asking the same thing. Can I merge these two questions? ||| {{answer_choices[questions_meta['paraphrase_label'][0]]}}",answer_choices='no ||| yes',reference=''),
        'duplicate or not':Template(name='duplicate or not',jinja='{{question}}\n{{questions_meta["similar_sentence"][0]}}\nPick one: These questions are "{{"duplicates"}}" or "{{"not duplicates"}}".\n|||\n{{answer_choices[questions_meta["paraphrase_label"][0]]}}',answer_choices='not duplicates ||| duplicates',reference=''),
        'same thing':Template(name='same thing',jinja='Are the questions "{{question}}" and "{{questions_meta["similar_sentence"][0]}}" asking the same thing? ||| {{answer_choices[questions_meta["paraphrase_label"][0]]}}',answer_choices='no ||| yes',reference=''),
        'answer':Template(name='answer',jinja='Can an answer to "{{question}}" also be used to answer "{{questions_meta["similar_sentence"][0]}}"? ||| {{answer_choices[questions_meta["paraphrase_label"][0]]}}',answer_choices='no ||| yes',reference=''),
        'meaning':Template(name='meaning',jinja='Question 1: {{question}}\nQuestion 2: {{questions_meta["similar_sentence"][0]}}\n\nDo these two questions convey the same meaning? Yes or no? ||| {{answer_choices[questions_meta["paraphrase_label"][0]]}}',answer_choices='No ||| Yes',reference=''),
        'duplicate':Template(name='duplicate',jinja='I received the questions "{{question}}" and "{{questions_meta["similar_sentence"][0]}}". Are they duplicates? ||| {{answer_choices[questions_meta["paraphrase_label"][0]]}}',answer_choices='no ||| yes',reference=''),
    },
    'question_to_answer':{
        'complex_question':Template(name='complex_question',jinja='Here\'s a complex question that requires someone to reason about the input, can you answer it?\n{{question}}\n|||\n{{answer}}\n',answer_choices=None,reference=''),
        'combining_facts':Template(name='combining_facts',jinja='Combine facts and answer this: {{question}}\n|||\n{{answer}}\n',answer_choices=None,reference=''),
        'formulate':Template(name='formulate',jinja='Formulate an answer to this elaborate question: {{question}}\n|||\n{{answer}}\n',answer_choices=None,reference=''),
        'final_exam':Template(name='final_exam',jinja='FINAL EXAM\n\nQuestion 1. {{question}}\n|||\n{{answer}}\n',answer_choices=None,reference=''),
        'straighforward_qa':Template(name='straighforward_qa',jinja='{{question}}\n|||\n{{answer}}\n',answer_choices=None,reference=''),
        'Direct Answer to Question':Template(name='Direct Answer to Question',jinja='Answer this question: {{question}}?|||\n{{answer}}\n',answer_choices=None,reference=''),
        'Direct Question (Closed Book)':Template(name='Direct Question (Closed Book)',jinja='Q: {{question}}\n\n\nA:|||{{answer}}',answer_choices=None,reference=''),
        # begin cosmos_qa
        'only_question_answer':Template(name='only_question_answer',jinja='{{question}} \n|||\n{{answer}}',answer_choices=None,reference=''),
        # end cosmos_qa
    },
    'question_to_answer_tf':{
        'Is This True?':Template(name='Is This True?',jinja='Question: {{question}}?\nWould "{{answer}}" be a reasonable answer? |||\n{{answer_choices[answer_label]}}',answer_choices='No ||| Yes',reference=''),
        'automatic_system':Template(name='automatic_system',jinja='I am verifying the answers generated by an automatic system to the following question: {{question}}\nSuggested answer: {{answer}}\nShould I validate this answer?\n|||\n{{answer_choices[answer_label]}}',answer_choices='No ||| Yes',reference=''),
        'found_on_google':Template(name='found_on_google',jinja='Question: {{question}}\nI found the following answer on Google: {{answer}}\nIs that a correct answer? Yes or no.\n|||\n{{answer_choices[answer_label]}}',answer_choices='No ||| Yes',reference=''),
        'exercise':Template(name='exercise',jinja='The exercise is to decide whether the question accepts the proposed suggestion as a correct answer. If yes, write "{{answer_choices[1]}}", otherwise write "{{answer_choices[0]}}".\nQuestion: {{question}}\nSuggestion: {{answer}}\n|||\n{{answer_choices[answer_label]}}',answer_choices='False ||| True',reference=''), #jing
    },
    'title_question_to_answer_tf':{
        'Decide_good_answer':Template(name='Decide_good_answer',jinja='This is a correct answer to the following question about {{attributes["title"]["answer"]}}. Yes or no?\nAnswer: {{answer}}\nQuestion: {{question}}\n|||\n{{answer_choices[answer_label]}}',answer_choices='No ||| Yes',reference=''),
    },
    'paragraph_hints_question_to_answer':{
        # begin ropes
        'prompt_beginning':Template(name='prompt_beginning',jinja='{% if answer != "no answer" %}\nPlease answer correctly the following question related to the paragraph below. \n\n{{question}}\n\n{{para1}}\n\nHint: {{attributes["hints"][0]}}\n|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'prompt_bottom_hint_beginning': Template(name='prompt_bottom_hint_beginning',jinja="{% if answer!='no answer' %}\nBackground: {{attributes['hints'][0]}}\n\nParagraph: {{para1}}\n\nGiven the paragraph above, please answer correctly the following question: {{question}}\n|||\n{{answer}}\n{% endif %}",answer_choices=None,reference=''),
        'given_background_situation':Template(name='given_background_situation',jinja='{% if answer!="no answer" %}\nGiven the background: {{attributes["hints"][0]}}\n\nand the situation: {{para1}}\n\nAnswer the following question: {{question}}|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'plain_bottom_hint': Template(name='plain_bottom_hint',jinja="{% if answer!='no answer' %}\n{{para1}}\n\n{{question}}\n\nHint: {{attributes['hints'][0]}}\n|||\n{{answer}}\n{% endif %}",reference=''),
        'plain_background_situation':Template(name='plain_background_situation',jinja='{% if answer!="no answer" %}\n{{attributes["hints"][0]}}\n\n{{para1}}\n\n{{question}}\n|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'background_new_situation_answer':Template(name='background_new_situation_answer',jinja='{% if answer!="no answer" %}\nI can use this background: {{attributes["hints"][0]}}\n\nNow, I have a new situation: {{para1}}\n\nAnswer this question please: {{question}}|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'background_situation_middle':Template(name='background_situation_middle',jinja='{% if answer!="no answer" %}\nYou are given a new situation: {{para1}}\n\nand a hint : {{attributes["hints"][0]}}\n\nPlease answer this question : {{question}}|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'new_situation_background_answer':Template(name='new_situation_background_answer',jinja='{% if answer!="no answer" %}\nI have a new situation: {{para1}}\n\nBut I can use this background: {{attributes["hints"][0]}}\n\nWhat is an answer for this question: {{question}}|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'prompt_mix':Template(name='prompt_mix',jinja='{% if answer!="no answer" %}\n{{para1}}\n\nGiven the paragraph above, please answer correctly the following question: \n\n{{question}}\n\nHint: {{attributes["hints"][0]}}\n|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'read_background_situation':Template(name='read_background_situation',jinja='{% if answer!="no answer" %}\nI read this background article the other day: {{attributes["hints"][0]}}\n\nI am facing a new situation today: {{para1}}\n\nUsing the knowledge I acquired from the background article, how should I answer correctly the following question regarding my new situation: {{question}}|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        # end ropes
    },
    'paragraph_question_to_answer':{
        'based_on':Template(name='based_on',jinja='Extract the answer to the question from the following context.\nQuestion: {{question}}\nContext: {{para1}}|||\n{{answer}}',answer_choices=None,reference=''),
        'answer_the_following_q':Template(name='answer_the_following_q',jinja='Given the following passage\n\n"{{para1}}",\n\nanswer the following question. Note that the answer is present within the text.\n\nQuestion: {{question}} |||\n{{answer}}',answer_choices=None,reference=''),
        'tell_what_it_is':Template(name='tell_what_it_is',jinja='I know that the answer to the question "{{question}}" is in "{{para1}}". Can you tell me what it is? |||\n\n{{answer}}',answer_choices=None,reference=''),
        'question_context_answer':Template(name='question_context_answer',jinja='Question: "{{question}}"\n\nContext: "{{para1}}"\n\nAnswer:\n|||\n{{answer}}',answer_choices=None,reference=''),
        # begin ropes
        'prompt_bottom_no_hint':Template(name='prompt_bottom_no_hint',jinja='{% if answer!="no answer" %}\n{{para1}}\n\nGiven the paragraph above, please answer correctly the following question: \n\n{{question}}\n|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'plain_no_background':Template(name='plain_no_background',jinja='{% if answer!="no answer" %}\n{{para1}}\n\n{{question}}\n|||\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        # end ropes, begin quoref
        'Guess Answer':Template(name='Guess Answer',jinja='The answer to the question: {{question}} is inside the article: {{para1}}, can you guess it ?\n\n|||\n{{answer}}\n',answer_choices=None,reference=''),
        'Answer Question Given Context':Template(name='Answer Question Given Context',jinja='Given the following context:\n\n{{para1}}\n\nanswer the following question:\n\n{{question}} |||\n{{answer}}',answer_choices=None,reference=''),
        'Find Answer':Template(name='Find Answer',jinja='The following article contains an answer for the question: {{question}} , can you please find it? \n\n{{para1}}|||\n{{answer}}',answer_choices=None,reference=''),
        'Context Contains Answer':Template(name='Context Contains Answer',jinja='This article: {{para1}} contains an answer for the question: {{question}}, what is it ?\n|||\n{{answer}}',answer_choices=None,reference=''),
        'Given Context Answer Question':Template(name='Given Context Answer Question',jinja='{{question}}\n\nAnswer the above question based on the context below:\n\n{{para1}} |||\n{{answer}}',answer_choices=None,reference=''),
        'What Is The Answer':Template(name='What Is The Answer',jinja='What is the answer for the question: {{question}} from the following article ?\n\n{{para1}}|||\n{{answer}}',answer_choices=None,reference=''),
        'Answer Test':Template(name='Answer Test',jinja='I have a test where I am given the following article, what is an answer for the question: {{question}} ?\n\n{{para1}}|||\n{{answer}}',answer_choices=None,reference=''),
        'Found Context Online':Template(name='Found Context Online',jinja='Found the following article online, use it to answer the question: {{question}}\n\n{{para1}}|||\n{{answer}}',answer_choices=None,reference=''),
        'Answer Friend Question':Template(name='Answer Friend Question',jinja='A friend asked me to answer this question: {{question}}, using the article: {{para1}}, what would be the answer ?\n\n|||\n{{answer}}',answer_choices=None,reference=''),
        'Read And Extract ':Template(name='Read And Extract ',jinja='Read the following paragraph and extract the answer for the question: {{question}}\n\n{{para1}} |||\n{{answer}}',answer_choices=None,reference=''),
        # end quoref, begin social_i_qa
        'I was wondering':Template(name='I was wondering',jinja='I heard that {{para1}}\n\nAnd I was wondering {{question}}\n\n|||\n\n{{answer}}',answer_choices=None,reference=''),
        'Generate answer':Template(name='Generate answer',jinja='{{para1}}\n\nGiven the context: {{question}}\n\n|||\n\n{{answer}}',answer_choices=None,reference=''),
        # end social_i_qa
        'Direct Question':Template(name='Direct Question',jinja='Answer the following question given this paragraph: \n\n{{para1}}\n\n\nQ: {{question}}\n\n\nA:|||{{answer}}\n',answer_choices=None,reference=''),
        # begin cosmos_qa
        'description_context_question_text':Template(name='description_context_question_text',jinja='Read the following context and answer the question.\nContext: {{para1}}\nQuestion: {{question}}\nAnswer:\n|||\n{{answer}}',answer_choices=None,reference=''),
        'context_description_question_text':Template(name='context_description_question_text',jinja='{{para1}}\nAccording to the above context, answer the following question.\n{{question}}\n|||\n{{answer}}',answer_choices=None,reference=''),
        'context_question_description_text':Template(name='context_question_description_text',jinja='{{para1}}\nQuestion: {{question}}\nThe answer to the above question:\n|||\n{{answer}}',answer_choices=None,reference=''),
        # end cosmos_qa, begin quartz
        'use_info_from_question_paragraph':Template(name='use_info_from_question_paragraph',jinja='Use information from the paragraph to answer the question.\n\nQuestion:\n\n\n{{question}} \n\n\nParagraph :\n\n{{para1}}\n|||\n{{answer}}',answer_choices=None,reference=''),
        'paragraph_question_plain_concat':Template(name='paragraph_question_plain_concat',jinja='{{para1}}\n\n{{question}}\n\n|||\n{{answer}}',answer_choices=None,reference=''),
        'use_info_from_paragraph_question':Template(name='use_info_from_paragraph_question',jinja='Use information from the paragraph to answer the question.\n\nParagraph :\n\n{{para1}}\n\nQuestion:\n\n\n{{question}}\n\n|||\n{{answer}}',answer_choices=None,reference=''),
        'answer_question_based_on':Template(name='answer_question_based_on',jinja='Answer the question based on the following text.\n\nQuestion:\n\n\n{{question}} \n\n\nText:\n\n{{para1}}|||\n{{answer}}',answer_choices=None,reference=''),
        'answer_question_below':Template(name='answer_question_below',jinja='Answer the question below:\n\n\n{{question}} \n\n\nAssuming that:\n\n{{para1}}|||\n{{answer}}',answer_choices=None,reference=''),
        'given_the_fact_answer_the_q':Template(name='given_the_fact_answer_the_q',jinja='Given the fact that:\n\n{{para1}}\n\nAnswer the question:\n\n\n{{question}}\n\n\n|||\n{{answer}}',answer_choices=None,reference=''),
        # end quartz, begin wiki_hop/original
        'explain_relation':Template(name='explain_relation',jinja='{% set supports= para1.split("\n\n") | reject("equalto", "") | list %}Information:\n{% for support in supports %}\n- {{ support }}\n{% endfor %}\n\n{{question}}\n\n|||\n{{answer}}',answer_choices=None,reference=''),
        'generate_subject':Template(name='generate_subject',jinja='{% set supports= para1.split("\n\n") | reject("equalto", "") | list %}Information:\n{% for support in supports %}\n- {{ support }}\n{% endfor %}\n\n\nGiven the paragraphs above, decide {{question}}.\n\n|||\n{{answer}}',answer_choices=None,reference=''),
        'generate_subject_and_object':Template(name='generate_subject_and_object',jinja='{% set supports= para1.split("\n\n") | reject("equalto", "") | list %}Information:\n{% for support in supports %}\n- {{ support }}\n{% endfor %}\nGiven the information, {{question}}\n\n|||\n{{answer}}',reference=""),
        # end wiki_hop/original
    },
    'paragraph_question_to_answer_tf':{
        # begin social_i_qa
        'Check if a random answer is valid or not':Template(name='Check if a random answer is valid or not',jinja='{{para1}}\n\nGiven the question "{{question}}", is "{{answer}}" a valid answer?\n\n|||\n\n{{answer_choices[answer_label]}}',answer_choices='No ||| Yes',reference=''),
        # end social_i_qa, begin qasc
        'is_correct_1':Template(name='is_correct_1',jinja='If I tell you that {{para1[0]|capitalize}}{{para1[1:]|trim(\'.\')}}, and ask you the question "{{question[0]|lower}}{{question[1:]}}", is the correct answer "{{answer}}"? \n\n||| \n\n{{answer_choices[answer_label]}}',answer_choices='No ||| Yes',reference=''),
        'is_correct_2':Template(name='is_correct_2',jinja='Do you think the right answer to the question "{{question[0]|lower}}{{question[1:]}}" is "{{answer}}", given that\n {{para1[0]|lower}}{{para1[1:]|trim(\'.\')}}?\n ||| \n\n{{answer_choices[answer_label]}}',answer_choices='No ||| Yes',reference=''),
        # end qasc
    },

    'paragraph_question_title_to_answer':{
        'movie_director':Template(name='movie_director',jinja='I am a movie director and I just received the following movie plot. Could you help me answer this question? If not, let me know by writing "{{"Not answerable"}}".\n\nPlot title: {{attributes["title"]["answer"]}}\nMovie plot: {{para1}}\nMy question: {{question}}\n|||\n{% if answer=="no answer" %}\nNot answerable\n{% else %}\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'extract_answer':Template(name='extract_answer',jinja='Extract the answer to the following question from the movie plot. If the question isn\'t answerable, please output "{{"Can\'t answer"}}".\nQuestion: {{question}}\nTitle: {{attributes["title"]["answer"]}}\nMovie plot: {{para1}}\n|||\n{% if answer=="no answer" %}\nCan\'t answer\n{% else %}\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'answer_question':Template(name='answer_question',jinja='Please answer the following question about this movie plot. If it\'s un-answerable, please output "{{"No answer"}}".\n\nQuestion: {{question}}\nMovie plot title: {{attributes["title"]["answer"]}}\nMovie plot: {{para1}}\n|||\n{% if answer=="no answer" %}\nNo answer\n{% else %}\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'question_answering':Template(name='question_answering',jinja='Question: {{question}}\nIf there is no answer, please output "{{"Insufficient information to provide an answer."}}".\nMovie title: {{attributes["title"]["answer"]}}\nContext: {{para1}}\n|||\n{% if answer=="no answer" %}\nInsufficient information to provide an answer.\n{% else %}\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
        'decide_worth_it':Template(name='decide_worth_it',jinja='I am trying to decide whether it\'s worth it to invest in this film proposal. Can you help me answer a few questions? If you can\'t, please say "{{"No I can\'t"}}".\n\nQuestion: {{question}}\nMovie title: {{attributes["title"]["answer"]}}\nMovie plot: {{para1}}\n|||\n{% if answer=="no answer" %}\nNo I can\'t\n{% else %}\n{{answer}}\n{% endif %}',answer_choices=None,reference=''),
    },

##############################################################################################################
    'answer_title_to_question':{
        'Jeopardy style':Template(name='Jeopardy style',jinja='What is the question to: "{{answer}}"? The topic is {{attributes["title"]["answer"]}}.|||\n"{{question}}?"',answer_choices=None,reference=''),
        'Generate Question from Topic':Template(name='Generate Question from Topic',jinja='Generate a question about the topic "{{attributes["title"]["answer"]}}" whose answer would be: {{answer}}.|||\n{{question}}?\n',answer_choices=None,reference=''),
    },
    'paragraph_to_question':{
        'generate_question1':Template(name='generate_question',jinja='I want to test the ability of students to read a passage and answer questions about it. Could you please come up with a good question for the passage "{{para1}}"? |||\n{{question}}',answer_choices=None,reference=''),
        'generate_question2':Template(name='generate_question',jinja='Generate a question about the following movie plot: {{para1}}\n|||\n{{question}}',answer_choices=None,reference=''),
    },
    'paragraph_answer_to_question':{
        # begin duorc
        'generate_question_by_answer':Template(name='generate_question_by_answer',jinja='{% if answer!="no answer" %}Generate a question that has the following answer: \n{{answer}} \nfor the following movie plot: \n{{para1}}\n|||\n{{question}}{% endif %}',answer_choices=None,reference=''),
        # end duorc, begin cosmos_qa
        'context_answer_to_question':Template(name='context_answer_to_question',jinja='Based on the context and the answer, generate a question. \n\nContext: {{para1}}\n\nAnswer:\n\n{{answer}}\n|||\n{{question}}',answer_choices=None,reference=''),
        # end cosmos_qa, begin social_i_qa
        'Generate the question from the answer':Template(name='Generate the question from the answer',jinja='{{para1}}\n\nGiven that the answer to a question is "{{answer}}", what is the question?\n\n|||\n\n{{question}}',answer_choices=None,reference=''),
        # end social_i_qa
    },

##############################################################################################################
    'question_answer_to_title':{
        'Topic Prediction - Question and Answer Pair':Template(name='Topic Prediction - Question and Answer Pair',jinja='Determine the topic of the question-answer pair.\nQuestion: "{{question}}?";  Answer: "{{answer}}"? Topic: |||\n{{attributes["title"]["answer"]}}\n',answer_choices=None,reference=''),
    },
    'question_to_title':{
        'Topic Prediction - Question Only':Template(name='Topic Prediction - Question Only',jinja='Determine the topic of the question.\nQuestion: "{{question}}?"\nTopic: |||\n{{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
    },
    'answer_to_title':{
        'Topic Prediction - Answer Only':Template(name='Topic Prediction - Answer Only',jinja='Determine the topic of the passage.\n"{{answer}}"\nTopic:|||\n{{attributes["title"]["answer"]}}\n',answer_choices=None,reference=''),
    },
    'paragraph_to_title':{
        'title_generation':Template(name='title_generation',jinja='Suggest a movie title for the following movie plot: {{para1}}\n|||\n{{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
        'Guess Title For Context':Template(name='Guess Title For Context',jinja='Given the below context:\n\n{{para1}}\n\nGuess a valid title for it! |||\n{{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
        # the following could also be title (from gigaword)
        'generate_summary_for_this':Template(name='generate_summary_for_this',jinja='{{para1}}\n\n===\n\nGenerate a title for this article: ||| {{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
        'make_a_title':Template(name='make_a_title',jinja='Make a title for this article: {{para1}} |||\n\n{{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
        'first_sentence_title':Template(name='first_sentence_title',jinja='First sentence of the article: {{para1}}\n\nTitle: ||| {{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
        'TLDR':Template(name='TLDR',jinja='{{para1}}\n\nTL;DR: ||| {{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
        'write_its_sentence':Template(name='write_its_sentence',jinja='{{para1}}\n\n===\n\nGiven the above sentence, write its title: ||| {{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
        'write_a_title_for_this_sentence':Template(name='write_a_title_for_this_sentence',jinja='Write a title for this sentence: {{para1}} \n\nTitle: ||| {{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
        'in_a_nutshell':Template(name='in_a_nutshell',jinja='{{para1}} In a nutshell, ||| {{attributes["title"]["answer"]}}',answer_choices=None,reference=''),
        'reverse_writing':Template(name='reverse_writing',jinja='Title: {{attributes["title"]["answer"]}}\n\n||| {{para1}}',answer_choices=None,reference=''),
        'write_an_article':Template(name='write_an_article',jinja='Title: {{attributes["title"]["answer"]}}\n\n===\n\nWrite an article with the given title: ||| {{para1}}',answer_choices=None,reference=''),
        # end gigaword
    },


    ##############################################################################################################
    'question_answer_to_paragraph':{
        'build_story_around_qa':Template(name='build_story_around_qa',jinja='{% if answer!="no answer" %}Build a movie plot around this: {{question}} {{answer}}\n|||\n{{para1}}{% endif %}',answer_choices=None,reference=''),
        'combining_facts':Template(name='combining_facts',jinja='Given the question "{{question}}" and the answer "{{answer}}", write a conversation that might have happened.\n|||\n{{para1}}',answer_choices=None,reference=''),
        'answer-to-dialogue':Template(name='answer-to-dialogue',jinja='Given the question "{{question}}" and the answer "{{answer}}", write a conversation that might have happened.\n|||\n{{para1}}',answer_choices=None,reference=''),
    },

    ##############################################################################################################

    'question_to_choose_answer':{
        'question_description_option_text':Template(name='question_description_option_text',jinja='{{question}}\nChoose the most suitable option to answer the above question.\nOptions:\n- {{answer_choices | join("\\n- ")}}\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'question_description_option_id':Template(name='question_description_option_id',jinja="{{question}}\nChoose the most suitable option to answer the above question.\nOptions：\n{% for k in range(choices | length) %}\n{{'. '.join([answer_choices[k], choices[k]])}}\n{% endfor %}\n|||\n{{answer_choices[choices.index(answer)]}}",answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=''),
        'question_option_description_text':Template(name='question_option_description_text',jinja='{{question}}\n- {{answer_choices | join("\\n- ")}}\n\nThe best answer is\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'description_question_option_id':Template(name='description_question_option_id',jinja="Pick the option in line with common sense to answer the question.\nQuestion: {{question}}\nOptions:\n{% for k in range(choices | length) %}\n{{'. '.join([answer_choices[k], choices[k]])}}\n{% endfor %}\n|||\n{{answer_choices[choices.index(answer)]}}",answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=''),
        'description_question_option_text':Template(name='description_question_option_text',jinja='Pick the option in line with common sense to answer the question.\nQuestions: {{question}}\nOptions:\n- {{answer_choices | join("\\n- ")}}\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'question_option_description_id':Template(name='question_option_description_id',jinja="{{question}}\n{% for k in range(choices | length) %}\n{{'. '.join([answer_choices[k], choices[k]])}}\n{% endfor %}\nThe best answer is\n|||\n{{answer_choices[choices.index(answer)]}}",answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=''),
        # begin quarel
        'do_not_use':Template(name='do_not_use',jinja='Question: {{question}}\n\nDo not use {{"A"}} and {{"B"}} to answer the question but instead, choose between "{{answer_choices[0]}}" and  "{{answer_choices[1]}}".\n|||\n{{answer}}',answer_choices='{{choices[0]}} ||| {{choices[1]}}',reference=''),
        'logic_test':Template(name='logic_test',jinja='Here\'s a logic test: {{question}}\n\nChoose the answer between "{{answer_choices[0]}}" and "{{answer_choices[1]}}".\n|||\n{{answer}}',answer_choices='{{choices[0]}} ||| {{choices[1]}}',reference=''),
        'heres_a_story':Template(name='heres_a_story',jinja='Here\'s a short story: {{question}}.\n\nWhat is the most sensical answer between "{{answer_choices[0]}}" and  "{{answer_choices[1]}}"?\n|||\n{{answer}}',answer_choices='{{choices[0]}} ||| {{choices[1]}}',reference=''),
        'choose_between':Template(name='choose_between',jinja='Choose between "{{answer_choices[0]}}" and  "{{answer_choices[1]}}".\nQuestion: {{question}}\n|||\n{{answer}}',answer_choices='{{choices[0]}} ||| {{choices[1]}}',reference=''),
        'testing_students':Template(name='testing_students',jinja='I am testing my students\' logic.\nWhat is the answer they should choose between "{{answer_choices[0]}}" and "{{answer_choices[1]}}"?\nLogic test: {{question}}\n|||\n{{answer}}',answer_choices='{{choices[0]}} ||| {{choices[1]}}',reference=''),
        # end quarel
        'Multiple Choice (Closed Book)':Template(name='Multiple Choice (Closed Book)',jinja='Q: {{question}}\n\n\n Choices:\n\n- {{answer_choices | join("\\n - ")}}\n\nA:|||{{answer}}',answer_choices='{{choices[0]}} ||| {{choices[1]}}',reference=''),
    },
    'paragraph_question_to_choose_answer':{
        # begin cosmos_qa
        'description_context_question_answer_text':Template(name='description_context_question_answer_text',jinja='Read the following context and choose the best option to answer the question.\nContext: {{para1}}\nQuestion: {{question}}\nOptions: \n- {{answer_choices | join("\\n - ")}}\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'description_context_question_answer_id':Template(name='description_context_question_answer_id',jinja="Read the following context and choose the best option to answer the question.\nContext: {{para1}}\nQuestion: {{question}}\nOptions: \n{% for k in range(choices | length) %}{{'. '.join([answer_choices[k], choices[k]])}}\n{% endfor %}\n|||\n{{answer_choices[choices.index(answer)]}}",answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=''),
        'no_prompt_id':Template(name='no_prompt_id',jinja="{{para1}}\n{{question}}\n{% for k in range(choices | length) %}{{'. '.join([answer_choices[k], choices[k]])}}\n{% endfor %}\n|||\n{{answer_choices[choices.index(answer)]}}",answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=''),
        'no_prompt_text':Template(name='no_prompt_text',jinja='{{para1}}\n{{question}}\n- {{answer_choices | join("\\n - ")}}\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'context_description_question_answer_id':Template(name='context_description_question_answer_id',jinja="{{para1}}\nAccording to the above context, choose the best option to answer the following question.\nQuestion: {{question}}\nOptions:\n{% for k in range(choices | length) %}{{'. '.join([answer_choices[k], choices[k]])}}\n{% endfor %}\n|||\n{{answer_choices[choices.index(answer)]}}",answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=''),
        'context_description_question_answer_text':Template(name='context_description_question_answer_text',jinja='{{para1}}\nAccording to the above context, choose the best option to answer the following question.\nQuestion: {{question}}\nOptions:\n- {{answer_choices | join("\\n - ")}}\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'context_question_description_answer_id':Template(name='context_question_description_answer_id',jinja="{{para1}}\n{{question}}\nPick the best answer from the following options:\n{% for k in range(choices | length) %}{{'. '.join([answer_choices[k], choices[k]])}}\n{% endfor %}\n|||\n{{answer_choices[choices.index(answer)]}}",answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=''),
        'context_question_description_answer_text':Template(name='context_question_description_answer_text',jinja='{{para1}}\n{{question}}\nPick the best answer from the following options:\n- {{answer_choices | join("\\n - ")}}\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # end cosmos_qa, begin dream
        'baseline':Template(name='baseline',jinja='Dialogue:\n\n{{para1}}\n\nQuestion: {{question}} \n\n- {{answer_choices | join("\n\n- ")}}\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'read_the_following_conversation_and_answer_the_question':Template(name='read_the_following_conversation_and_answer_the_question',jinja='Read the following conversation and answer the question.\n\n{{para1}}\n\nQuestion: {{question}} \n\n- {{answer_choices | join("\n\n- ")}}\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # end dream, begin qasc
        'qa_with_combined_facts_1': Template(name='qa_with_combined_facts_1',jinja='If {{para1[0]|lower}}{{para1[1:]|trim|trim(\'.\')}}, then {{question[0]|lower}}{{question[1:]|trim|trim(\'?\')}}?\n\nAnswer choices:\n- {{answer_choices | join("\\n - ")}}\n||| {{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # end qasc, begin quail
        'context_question_answer_description_id':Template(name='context_question_answer_description_id',jinja='{{para1}}\nQuestion: {{question}}\nOptions:\n{% for k in range(choices | length) %}\n{{". ".join([answer_choices[k], choices[k]])}}\n{% endfor %}\n===\nThe correct answer is\n|||\n{{answer_choices[choices.index(answer)]}}',answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=''),
        'context_question_answer_description_text':Template(name='context_question_answer_description_text',jinja='{{para1}}\nQuestion: {{question}}\nOptions:\n- {{ answer_choices | join(" \\n - ") }}\n===\nThe correct answer is\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # end quail, begin quartz
        # 'use_info_from_question_paragraph':Template(name='use_info_from_question_paragraph',jinja='Use information from the paragraph to answer the question.\n\nQuestion:\n\n\n{{question}}\n\n\nParagraph :\n\n{{para1}}\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # 'paragraph_question_plain_concat':Template(name='paragraph_question_plain_concat',jinja='{{para1}}\n\n{{question}}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # 'use_info_from_paragraph_question':Template(name='use_info_from_paragraph_question',jinja='Use information from the paragraph to answer the question.\n\nParagraph :\n\n{{para1}}\n\nQuestion:\n\n\n{{question}}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # 'answer_question_based_on':Template(name='answer_question_based_on',jinja='Answer the question based on the following text.\n\nQuestion:\n\n\n{{question}}\n\n\nText:\n\n{{para1}}|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # 'answer_question_below':Template(name='answer_question_below',jinja='Answer the question below:\n\n{% if \'_____\' in question %}\n{{question | trim(".?!") | replace("_____", answer_choices | join(" or "))}}{{"?"}} \n{% else %}\n{{question | trim(".?!")}} {{ answer_choices | join(" or ")}}{{"?"}} \n{% endif %}\n\nAssuming that:\n\n{{para1}}|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # 'read_passage_below_choose':Template(name='read_passage_below_choose',jinja='Read the passage below and choose the right answer to the following question (choices are {{answer_choices | join(" or ")}} ):\n\n{{para1}}\n\n\n{{question}}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # 'having_read_above_passage':Template(name='having_read_above_passage',jinja='{{para1}}\n\nHaving read the above passage, choose the right answer to the following question (choices are {{answer_choices | join(" or ")}} ):\n\n\n{{question}}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # 'given_the_fact_answer_the_q':Template(name='given_the_fact_answer_the_q',jinja='Given the fact that:\n\n{{para1}}\n\nAnswer the question:\n\n\n{{question}}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'read_passage_below_choose':Template(name='read_passage_below_choose',jinja='Read the passage below and choose the right answer to the following question (choices are {{answer_choices | join(" or ")}} ):\n\n{{para1}}\n\n\n{{question}}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'having_read_above_passage':Template(name='having_read_above_passage',jinja='{{para1}}\n\nHaving read the above passage, choose the right answer to the following question (choices are {{answer_choices | join(" or ")}} ):\n\n\n{{question}}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # end quartz, begin sciq
        'Multiple Choice Question First':Template(name='Multiple Choice Question First',jinja='Q: {{question}}\n\n\nRead this paragraph and choose the correct option from the provided answers:\n\n{{para1}}\n\n Choices:\n\n- {{answer_choices | join("\\n - ")}}\n\nA:|||{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'Multiple Choice':Template(name='Multiple Choice',jinja='Answer the following question given this paragraph: \n\n{{para1}}\n\n\nQ: {{question}}\n\n Choices:\n\n- {{answer_choices | join("\\n - ")}}\n\nA:|||{{answer_choices[3]}}\n\n',answer_choices='{{choices | join("|||")}}',reference=''),
        # end sciq, begin social_i_qa
        'Show choices and generate answer':Template(name='Show choices and generate answer',jinja='{{para1}}\n\nGiven the context: {{question}}\n\nPossible answers: {{answer_choices | join(", ")}}\n\n|||\n\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'Show choices and generate index':Template(name='Show choices and generate index',jinja='Context: {{para1}}\n\nQuestion: {{question}}\n\nWhich one of these answers best answers the question according to the context?\n\n{% for k in range(choices | length) %}{{\': \'.join([answer_choices[k], choices[k]])}}\n\n{% endfor %}\n\n|||\n\n{{answer_choices[choices.index(answer)]}}',answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=''),
        # end social_i_qa, begin wiki_hop/original
        'choose_best_object_interrogative_1':Template(name='choose_best_object_interrogative_1',jinja='{% set supports= para1.split("\n\n") | reject("equalto", "") | list %}Information:\n{% for support in supports %}\n- {{ support }}\n{% endfor %}\n\n\n{{question}} \n\nChoices:\n- {{answer_choices | join("\\n - ") }}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'choose_best_object_affirmative_1':Template(name='choose_best_object_affirmative_1',jinja='{% set supports= para1.split("\n\n") | reject("equalto", "") | list %}Information:\n{% for support in supports %}\n- {{ support }}\n{% endfor %}\n\n\nGiven the information above, choose from the list below {{question}}.\n\nChoices:\n- {{answer_choices | join("\\n - ") }}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'choose_best_object_affirmative_3':Template(name='choose_best_object_affirmative_3',jinja='{% set supports= para1.split("\n\n") | reject("equalto", "") | list %}Information:\n{% for support in supports %}\n- {{ support }}\n{% endfor %}\n\n\nAfter reading the paragraphs above, we are interested in knowing {{question}}. Find the answer from the choices below.\n\nChoices:\n- {{answer_choices | join("\\n - ") }}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'choose_best_object_affirmative_2':Template(name='choose_best_object_affirmative_2',jinja='{% set supports= para1.split("\n\n") | reject("equalto", "") | list %}Information:\n{% for support in supports %}\n- {{ support }}\n{% endfor %}\n\n\nAfter reading the paragraphs above, {{question}}\n\nChoices:\n- {{answer_choices | join("\\n - ") }}\n\n|||\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # end wiki_hop/original, begin wiqa
         'effect_with_string_answer':Template(name='effect_with_string_answer',jinja='{% set question_para_step = para1.split("\n\n") | reject("equalto", "") | list %}Process:\n- {{ question_para_step | join("\\n- ")}}\n\nQuestion:\n{{question}} Answer by {{answer_choices[:-1] | join(", ") }} or {{answer_choices[-1]}}\n\n|||\n\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
         'which_of_the_following_is_the_supposed_perturbation':Template(name= 'which_of_the_following_is_the_supposed_perturbation',jinja='{% set question_para_step = para1.split("\n\n") | reject("equalto", "") | list %}Process:\n\n- {{ question_para_step | join("\\n- ") }}\n\n{{question}}\n\n- {{answer_choices | join("\\n - ") }}\n\n\n|||\n\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
         'effect_with_label_answer':Template(name='effect_with_label_answer',jinja='{% set question_para_step = para1.split("\n\n") | reject("equalto", "") | list %}Process:\n- {{ question_para_step | join("\\n- ")}}\n\nQuestion:\n{{question}}\n{% for k in range(choices | length) %}\n- {{\': \'.join([answer_choices[k], choices[k]])}}{% endfor %}\n\n|||\n\n{{answer_choices[choices.index(answer)]}}',answer_choices='A ||| B ||| C ||| D ||| E ||| F ||| G ||| H ||| I ||| J ||| K ||| L ||| M ||| N',reference=""), 
        # end wiqa
    },
    'paragraph_hints_question_to_choose_answer':{
        # begin qasc
        'qa_with_separated_facts_1':Template(name='qa_with_separated_facts_1',jinja='{% set fact1=attributes["hints"][0]%}{% set fact2=attributes["hints"][1]%}{{ fact1[0]|capitalize }}{{ fact1[1:]|trim|trim(\'.\') }}, and {{fact2[0]|lower }}{{ fact2[1:]|trim|trim(\'.\') }}. Given these facts, {{ question[0]|lower }}{{question[1:]|trim(\'?\') }} among the following options:\n- {{answer_choices | join("\\n - ") }}\n\n||| \n\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'qa_with_separated_facts_2':Template(name='qa_with_separated_facts_2',jinja='{% set fact1=attributes["hints"][0]%}{% set fact2=attributes["hints"][1]%}Fact 1: {{ fact1[0]|capitalize }}{{ fact1[1:]|trim|trim(\'.\') }}.\n\nFact 2: {{fact2[0]|capitalize }}{{ fact2[1:]|trim|trim(\'.\') }}.\n\nGiven the two facts above, answer the question "{{ question }}" with the following options: \n- {{answer_choices | join("\\n - ") }}\n\n||| \n\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'qa_with_separated_facts_3':Template(name='qa_with_separated_facts_3',jinja='{% set fact1=attributes["hints"][0]%}{% set fact2=attributes["hints"][1]%}Fact 1: {{ fact1[0]|capitalize }}{{ fact1[1:]|trim|trim(\'.\') }}.\n\nFact 2: {{fact2[0]|capitalize }}{{ fact2[1:]|trim|trim(\'.\') }}.\n\nGiven the two facts above, {{ question[0]|lower }}{{question[1:]|trim(\'?\') }}?\n\n||| \n\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'qa_with_separated_facts_4':Template(name='qa_with_separated_facts_4',jinja='{% set fact1=attributes["hints"][0]%}{% set fact2=attributes["hints"][1]%}You are presented with the question "{{ question }}" and the following answer choices: \n- {{answer_choices | join("\\n - ") }}\n\nNow knowing that {{ fact1[0]|lower }}{{ fact1[1:]|trim|trim(\'.\') }} and {{fact2[0]|lower }}{{ fact2[1:]|trim|trim(\'.\') }}, choose the best answer.\n\n||| \n\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        'qa_with_separated_facts_5':Template(name='qa_with_separated_facts_5',jinja='{% set fact1=attributes["hints"][0]%}{% set fact2=attributes["hints"][1]%}You are presented with the quiz "{{ question }}" \n\nBut you don\'t know the answer, so you turn to your teacher to ask for hints. He says that "{{ fact1[0]|lower }}{{ fact1[1:]|trim|trim(\'.\') }}" and "{{fact2[0]|lower }}{{ fact2[1:]|trim|trim(\'.\') }}". \n\nSo, what\'s the best answer to the question?\n\n||| \n\n{{answer}}',answer_choices='{{choices | join("|||")}}',reference=''),
        # end qasc
    },
    'paragraph_to_paragraph':{
        # begin dream
        'generate-last-utterance':Template(name='generate-last-utterance',jinja='{% set dialogue = para1.split("\n\n") | reject("equalto", "") | list %}{% if (dialogue|length)>1 %}Read the below conversation.\n\n{{dialogue[:-1] | join("\\n\\n")}}\n\nWhat would the listener say?\n|||\n{{dialogue[-1]}}{% endif %}',answer_choices=None,reference=''),
        'generate-first-utterance':Template(name='generate-first-utterance',jinja='{% set dialogue = para1.split("\n\n") | reject("equalto", "") | list %}{% if (dialogue|length)>1 %}{{dialogue[1:] | join("\\n\\n")}}\n\nWhat was said before this conversation?\n|||\n{{dialogue[0]}}{% endif %}',answer_choices=None,reference=''),
        # end dream, begin wiqa
        'what_might_be_the_first_step_of_the_process':Template(name='what_might_be_the_first_step_of_the_process',jinja='{% set question_para_step = para1.split("\n\n") | reject("equalto", "") | list %}{% if (question_para_step|length)>1 %}-  {{ question_para_step[1:] | join("\\n- ") }}\n\nWhat might be the first step of the process?\n\n|||\n\n{{ question_para_step | first }}\n{% endif %}',answer_choices=None,reference=''),
        'what_might_be_the_last_step_of_the_process':Template(name='what_might_be_the_last_step_of_the_process',jinja='{% set question_para_step = para1.split("\n\n") | reject("equalto", "") | list %}{% if (question_para_step|length)>1 %}{% set process_list = question_para_step[:-1] if question_para_step[-1] == "" else question_para_step %}\n-  {{ process_list[:-1] | join("\\n- ") }}\n\nWhat might be the last step of the process?\n\n|||\n\n{{ process_list | last }}\n{% endif %}',answer_choices=None,reference=''),
        'what_is_the_missing_first_step':Template(name='what_is_the_missing_first_step',jinja='{% set question_para_step = para1.split("\n\n") | reject("equalto", "") | list %}{% if (question_para_step|length)>1 %}What is the missing first step of the following process:\n\n-  {{ question_para_step[1:] | join("\\n- ") }}\n\n|||\n\n{{ question_para_step | first }}{% endif %}',answer_choices=None,reference=''),
        'what_is_the_final_step_of_the_following_process':Template(name='what_is_the_final_step_of_the_following_process',jinja= '{% set question_para_step = para1.split("\n\n") | reject("equalto", "") | list %}{% if (question_para_step|length)>1 %} {% set process_list = question_para_step[:-1] if question_para_step[-1] == "" else question_para_step %}\nWhat is the final step of the following process:\n-  {{ process_list[:-1] | join("\\n- ") }}\n\n|||\n\n{{ process_list | last }}\n{% endif %}',reference=''),
        # end wiqa
    },
    ##############################################################################################################
    'paragraph_to_sentiment':{
        'User_recommend_this_product':Template(name='User_recommend_this_product',jinja='Based on this review, would the user recommend this product?\n===\nReview: {{para1}}\nAnswer: |||\n{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='No ||| Yes',reference=''),
        'categorize_rating_using_review':Template(name='categorize_rating_using_review',jinja='Given this review: "{{para1}}"\nWould you recommend this app to a friend? {{answer_choices[0]}}, {{answer_choices[1]}}, {{answer_choices[2]}}, {{answer_choices[3]}}, or {{answer_choices[4]}}?\n|||\n{{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='Not at all ||| No ||| Maybe ||| Yes ||| Definitely',reference=''),
        'convert_to_star_rating':Template(name='convert_to_star_rating',jinja='What would be the ★-rating of this review (★ being the lowest and ★★★★★ being the highest)? "{{para1}}"\n|||\n{{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='★ ||| ★★ ||| ★★★ ||| ★★★★ ||| ★★★★★',reference=''),
        'convert_to_rating':Template(name='convert_to_rating',jinja='On a scale of 1-5 (with 1 being least favorable and 5 being most favorable), how would you rate this review? "{{para1}}"\n|||\n{{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='1 ||| 2 ||| 3 ||| 4 ||| 5',reference=''),
        'Movie Expressed Sentiment 2':Template(name='Movie Expressed Sentiment 2',jinja='The following movie review expresses what sentiment? {{para1}} |||{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='negative ||| positive',reference=''),
        'Reviewer Opinion bad good choices':Template(name='Reviewer Opinion bad good choices',jinja='{{para1}} Did the reviewer find this movie {{"good or bad"}}? ||| {{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='bad ||| good',reference=''),
        'Sentiment with choices ':Template(name='Sentiment with choices ',jinja='{{para1}} \nIs this review {{"positive or negative"}}? ||| \n{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='negative ||| positive',reference=''),
        'Reviewer Sentiment Feeling':Template(name='Reviewer Sentiment Feeling',jinja='{{para1}} How does the viewer feel about the movie? ||| {{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='negative ||| positive',reference=''),
        'Writer Expressed Sentiment':Template(name='Writer Expressed Sentiment',jinja='{{para1}} What sentiment does the writer express for the movie? ||| {{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='negative ||| positive',reference=''),
        'Movie Expressed Sentiment':Template(name='Movie Expressed Sentiment',jinja='{{para1}} The sentiment expressed for the movie is ||| {{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='negative ||| positive',reference=''),
        'Text Expressed Sentiment':Template(name='Text Expressed Sentiment',jinja='{{para1}} What is the sentiment expressed in this text? ||| {{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='negative ||| positive',reference=''),
        'Negation template for positive and negative':Template(name='Negation template for positive and negative',jinja='{{para1}} This is definitely not a ||| {{answer_choices[1-attributes["sentiment_2"]["answer"]]}} review.',answer_choices='negative ||| positive',reference=''),
        'Reviewer Enjoyment Yes No':Template(name='Reviewer Enjoyment Yes No',jinja='{{para1}} Did the reviewer enjoy the movie? ||| {{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='No ||| Yes',reference=''),
        'Reviewer Expressed Sentiment':Template(name='Reviewer Expressed Sentiment',jinja='{{para1}} What is the sentiment expressed by the reviewer for the movie? ||| {{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='negative ||| positive',reference=''),
        'Reviewer Enjoyment':Template(name='Reviewer Enjoyment',jinja='{{para1}} How does the reviewer feel about the movie? ||| {{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices="They didn't like it! ||| They loved it",reference=''),
        'so_i_would':Template(name='so_i_would',jinja='{{para1}}\nSo I would like to give it ||| {{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='1 star ||| 2 stars ||| 3 stars ||| 4 stars ||| 5 stars',reference=''),
        'based_on_that':Template(name='based_on_that',jinja='{{para1}}\n===\nBased on that, my rating is ||| {{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='1 star ||| 2 stars ||| 3 stars ||| 4 stars ||| 5 stars',reference=''),
        'format_star':Template(name='format_star',jinja='Review text:\n{{para1}}\n\nStars: |||\n{{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='1 star ||| 2 stars ||| 3 stars ||| 4 stars ||| 5 stars',reference=''),
        'this_place':Template(name='this_place',jinja='{{para1}} My rating for this place is ||| {{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='1 star ||| 2 stars ||| 3 stars ||| 4 stars ||| 5 stars',reference=''),
        'format_score':Template(name='format_score',jinja='Review text:\n{{para1}}\n\nReview score (between 1 and 5): |||\n{{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='1 ||| 2 ||| 3 ||| 4 ||| 5',reference=''),
        'on_a_scale':Template(name='on_a_scale',jinja='Review: {{para1}}\nOn a scale of 1 to 5, I would give this product ||| {{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='1 ||| 2 ||| 3 ||| 4 ||| 5',reference=''),
        'format_rating':Template(name='format_rating',jinja='Review text:\n{{para1}}\n\nReview rating: |||\n{{answer_choices[attributes["sentiment_5"]["answer"]-1]}}',answer_choices='1 star ||| 2 stars ||| 3 stars ||| 4 stars ||| 5 stars',reference=''),
    },
    'paragraph_title_to_sentiment':{
        'user_satisfied':Template(name='user_satisfied',jinja='Here is a review left by a customer on a product. Would you say he was {{answer_choices[1]}} or {{answer_choices[0]}}?\nTitle: {{attributes["title"]["answer"]}}\nReview: {{para1}}\n|||\n{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='dissatisfied ||| satisfied',reference=''),
        'Is_this_review':Template(name='Is_this_review',jinja='Title: {{attributes["title"]["answer"]}}\nReview: {{para1}}\nIs the review positive or negative? |||{{answer_choices[attributes["sentiment_2"]["answer"]]}}\n',answer_choices='Negative ||| Positive',reference=''),
        'Is_this_product_review_positive':Template(name='Is_this_product_review_positive',jinja='Is this product review positive?\nTitle: {{attributes["title"]["answer"]}}\nReview: {{para1}}\nAnswer: |||{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='No ||| Yes',reference=''),
        'Is_this_review_negative':Template(name='Is_this_review_negative',jinja='Title: {{attributes["title"]["answer"]}}\nReview: {{para1}}\nIs this product review negative?|||\n{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='Yes ||| No',reference=''),
        'convey_negative_or_positive_sentiment':Template(name='convey_negative_or_positive_sentiment',jinja='Title: {{attributes["title"]["answer"]}}\nReview: {{para1}}\nDoes this product review convey a negative or positive sentiment?|||\n{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='Negative ||| Positive',reference=''),
        'negative_or_positive_tone':Template(name='negative_or_positive_tone',jinja='Is there a negative or positive tone to this product review?\n===\nTitle: {{attributes["title"]["answer"]}}\nReview: {{para1}}\nAnswer: |||\n{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='Negative ||| Positive',reference=''),
        'would_you_buy':Template(name='would_you_buy',jinja='You are considering whether to buy a product. You look at the reviews. Would the following review {{answer_choices[0]}} or {{answer_choices[1]}} the chances of you buying the product?\nReview title: {{attributes["title"]["answer"]}}\nProduct review: {{para1}}\n|||\n{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='decrease ||| increase',reference=''),
        'flattering_or_not':Template(name='flattering_or_not',jinja='Title: {{attributes["title"]["answer"]}}\nProduct review: {{para1}}\nWould you say this review depicts the product in a {{answer_choices[1]}} or {{answer_choices[0]}} light?\n|||\n{{answer_choices[attributes["sentiment_2"]["answer"]]}}',answer_choices='unflattering ||| flattering',reference=''),
    },
    'sentiment_title_to_paragraph':{
        'generate_review':Template(name='generate_review',jinja='Generate a {{attributes["sentiment_5"]["answer"]}}-star review (1 being lowest and 5 being highest) about an app with package {{attributes["title"]["answer"]}}.\n|||\n{{para1}}',answer_choices=None,reference=''),
    },
    ##############################################################################################################
    'keywords_to_paragraph':{
        'Given concepts - type 2':Template(name='Given concepts - type 2',jinja='Ignoring the order of the concepts: {{attributes["keywords"]["answer"] | join(", ")}}; \nGenerate a sentence with all the concepts :\n|||\n{{para1}}',answer_choices=None,reference=''),
        'Put together':Template(name='Put together',jinja='Put the concepts together to form a sentence: {{attributes["keywords"]["answer"] | join(", ")}}.\n|||\n{{para1}}',answer_choices=None,reference=''),
        'choice in concept centric sentence generation':Template(name='choice in concept centric sentence generation',jinja='Construct a sentence with the word {{attributes["keywords"]["answer"] | choice}}. \n\nHint: Use {{attributes["keywords"]["answer"] | join(", ")}} to restrict the output sentence.\n|||\n{{para1}}',answer_choices=None,reference=''),
        'random task template prompt':Template(name='random task template prompt',jinja='{% set seq = [ \n\'From the concepts mentioned below, generate a sentence:\', \n\'Convert the concepts to a sentence:\', \n\'Given the list of concepts, write a sentence:\'\n] %} \n{{seq | choice}}\n{{attributes["keywords"]["answer"] | join(", ")}}\n|||\n{{para1}}',answer_choices=None,reference=''),
        'Example prompt':Template(name='Example prompt',jinja='Humans can easily string together abstract concepts to form a coherent sentence. \nFor example, with the concepts {{attributes["keywords"]["answer"] | join(", ")}}, a simple sentence can be  \n|||\n{{para1}}',answer_choices=None,reference=''),
        'Given concepts type 1':Template(name='Given concepts type 1',jinja='Given the list of concepts: {{attributes["keywords"]["answer"] | join(", ")}}; \nGenerate a sentence with all the concepts :\n|||\n{{para1}}',answer_choices=None,reference=''),
        'topic to sentence':Template(name='topic to sentence',jinja='Can you write a sentence about the topic {{attributes["keywords"]["answer"] | choice}}?\n|||\n{{para1}}',answer_choices=None,reference='')
    },
    'paragraph_to_keywords':{
        'topics from the sentence':Template(name='topics from the sentence',jinja='What are the topics in the sentence: {{para1}}\n|||\n{{attributes["keywords"]["answer"] | join(", ")}}',answer_choices=None,reference=''),
        'sentence to concepts':Template(name='sentence to concepts',jinja='We have the sentence: {{para1}}; \nExtract all the key concepts: \n|||\n{{attributes["keywords"]["answer"] | join(", ")}}',answer_choices=None,reference=''),
    },
    ##############################################################################################################
    'paragraph3_to_paragraph1':{
        'write_an_outline':Template(name='write_an_outline',jinja='Can you write an outline of the following article in a few points?\n\nArticle: {{para2}}|||\n{{para1}}',answer_choices=None,reference=''),
        'news_summary':Template(name='news_summary',jinja='Summarise the article:\n\n{{para2}} |||\n{{para1}}',answer_choices=None,reference=''),
        '2_or_3_sentences':Template(name='2_or_3_sentences',jinja='In 2 or 3 sentences, what are the main points one should remember from this news article?\n\nArticle: {{para2}} |||\n{{para1}}',answer_choices=None,reference=''),
        'tldr_summary':Template(name='tldr_summary',jinja="Could you please generate a TLDR (Too Long Didn't Read) summary of the following news article?\n\nArticle: {{para2}} |||\n{{para1}}",answer_choices=None,reference=''),
        'news_card_view':Template(name='news_card_view',jinja='Condense the article down to the essentials to present it in the form of short cards in mobile news apps:\n\n{{para2}} |||\n{{para1}}',answer_choices=None,reference=''),
        'sum_in_brief':Template(name='sum_in_brief',jinja='Sum the following article in brief: {{para2}}|||{{para1}}',answer_choices=None,reference=''),
        'news_stock':Template(name='news_stock',jinja='Extract key points from the article based on which the stock market could react:\n\n{{para2}} |||\n{{para1}}',answer_choices=None,reference=''),
        'what are the key points':Template(name='what are the key points',jinja='{% set docs = para2.split("3ed2dface8203c4c9dfb1a5dc58e41e0||") | reject("equalto", "") | list %}\nWhat are the key points across these news articles:\n{% for doc in docs %}\n\nArticle: {{doc}}\n{% endfor %}\n|||\n{{para1}}',answer_choices=None,reference=''),
        'synthesize':Template(name='synthesize',jinja='{% set docs = para2.split("3ed2dface8203c4c9dfb1a5dc58e41e0||") | reject("equalto", "") | list %}\nSynthesize these documents into a single one:\n{% for doc in docs %}\n\n- {{doc}}\n{% endfor %}\n|||\n{{para1}}',answer_choices=None,reference=''),
        'summary scenario':Template(name='summary scenario',jinja='{% set docs = para2.split("3ed2dface8203c4c9dfb1a5dc58e41e0||") | reject("equalto", "") | list %}\nI want to edit the following articles into a more concise summary:\n{% for doc in docs %}\n\nArticle: {{doc}}\n{% endfor %}\n|||\n{{para1}}',answer_choices=None,reference=''),
        'summarize':Template(name='summarize',jinja='{% set docs = para2.split("3ed2dface8203c4c9dfb1a5dc58e41e0||") | reject("equalto", "") | list %}\nWrite a summary of the following articles:\n{% for doc in docs %}\n\nDocument: {{doc}}\n{% endfor %}\n|||\n{{para1}}',answer_choices=None,reference=''),
        'distill':Template(name='distill',jinja='{% set docs = para2.split("3ed2dface8203c4c9dfb1a5dc58e41e0||") | reject("equalto", "") | list %}\nI\'m trying to distill these articles down into one:\n{% for doc in docs %}\n\nArticle: {{doc}}\n{% endfor %}\n|||\n{{para1}}',answer_choices=None,reference=''),
        'Summarize this dialogue:':Template(name='Summarize this dialogue:',jinja='Summarize this dialogue: {{para2}} |||\n{{para1}}',answer_choices=None,reference=''),
        'Given the above dialogue write a summary':Template(name='Given the above dialogue write a summary',jinja='{{para2}}\nGiven the above dialogue, write a summary. |||\n{{para1}}',answer_choices=None,reference=''),
        'Summarize:':Template(name='Summarize:',jinja='Summarize: {{para2}}|||\n{{para1}}',answer_choices=None,reference=''),
        'To sum up this dialog':Template(name='To sum up this dialog',jinja='{{para2}}\nTo sum up this dialog:\n|||{{para1}}',answer_choices=None,reference=''),
        'Generate a summary for this dialogue':Template(name='Generate a summary for this dialogue',jinja='Generate a summary for this dialogue:\n{{para2}}\n|||{{para1}}',answer_choices=None,reference=''),
        'Sum up the following dialogue':Template(name='Sum up the following dialogue',jinja='Sum up the following dialogue: \n{{para2}}\n|||{{para1}}',answer_choices=None,reference=''),
        'DOC_write_summary_of_above':Template(name='DOC_write_summary_of_above',jinja='{{para2}}\n\n===\n\nWrite a summary of the text above : ||| {{para1}}',answer_choices=None,reference=''),
        'article_DOC_summary':Template(name='article_DOC_summary',jinja='Article: {{para2}}\n\nSummary: ||| {{para1}}',answer_choices=None,reference=''),
        'DOC_how_would_you_rephrase_few_words':Template(name='DOC_how_would_you_rephrase_few_words',jinja='{{para2}}\nHow would you rephrase that in a few words? ||| {{para1}}',answer_choices=None,reference=''),
        'college_roommate_asked_DOC_so_I_recap':Template(name='college_roommate_asked_DOC_so_I_recap',jinja="My college roommate asked me what this article means:\n\n{{para2}}\n\nSo I recapped it in layman's terms: ||| {{para1}}",answer_choices=None,reference=''),
        'DOC_boils_down_to_simple_idea_that':Template(name='DOC_boils_down_to_simple_idea_that',jinja='{{para2}}\nThis boils down to the simple idea that ||| {{para1}}',answer_choices=None,reference=''),
        'summarize_DOC':Template(name='summarize_DOC',jinja='Summarize: {{para2}}|||\n{{para1}}',answer_choices=None,reference=''),
        'summarize_this_DOC_summary':Template(name='summarize_this_DOC_summary',jinja='Summarize this document: {{para2}}\nSummary: ||| {{para1}}',answer_choices=None,reference=''),
        'DOC_given_above_write_one_sentence':Template(name='DOC_given_above_write_one_sentence',jinja='{{para2}}\n\n===\n\nGiven the above document, write one sentence to summarize: ||| {{para1}}',answer_choices=None,reference=''),
        'read_below_DOC_write_abstract':Template(name='read_below_DOC_write_abstract',jinja='First, please read the article below.\n\n{{para2}}\n\nNow, can you write me an extremely short abstract for it?  ||| {{para1}}',answer_choices=None,reference=''),
        'DOC_tldr':Template(name='DOC_tldr',jinja='{{para2}}\n\nTL;DR: ||| {{para1}}',answer_choices=None,reference=''),
    },

    'paragraph1_to_paragraph3':{
        'generate_story':Template(name='generate_story',jinja='Generate a story from key plot points:\n\n{{para1}} |||\n{{para2}}',answer_choices=None,reference=''),
        'spice_up_story':Template(name='spice_up_story',jinja='What details would you include in a storyline to make it more engaging and informative?\n\n{{para1}} |||\n{{para2}}',answer_choices=None,reference=''),
        'expand (reverse task)':Template(name='expand (reverse task)',jinja='{% set docs = para2.split("3ed2dface8203c4c9dfb1a5dc58e41e0||") | reject("equalto", "") | list%}\nWrite an expanded news article with plausible details from the following summary:\n{{para1}}\n|||\n{{docs | choice}}', answer_choices=None, reference=''),
        'Write a dialogue that match this summary':Template(name='Write a dialogue that match this summary',jinja='Write a dialogue that matches this summary: {{para1}} |||\n{{para2}}',answer_choices=None,reference=''),
    },

    'paragraph_to_topic1':{
        'classify_question_first':Template(name='classify_question_first',jinja='What label best describes this news article?\n{{para1}} ||| \n{{answer_choices[attributes["topic1"]["candidates"].index(attributes["topic1"]["answer"])]}}',answer_choices='World politics ||| Sports ||| Business ||| Science and technology',reference=''),
        'classify_with_choices_question_first':Template(name='classify_with_choices_question_first',jinja='Is this a piece of news regarding {{"world politics, sports, business, or science and technology"}}?\n{{para1}} \n||| \n{{answer_choices[attributes["topic1"]["candidates"].index(attributes["topic1"]["answer"])]}}',answer_choices='World politics ||| Sports ||| Business ||| Science and technology',reference=''),
        'recommend':Template(name='recommend',jinja='Would you recommend the following article to a {{"politician"}}, an {{"athlete"}}, a {{"business executive"}}, or a {{"scientist"}}?\n\n{{para1}}\n|||\n{{answer_choices[attributes["topic1"]["candidates"].index(attributes["topic1"]["answer"])]}}',answer_choices='Politician ||| Athlete ||| Business executive ||| Scientist',reference=''),
        'which_section_choices':Template(name='which_section_choices',jinja='{{para1}} \n\nWhich of the following sections of a newspaper would this article likely appear in? {{"World News"}}, {{"Sports"}}, {{"Business"}}, or {{"Science and Technology"}}? ||| \n{{answer_choices[attributes["topic1"]["candidates"].index(attributes["topic1"]["answer"])]}}',answer_choices='World News ||| Sports ||| Business ||| Science and Technology',reference=''),
        'which_section':Template(name='which_section',jinja='{{para1}} \n\nWhich section of a newspaper would this article likely appear in? ||| \n{{answer_choices[attributes["topic1"]["candidates"].index(attributes["topic1"]["answer"])]}}',answer_choices='World News ||| Sports ||| Business ||| Science and Technology',reference=''),
        'classify_with_choices':Template(name='classify_with_choices',jinja='{{para1}} \nIs this a piece of news regarding {{"world politics, sports, business, or science and technology"}}? ||| \n{{answer_choices[attributes["topic1"]["candidates"].index(attributes["topic1"]["answer"])]}}',answer_choices='World politics ||| Sports ||| Business ||| Science and technology',reference=''),
        'classify':Template(name='classify',jinja='{{para1}} \nWhat label best describes this news article? ||| \n{{answer_choices[attributes["topic1"]["candidates"].index(attributes["topic1"]["answer"])]}}',answer_choices='World politics ||| Sports ||| Business ||| Science and technology',reference=''),
    },
    'paragraph_to_topic2':{
        'given_list_what_category_does_the_paragraph_belong_to':Template(name='given_list_what_category_does_the_paragraph_belong_to',jinja='{{para1}} Given a list of categories: {{"company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work"}}, what category does the paragraph belong to? ||| {{answer_choices[attributes["topic2"]["candidates"].index(attributes["topic2"]["answer"])]}}\n\n',answer_choices='Company ||| Educational Institution ||| Artist ||| Athlete ||| Office Holder ||| Mean Of Transportation ||| Building ||| Natural Place ||| Village ||| Animal ||| Plant ||| Album ||| Film ||| Written Work',reference=''),
    },
    'title_to_topic2':{
        'given_a_list_of_category_what_does_the_title_belong_to':Template(name='given_a_list_of_category_what_does_the_title_belong_to',jinja='"{{attributes["title"]["answer"]}}", given a list of categories: {{"company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work"}}, what category does the title belong to? ||| {{answer_choices[attributes["topic2"]["candidates"].index(attributes["topic2"]["answer"])]}}\n\n',answer_choices='Company ||| Educational Institution ||| Artist ||| Athlete ||| Office Holder ||| Mean Of Transportation ||| Building ||| Natural Place ||| Village ||| Animal ||| Plant ||| Album ||| Film ||| Written Work',reference=''),
    },
    'paragraph_title_to_topic2':{
        'pick_one_category_for_the_following_text':Template(name='pick_one_category_for_the_following_text',jinja='Pick one category for the following text. The options are - {{"company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work"}}. {{attributes["title"]["answer"]}} - {{para1}} ||| {{answer_choices[attributes["topic2"]["candidates"].index(attributes["topic2"]["answer"])]}}',answer_choices='Company ||| Educational Institution ||| Artist ||| Athlete ||| Office Holder ||| Mean Of Transportation ||| Building ||| Natural Place ||| Village ||| Animal ||| Plant ||| Album ||| Film ||| Written Work',reference=''),
        'given_a_choice_of_categories ':Template(name='given_a_choice_of_categories ',jinja='{{attributes["title"]["answer"]}} - {{para1}} Given a choice of categories {{"company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work"}}, the text refers to which one? ||| {{answer_choices[attributes["topic2"]["candidates"].index(attributes["topic2"]["answer"])]}}',answer_choices='Company ||| Educational Institution ||| Artist ||| Athlete ||| Office Holder ||| Mean Of Transportation ||| Building ||| Natural Place ||| Village ||| Animal ||| Plant ||| Album ||| Film ||| Written Work',reference=''),
    }
}
prompt_name_type_map={}
for (prompt_type,prompts) in uniformed_prompt_templates.items():
    for prompt_name in prompts:
        prompt_name_type_map[prompt_name]=prompt_type

def enlarge_prompts(prompts,only_source=True):
    # prompts: [('paragraph_to_topic1',prompts)]
    ret_prompts=[]
    for (prompt_type,sub_prompts) in prompts:
        new_prompts=[]
        if prompt_type in ['paragraph1_to_paragraph3','paragraph3_to_paragraph1']:
            for prompt_name,prompt in sub_prompts.items():
                new_prompts.append(prompt)
        else:
            for prompt_name,prompt in sub_prompts.items():
                new_prompts.append(prompt)
                if only_source==True:
                    flag='{{para1}}' in prompt.__dict__['jinja'].split('|||')[0] and 'para1_meta' not in prompt.__dict__['jinja'].split('|||')[0]
                else:
                    flag='{{para1}}' in prompt.__dict__['jinja'] and 'para1_meta' not in prompt.__dict__['jinja']
                if flag:
                    new_prompt=copy.deepcopy(prompt)
                    new_prompt.__dict__['jinja']=prompt.__dict__['jinja'].replace('{{para1}}','{{para2}}')
                    new_prompts.append(new_prompt)
                    new_prompt=copy.deepcopy(prompt)
                    new_prompt.__dict__['jinja']=prompt.__dict__['jinja'].replace('{{para1}}','{{para3}}')
                    new_prompts.append(new_prompt)
        ret_prompts.append((prompt_type,new_prompts))
    return ret_prompts

def prompts_to_list(prompts):
    return sum([x for (x,y) in prompts],[])

Direct_Prompts=uniformed_prompt_templates.keys()

CLS_Prompts=['paragraph_question_tf','question_paraphrase_tf','question_to_answer_tf','title_question_to_answer_tf','question_to_choose_answer','paragraph_question_to_choose_answer','paragraph_to_sentiment','paragraph_title_to_sentiment','paragraph_to_topic1','paragraph_to_topic2','title_to_topic2','paragraph_title_to_topic2']
GEN_Prompts=[k for k in uniformed_prompt_templates.keys() if k not in CLS_Prompts]
Extended_CLS_Prompts=sum([list(uniformed_prompt_templates[p].keys()) for p in CLS_Prompts],[])
Extended_GEN_Prompts=sum([list(uniformed_prompt_templates[p].keys()) for p in GEN_Prompts],[])
