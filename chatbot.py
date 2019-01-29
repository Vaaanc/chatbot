# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 22:51:54 2019

@author: Vaaan
"""

import numpy as np
import re
import tensorflow as tf
import time

# Open the datasets
movie_lines = open(
        'movie_lines.txt',
        encoding='utf-8',
        errors='ignore').read().split('\n')
movie_conversations = open(
        'movie_conversations.txt',
        encoding='utf-8',
        errors='ignore').read().split('\n')

# Create dictionary of the movie lines and it's id
id_to_line = {}
for line in movie_lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id_to_line[_line[0]] = _line[4]
        
# Create list of conversation id's
conversation_ids = []
for conversation in movie_conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversation_ids.append(_conversation.split(','))
    
# Separate the question and answer
questions = []
answers = []
for conversation in conversation_ids:
    for i in range(len(conversation) - 1):
        questions.append(id_to_line[conversation[i]])
        answers.append(id_to_line[conversation[i+1]])
        
def clean(text):
    text = text.lower()
    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"he's", 'he is', text)
    text = re.sub(r"she's", 'she is', text)
    text = re.sub(r"that's", 'that is', text)
    text = re.sub(r"what's", 'what is', text)
    text = re.sub(r"where's", 'where is', text)
    text = re.sub(r"how's", 'how is', text)
    text = re.sub(r"here's", 'here is', text)
    
    text = re.sub(r"\'ll", ' will', text)
    text = re.sub(r"\'ve", ' have', text)
    text = re.sub(r"\'re", ' you are', text)
    text = re.sub(r"\'d", ' would', text)
    
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"can't", 'can not', text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", '', text)
                     
    return text

# Clean Questions
cleaned_questions = []
for question in questions:
    cleaned_questions.append(clean(question))
    
# Clean Answers
cleaned_answers = []
for answer in answers:
    cleaned_answers.append(clean(answer))
    
# Dictionary of word with it's number of occurences
word_to_count = {}
for question in cleaned_questions:
    for word in question.split():
        if word not in word_to_count:
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1

for answer in cleaned_answers:
    for word in question.split():
        if word not in word_to_count:
            word_to_count[word] = 1
        else:
            word_to_count[word] += 1

# Dictionary that map the questions words and the answer words to a unique key
threshold = 20

question_words_to_int = {}
counter = 0
for work, word_count in word_to_count.items():
    if word_count >= threshold:
        question_words_to_int[word] = counter
        counter += 1
        
answer_words_to_int = {}
counter = 0
for work, word_count in word_to_count.items():
    if word_count >= threshold:
        answer_words_to_int[word] = counter
        counter += 1


# Add Token to the two dictionary for encoder & decoder
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    question_words_to_int[token] = len(question_words_to_int) + 1
    answer_words_to_int[token] = len(answer_words_to_int) + 1


# Inverse answer_words_to_int dictionary
answers_int_to_word = {word_int: word for word, word_int in answer_words_to_int.items()}


# Translate all the questions and the answers into unique key
# And replace all the words that were filtered by <OUT>
question_to_int = []
for question in cleaned_questions:
    question_ints = []
    for word in question.split():
        if word not in question_words_to_int:
            question_ints.append(question_words_to_int['<OUT>'])
        else:
            question_ints.append(question_words_to_int[word])
            
    question_to_int.append(question_ints)

answer_to_int = []
for answer in cleaned_answers:
    answer_ints = []
    for word in answer.split():
        if word not in answer_words_to_int:
            answer_ints.append(answer_words_to_int['<OUT>'])
        else:
            answer_ints.append(answer_words_to_int[word])
            
    answer_to_int.append(answer_ints)

# Sort question and answers by length of question to speed up the training
max_length_of_question = 25
sorted_cleaned_questions = []
sorted_cleaned_answers = []
for length in range(1, max_length_of_question + 1):
    for i, value in enumerate(question_to_int):
        if len(value) == length:
            sorted_cleaned_questions.append(question_to_int[i])
            sorted_cleaned_answers.append(answer_to_int[i])






