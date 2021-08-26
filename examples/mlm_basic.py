from embedding import BertHuggingfaceMLM
import torch
from transformers import AdamW
from tqdm import tqdm
from random import random
import json
import codecs
from operator import itemgetter
import pandas as pd
import numpy as np


bert = BertHuggingfaceMLM(model_name='bert-base-uncased', batch_size=8)

pronouns = ['he', 'she']
pronouns2 = ['his', 'her']

# uncomment for more training data
templates = ['[MASK]\'s a OCCUPATION']#, '[MASK] is a OCCUPATION', '[MASK] will become a OCCUPATION', '[MASK] was a OCCUPATION', '[MASK] is working as a OCCUPATION', '[MASK] just started working as a OCCUPATION', 'the OCCUPATION was in a hurry because [MASK] was late for work', '[MASK] is interested in becoming a OCCUPATION', '[MASK] does not like being a OCCUPATION', '[MASK] likes being a OCCUPATION',
            #'[MASK] always wanted to be a OCCUPATION', '[MASK] never wanted to be a OCCUPATION', '[MASK] had an interview for a position as a OCCUPATION', 'is [MASK] a OCCUPATION', '[MASK] is a OCCUPATION, right?']
templates2 = ['the OCCUPATION enjoyed [MASK] lunch']#, 'the OCCUPATION missed [MASK] bus', 'the OCCUPATION arrived in [MASK] car', 'the OCCUPATION asked [MASK] boss for a promotion', 'the OCCUPATION collected [MASK] check']

with codecs.open('professions.json') as f:
    occupations = json.load(f)
# format is:
# [job, def_gender, stereotype_gender] - gender relation: -1 female, +1 male

# get two equal-sized sets with female/male stereotyped jobs
sorted_jobs = sorted(occupations, key=itemgetter(2))
jobs_f = [tup[0] for tup in sorted_jobs[:50]]
jobs_m = [tup[0] for tup in sorted_jobs[-50:]]

# with max stereotype
jobs_max_stereo = [[job, 0.0, -1.0] for job in jobs_f]+[[job, 0.0, 1.0] for job in jobs_m]


def insert_job(template, occupation):
    sent = template.replace('OCCUPATION', occupation)
    if occupation[0] in "aeiou":
        sent = sent.replace(' a ', ' an ')
    return sent


# default first, then other
def insert_pronoun(sent, pronouns=['he', 'she'], prob=0.5):
    if random() > prob:
        sent = sent.replace('[MASK]', pronouns[1])
    else:

        sent = sent.replace('[MASK]', pronouns[0])
    return sent


# randomized pronoun selection, stereotype=0.5 is fair on average, 1.0 maximally stereotyped as implied by the lists, 0.0 counterwise stereotyped
def create_occupation_data(jobs):
    masked_texts = []
    labels = []
    for tup in jobs:
        stereo_prob = (tup[2] + 1) / 2.0
        occupation = tup[0]
        for template in templates:
            sent = insert_job(template, occupation)
            masked_texts.append(sent)
            labels.append(insert_pronoun(sent, ['he', 'she'], prob=stereo_prob))
        for template in templates2:
            sent = insert_job(template, occupation)
            masked_texts.append(sent)
            labels.append(insert_pronoun(sent, ['his', 'her'], prob=stereo_prob))
    return masked_texts, labels


# creates both gendered versions of each sentence
def create_balanced_occupation_data():
    masked_texts = []
    labels = []
    for template in templates:
        for occupation in jobs_f + jobs_m:
            sent = insert_job(template, occupation)
            masked_texts.append(sent)
            labels.append(sent.replace('[MASK]', 'she'))
            masked_texts.append(sent)
            labels.append(sent.replace('[MASK]', 'he'))
    for template in templates2:
        for occupation in jobs_f + jobs_m:
            sent = insert_job(template, occupation)
            masked_texts.append(sent)
            labels.append(sent.replace('[MASK]', 'her'))
            masked_texts.append(sent)
            labels.append(sent.replace('[MASK]', 'his'))
    return masked_texts, labels


def train_and_log(job_tuples, reset_bert=True, balanced=False):
    if reset_bert:  # we want to use the same baseline model for any training
        bert.load('bert-base-uncased')

    # create training data
    if balanced:  # in case we want perfectly balanced training data
        masked_texts, labels = create_balanced_occupation_data()
    else:  # normally use defintional stereotype of job_tuples
        masked_texts, labels = create_occupation_data(job_tuples)

    # training with torch
    bert.retrain(masked_texts[:50], labels[:50], 1)
    bert.save('test/')


train_and_log(jobs_max_stereo)
train_and_log(jobs_max_stereo, balanced=True)

