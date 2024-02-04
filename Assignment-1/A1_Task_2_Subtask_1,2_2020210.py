import numpy as np
from collections import defaultdict
import random
from utils import emotion_scores
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

class BigramLM:
    def __init__(self):
        self.vocab = set()
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.bigram_probs = None
        self.beta_values = None
        self.emotion_dict = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}
        self.sentence_emotions = []
        self.bigram_sentences = defaultdict(lambda: defaultdict(list))

    def learn_from_dataset(self, dataset):
        for x, sentence in enumerate(dataset):
            emotion = emotion_scores(sentence)
            self.sentence_emotions.append(emotion)
            tokens = sentence.split()                        
            for i in range(len(tokens) - 1):
                word1, word2 = tokens[i], tokens[i + 1]
                self.vocab.add(word1)
                self.vocab.add(word2)
                self.bigram_counts[word1][word2] += 1
                self.bigram_sentences[word1][word2].append(x)
                self.unigram_counts[word1] += 1                            
                
        self.vocab = list(self.vocab)
        print(f"Vocabulary size: {len(self.vocab)}")

    def calculate_beta_values_sentence(self):  
        num_words = len(self.vocab)
        self.beta_values_sentence = np.zeros((num_words, num_words, 6))

        for i, word1 in tqdm(enumerate(self.vocab)):
            if word1 not in self.bigram_counts.keys():
                continue
            for j, word2 in enumerate(self.vocab):
                if word2 not in self.bigram_counts[word1].keys():
                    continue
                self.beta_values_sentence[i][j] = np.array([np.mean([self.sentence_emotions[sentence][k]['score'] for sentence in self.bigram_sentences[word1][word2]]) for k in range(6)])
                

    def calculate_beta_values(self):
            num_words = len(self.vocab)
            self.beta_values = np.zeros((num_words, num_words, 6))
            for i, word1 in tqdm(enumerate(self.vocab)):
                if word1 not in self.bigram_counts.keys():
                    continue
                for j, word2 in enumerate(self.vocab):
                    if word2 not in self.bigram_counts[word1].keys():
                        continue
                    emotions = emotion_scores(word1 + " " + word2)
                    self.beta_values[i][j] = np.array([emotions[k]['score'] for k in range(6)])
                    
    def calculate_bigram_probs(self):
        num_words = len(self.vocab)
        self.bigram_probs = np.zeros((num_words, num_words))        
        
        for i, word1 in tqdm(enumerate(self.vocab)):
            for j, word2 in enumerate(self.vocab):
                if self.unigram_counts[word1] > 0:
                    self.bigram_probs[i][j] = float(self.bigram_counts[word1][word2]) / float(self.unigram_counts[word1])

                    
    def calculate_bigram_probs_laplace(self):
        num_words = len(self.vocab)
        self.bigram_probs = np.zeros((num_words, num_words))

        for i, word1 in enumerate(self.vocab):
            for j, word2 in enumerate(self.vocab):
                self.bigram_probs[i][j] = (self.bigram_counts[word1][word2] + 1) / (self.unigram_counts[word1] + num_words)
                
    
    def calculate_bigram_probs_kneser_ney(self, discount=0.75):
        num_words = len(self.vocab)
        self.bigram_probs = np.zeros((num_words, num_words))

        continuation_counts = defaultdict(set)
   
        for word1, word2_dict in self.bigram_counts.items():
            for word2 in word2_dict:
                continuation_counts[word2].add(word1)

        total_continuations = {word2: len(word1s) for word2, word1s in continuation_counts.items()}

        for i, word1 in tqdm(enumerate(self.vocab)):            
            for j, word2 in enumerate(self.vocab):
                if word2 in total_continuations.keys():
                    adjusted_count = max(self.bigram_counts[word1][word2] - discount, 0)
                    continuation_prob = total_continuations[word2] / sum(total_continuations.values()) if sum(total_continuations.values()) > 0 else 0
                    lower_order_weight = (discount * continuation_prob) / self.unigram_counts[word1] if self.unigram_counts[word1] > 0 else 0

                    self.bigram_probs[i][j] = adjusted_count / self.unigram_counts[word1] + lower_order_weight if self.unigram_counts[word1] > 0 else 0                                            

corpus = open('../Dataset/corpus.txt')
dataset = []
for i in corpus.readlines():
    dataset.append('<SOS> ' + i + ' <EOS>')

model = BigramLM()
model.learn_from_dataset(dataset)
pickle.dump(model.vocab, open('Checkpoints/vocab.pkl', 'wb'))

model.calculate_beta_values_sentence()
pickle.dump(model.beta_values_sentence, open('Checkpoints/beta_values_sentence.pkl', 'wb'))

model.calculate_beta_values()
pickle.dump(model.beta_values, open('Checkpoints/beta_values.pkl', 'wb'))

model.calculate_bigram_probs()
pickle.dump(model.bigram_probs, open('Checkpoints/bigram_probs.pkl', 'wb'))

model.calculate_bigram_probs_kneser_ney()
pickle.dump(model.bigram_probs, open('Checkpoints/bigram_probs_kneser.pkl', 'wb'))

model.calculate_bigram_probs_laplace()
pickle.dump(model.bigram_probs, open('Checkpoints/bigram_probs_laplace.pkl', 'wb'))

normal_bigram_probs = pickle.load(open('Checkpoints/bigram_probs.pkl', 'rb'))
laplace_bigram_probs = pickle.load(open('Checkpoints/bigram_probs_laplace.pkl', 'rb'))
kneser_ney_bigram_probs = pickle.load(open('Checkpoints/bigram_probs_kneser.pkl', 'rb'))

vocab = pickle.load(open('Checkpoints/vocab.pkl', 'rb'))

def perplexity_without_smoothing(dataset):   
    perplexity_score = 0 
    for sentence in dataset:        
        sentence = sentence.split()
        log_sum = 0
        for i in range(len(sentence) - 1):
            word1, word2 = sentence[i], sentence[i+1]            
            idx1, idx2 = vocab.index(word1), vocab.index(word2)            
            prob = normal_bigram_probs[idx1][idx2]
            log_sum += (np.log(prob))*-1
        perplexity_score += log_sum/len(sentence)    
    perplexity_score/=len(dataset)
    return perplexity_score        

def perplexity_laplace(dataset):   
    perplexity_score = 0 
    for sentence in dataset:        
        sentence = sentence.split()
        log_sum = 0
        for i in range(len(sentence) - 1):
            word1, word2 = sentence[i], sentence[i+1]            
            idx1, idx2 = vocab.index(word1), vocab.index(word2)            
            prob = laplace_bigram_probs[idx1][idx2]
            log_sum += (np.log(prob))*-1
        perplexity_score += log_sum/len(sentence)    
    perplexity_score/=len(dataset)
    return perplexity_score        

def perplexity_kneser_ney(dataset):   
    perplexity_score = 0 
    for sentence in dataset:        
        sentence = sentence.split()
        log_sum = 0
        for i in range(len(sentence) - 1):
            word1, word2 = sentence[i], sentence[i+1]            
            idx1, idx2 = vocab.index(word1), vocab.index(word2)            
            prob = kneser_ney_bigram_probs[idx1][idx2]
            log_sum += (np.log(prob))*-1
        perplexity_score += log_sum/len(sentence)    
    perplexity_score/=len(dataset)
    return perplexity_score        

print(f"No smoothing perplexity: {perplexity_without_smoothing(dataset)}")
print(f"Laplace perplexity: {perplexity_laplace(dataset)}")
print(f"Kneser-ney perplexity: {perplexity_kneser_ney(dataset)}")

import os

all_samples = []
dir = 'Test Samples/coeff_1_kneser/'
for file in os.listdir(dir):
    current_sample = []
    f = open(dir + file)
    for line in f.readlines():
        current_sample.append(line)        
    print(f"Perplexity of {dir + file}: {perplexity_kneser_ney(current_sample)}")
    all_samples.extend(current_sample)
print(f"Perplexity of all files in {dir}: {perplexity_kneser_ney(all_samples)}")

print()

all_samples = []
dir = 'Test Samples/coeff_1_laplace/'
for file in os.listdir(dir):
    current_sample = []
    f = open(dir + file)
    for line in f.readlines():
        current_sample.append(line)        
    print(f"Perplexity of {dir + file}: {perplexity_laplace(current_sample)}")
    all_samples.extend(current_sample)
print(f"Perplexity of all files in {dir}: {perplexity_laplace(all_samples)}")

print()

all_samples = []
dir = 'Test Samples/coeff_1_no/'
for file in os.listdir(dir):
    current_sample = []
    f = open(dir + file)
    for line in f.readlines():
        current_sample.append(line)        
    print(f"Perplexity of {dir + file}: {perplexity_without_smoothing(current_sample)}")
    all_samples.extend(current_sample)
print(f"Perplexity of all files in {dir}: {perplexity_without_smoothing(all_samples)}")


def calc_top_5(bigram_probs):
    vocab = pickle.load(open('Checkpoints/vocab.pkl', 'rb'))
    flattened_array = bigram_probs.flatten()
    sorted_indices = np.argsort(flattened_array)
    top5_indices = sorted_indices[-5:]
    top5_indices_2d = np.unravel_index(top5_indices, bigram_probs.shape)
    top5_indices_2d = np.column_stack((top5_indices_2d[0], top5_indices_2d[1]))

    for i in top5_indices_2d:
        print(vocab[i[0]], vocab[i[1]], bigram_probs[i[0]][i[1]])

normal_bigram_probs = pickle.load(open('Checkpoints/bigram_probs.pkl', 'rb'))
laplace_bigram_probs = pickle.load(open('Checkpoints/bigram_probs_laplace.pkl', 'rb'))
kneser_ney_bigram_probs = pickle.load(open('Checkpoints/bigram_probs_kneser.pkl', 'rb'))

print("Top 5 bigrams before smoothing")
calc_top_5(normal_bigram_probs)

print("Top 5 bigrams after laplace smoothing")
calc_top_5(laplace_bigram_probs)

print("Top 5 bigrams after kneser-ney smoothing")
calc_top_5(kneser_ney_bigram_probs)