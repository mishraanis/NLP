
import pickle
import random
from utils import emotion_scores
import numpy as np


beta_values = pickle.load(open('Checkpoints/beta_values.pkl', 'rb'))
bigram_probs = pickle.load(open('Checkpoints/bigram_probs.pkl', 'rb'))
bigram_probs_laplace = pickle.load(open('Checkpoints/bigram_probs_laplace.pkl', 'rb'))
bigram_probs_kneser = pickle.load(open('Checkpoints/bigram_probs_kneser.pkl', 'rb'))
vocab = pickle.load(open('Checkpoints/vocab.pkl', 'rb'))
beta_values_sentence = pickle.load(open('Checkpoints/beta_values_sentence.pkl', 'rb'))


emotion_dict = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}


def generate_sentence(emotion, coeff, max_length=20):
    sentence = []
    current_word = '<SOS>'

    for i in range(max_length):
        current_word = generate_next_word(current_word, emotion, coeff)
        if current_word == "" or current_word == '<EOS>':
            break
        sentence.append(current_word)
    
    return " ".join(sentence)

def generate_next_word(current_word, emotion, coeff):
    if current_word not in vocab:
        raise ValueError(f"{current_word} not found in the vocabulary.")

    word_index = vocab.index(current_word)
    if current_word == '<SOS>':
        next_word_probs = bigram_probs_kneser[word_index]
    else:
        next_word_probs = bigram_probs_kneser[word_index] + (coeff * beta_values[word_index, :, emotion_dict[emotion]])
        next_word_probs -= np.min(next_word_probs)
        next_word_probs /= np.max(next_word_probs)


    # next_word_index = list(next_word_probs).index(max(next_word_probs))
    try:
        next_word_index = random.choices(range(len(next_word_probs)), weights=next_word_probs)[0]
    except:
        return ""

    next_word = list(vocab)[next_word_index]

    return next_word


generate_sentence('joy', 0.5)


# #### sample level beta values


for emotion in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']:
    with open('Test Samples/sample_level/gen_' + emotion + '.txt', 'w') as f:
        for i in range(50):
            while True:
                sentence = generate_sentence(emotion, 1)
                if len(sentence.split()) > 6:
                    f.write(sentence + '\n')
                    break


count = 0
with open('Test Samples/sample_level/gen_sadness.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['sadness']]['score']
        
        if(score > 0.5):
            count += 1

print('sadness: ' + str(count))

count = 0
with open('Test Samples/sample_level/gen_joy.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['joy']]['score']
        
        if(score > 0.5):
            count += 1

print('joy: ' + str(count))

count = 0
with open('Test Samples/sample_level/gen_love.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['love']]['score']
        
        if(score > 0.5):
            count += 1

print('love: ' + str(count))

count = 0
with open('Test Samples/sample_level/gen_anger.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['anger']]['score']
        
        if(score > 0.5):
            count += 1

print('anger: ' + str(count))

count = 0
with open('Test Samples/sample_level/gen_fear.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['fear']]['score']
        
        if(score > 0.5):
            count += 1

print('fear: ' + str(count))

count = 0
with open('Test Samples/sample_level/gen_surprise.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['surprise']]['score']
        
        if(score > 0.5):
            count += 1

print('surprise: ' + str(count))


        





for emotion in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']:
    with open('Test Samples/coeff_0_no/gen_' + emotion + '.txt', 'w') as f:
        for i in range(50):
            while True:
                sentence = generate_sentence(emotion, 0)
                if len(sentence.split()) > 6:
                    f.write(sentence + '\n')
                    break


count = 0
with open('Test Samples/coeff_0_no/gen_sadness.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['sadness']]['score']
        
        if(score > 0.5):
            count += 1

print('sadness: ' + str(count))

count = 0
with open('Test Samples/coeff_0_no/gen_joy.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['joy']]['score']
        
        if(score > 0.5):
            count += 1

print('joy: ' + str(count))

count = 0
with open('Test Samples/coeff_0_no/gen_love.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['love']]['score']
        
        if(score > 0.5):
            count += 1

print('love: ' + str(count))

count = 0
with open('Test Samples/coeff_0_no/gen_anger.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['anger']]['score']
        
        if(score > 0.5):
            count += 1

print('anger: ' + str(count))

count = 0
with open('Test Samples/coeff_0_no/gen_fear.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['fear']]['score']
        
        if(score > 0.5):
            count += 1

print('fear: ' + str(count))

count = 0
with open('Test Samples/coeff_0_no/gen_surprise.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['surprise']]['score']
        
        if(score > 0.5):
            count += 1

print('surprise: ' + str(count))



for emotion in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']:
    with open('Test Samples/coeff_1_no/gen_' + emotion + '.txt', 'w') as f:
        for i in range(50):
            while True:
                sentence = generate_sentence(emotion, 1)
                if len(sentence.split()) > 6:
                    f.write(sentence + '\n')
                    break

count = 0
with open('Test Samples/coeff_1_no/gen_sadness.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['sadness']]['score']
        
        if(score > 0.5):
            count += 1

print('sadness: ' + str(count))

count = 0
with open('Test Samples/coeff_1_no/gen_joy.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['joy']]['score']
        
        if(score > 0.5):
            count += 1

print('joy: ' + str(count))

count = 0
with open('Test Samples/coeff_1_no/gen_love.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['love']]['score']
        
        if(score > 0.5):
            count += 1

print('love: ' + str(count))

count = 0
with open('Test Samples/coeff_1_no/gen_anger.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['anger']]['score']
        
        if(score > 0.5):
            count += 1

print('anger: ' + str(count))

count = 0
with open('Test Samples/coeff_1_no/gen_fear.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['fear']]['score']
        
        if(score > 0.5):
            count += 1

print('fear: ' + str(count))

count = 0
with open('Test Samples/coeff_1_no/gen_surprise.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['surprise']]['score']
        
        if(score > 0.5):
            count += 1

print('surprise: ' + str(count))



for emotion in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']:
    with open('Test Samples/coeff_1_laplace/gen_' + emotion + '.txt', 'w') as f:
        for i in range(50):
            while True:
                sentence = generate_sentence(emotion, 1)
                if len(sentence.split()) > 6:
                    f.write(sentence + '\n')
                    break


count = 0
with open('Test Samples/coeff_1_laplace/gen_sadness.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['sadness']]['score']
        
        if(score > 0.5):
            count += 1

print('sadness: ' + str(count))

count = 0
with open('Test Samples/coeff_1_laplace/gen_joy.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['joy']]['score']
        
        if(score > 0.5):
            count += 1

print('joy: ' + str(count))

count = 0
with open('Test Samples/coeff_1_laplace/gen_love.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['love']]['score']
        
        if(score > 0.5):
            count += 1

print('love: ' + str(count))

count = 0
with open('Test Samples/coeff_1_laplace/gen_anger.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['anger']]['score']
        
        if(score > 0.5):
            count += 1

print('anger: ' + str(count))

count = 0
with open('Test Samples/coeff_1_laplace/gen_fear.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['fear']]['score']
        
        if(score > 0.5):
            count += 1

print('fear: ' + str(count))

count = 0
with open('Test Samples/coeff_1_laplace/gen_surprise.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['surprise']]['score']
        
        if(score > 0.5):
            count += 1

print('surprise: ' + str(count))




for emotion in ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']:
    with open('Test Samples/coeff_1_kneser/gen_' + emotion + '.txt', 'w') as f:
        for i in range(50):
            while True:
                sentence = generate_sentence(emotion, 1)
                if len(sentence.split()) > 6:
                    f.write(sentence + '\n')
                    break


count = 0
with open('Test Samples/coeff_1_kneser/gen_sadness.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['sadness']]['score']
        
        if(score > 0.5):
            count += 1

print('sadness: ' + str(count))

count = 0
with open('Test Samples/coeff_1_kneser/gen_joy.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['joy']]['score']
        
        if(score > 0.5):
            count += 1

print('joy: ' + str(count))

count = 0
with open('Test Samples/coeff_1_kneser/gen_love.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['love']]['score']
        
        if(score > 0.5):
            count += 1

print('love: ' + str(count))

count = 0
with open('Test Samples/coeff_1_kneser/gen_anger.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['anger']]['score']
        
        if(score > 0.5):
            count += 1

print('anger: ' + str(count))

count = 0
with open('Test Samples/coeff_1_kneser/gen_fear.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['fear']]['score']
        
        if(score > 0.5):
            count += 1

print('fear: ' + str(count))

count = 0
with open('Test Samples/coeff_1_kneser/gen_surprise.txt', 'r') as f:
    for line in f.readlines():
        score = emotion_scores(line)[emotion_dict['surprise']]['score']
        
        if(score > 0.5):
            count += 1

print('surprise: ' + str(count))




