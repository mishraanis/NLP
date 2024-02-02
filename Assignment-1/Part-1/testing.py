import collections, re

class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = collections.defaultdict(int)
        self.tokens = []

def learn_vocabulary(self, corpus, num_merges):
    # Count character frequencies in the corpus
    word_freqs = collections.defaultdict(int)
    for word in corpus:
        # print("word: ", word)
        _word = ' '.join(list(word)) + ' </w>'
        # print("_word: ", _word)
        word_freqs[_word] += 1

    # Create the initial vocabulary
    for word, freq in word_freqs.items():
        self.vocab[word] = freq

    # print("self.vocab", self.vocab)
    for word in self.vocab:
        self.tokens.extend(word.split())
    # Learn the split rules and frequencies
    for _ in range(num_merges):
        pairs = collections.defaultdict(int)
        for word, freq in self.vocab.items():
            symbols = word.split()
            # print("symbols: ", symbols)
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq

        # print("pairs: ", pairs)
        if not pairs:
            break

        # Get the most frequent pair
        most_frequent_pair = max(pairs, key=pairs.get)
        # print("most_frequent_pair: ", most_frequent_pair)
        self.merges[most_frequent_pair] += 1
        # print("self.merges: ", self.merges)
        # Merge the most frequent pair in the vocabulary
        new_vocab = {}
        bigram = re.escape(' '.join(most_frequent_pair)) 
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') 
        for word in self.vocab:
            # new_vocab[word] = self.vocab[word]
            new_word = p.sub(''.join(most_frequent_pair), word)
            new_vocab[new_word] = self.vocab[word]
        self.vocab = new_vocab
        
        # traverse the vocab and add the new words to mastervocab
        for word in self.vocab:
            self.tokens.extend(word.split())
        
def _apply_merge_rules(self, sentence):
    for merge in self.merges:
        sentence = sentence.replace(' '.join(merge), ''.join(merge))
    return sentence

def _split_word(self, word):
    if len(word) == 1:
        return [word]

    split_tokens = []
    for i in range(len(word) - 1):
        pair = (word[i], word[i+1])
        if pair in self.merges:
            split_tokens.append(''.join(pair))
        else:
            split_tokens.append(word[i])
    split_tokens.append(word[-1])
    return split_tokens

def tokenize(self, sentence):
    # Apply the learned merge rules which are stored in self.merges to the sentence to be tokenized
    tokens = []
    sentence = list('$'.join(list(sentence.split())))
    # replace $ with </w>
    for i in range(len(sentence)):
        if(sentence[i] == '$'):
            sentence[i] = '</w>'
    # convert list to a comma separated string
    sentence = '_'+'_,_'.join(sentence)
    print("sentence: ", sentence)
    # sentence = sentence.replace(' ', ',')
    # print("sentence: ", sentence)
    for merge_rules in self.merges:
        print("merge_rules: ", merge_rules)
        rule_with_spaces = '_'+'_,_'.join(merge_rules)+'_'
        # Remove the comma from the rule
        rule_without_comma = '_'+(''.join(merge_rules))+ '_'
        print("rule_with_spaces:", rule_with_spaces, ", rule_without_comma: ", rule_without_comma)
        # Replace the rule in the sentence
        sentence = sentence.replace(rule_with_spaces, rule_without_comma)
        if(sentence[0] != '_'):
            sentence = '_' + sentence
        # sentence = sentence.replace(','.join(merge_rules), ''.join(merge_rules))
        # sentence = sentence.replace(' ' + merge_rules + ' ', ' ' + ''.join(merge_rules.split(',')) + ' ')
        # sentence = sentence.replace(' ', '')
        print("sentence: ", sentence)
    return sentence.replace('_', '')
    # sentence = sentence.replace('_', '')
    # print("sentence: ", sentence)
    # tokens = []
    # for word in sample.split():
    #     word_tokens = []
    #     for char in word:
    #         if char in self.vocab:
    #             word_tokens.append(char)
    #         else:
    #             word_tokens.extend(self._split_word(char))
    #     tokens.extend(word_tokens)
    # return tokens
        
Tokenizer.learn_vocabulary = learn_vocabulary
Tokenizer.tokenize = tokenize
Tokenizer._apply_merge_rules = _apply_merge_rules

corpus = []
with open('../Assignment-1/Dataset/corpus.txt', 'r') as f:
    for line in f:
        # print("line.strip()", line.strip())
        corpus.extend(line.strip().split())

tokenizer = Tokenizer()
tokenizer.learn_vocabulary(corpus, 1000)

tokens = set(tokenizer.tokens)
# write tokens to tokens.txt
with open('./Part-1/tokens.txt', 'w') as f:
    for token in tokens:
        # replace </w> with $
        token = token.replace('</w>', '$')
        f.write(token + '\n')

with open('./Part-1/merges.txt', 'w') as f:
    for merge, freq in tokenizer.merges.items():
        f.write(merge[0] + ',' + merge[1] + '\n')

sample_corpus = []
with open('./Part-1/sample_corpus.txt', 'r') as f:
    for line in f:
        sample_corpus.append(line.strip())

for sentence in sample_corpus:
    # print("sentence: ", sentence)
    with open('./Part-1/tokenized_samples.txt', 'a') as f:
        f.write(tokenizer.tokenize(sentence) + '\n')
        # break
