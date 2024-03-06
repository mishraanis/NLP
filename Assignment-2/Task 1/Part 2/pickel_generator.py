import gensim
import pickle
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Load Word2Vec
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./Word Embeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
with open('./Word Embeddings/word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump(word2vec_model, f)

# Load GloVe
# glove_file = 'path/to/glove_model.txt'
# word2vec_output_file = 'glove2word2vec.txt'
# glove2word2vec(glove_file, word2vec_output_file)
# glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# with open('glove_embeddings.pkl', 'wb') as f:
#     pickle.dump(glove_model, f)

# Load FastText
# fasttext_model = gensim.models.KeyedVectors.load_word2vec_format('path/to/fasttext_model.bin', binary=True)
# with open('fasttext_embeddings.pkl', 'wb') as f:
#     pickle.dump(fasttext_model, f)


# also there code mein inn bin files ko as pickes save krne ka aisa sa code mil nhi rha
# ho skta h word embeddings folder mein ho not sure