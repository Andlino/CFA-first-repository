# new attempt at word embeddings in python
# Tim Runck
# November 2019

# Python program to generate word vectors using Word2Vec 
  
# importing all necessary modules 
#%%
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 

import os
import pandas as pd
import feather as f

#%%

testframe = f.read_dataframe('C:/Users/au615270/Documents/R/sampledatanew.feather')

#%%

test = pd.DataFrame(testframe['referat'][1:1000])

data = []

for g in test['referat']:
    for i in sent_tokenize(g): 
        temp = [] 
      
        # tokenize the sentence into words 
        for j in word_tokenize(i): 
            temp.append(j.lower()) 
  
        data.append(temp) 


#%%

model = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                             window = 5, sg = 5) 

#model.save("modelnewest.bin")
#%%

from sklearn.decomposition import PCA
from matplotlib import pyplot

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()

###################################################################

#%%
import multiprocessing
from gensim.models import Word2Vec


def train_word2vec(data):
    data = gensim.models.word2vec.LineSentence(data)
    return Word2Vec(data, size=200, window=5, min_count=5, workers=multiprocessing.cpu_count())




#%%

keys = ['økonomi', 'administrationen', 'risikobaserede', 'indstilles', 'ærøskøbing']

embedding_clusters = []
word_clusters = []
for word in keys:
    embeddings = []
    words = []
    for similar_word, _ in model.most_similar(word, topn=60):
        words.append(similar_word)
        embeddings.append(model[similar_word])
    embedding_clusters.append(embeddings)
    word_clusters.append(words)

#%%

from sklearn.manifold import TSNE
import numpy as np

embedding_clusters = np.array(embedding_clusters)
n, m, k = embedding_clusters.shape
tsne_model_en_2d = TSNE(perplexity=35, n_components=2, init='pca', n_iter=5000, random_state=32)
embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)

#%%

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


tsne_plot_similar_words('Similar words from 1000 referat sample', keys, embeddings_en_2d, word_clusters, 0.7,
                        'similar_words.png')

#%%

vectors = model.wv
vectors.save_word2vec_format("vect.txt",binary = False)
vectors.save_word2vec_format("vect.tsv",binary = False)

print("Vocabulary Size: {0}".format(len(vectors.vocab)))

for i,w in enumerate(vectors.vocab):
    print(w)
    if i>4:
        break

#%%

VOCAB_SIZE = len(vectors.vocab)
EMBEDDING_DIM = vectors["ad"].shape[0]

w2v = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))

#%%

tsv_file_path = FOLDER_PATH+"/metadata.tsv"
with open(tsv_file_path,'w+', encoding='utf-8') as file_metadata:
    for i,word in enumerate(vectors.index2word[:VOCAB_SIZE]):
        w2v[i] = model[word]
        file_metadata.write(word+'\n')

#%%
