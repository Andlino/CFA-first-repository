#%%
import scattertext as st
import spacy
from pprint import pprint
import feather as fea


#%%

df = fea.read_dataframe("C:/Users/au615270/Dropbox/CROW_FAR/First_Repository_CROW_FAR/full model files/GloVe Model/dataframe.feather")

df.iloc[0]

# %%

nlp = spacy.load('en_core_web_sm')
corpus = st.CorpusFromPandas(df, 
                              category_col='RU', 
                              text_col='referat',
                              nlp=nlp).build()

#%%
print(list(corpus.get_scaled_f_scores_vs_background().index[:10]))