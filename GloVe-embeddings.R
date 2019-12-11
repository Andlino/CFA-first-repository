# Word Embedding Project Tim Runck, Massimo Graae Losinno, Matt Loftis
# November 2019
# Aarhus University


#load in data from archive


library(keras)
library(stringr)
library(purrr)
library(tm)

#preprocess data
out <- combo_set[combo_set$referat != "", ]

wordcounts <- out$referat %>% 
  map(function(x) str_count(x, boundary("word"))) %>%
  unlist()

out$referat = removeWords(out$referat, stopwords("danish"))
out$referat = removeNumbers(out$referat)
out$referat <- gsub("\\s*(?<!\\S)[a-zA-Z]{1,2}(?!\\S)", "", out$referat, perl=T)
out$referat <- gsub("[^\\w\\s]", "", out$referat, perl=T)


library(tm)
library(SnowballC)
library(quanteda)
library(stringr)
texts <- gsub(":", " ", df$referat, fixed = T)

quanteda_options("language_stemmer" = "danish")

texts <- tokens(texts, what = "word",
                remove_numbers = T,
                remove_punct = T,
                remove_symbols = T,
                remove_separators = T,
                remove_hyphens = T,
                remove_url = T,
                verbose = T)

texts <- tokens_tolower(texts)
texts <- tokens_remove(texts, stopwords("danish"))
#texts <- tokens_wordstem(texts)
texts <- tokens_remove(texts, stopwords("danish"))

texts <- sapply(texts, function(x) {
  if (length(x) > 200) x <- x[1:200]
  return(paste(x, collapse = " "))
})

texts <- tokens(texts)

# get actual dfm
txt.mat <- dfm(texts)
txt.mat <- txt.mat[, colSums(txt.mat) > 3]

# # filter one-letter words
txt.mat <- txt.mat[, str_length(colnames(txt.mat)) > 3]

topfeatures(txt.mat, n = 200)
names(topfeatures(txt.mat, n = 300))

tokens <- as.list(texts)

#Glove embeddings

library(text2vec)


# Create iterator over tokens
tokens <- space_tokenizer(out$referat)
# Create vocabulary. Terms will be unigrams (simple words).
it = itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)

vocab <- prune_vocabulary(vocab, term_count_min = 15L)

# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)
# use window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)

glove = GlobalVectors$new(word_vectors_size = 50, vocabulary = vocab, x_max = 10)
#glove$fit(tcm, n_iter = 20)
wv_main = fit_transform(tcm, glove, n_iter = 20)


wv_context = glove$components
dim(wv_context)

word_vectors = wv_main + t(wv_context)

#find_similar_words("", word_vectors)

library(dplyr)
library(text2vec)

find_similar_words <- function(word, word_vectors, n = 5) {
  similarities <- word_vectors[word, , drop = FALSE] %>%
    sim2(word_vectors, y = ., method = "cosine")
  
  similarities[,1] %>% sort(decreasing = TRUE) %>% head(n)
}

find_similar_words("digitalisering", word_vectors, n=50)

word_vectors = wv_main + t(wv_context)


## Testing cosine sim on resp terms

responsiveness <- c("vælger", "klage", "protest", "borger", "forælder", "pendlere", "modtager", "rettigheder", "forpligtelse", "sagsbehandler")

table(responsiveness %in% rownames(word_vectors))

library(data.table)

responsiveness <- c("vælger", "klage", "protest", "borger", "forælder", "pendlere", "modtager", "rettigheder", "forpligtelse", "sagsbehandler")

dfs <- list()
for (i in responsiveness) {
  
  cosinesimilar <- tryCatch(as.data.frame(find_similar_words(c(i), word_vectors, n=20), error = function(e) print(NA)))
  key <- i
  df <- data.frame(consinesimilar = cosinesimilar, key = key, stringsAsFactors = F)
  
  dfs[[length(dfs) + 1]] <- df
}

dfcosine <- do.call(rbind, setNames(dfs, NULL))
dfcosine$similar <- rownames(dfcosine)
dfcosine$similar <- removeNumbers(dfcosine$similar)
responsivenessdf <- as.data.frame(unique(dfcosine$similar))

#Plot vectors of interest


### VISUALIZATION ###


library(ggplot2)
library(ggridges)
library(ggpointdensity)
library(viridis)
library(Rtsne)
library(ggplot2)
library(hrbrthemes)
library(plotly)

# HRBR Themes prep ######################################
hrbrthemes::import_roboto_condensed()
d <- read.csv(extrafont:::fonttable_file(), stringsAsFactors = FALSE)
d[grepl("Light", d$FontName),]$FamilyName <- font_rc_light
write.csv(d,extrafont:::fonttable_file(), row.names = FALSE)
extrafont::loadfonts()
#########################################################

# SIMPLE-TSNE

lookup <- c(dfcosine$similar)

responsevectors <- wv_main[row.names(wv_main) %in% lookup, ]
tsne <- Rtsne(word_vectors, perplexity = 50, pca = FALSE)

tsne_plot <- tsne$Y %>%
  as.data.frame() %>%
  mutate(word = row.names(responsevectors)) %>%
  ggplot(aes(x = V1, y = V2, label = word)) + 
  geom_text(size = 3, alpha = .6) +
  theme_ipsum_rc()
tsne_plot

# TSNE with density


tsne_plot <- tsne$Y %>%
  as.data.frame() %>%
  mutate(word = row.names(uncertvec)) %>%
  ggplot(aes(x = V1, y = V2, label = word)) + 
  geom_text(size = 3, alpha = .6) +
  geom_pointdensity() +
  scale_color_viridis() +
  theme_ipsum_rc()

tsne_plot

# CONVEX HULL

#most outer points
convexhull <- chull(tsne$Y)

#plot
plot(tsne$Y, cex = 0.5)
hpts <- chull(tsne$Y)
hpts <- c(hpts, hpts[1])
lines(tsne$Y[hpts, ])

