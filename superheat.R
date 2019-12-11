library(data.table)
write.table(rownames(word_vectors), row.names = F, col.names = F, file = "labels.txt")
write.table(word_vectors, row.names = F, col.names = F, sep = "\t", file = "vectors.txt")

# install devtools if you don't have it already
install.packages("devtools")
# install the development version of superheat
devtools::install_github("rlbarter/superheat")
library(superheat)


tstat_freq <- textstat_frequency(txt.mat)
top200 <- head(tstat_freq, 200)

forty.most.commmon.words.df <- tstat_freq[1:40]
forty.most.commmon.words <- as.character(forty.most.commmon.words.df$feature)

onek.most.commmon.words.df <- as.character(tstat_freq$feature)

freq <- colSums(as.matrix(txt.mat))   
freq.df <- data.frame(word = names(freq), count = freq) %>%
  arrange(desc(count))
# view the 10 most frequent words
kable(head(freq.df, 10))


freqtest <- top200$feature

mostsimilar <- test
mostsimilar <- as.data.frame(mostsimilar)
mostsimilar <- mostsimilar$mostsimilar[-c(freqtest)]
c <- which(mostsimilar %in% freqtest)
mostsimilar <- mostsimilar[-c(c)]

library(stringr)



CosineFun <- function(x, y){
  # calculate the cosine similarity between two vectors: x and y
  c <- sum(x*y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)))
  return(c)
}

CosineSim <- function(X) {
  # calculate the pairwise cosine similarity between columns of the matrix X.
  # initialize similarity matrix
  m <- matrix(NA, 
              nrow = ncol(X),
              ncol = ncol(X),
              dimnames = list(colnames(X), colnames(X)))
  cos <- as.data.frame(m)
  
  # calculate the pairwise cosine similarity
  for(i in 1:ncol(X)) {
    for(j in i:ncol(X)) {
      co_rate_1 <- X[which(X[, i] & X[, j]), i]
      co_rate_2 <- X[which(X[, i] & X[, j]), j]  
      cos[i, j] <- CosineFun(co_rate_1, co_rate_2)
      # fill in the opposite diagonal entry
      cos[j, i] <- cos[i, j]        
    }
  }
  return(cos)
}

mostsimilar <- as.character(mostsimilar)

cosine.similarity <- CosineSim(t(word_vectors[mostsimilar, ]))

diag(cosine.similarity) <- NA

superheat(cosine.similarity, 
          
          # place dendrograms on columns and rows 
          row.dendrogram = T, 
          col.dendrogram = T,
          
          # make gridlines white for enhanced prettiness
          grid.hline.col = "white",
          grid.vline.col = "white",
          
          # rotate bottom label text
          bottom.label.text.angle = 90,
          
          legend.breaks = c(-0.1, 0.1, 0.3, 0.5))


cosine.similarity.full <- CosineSim(t(word_vectors[mostsimilar, ]))
dim(cosine.similarity.full)

# (1) the lowest average dissimilarity of the data point to any other cluster, 
#  minus
# (2) the average dissimilarity of the data point to all other data points in 
#     the same cluster
cosineSilhouette <- function(cosine.matrix, membership) {
  # Args:
  #   cosine.matrix: the cosine similarity matrix for the words
  #   membership: the named membership vector for the rows and columns. 
  #               The entries should be cluster centers and the vector 
  #               names should be the words.
  if (!is.factor(membership)) {
    stop("membership must be a factor")
  }
  # note that there are some floating point issues:
  # (some "1" entires are actually sliiightly larger than 1)
  cosine.dissim <- acos(round(cosine.matrix, 10)) / pi
  widths.list <- lapply(levels(membership), function(clust) {
    # filter rows of the similarity matrix to words in the current cluster
    # filter cols of the similarity matrix to words in the current cluster
    cosine.matrix.inside <- cosine.dissim[membership == clust, 
                                          membership == clust]
    # a: average dissimilarity of i with all other data in the same cluster
    a <- apply(cosine.matrix.inside, 1, mean)
    # filter rows of the similarity matrix to words in the current cluster
    # filter cols of the similarity matrix to words NOT in the current cluster
    other.clusters <- levels(membership)[levels(membership) != clust]
    cosine.matrix.outside <- sapply(other.clusters, function(other.clust) {
      cosine.dissim[membership == clust, membership == other.clust] %>%
        apply(1, mean) # average over clusters
    })
    # b is the lowest average dissimilarity of i to any other cluster of 
    # which i is not a member
    b <- apply(cosine.matrix.outside, 1, min)
    # silhouette width is b - a
    cosine.sil.width <- b - a
    data.frame(word = names(cosine.sil.width), width = cosine.sil.width)
  })
  widths.list <- do.call(rbind, widths.list)
  # join membership onto data.frame
  membership.df <- data.frame(word = names(membership), 
                              membership = membership)
  widths.list <- left_join(widths.list, membership.df, by = "word")
  return(widths.list)
}

#Using the cosineSilhouette() function to calculating the average cosine silhouette width for each k, we can plot k versus average cosine silhouette width across all observations for each number of clusters, k.
######################################################################################
######################################################################################
library(cluster)
set.seed(238942)
# calculate the average silhouette width for k=5, ..., 20
sil.width <- sapply(5:20, function(k) {
  # generate k clusters
  membership <- pam(cosine.similarity.full, k = k)
  # calcualte the silhouette width for each observation
  width <- cosineSilhouette(cosine.similarity.full, 
                            membership = factor(membership$clustering))$width
  return(mean(width))
})

######################################################################################
######################################################################################
library(ggplot2)
# plot k verus silhouette width
data.frame(k = 5:20, width = sil.width) %>%
  ggplot(aes(x = k, y = width)) +
  geom_line() + 
  geom_point() +
  scale_y_continuous(name = "Avergae silhouette width")

######################################################################################
######################################################################################
library(cluster)

# perform clustering for k in k.range clusters over N 90% sub-samples 
generateClusters <- function(similarity.mat, k.range, N) {
  random.subset.list <- lapply(1:100, function(i) {
    sample(1:nrow(similarity.mat), 0.9 * nrow(similarity.mat))
  })
  lapply(k.range, function(k) {
    print(paste("k =", k))
    lapply(1:N, function(i) {
      # randomly sample 90% of words
      cosine.sample <- similarity.mat[random.subset.list[[i]], random.subset.list[[i]]]
      # perform clustering
      pam.clusters <- pam(1 - cosine.sample, k = k, diss = TRUE)
    })
  })
}

######################################################################################
######################################################################################
cluster.iterations <- generateClusters(cosine.similarity.full, 
                                       k.range = 5:20, 
                                       N = 100)

######################################################################################
######################################################################################
join.cluster.iterations <- lapply(cluster.iterations, function(list) {
  # for each list of iterations (for a specific k), 
  # full-join the membership vectors into a data frame 
  # (there will be missing values in each column)
  Reduce(function(x, y) full_join(x, y, by = "words"), 
         lapply(list, function(cluster.obj) {
           df <- data.frame(words = names(cluster.obj$clustering), 
                            clusters = cluster.obj$clustering)
         }))
})
# clean column names 
join.cluster.iterations <- lapply(join.cluster.iterations, function(x) {
  colnames(x) <- c("words", paste0("membership", 1:100))
  return(x)
})

library(knitr)
kable(head(join.cluster.iterations[[1]][, 1:8]))



# calculate the pairwise jaccard similarity between each of the cluster 
# memberships accross the common words
# to avoid correlation, we do this pairwise between simulations 1 and 2, 
# and then between simulations 3 and 4, and so on
library(Rcpp)
library(reshape2)
# use Rcpp to speed up the computation
sourceCpp('code/Rcpp_similarity.cpp')
jaccard.similarity <- sapply(join.cluster.iterations, 
                             function(cluster.iteration) {
                               sapply(seq(2, ncol(cluster.iteration) - 1, by = 2), 
                                      function(i) {
                                        # calculate the Jaccard similarity between each pair of columns
                                        cluster.iteration.pair <- cluster.iteration[ , c(i, i + 1)]
                                        colnames(cluster.iteration.pair) <- c("cluster1", "cluster2")
                                        # remove words that do not appear in both 90% sub-samples
                                        cluster.iteration.pair <- cluster.iteration.pair %>%
                                          filter(!is.na(cluster1), !is.na(cluster2))
                                        # Calcualte the Jaccard similarity between the two cluster vectors
                                        RcppSimilarity(cluster.iteration.pair[ , 1], 
                                                       cluster.iteration.pair[ , 2])
                                      })
                             })


# average similarity over simulations
jaccard.similarity.long <- melt(jaccard.similarity)
colnames(jaccard.similarity.long) <- c("iter", "k", "similarity")
# k is the number of clusters
jaccard.similarity.long$k <- jaccard.similarity.long$k + 4
jaccard.similarity.long <- jaccard.similarity.long %>% 
  filter(k <= 20)
# average over iterations
jaccard.similarity.avg <- jaccard.similarity.long %>% 
  group_by(k) %>% 
  summarise(similarity = mean(similarity))


ggplot(jaccard.similarity.long) + 
  geom_boxplot(aes(x = k, y = similarity, group = k)) +
  geom_line(aes(x = k, y = similarity), 
            linetype = "dashed",
            data = jaccard.similarity.avg) +
  ggtitle("Jaccard similarity versus k")


# note that there are some floating point issues in the similarity matrix:
# some "1" entires are actually sliiightly larger than 1, so we round to 
# the nearest 10 dp when calcualting the distance matrix
word.clusters <- pam(acos(round(cosine.similarity.full, 10)) / pi, k = 11, diss = TRUE)
word.clusters.12 <- pam(acos(round(cosine.similarity.full, 10)) / pi, k = 12, diss = TRUE)


# print the cluster medoids
word.clusters$medoids
##  [1] "just"       "struggle"   "american"   "north"      "government"
##  [6] "pushes"     "give"       "murder"     "proposal"   "bombing"   
## [11] "companies"
# convert the membership vector to a factor
word.membership <- factor(word.clusters$clustering)

# print the cluster medoids
word.clusters.12$medoids
##  [1] "just"       "struggle"   "american"   "north"      "government"
##  [6] "pushes"     "children"   "give"       "murder"     "proposal"  
## [11] "bombing"    "companies"
# convert the membership vector to a factor
word.membership.12 <- factor(word.clusters.12$clustering)


# replace integer membership by medoid membership
levels(word.membership) <- word.clusters$medoids
# replace integer membership by medoid membership
levels(word.membership.12) <- word.clusters.12$medoids


# compare the membership vectors with 12 and 13 clusters
word.membership.split <- split(word.membership, word.membership)
word.membership.split.12 <- split(word.membership.12, word.membership.12)
compare.11.12 <- sapply(word.membership.split, function(i) {
  sapply(word.membership.split.12, function(j) {
    sum(names(i) %in% names(j)) / length(i)
  })
})


superheat(compare.11.12, 
          heat.pal = c("white", "grey", "black"),
          heat.pal.values = c(0, 0.1, 1),
          column.title = "11 clusters",
          row.title = "12 clusters",
          bottom.label.text.angle = 90,
          bottom.label.size = 0.4)


# calcualte the cosine silhouette width
cosine.silhouette <- 
  cosineSilhouette(cosine.similarity.full, word.membership)
# arrange the words in the same order as the original matrix
rownames(cosine.silhouette) <- cosine.silhouette$word
cosine.silhouette <- cosine.silhouette[rownames(cosine.similarity.full), ]


# calculate the average width for each cluster
avg.sil.width <- cosine.silhouette %>% 
  group_by(membership) %>% 
  summarise(avg.width = mean(width)) %>% 
  arrange(avg.width)
# add a blank space after each word (for aesthetic purposes)
word.membership.padded <- paste0(word.membership, " ")
# reorder levels based on increasing separation
word.membership.padded <- factor(word.membership.padded, 
                                 levels = paste0(avg.sil.width$membership, " "))




superheat(cosine.similarity.full,
          
          # row and column clustering
          membership.rows = word.membership.padded,
          membership.cols = word.membership.padded,
          
          # top plot: silhouette
          yt = cosine.silhouette$width,
          yt.axis.name = "Cosine\nsilhouette\nwidth",
          yt.plot.type = "bar",
          yt.bar.col = "grey35",
          
          # order of rows and columns within clusters
          order.rows = order(cosine.silhouette$width),
          order.cols = order(cosine.silhouette$width),
          
          # bottom labels
          bottom.label.col = c("grey95", "grey80"),
          bottom.label.text.angle = 90,
          bottom.label.text.alignment = "right",
          bottom.label.size = 0.28,
          
          # left labels
          left.label.col = c("grey95", "grey80"),
          left.label.text.alignment = "right",
          left.label.size = 0.26,
          
          # title
          title = "(a)")




superheat(cosine.similarity.full, 
          
          # row and column clustering
          membership.rows = word.membership.padded,
          membership.cols = word.membership.padded,
          
          # top plot: silhouette
          yt = cosine.silhouette$width,
          yt.axis.name = "Cosine\nsilhouette\nwidth",
          yt.plot.type = "bar",
          yt.bar.col = "grey35",
          
          # order of rows and columns within clusters
          order.rows = order(cosine.silhouette$width),
          order.cols = order(cosine.silhouette$width),
          
          # bottom labels
          bottom.label.col = c("grey95", "grey80"),
          bottom.label.text.angle = 90,
          bottom.label.text.alignment = "right",
          bottom.label.size = 0.28,
          
          # left labels
          left.label.col = c("grey95", "grey80"),
          left.label.text.alignment = "right",
          left.label.size = 0.26,
          
          # smooth heatmap within clusters
          smooth.heat = T,
          
          # title
          title = "(b)")

freq <- data.frame(word = tstat_freq$feature, count = tstat_freq$frequency, stringsAsFactors = F)
freq <- as.matrix(freq)


library(RColorBrewer)
library(wordcloud)
# define a function that takes the cluster name and the membership vector 
# and returns a word cloud
makeWordCloud <- function(cluster, word.membership, words.freq) {
  words <- names(word.membership[word.membership == cluster])
  words.freq <- words.freq[words]
  # make all words black except for the cluster center
  words.col <- rep("black", length = length(words.freq))
  words.col[words == cluster] <- "red"
  # the size of the words will be the frequency from the NY Times headlines
  wordcloud(words, words.freq, colors = words.col, 
            ordered.colors = TRUE, random.order = FALSE, max.words = 80)
}


set.seed(52545)
for (word in levels(word.membership)) {
  makeWordCloud(word, word.membership, words.freq = freq)
}





freplot <- factor(tstat_freq$frequency)
rankplot <- factor(tstat_freq$rank) 
plot(x = freplot[1:50], y = rankplot[1:50], type = "o")
