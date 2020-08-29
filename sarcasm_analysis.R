setwd('/Users/sean/Desktop/Syracuse University/Semester 2/IST707 Data Analytics/project')

library(jsonlite)
library(tm)
library("wordcloud")

headline_document <- stream_in(file('Sarcasm_Headlines_Dataset_v2.json'))
str(headline_document)

##########################################################################################################################################

## Data Preprocessing
headline_document <- headline_document[,-3] # getting rid of the source URL
headline_document

words.vec <- VectorSource(headline_document$headline)
words.corpus <- Corpus(words.vec)
words.corpus <- tm_map(words.corpus, removePunctuation)

# want to know which headlines have numeric values in them: 1 for yes, 0 for no
numbers <- ifelse(grepl("\\d", headline_document$headline), 1, 0)

headline_tdm <- DocumentTermMatrix(words.corpus, control = list(removeNumbers = T))
headline_tdm <- removeSparseTerms(headline_tdm, .995)

headline_tdm_df <- as.data.frame(as.matrix(headline_tdm))
headline_tdm_df <- cbind(has_numbers = numbers, headline_tdm_df)
headline_tdm_df <- cbind(is_sarcastic = headline_document$is_sarcastic, headline_tdm_df)
headline_tdm_df$has_numbers <- as.factor(headline_tdm_df$has_numbers)
headline_tdm_df$is_sarcastic <- as.factor(headline_tdm_df$is_sarcastic)
length(which(rowSums(headline_tdm_df[,-c(1,2)]) < 2))
headline_tdm_df <- headline_tdm_df[-which(rowSums(headline_tdm_df[,-c(1,2)]) < 2),] # get rid of observations with one or less words

# want to get sentiment score for each headline since many vars will be removed due to high sparsity, we want to keep as much info as possible using sentiment of full sentence
#install.packages("sentimentr")
library(sentimentr)
sentiment_scores <- sapply(headline_document[rownames(headline_tdm_df),]$headline, function(x) sentiment(x)$sentiment, USE.NAMES = FALSE)
sentiment_scores_sums <- sapply(sentiment_scores, sum)
sentiment_scores_scaled <- sentiment_scores_sums * 10

headline_tdm_df$sentiment_score <- sentiment_scores_scaled

str(headline_tdm_df)

#headline_tfidf <- DocumentTermMatrix(words.corpus, control = list(weighting = weightTfIdf, removeNumbers = T))
#headline_tfidf <- removeSparseTerms(headline_tfidf, .995)

#headline_tfidf_df <- as.data.frame(as.matrix(headline_tfidf))
#headline_tfidf_df <- cbind(has_numbers = numbers, headline_tfidf_df)
#headline_tfidf_df <- cbind(is_sarcastic = headline_document$is_sarcastic, headline_tfidf_df)
#headline_tfidf_df$has_numbers <- as.factor(headline_tfidf_df$has_numbers)
#headline_tfidf_df$is_sarcastic <- as.factor(headline_tfidf_df$is_sarcastic)
#length(which(rowSums(headline_tfidf_df[,-c(1,2)]) < 2))
#headline_tfidf_df <- headline_tfidf_df[-which(rowSums(headline_tfidf_df[,-c(1,2)]) < 1),]
#str(headline_tfidf_df)

set.seed(11)
training_indexes <- sample.int(nrow(headline_tdm_df),0.8*nrow(headline_tdm_df))
training_set <- headline_tdm_df[training_indexes,]
testing_set <- headline_tdm_df[-training_indexes,]
testing_set_no_label <- testing_set[,-c(1)]
dim(training_set)
dim(testing_set)

summary(training_set$is_sarcastic)
summary(testing_set$is_sarcastic)

pie(table(training_set$is_sarcastic), main = "Sarcastic to Non-Sarcastic Traning Set Ratio")
pie(table(testing_set$is_sarcastic), main = "Sarcastic to Non-Sarcastic Testing Set Ratio")

##########################################################################################################################################

## EDA

headline_words <- colnames(headline_tdm_df)
stopwords("english")[stopwords("english") %in% headline_words]
# many of these words, especially pronouns may be defining words so we will not remove stopwords

document_sizes <- rowSums(headline_tdm_df[,3:ncol(headline_tdm_df)-1])
summary(document_sizes)

tapply(headline_tdm_df$sentiment_score, headline_tdm_df$is_sarcastic, mean)
#0           1 
#0.010063051 0.006948815 
summary(headline_tdm_df$sentiment_score)
#Min.   1st Qu.    Median      Mean   3rd Qu.      Max. 
#-18.33333  -1.66410   0.00000   0.08677   1.88982  19.20000 
ggplot(headline_tdm_df, aes(x=sentiment_score, fill=is_sarcastic)) + geom_histogram()

sarcastic_df <- headline_tdm_df[headline_tdm_df$is_sarcastic==1,]
ggplot(sarcastic_df, aes(x=sentiment_score)) + geom_histogram(binwidth = 1) + xlim(-20,20) + ggtitle("Sentiment Scores for Sarcastic Headlines")

real_df <- headline_tdm_df[headline_tdm_df$is_sarcastic==0,]
ggplot(real_df, aes(x=sentiment_score)) + geom_histogram(binwidth = 1) + xlim(-20,20) + ggtitle("Sentiment Scores for Non-Sarcastic Headlines")
  
sort(colSums(sarcastic_df[,-c(1,2,ncol(sarcastic_df))]))[1:20] # least common sarcastic headline words
sort(colSums(sarcastic_df[,-c(1,2,ncol(sarcastic_df))]),decreasing =TRUE)[1:20] # most common sarcastic headline words

sort(colSums(real_df[,-c(1,2,ncol(real_df))]))[1:50] # least common real headline words
sort(colSums(real_df[,-c(1,2,ncol(real_df))]),decreasing =TRUE)[1:20] # most common real headline words

wordcloud(words = colnames(sarcastic_df[,-c(1,2,ncol(sarcastic_df))]), freq = colSums(sarcastic_df[,-c(1,2,ncol(sarcastic_df))]), min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

wordcloud(words = colnames(real_df[,-c(1,2,ncol(real_df))]), freq = colSums(real_df[,-c(1,2,ncol(real_df))]), min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

##########################################################################################################################################

#PCA dimensionality reduction
#library(dplyr)
#library(FactoMineR)
#library(factoextra)
#words_pca_results <- PCA(headline_tdm_df[,-c(1,2)])
#pca_eigen_values <- get_eigenvalue(words_pca_results)
#head(pca_eigen_values, 150)

##########################################################################################################################################

##  ARM

library(arules)
library(arulesViz)
arm_matrix <- headline_tdm_df
arm_matrix[,-c(1,2)] <- ifelse(arm_matrix[,-c(1,2)] > 0, TRUE, FALSE) # changing factor vars to boolean

# need to bin the sentiment scores into 5 categories, v.negative, negative, neutral, positive, v.positive
binned_sentiments <- cut(headline_tdm_df$sentiment_score, breaks = c(-50,-6,-1,1,6,50), labels = c("v.negative"," negative","neutral","positive","v.positive"))
arm_matrix$sentiment_score <- binned_sentiments

headlines_transactions <- as(arm_matrix,"transactions")
arules::inspect(head(headlines_transactions,10))

word_rules <- apriori(headlines_transactions, parameter = list(supp = 0.001, conf = 0.8))
word_rules <-sort(word_rules, by="confidence", decreasing=TRUE) # sort by confidence in descending order
inspect(word_rules[1:25]) # print top 25
plot(word_rules)

sarcastic_rules<-apriori(data=headlines_transactions, parameter=list(supp=0.001,conf = 0.8), 
                   appearance = list(default="lhs",rhs="is_sarcastic=1"),
                   control = list(verbose=F))
sarcastic_rules <-sort(sarcastic_rules, by="confidence", decreasing=TRUE) # sort by confidence in descending order
inspect(sarcastic_rules[1:25]) # print top 25

non_sarcastic_rules<-apriori(data=headlines_transactions, parameter=list(supp=0.001,conf = 0.8), 
                         appearance = list(default="lhs",rhs="is_sarcastic=0"),
                         control = list(verbose=F))
non_sarcastic_rules <-sort(non_sarcastic_rules, by="confidence", decreasing=TRUE) # sort by confidence in descending order
inspect(non_sarcastic_rules[1:25]) # print top 25

word_rules_low_conf <- apriori(headlines_transactions, parameter = list(supp = 0.01, conf = 0.5))
word_rules_low_conf <-sort(word_rules_low_conf, by="confidence", decreasing=TRUE) # sort by confidence in descending order
trump_rules <- subset(word_rules_low_conf, lhs %in% c("trump", "trumps", "donald"))
inspect(trump_rules[1:20])

trump_nonsarcastic_rules <- subset(non_sarcastic_rules, lhs %in% c("trump", "trumps", "donald"))
inspect(trump_nonsarcastic_rules[1:25])

negative_rules <- subset(word_rules_low_conf, lhs %in% c("sentiment_score=v.negative"))
inspect(negative_rules)

positive_rules <- subset(word_rules_low_conf, lhs %in% c("sentiment_score=v.positive"))
inspect(positive_rules)

##########################################################################################################################################

## Clustering

#k-means
set.seed(10)
kmeans1 <- kmeans(headline_tdm_df[,-c(1)], 2)
kmeans2 <- kmeans(headline_tdm_df[,-c(1)], 4)
kmeans3 <- kmeans(headline_tdm_df[,-c(1)], 8)
kmeans4 <- kmeans(headline_tdm_df[,-c(1)], 16)
kmeans5 <- kmeans(headline_tdm_df[,-c(1)], 32)
kmeans6 <- kmeans(headline_tdm_df[,-c(1)], 64)
kmeans7 <- kmeans(headline_tdm_df[,-c(1)], 128)
kmeans8 <- kmeans(headline_tdm_df[,-c(1)], 256)

kmeans_sse_results <- data.frame(
  kmeans1=kmeans1$tot.withinss,
  kmeans2=kmeans2$tot.withinss,
  kmeans3=kmeans3$tot.withinss,
  kmeans4=kmeans4$tot.withinss,
  kmeans5=kmeans5$tot.withinss,
  kmeans6=kmeans6$tot.withinss,
  kmeans7=kmeans7$tot.withinss,
  kmeans8=kmeans7$tot.withinss)
plot(t(kmeans_sse_results),type = "o",xlab = "K number", ylab = "SSE", main="SSE for different K",x=c(2,4,8,16,32,64,128,256))

kmeans_prediction_df <- data.frame(is_sarcastic = headline_tdm_df$is_sarcastic, prediction = kmeans4$cluster)
table(kmeans_prediction_df)

##############################################

# k-medoid
#install.library("cluster")
library(cluster)
kmedoid1 <- pam(headline_tdm_df[,-c(1)], k=2)
kmedoid2 <- pam(headline_tdm_df[,-c(1)], k=4)
kmedoid3 <- pam(headline_tdm_df[,-c(1)], k=8)
kmedoid4 <- pam(headline_tdm_df[,-c(1)], k=16)
kmedoids_prediction_df <- data.frame(is_sarcastic = headline_tdm_df$is_sarcastic, prediction = kmedoid4$clustering)
table(kmedoids_prediction_df)

##########################################################################################################################################

## Decision Tree
library(rpart)
library(rpart.plot)
library(rattle)

set.seed(11)
dt_model_1 <- rpart(is_sarcastic ~ ., data=training_set, method="class", control=rpart.control(cp=0.001))
plotcp(dt_model_1)
printcp(dt_model_1)

pruned_model_1 <- prune(dt_model_1, cp = 0.0012907)
fancyRpartPlot(pruned_model_1, caption = NULL)

dt_1_predictions <- predict(dt_model_1, testing_set, type="class")
dt_1_confusion_matrix <- table(real = testing_set$is_sarcastic, pred = dt_1_predictions)
accuracy_dt_1 <- sum(diag(dt_1_confusion_matrix)) / sum(rowSums(dt_1_confusion_matrix))
accuracy_dt_1 # 75.4% 

min_bucket_options <- c(2,10,20,40)
for (min_buck in min_bucket_options) {
  dt_model <- rpart(is_sarcastic ~ ., data=training_set, method="class", control=rpart.control(cp=0.0012907, minbucket = min_buck))
  dt_predictions <- predict(dt_model, testing_set, type="class")
  dt_confusion_matrix <- table(real = testing_set$is_sarcastic, pred = dt_predictions)
  accuracy_dt <- sum(diag(dt_confusion_matrix)) / sum(rowSums(dt_confusion_matrix))
  print(accuracy_dt) 
}
#[1] 0.7494888
#[1] 0.7494888
#[1] 0.7494888
#[1] 0.7464213

# want to lower cp to find most important variables. we can then use this for dimensionality reduction
dt_model_low_cp <- rpart(is_sarcastic ~ ., data=training_set, method="class", control=rpart.control(cp=0.0001))
printcp(dt_model_low_cp)

# lowest xerror occurs for cp = 0.00035853
dt_model_2 <- rpart(is_sarcastic ~ ., data=training_set, method="class", control=rpart.control(cp=0.00035853))
dt_2_predictions <- predict(dt_model_2, testing_set_no_label, type="class")
dt_2_confusion_matrix <- table(real = testing_set$is_sarcastic, pred = dt_2_predictions)
accuracy_dt_2 <- sum(diag(dt_2_confusion_matrix)) / sum(rowSums(dt_2_confusion_matrix))
accuracy_dt_2 # 75.5%
fancyRpartPlot(dt_model_2, caption = NULL)

##########################################################################################################################################

# Random Forest
library(randomForest)
rf_model <- randomForest(training_set[,2:ncol(training_set)], training_set$is_sarcastic)
rf_predict <- predict(rf_model, testing_set_no_label, type = "class")
rf_confusion_matrix <- table(real = testing_set$is_sarcastic, pred = rf_predict)
accuracy_rf <- sum(diag(rf_confusion_matrix)) / sum(rowSums(rf_confusion_matrix))
accuracy_rf 
# 78.4%

##########################################################################################################################################


## SVM
library(e1071)
#tune_results_svm1 <- tune(svm, is_sarcastic ~ ., data=training_set, kernel="linear", ranges=list(cost=c(0.01, 0.1, 1, 10)))
#optimal_c_svm1 <- tune_results_svm1$best.performance
# ^ too computationally expensive. could not run with all variables. try using vars from decision tree

# using dt_model_2 's most important variables (78) dt_model_2$variable.importance we can reduce the dimensionality of the data
important_vars <- labels(dt_model_2$variable.importance)
important_vars <- append(important_vars,'is_sarcastic',0)
training_set_reduced <- subset(training_set, select=important_vars)
testing_set_reduced <- subset(testing_set, select=important_vars)
testing_set_reduced_no_label <- testing_set_reduced[,-c(1)]

# Linear SVM

set.seed(12)
svm_linear_model_1 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="linear")
svm_pred_1 <- predict(svm_linear_model_1, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_1 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_1)
accuracy_svm_1 <- sum(diag(svm_confusion_matrix_1)) / sum(rowSums(svm_confusion_matrix_1))
accuracy_svm_1
# 75.8%

# now lowering the cost to 0.1
set.seed(13)
svm_linear_model_2 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="linear", cost=0.1)
svm_pred_2 <- predict(svm_linear_model_2, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_2 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_2)
accuracy_svm_2 <- sum(diag(svm_confusion_matrix_2)) / sum(rowSums(svm_confusion_matrix_2))
accuracy_svm_2
# 75.8%

# now increasing the cost to 10
set.seed(14)
svm_linear_model_3 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="linear", cost=10)
svm_pred_3 <- predict(svm_linear_model_3, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_3 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_3)
accuracy_svm_3 <- sum(diag(svm_confusion_matrix_3)) / sum(rowSums(svm_confusion_matrix_3))
accuracy_svm_3
# 75.8 %

# now decrease the cost to 0.01
set.seed(14)
svm_linear_model_4 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="linear", cost=0.01)
svm_pred_lin4 <- predict(svm_linear_model_4, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_lin4 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_lin4)
accuracy_svm_lin4 <- sum(diag(svm_confusion_matrix_lin4)) / sum(rowSums(svm_confusion_matrix_lin4))
accuracy_svm_lin4
# 75.9 %

##############################################

# RBF

# cost = 1
set.seed(14)
svm_radial_model_1 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="radial")
svm_pred_4 <- predict(svm_radial_model_1, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_4 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_4)
accuracy_svm_4 <- sum(diag(svm_confusion_matrix_4)) / sum(rowSums(svm_confusion_matrix_4))
accuracy_svm_4
# 75.5%

# cost = 0.1
set.seed(14)
svm_radial_model_2 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="radial", cost=0.1)
svm_pred_5 <- predict(svm_radial_model_2, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_5 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_5)
accuracy_svm_5 <- sum(diag(svm_confusion_matrix_5)) / sum(rowSums(svm_confusion_matrix_5))
accuracy_svm_5
# 74.4%

# cost = 10
set.seed(14)
svm_radial_model_3 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="radial", cost=10)
svm_pred_6 <- predict(svm_radial_model_3, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_6 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_6)
accuracy_svm_6 <- sum(diag(svm_confusion_matrix_6)) / sum(rowSums(svm_confusion_matrix_6))
accuracy_svm_6
# 74.9%

##############################################

# Polynomial degree 3
# cost = 1
set.seed(15)
svm_polynomial_model_1 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="polynomial", degree=3)
svm_pred_poly1 <- predict(svm_polynomial_model_1, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_poly1 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_poly1)
accuracy_svm_poly1 <- sum(diag(svm_confusion_matrix_poly1)) / sum(rowSums(svm_confusion_matrix_poly1))
accuracy_svm_poly1
# 71.4%

# cost = 0.1
set.seed(16)
svm_polynomial_model_2 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="polynomial", degree=3, cost=0.1)
svm_pred_poly2 <- predict(svm_polynomial_model_2, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_poly2 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_poly2)
accuracy_svm_poly2 <- sum(diag(svm_confusion_matrix_poly2)) / sum(rowSums(svm_confusion_matrix_poly2))
accuracy_svm_poly2
# 65.9%

# cost = 10
set.seed(16)
svm_polynomial_model_3 <- svm(is_sarcastic ~ ., data=training_set_reduced, kernel="polynomial", degree=3, cost=10)
svm_pred_poly3 <- predict(svm_polynomial_model_3, testing_set_reduced_no_label, type ="class")
svm_confusion_matrix_poly3 <- table(real = testing_set_reduced$is_sarcastic, pred = svm_pred_poly3)
accuracy_svm_poly3 <- sum(diag(svm_confusion_matrix_poly3)) / sum(rowSums(svm_confusion_matrix_poly3))
accuracy_svm_poly3
# 74.0%

##########################################################################################################################################

# Naive Bayes
# running nb on the reduced dimension training and testing sets
set.seed(17)
nb_model_1 <- naiveBayes(is_sarcastic ~ ., data=training_set_reduced)
nb_model_pred_1 <- predict(nb_model_1, testing_set_reduced_no_label, type ="class")
nb_confusion_matrix_1 <- table(real = testing_set_reduced$is_sarcastic, pred = nb_model_pred_1)
accuracy_nb_1 <- sum(diag(nb_confusion_matrix_1)) / sum(rowSums(nb_confusion_matrix_1))
accuracy_nb_1
# 72.0 %

# quite fast for model build time so will try on the original training and testing sets
set.seed(17)
nb_model_2 <- naiveBayes(is_sarcastic ~ ., data=training_set)
nb_model_pred_2 <- predict(nb_model_2, testing_set_no_label, type ="class")
nb_confusion_matrix_2 <- table(real = testing_set$is_sarcastic, pred = nb_model_pred_2)
accuracy_nb_2 <- sum(diag(nb_confusion_matrix_2)) / sum(rowSums(nb_confusion_matrix_2))
accuracy_nb_2
# 72.7 %





