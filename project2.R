rm(list=ls())

library(tm)
library(wordcloud)
library(ggplot2)
library(scales)
library(xgboost)

#setting the Directory where the file have been placed.
setwd("E:/data/project 2/Text Classification - HealthCare")
hospdata=read.csv("TextClassification_Data.csv")

#Exploatory Data analysis

dim(hospdata)
str(hospdata)
summary(hospdata)
head(hospdata)
colnames(hospdata)
levels(hospdata$sub_categories)
levels(hospdata$categories)
levels(hospdata$previous_appointment)



#feature engineering

#removing the unimportant variables
hospdata$fileid<-NULL
hospdata$ID<-NULL

#reducing to apropriate categories and sub categores
hospdata$previous_appointment<-as.factor(toupper(hospdata$previous_appointment))
hospdata$categories<-as.factor(toupper(hospdata$categories))
hospdata$sub_categories<-as.factor(toupper(hospdata$sub_categories))

table(hospdata$sub_categories)
table(hospdata$categories)

#removing the least frequent category
notusefull<-which(hospdata$sub_categories=="JUNK")
hospdata<-hospdata[-notusefull,]

#Reducing levels of categories and sub_category
hospdata$sub_categories<-droplevels(hospdata$sub_categories)
hospdata$categories<-droplevels(hospdata$categories)

table(hospdata$previous_appointment)

#Converting Nes/No to binary 
hospdata$previous_appointment<-ifelse(hospdata$previous_appointment=="YES",1,0)
table(hospdata$previous_appointment)
hospdata$previous_appointment<-as.integer(hospdata$previous_appointment)

#converting text data from factor to character
hospdata$SUMMARY<-as.character(hospdata$SUMMARY)
hospdata$DATA<-as.character(hospdata$DATA)

#Removing missing data in SUMMARY
r1<-which(hospdata$SUMMARY=="")
which(hospdata$SUMMARY=="")
length(unique(r1))
hospdata<-hospdata[-r1,]

#plots

ggplot(hospdata,aes(x=categories)) + geom_bar(fill= "DarkSlateBlue")
ggplot(hospdata,aes(x=as.factor(previous_appointment)))+ggtitle("'0' = NO , '1' = YES")+xlab("previous_appointment")+ geom_bar(fill= "Red")
ggplot(hospdata,aes(x=as.numeric(sub_categories)))+xlab("sub_categories")+ geom_bar(fill="Black")+scale_x_continuous(breaks=pretty_breaks(n=20))





#textmining
summarycorpus = Corpus(VectorSource(hospdata$SUMMARY))
summarycorpus = tm_map(summarycorpus, stripWhitespace)
summarycorpus = tm_map(summarycorpus, removePunctuation)
summarycorpus = tm_map(summarycorpus, removeNumbers)
summarycorpus = tm_map(summarycorpus, content_transformer(tolower))
summarycorpus = tm_map(summarycorpus, removeWords, stopwords('english'))
summarycorpus = tm_map(summarycorpus, stemDocument, language="english")


Datacorpus = Corpus(VectorSource(hospdata$DATA))
Datacorpus = tm_map(Datacorpus, stripWhitespace)
Datacorpus = tm_map(Datacorpus, removePunctuation)
Datacorpus = tm_map(Datacorpus, removeNumbers)
Datacorpus = tm_map(Datacorpus, content_transformer(tolower))
Datacorpus = tm_map(Datacorpus, removeWords, stopwords('english'))
Datacorpus = tm_map(Datacorpus, stemDocument, language="english")

#wordcloud
graphics.off()
wordcloud(summarycorpus, max.words = 200, scale=c(3, .1), colors=brewer.pal(6, "Dark2"))
wordcloud(Datacorpus, max.words = 200, scale=c(3, .1), colors=brewer.pal(6, "Dark2"))


#Converting the corpus into document term matrix
DATAdtm <- DocumentTermMatrix(Datacorpus, control = list(weighting = weightTfIdf))
DATAdtm <- removeSparseTerms(DATAdtm, 0.99)
DATAdtms <- as.matrix(DATAdtm)



#Converting the corpus into document term matrix(SUMMRY)
summarydtm <- DocumentTermMatrix(summarycorpus, control = list(weighting = weightTfIdf))
summarydtms <- removeSparseTerms(summarydtm, 0.99)
summarydtms <- as.matrix(summarydtms)

#cbinding dtm with orignal data
final<-cbind(DATAdtms,summarydtms,hospdata[,c(3:5)])

#train test split

library(caTools)
set.seed(101) 
sample = sample.split(final$categories, SplitRatio = 0.7)
trainmodel = subset(final, sample == TRUE)
testmodel  = subset(final, sample == FALSE)

#changing datatypes for fiiting the model
trainmodel$sub_categories<-as.numeric(trainmodel$sub_categories)
testmodel$sub_categories<-as.numeric(testmodel$sub_categories)
trainmodel$categories<-as.numeric(trainmodel$categories)
testmodel$categories<-as.numeric(testmodel$categories)

testmodel$categories<-as.factor(testmodel$categories)
trainmodel$sub_categories<-as.factor(trainmodel$sub_categories)
testmodel$sub_categories<-as.factor(testmodel$sub_categories)
trainmodel$categories<-as.factor(trainmodel$categories)

##Naive bayes model building for categories class
library(e1071)
#naive bayes 
t1<-Sys.time()
model_cat = naiveBayes(categories ~ ., data = trainmodel,type="raw")
t2<-Sys.time()
tnaivecat=t2-t1
#predict using test data
test_pred_cat = predict(model_cat, testmodel[,-749], type = "class")

#Visualizing the confusion matrix
xtab_cat = table(observed = testmodel[,749], predicted = test_pred_cat)
library(caret)
confusionMatrix(xtab_cat)

##Naive bayes model building for sub_category class
library(e1071)
#naive bayes 
t1<-Sys.time()
model_subcat = naiveBayes(sub_categories ~ ., data = trainmodel,type="raw")
t2<-Sys.time()
tnaivesubcat=t2-t1
#predict using test data
test_pred_subcat = predict(model_subcat, testmodel[,-750], type = "class")

#Visualizing the confusion matrix
xtab_subcat = table(observed = testmodel[,750], predicted = test_pred_subcat)

confusionMatrix(xtab_subcat)



#Xgboost model building for categories class

#changing datatypes for fiiting the model
trainmodel$sub_categories<-as.numeric(trainmodel$sub_categories)
testmodel$sub_categories<-as.numeric(testmodel$sub_categories)
trainmodel$categories<-as.numeric(trainmodel$categories)
testmodel$categories<-as.numeric(testmodel$categories)

train_matrix_cat <- xgb.DMatrix(as.matrix(trainmodel[,-749]), label = trainmodel$categories )
t1<-Sys.time()
xgb_cat 	<- xgboost(train_matrix_cat, max.depth = 40, eta = 0.3, nround = 13, objective = "multi:softmax", num_class =6)
t2<-Sys.time()
txgbcat=t2-t1
#Applying model to test data
test_pred_catxg = predict(xgb_cat, as.matrix(testmodel[,-749]), type = "class")
#confusion matirx
xtab_catxg = table(observed = as.numeric(testmodel[,749]), predicted = test_pred_catxg)

confusionMatrix(xtab_catxg)


#Xgboost model building for sub_categories class
train_matrix_subcat <- xgb.DMatrix(as.matrix(trainmodel[,-750]), label =trainmodel$sub_categories)
t1<-Sys.time()
xgb_subcat 	<- xgboost(train_matrix_subcat, max.depth = 45, eta = 0.3, nround = 35, objective = "multi:softmax", num_class =30)
t2<-Sys.time()
xgbsubcat=t2-t1
#Applying model to test data
test_pred_subcatxg = predict(xgb_subcat, as.matrix(testmodel[,-750]), type = "class")
#confusion Matirx
xtab_subcatxg = table(observed = as.numeric(testmodel[,750]), predicted = test_pred_subcatxg)

confusionMatrix(xtab_subcatxg)

