#====================load dset ===============================================
library(readxl)
library(keras)
library(caret)
df <- read_excel("D:/dataseq_2.xlsx")

summary(df)
dt<-df[c("ID_APEL", "CONTENT", "SSB_CARD", 
         "CATEGORY", "SUBCATEGORY", "RESPONSE")]
summary(dt)

library(data.table)
setnames(dt, "RESPONSE", "y")
setnames(dt, "CONTENT", "x")

colnames(dt)

dvec<-dt[c("x", "y")] 
summary(dvec)
dvec$y<-as.factor(dvec$y)
summary(dvec$y)
length(unique(dvec$y))
down_train<-downSample(x=dvec$x, y=dvec$y)


#===========split train and test =====================================================
set.seed(9000)
maxlen<-50 #max no of words to be considered from each complaint entry
max_words<-20000 #max no of words in the vocabulary
tokenizer<-text_tokenizer(num_words = max_words)%>%
  fit_text_tokenizer(dvec$x)
tokenizer$fit_on_texts(dvec)
sequences<-texts_to_sequences(tokenizer = tokenizer, dvec$x)
typeof(sequences)


train_index<-sample(1:length(sequences), size = 0.75*length(sequences), replace=FALSE)
x_test<-sequences[-train_index]
x_train<-sequences[train_index]
y_train<-dvec[train_index,]$y 
y_test <-  dvec[-train_index,]$y


cat(length(x_train), 'train sequences\n')
cat(length(x_test), 'test sequences\n')
summary(dvec$y)

num_classes <- length(unique(y_train)) + 1
cat(num_classes, '\n')
cat('Vectorizing sequence data...\n')

x_train <- sequences_to_matrix(tokenizer, x_train, mode = 'binary')
x_test <- sequences_to_matrix(tokenizer, x_test, mode = 'binary')

cat('x_train shape:', dim(x_train), '\n')
cat('x_test shape:', dim(x_test), '\n')

cat('Convert class vector to binary class matrix',
    '(for use with categorical_crossentropy)\n')
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)
cat('y_train shape:', dim(y_train), '\n')
cat('y_test shape:', dim(y_test), '\n')

batch_size <- 64
epochs <- 10

cat('Building model...\n')
model <- keras_model_sequential()
model %>%
  layer_dense(units = 512, input_shape = c(max_words)) %>% 
  layer_activation(activation = 'relu') %>% 
  layer_dropout(rate = 0.1) %>% 
  layer_dense(units = num_classes) %>% 
  layer_activation(activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

history <- model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_split = 0.1
)

score <- model %>% evaluate(
  x_test, y_test,
  batch_size = batch_size,
  verbose = 1
)

cat('Test score:', score[[1]], '\n')
cat('Test accuracy', score[[2]], '\n')