import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dropout, Bidirectional, LSTM, Reshape, Conv2D, MaxPool2D, Flatten, Dense, Concatenate
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
tf.__version__

import pandas as pd
from sklearn.model_selection import train_test_split

from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import numpy as np

import re
from tqdm import tqdm

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
###################################################################################
#Tutorials referenced and dataset links
#https://github.com/yoonkim/CNN_sentence
#https://github.com/dennybritz/cnn-text-classification-tf/blob/master/text_cnn.py
#
#url = 'https://query.data.world/s/oxsccggigfhg7gcpovieducktnk2oa'
#dataset = tf.keras.utils.get_file('text_emotion.csv', url, cache_dir='.', cache_subdir='')
###################################################################################
#Functions
def preprocess(message):
    """
    This function takes a string as input, then performs these operations: 
        - lowercase
        - remove URLs
        - remove ticker symbols 
        - removes punctuation
        - removes any single character tokens
    Parameters
    ----------
        message : The text message to be preprocessed
    Returns
    -------
        text: The preprocessed text
    """ 
    # Lowercase the twit message
    text = message.lower()
    # Replace URLs with a space in the message
    text = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', text)
    # Replace ticker symbols with a space. The ticker symbols are any stock symbol that starts with $.
    text = re.sub('\$[a-zA-Z0-9]*', ' ', text)
    # Replace StockTwits usernames with a space. The usernames are any word that starts with @.
    text = re.sub('\@[a-zA-Z0-9]*', ' ', text)
    # Replace everything not a letter or apostrophe with a space
    text = re.sub('[^a-zA-Z\']', ' ', text)
    # Remove single letter words
    text = ' '.join( [w for w in text.split() if len(w)>1] )
    
    return text

def tokenize_text(text):
  return [word for word in word_tokenize(text) if (word.isalpha()==1)]

def build_classifier_model(embedding_dim, sentence_length, lstm_units, num_filters, conv_kernel_size, conv_strides, pool_kernel_size, pool_strides, pooling_drop_rate):
    text_input = tf.keras.layers.Input(shape=(sentence_length), dtype=tf.float64, name='text')

    cnn_embedding = Embedding(input_dim=len(word_index) + 1,
                                    output_dim=embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=sentence_length,
                                    trainable=False, name="Embedding_layer")(text_input)

    reshape = Reshape(target_shape=[sentence_length, embedding_dim, -1], name="embedding_reshape")(cnn_embedding)

    conv_kernel_size = 3
    cnn_conv_1 = Conv2D(filters=num_filters, kernel_size=(conv_kernel_size, embedding_dim), strides=conv_strides, padding="valid", use_bias=True, name="Conv2D_layer_1")(reshape)
    cnn_pooling_1 = MaxPool2D(pool_size=(sentence_length - conv_kernel_size + 1, 1), strides=(1, 1), padding='valid')(cnn_conv_1)

    conv_kernel_size = 4
    cnn_conv_2 = Conv2D(filters=num_filters, kernel_size=(conv_kernel_size, embedding_dim), strides=conv_strides, padding="valid", use_bias=True, name="Conv2D_layer_2")(reshape)
    cnn_pooling_2 = MaxPool2D(pool_size=(sentence_length - conv_kernel_size + 1, 1), strides=(1, 1), padding='valid')(cnn_conv_2)

    conv_kernel_size = 5
    cnn_conv_3 = Conv2D(filters=num_filters, kernel_size=(conv_kernel_size, embedding_dim), strides=conv_strides, padding="valid", use_bias=True, name="Conv2D_layer_3")(reshape)
    cnn_pooling_3 = MaxPool2D(pool_size=(sentence_length - conv_kernel_size + 1, 1), strides=(1, 1), padding='valid')(cnn_conv_3)

    cnn_concat = layers.concatenate(inputs=[cnn_pooling_1, cnn_pooling_2, cnn_pooling_3], axis=3)

    flattened_cnn_output = Flatten()(cnn_concat)
    pooling_dropout = Dropout(rate=pooling_drop_rate)(flattened_cnn_output)

    output = Dense(num_classes, activation='softmax', name="Output_Softmax_layer")(pooling_dropout)
    return tf.keras.Model(text_input, output)

def train(name, train_ds, test_ds, embedding_dim, sentence_length, lstm_units, num_filters, conv_kernel_size, conv_strides, pool_kernel_size, pool_strides, pooling_drop_rate, epochs, batch_size):
    K.clear_session()
    
    ##Construct the classifier model
    model = build_classifier_model(embedding_dim, sentence_length, lstm_units, num_filters, conv_kernel_size, conv_strides, pool_kernel_size, pool_strides, pooling_drop_rate)
    model.summary()

    ##Training
    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.losses.CategoricalCrossentropy(name="ce"), 'accuracy']
    metric_monitor = ('val_ce', 'min')

    #ModelCheckpoint callback for saving best model
    checkpoint_filepath = './checkpoints/{}'.format(name)
    #Monitor test RMSE values for each epoch and save model IF value < current min test RMSE value
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, #filepath where the model saved is to be stored
        save_weights_only=False, #save the entire model for simplicity
        monitor=metric_monitor[0], #monitor test RMSE values as required 
        mode=metric_monitor[1], #monitor to find the model with min RMSE value
        save_best_only=True) #to save only the best model, not all models

    callbacks = [model_checkpoint_callback]

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=metrics)
    history = model.fit(x=train_ds,
                        validation_data=test_ds,
                        callbacks=callbacks,
                        batch_size=batch_size,
                        epochs=epochs)
    return history

###################################################################################
##Configure model params
#Global parameters
embedding_dim = 100 #acceptable values: 50, 100, 200, 300
sentence_length = 100
lstm_units = 4

num_filters = 128
conv_kernel_size = (3,3)
conv_strides = (1,1)
pool_kernel_size = (1,1)
pool_strides = (2,2)
pooling_drop_rate = 0.5

epochs = 200
batch_size = 64
###################################################################################
##Import dataset
df = pd.read_csv("./dataset/text_emotion.csv")
print(f"Original length: {len(df)}")

#drop irrelevant features
df.drop(labels=["tweet_id", "author"], axis=1, inplace=True)
#convert content to str type
df.content = df.content.astype(str)
#drop any duplicate content
df.drop_duplicates(subset=['content'],inplace=True)
#drop rows with NA entries
df.dropna(axis=0, how='any', inplace=True)

##Remove noisy words from text
df["text"] = [preprocess(message) for message in tqdm(df.content.values)]
df.drop(columns=["content"], inplace=True)

##Tokenise text
df["text"] = [tokenize_text(message) for message in tqdm(df.text.values)]

X = df['text']

##Process true labels into categorical
class_list = df.sentiment.unique()
label_dict = {}
for index, class_label in enumerate(class_list):
    label_dict[class_label] = index

#numerise the class labels
df['label'] = df.sentiment.replace(label_dict)
df.drop(columns=["sentiment"], inplace=True)

y = df.pop("label")

#print(y)
#print(X)

###################################################################################
#Training experiments
num_classes = len(class_list)

seed = 42
kf = KFold(n_splits=3, shuffle=True, random_state=seed) #initialise KFold tool to create 3 folds with shuffling

histories = {}
train_acc_cv = []
test_acc_cv = []
train_loss_cv = []
test_loss_cv = []

k=0
for train_index, test_index in kf.split(X):
    k+=1
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    y_train = tf.keras.utils.to_categorical(y_train.values, num_classes=len(class_list))
    y_test = tf.keras.utils.to_categorical(y_test.values, num_classes=len(class_list))

    ##Embed input texts for train and test
    embeddings_index = {}
    f = open('./dataset/glove/glove.6B.{}d.txt'.format(embedding_dim))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    vocab_size = 10000
    oov_token = "<OOV>"
    tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(X_train) #Only fit X_train!!!
    word_index = tokenizer.word_index
    print("tokenizer: ", len(word_index))
    #Length 25905

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    padding_type='post'
    truncation_type='post'
    X_test_padded = pad_sequences(X_test_sequences,maxlen=sentence_length, padding=padding_type, truncating=truncation_type)
    X_train_padded = pad_sequences(X_train_sequences,maxlen=sentence_length, padding=padding_type, truncating=truncation_type)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train_padded, y_train))
    train_ds = train_ds.shuffle(buffer_size=len(X_train_padded))
    train_ds = train_ds.batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((X_test_padded, y_test))
    test_ds = test_ds.shuffle(buffer_size=len(X_test_padded))
    test_ds = test_ds.batch(batch_size)

    name = "original_cnn_fold{}".format(k)
    histories[name] = train(name, train_ds, test_ds, embedding_dim, sentence_length, lstm_units, num_filters, conv_kernel_size, conv_strides, pool_kernel_size, pool_strides, pooling_drop_rate, epochs, batch_size)

    train_acc_cv.append(histories[name].history["accuracy"])
    test_acc_cv.append(histories[name].history["val_accuracy"])
    train_loss_cv.append(histories[name].history["loss"])
    test_loss_cv.append(histories[name].history["val_loss"])

##Analyse experimental results
results = {}
results["accuracy"] = np.mean(np.array(train_acc_cv), axis = 0)
results["val_accuracy"] = np.mean(np.array(test_acc_cv), axis = 0)
results["loss"] = np.mean(np.array(train_loss_cv), axis = 0)
results["val_loss"] = np.mean(np.array(test_loss_cv), axis = 0)
results_df = pd.DataFrame(results)
results_df.to_csv("cnn.csv", index=False)

#Plot accuracies
plot_X = [x for x in range(1,epochs+1)]
plot_y1 = tfdocs.plots._smooth(results["accuracy"], std=10)
plot_y2 = tfdocs.plots._smooth(results["val_accuracy"], std=10)
#draw the plots
plt.title("Plot of model accuracy against training epoch")
plt.plot(plot_X, plot_y1, label="Train")
plt.plot(plot_X, plot_y2, label="Test")
plt.legend()
plt.ylabel("Accuracy")
plt.xlabel("Training epoch")
plt.savefig("accuracy_cnn.png")

plt.clf()

#Plot losses
plot_X = [x for x in range(1,epochs+1)]
plot_y1 = tfdocs.plots._smooth(results["loss"], std=10)
plot_y2 = tfdocs.plots._smooth(results["val_loss"], std=10)
#draw the plots
plt.title("Plot of model loss against training epoch")
plt.plot(plot_X, plot_y1, label="Train")
plt.plot(plot_X, plot_y2, label="Test")
plt.legend()
plt.ylabel("Loss")
plt.xlabel("Training epoch")
plt.savefig("loss_cnn.png")
