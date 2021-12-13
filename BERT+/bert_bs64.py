#pip install -q -U tensorflow-text
#pip install -q tf-models-official

import os
import shutil
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_hub as hub
import tensorflow_text as text
from models.official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

tf.get_logger().setLevel('ERROR')

###################################################################################
#Tutorials referenced and dataset links
#https://www.tensorflow.org/text/tutorials/classify_text_with_bert
#https://www.analyticsvidhya.com/blog/2021/08/training-bert-text-classifier-on-tensor-processing-unit-tpu/

#url = 'https://query.data.world/s/oxsccggigfhg7gcpovieducktnk2oa'
#dataset = tf.keras.utils.get_file('text_emotion.csv', url, cache_dir='.', cache_subdir='')
###################################################################################
#Functions
def build_classifier_model_no_linear():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  #tfhub performs all the preprocessing of the input text before it can be sent into the encoder.
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  #output from encoder.

  #decoder ie. classifier portion
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(len(class_list), activation=None, name='13_classifier')(net)
  return tf.keras.Model(text_input, net)

def build_classifier_model_1_linear(num_neurons):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  #tfhub performs all the preprocessing of the input text before it can be sent into the encoder.
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  #output from encoder.

  #decoder ie. classifier portion
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(num_neurons, activation="relu", name="linear_layer")(net)
  net = tf.keras.layers.Dense(len(class_list), activation=None, name='13_classifier')(net)
  return tf.keras.Model(text_input, net)

def build_classifier_model_1_linear_dropout(num_neurons):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  #tfhub performs all the preprocessing of the input text before it can be sent into the encoder.
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  #output from encoder.

  #decoder ie. classifier portion
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(num_neurons, activation="relu", name="linear_layer")(net)
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(len(class_list), activation=None, name='13_classifier')(net)
  return tf.keras.Model(text_input, net)

def build_classifier_model_2_linear_dropout(num_neurons):
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  #tfhub performs all the preprocessing of the input text before it can be sent into the encoder.
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  #output from encoder.

  #decoder ie. classifier portion
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(num_neurons, activation="relu", name="linear_layer")(net)
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(num_neurons, activation="relu", name="linear_layer2")(net)
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(len(class_list), activation=None, name='13_classifier')(net)
  return tf.keras.Model(text_input, net)

###################################################################################

def train(name, train_ds, test_ds, mode, num_neurons, epochs, batch_size):
  K.clear_session()

  ##Construct the classifier model
  if mode == 0:
    classifier_model = build_classifier_model_no_linear()
  elif mode == 1:
    classifier_model = build_classifier_model_1_linear(num_neurons)
  elif mode == 2:
    classifier_model = build_classifier_model_1_linear_dropout(num_neurons)
  elif mode == 3:
    classifier_model = build_classifier_model_2_linear_dropout(num_neurons)
  classifier_model.summary()

  ##Training
  #logits true because sigmoid not applied yet so not considered probability.
  loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  metrics = [tf.keras.losses.CategoricalCrossentropy(from_logits=True, name="ce"), 'accuracy']
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

  #Create custom optimiser
  steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(0.1*num_train_steps)

  init_lr = 3e-5
  optimizer = optimization.create_optimizer(init_lr=init_lr,
                                            num_train_steps=num_train_steps,
                                            num_warmup_steps=num_warmup_steps,
                                            optimizer_type='adamw')

  classifier_model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=metrics)

  print(f'Training model with {tfhub_handle_encoder}')
  history = classifier_model.fit(x=train_ds,
                                validation_data=test_ds,
                                callbacks=callbacks,
                                batch_size=batch_size,
                                epochs=epochs)

  return history

###################################################################################
##Configure model params
#Global parameters
num_neurons = 0 #[16, 32, 64, 128]
mode = 0 #[0,1,2,3]
batch_size = 64
epochs = 10
###################################################################################
#Get pretrained BERT encoder and preprocessors 
tfhub_handle_encoder = './dataset/bert_en_uncased_L-12_H-768_A-12_4'
tfhub_handle_preprocess = './dataset/bert_en_uncased_preprocess_3'
#print(f'BERT model selected           : {tfhub_handle_encoder}')
#print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)
###################################################################################
##Import data
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

#get class label list
class_list = df.sentiment.unique()
label_dict = {}
for index, class_label in enumerate(class_list):
    label_dict[class_label] = index
#print(label_dict)

#numerise the class labels
df['label'] = df.sentiment.replace(label_dict)
df.drop(columns=["sentiment"], inplace=True)

y = df.pop("label")
X = df["content"]

print(y)
print(X)

###################################################################################
#Training experiments
AUTOTUNE = tf.data.AUTOTUNE
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

  train_ds = tf.data.Dataset.from_tensor_slices((X_train.squeeze(), y_train))
  train_ds = train_ds.shuffle(buffer_size=len(X_train))

  test_ds = tf.data.Dataset.from_tensor_slices((X_test.squeeze(), y_test))
  test_ds = test_ds.shuffle(buffer_size=len(X_test))

  raw_train_ds = train_ds.batch(batch_size)
  test_ds = test_ds.batch(batch_size)

  train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

  name = "bert_fold{}_bs{}".format(k, batch_size)
  histories[name] = train(name, train_ds, test_ds, mode, num_neurons, epochs, batch_size)

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
results_df.to_csv("bert_{}_bs{}.csv".format(mode, batch_size), index=False)

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
plt.savefig("accuracy_bert_{}_bs{}.png".format(mode, batch_size))

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
plt.savefig("loss_bert_{}_bs{}.png".format(mode, batch_size))
