import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.tokenize import word_tokenize
import numpy as np
import os
import zipfile
import urllib.request
import matplotlib.pyplot as plt
import json

data_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
urllib.request.urlretrieve(data_url, "annotations_trainval2014.zip")
with zipfile.ZipFile("annotations_trainval2014.zip", "r") as zip_ref:
    zip_ref.extractall("annotations")

# Load pre-trained VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False)

# Function to preprocess and extract image features
def extract_image_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    features = vgg_model.predict(img_array)
    features = np.reshape(features, (features.shape[0], -1))
    return features

# Function to preprocess captions
def preprocess_captions(captions_list, tokenizer, max_length):
    captions_seq = tokenizer.texts_to_sequences(captions_list)
    captions_padded = pad_sequences(captions_seq, maxlen=max_length, padding='post')
    return captions_padded

# Download NLTK tokenizer data
import nltk
nltk.download('punkt')

# Load MS-COCO captions and tokenize them
captions_file_path = "annotations/annotations/captions_train2014.json"
annotations = json.load(open(captions_file_path, 'r'))
captions_list = []
for annot in annotations['annotations']:
    captions_list.append(annot['caption'])

# Tokenize captions
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(captions_list)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions_list)
captions_padded = preprocess_captions(captions_list, tokenizer, max_length)

# Build captioning model
image_features_input = Input(shape=(4096,))
decoder_input = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=vocab_size, output_dim=256, input_length=max_length)(decoder_input)
lstm_layer = LSTM(256)(embedding_layer)
output_layer = Dense(vocab_size, activation='softmax')(lstm_layer)
captioning_model = Model(inputs=[image_features_input, decoder_input], outputs=output_layer)

# Compile the model
captioning_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Train the model (dummy data for demonstration)
image_features = np.random.randn(100, 4096)
decoder_input_data = np.random.randint(1, vocab_size, size=(100, max_length))
decoder_target_data = np.zeros((100, max_length, vocab_size))
for i, seq in enumerate(decoder_input_data):
    for j, idx in enumerate(seq):
        if j > 0:
            decoder_target_data[i, j - 1, idx] = 1

captioning_model.fit([image_features, decoder_input_data], decoder_target_data, epochs=10, batch_size=32)

# Function to generate caption for a given image
def generate_caption(image_path):
    image_features = extract_image_features(image_path)
    input_seq = np.zeros((1, max_length))
    input_seq[0, 0] = tokenizer.word_index['<start>']
    
    caption = []
    for i in range(1, max_length):
        output = captioning_model.predict([image_features, input_seq])
        predicted_word_index = np.argmax(output)
        if predicted_word_index == tokenizer.word_index['<end>']:
            break
        caption.append(tokenizer.index_word[predicted_word_index])
        input_seq[0, i] = predicted_word_index
    
    return ' '.join(caption)

# Example usage: generate caption for an image
image_path = 'example.jpg'
caption = generate_caption(image_path)
print("Generated Caption:", caption)