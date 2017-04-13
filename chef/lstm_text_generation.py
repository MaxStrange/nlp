'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import os
import random
import sys

class LossHistory(Callback):
    """
    A class that lets us save the data.
    """
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

# Create the checkpoint directory
checkpointdir = "checkpoints" + os.path.sep
if not os.path.exists(checkpointdir):
    os.makedirs(checkpointdir)

# Get the input datafile or use the default
if len(sys.argv) > 1:
    path = sys.argv[1]
    print("Using", path, "as training data.")
else:
    print("Using tiny-shakespeare.txt since no arg provided.")
    path = "./tiny-shakespeare.txt"
text = open(path).read().lower()
print('corpus length:', len(text))

# Make two dictionaries: {a: 0, b: 1, c: 2, ...} and {0: a, 1: b, 2: c, ...}
chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
# TODO: Figure out what these two lists are used for
sentences = []
next_chars = []
# Iterate i in steps of maxlen (so i = 0, i = 40, i = 80, etc.)
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

output = open("output.txt", 'w')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
maxits = 800
for iteration in range(1, maxits):
    print()
    print('-' * 50)
    print('Iteration', iteration, "of", maxits)
    history = LossHistory()
    model.fit(X, y,
              batch_size=128,
              epochs=1,
              callbacks=[history])

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(1000):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
        output.write('----- diversity: ' + str(diversity))
        output.write("\n\n")
        output.write('----- Generating with seed: "' + sentence + '"')
        output.write("\n\n")
        output.write(generated)
        output.write("\n\n::::::::::::::::::::::::::::::END::::::::::::::::::::::::::::::\n\n")


    chkpfilename = checkpointdir + "model" + str(iteration) + "_" + str(history.losses[-1])
    print("Checkpointing this model as:", chkpfilename)
    model.save(chkpfilename)

