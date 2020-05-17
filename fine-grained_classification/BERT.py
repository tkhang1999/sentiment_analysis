from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as tf_hub
import time
import numpy as np
from bert.tokenization import FullTokenizer
import tqdm
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns

tf.logging.set_verbosity(tf.logging.INFO)
SEED = 42
np.random.seed(SEED)
tf.set_random_seed(SEED)

# DATA PRE-PROCESSING

df = pd.read_csv('../yelp_food_review.csv')
df = df.loc[:, ['stars', 'text']]
df['sentiment'] = [2 if star > 3 else 1 if star == 3 else 0 for star in df['stars']]

print(df['stars'].value_counts())

# Get 10,000 sample data for faster training
rev_1 = df[df['stars'] == 1].sample(n = 2000)
rev_2 = df[df['stars'] == 2].sample(n = 2000)
rev_3 = df[df['stars'] == 3].sample(n = 2000)
rev_4 = df[df['stars'] == 4].sample(n = 2000)
rev_5 = df[df['stars'] == 5].sample(n = 2000)
dataset = pd.concat([rev_1, rev_2, rev_3, rev_4, rev_5]).sample(frac=1).reset_index(drop=True)

# Get 700,000 dataset
# dataset = df.sample(n = 700000).reset_index(drop=True)

dataset['stars'] = dataset['stars'] - 1
X_train, X_test, y_train, y_test = train_test_split(dataset['text'], dataset['stars'], test_size=0.1, random_state=SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1/9, random_state=SEED)

y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)
y_test = tf.keras.utils.to_categorical(y_test)

print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)

# BERT DATA

class PaddingInputExample(object):
  pass
    
    
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module =  tf_hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [
            tokenization_info["vocab_file"],
            tokenization_info["do_lower_case"],
        ]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=text, text_b=None, label=label)
        )
    return InputExamples

def convert_single_example(tokenizer, example, max_seq_length=256):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label

def convert_examples_to_features(tokenizer, examples, max_seq_length=256):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm.tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        # np.array(labels).reshape(-1, 1)
        np.array(labels)
    )

# Initialize session
sess = tf.Session()

# Params for bert model and tokenization
BERT_PATH = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
MAX_SEQ_LENGTH = 512

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module(bert_path=BERT_PATH)

# Convert data to InputExample format
train_examples = convert_text_to_examples(X_train, y_train)
val_examples = convert_text_to_examples(X_val, y_val)
test_examples = convert_text_to_examples(X_test, y_test)

(train_input_ids, train_input_masks, 
 train_segment_ids, train_labels) =  convert_examples_to_features(tokenizer=tokenizer, 
                                                                  examples=train_examples, 
                                                                  max_seq_length=MAX_SEQ_LENGTH)

(val_input_ids, val_input_masks, 
 val_segment_ids, val_labels) =  convert_examples_to_features(tokenizer=tokenizer, 
                                                              examples=val_examples, 
                                                              max_seq_length=MAX_SEQ_LENGTH)

(test_input_ids, test_input_masks, 
 test_segment_ids, test_labels) =  convert_examples_to_features(tokenizer=tokenizer, 
                                                                examples=test_examples, 
                                                                max_seq_length=MAX_SEQ_LENGTH)

print(train_input_ids.shape, val_input_ids.shape, test_input_ids.shape)
bm = tf_hub.Module(BERT_PATH, trainable=True, name=f"bert_module")

# BERT MODEL

class BertLayer(tf.keras.layers.Layer):
    
    def __init__(self, bert_path, n_fine_tune_encoders=10, **kwargs,):
        
        self.n_fine_tune_encoders = n_fine_tune_encoders
        self.trainable = True
        self.output_size = 768
        self.bert_path = bert_path
        super(BertLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_fine_tune_encoders': self.n_fine_tune_encoders,
            'trainable': self.trainable,
            'output_size': self.output_size,
            'bert_path': self.bert_path
        })

        return config
        
    def build(self, input_shape):
        self.bert = tf_hub.Module(self.bert_path,
                                  trainable=self.trainable, 
                                  name=f"{self.name}_module")

        # Remove unused layers
        trainable_vars = self.bert.variables
        trainable_vars = [var for var in trainable_vars 
                                  if not "/cls/" in var.name]
        trainable_layers = ["embeddings", "pooler/dense"]


        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_encoders+1):
            trainable_layers.append(f"encoder/layer_{str(10 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [var for var in trainable_vars
                                  if any([l in var.name 
                                              for l in trainable_layers])]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:# and 'encoder/layer' not in var.name:
                self._non_trainable_weights.append(var)
        print('Trainable layers:', len(self._trainable_weights))
        print('Non Trainable layers:', len(self._non_trainable_weights))

        super(BertLayer, self).build(input_shape)
        
    def call(self, inputs):
        
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(input_ids=input_ids, 
                           input_mask=input_mask, 
                           segment_ids=segment_ids)
        
        pooled = self.bert(inputs=bert_inputs, 
                           signature="tokens", 
                           as_dict=True)["pooled_output"]

        return pooled

    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)

# Build model
def build_model(bert_path, max_seq_length, n_fine_tune_encoders=10): 
    
    inp_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    inp_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    inp_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [inp_id, inp_mask, inp_segment]
    
    bert_output = BertLayer(bert_path=bert_path, 
                            n_fine_tune_encoders=n_fine_tune_encoders)(bert_inputs)
    
    dense = tf.keras.layers.Dense(256, activation='relu')(bert_output)
    # pred = tf.keras.layers.Dense(1, activation='sigmoid')(dense)

    # For fine-grained classification
    drop = tf.keras.layers.Dropout(0.1)(dense)
    pred = tf.keras.layers.Dense(5, activation='softmax')(drop)
    
    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss='categorical_crossentropy',
                #   loss='binary_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])    
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)

# Build model
bert_model = build_model(bert_path=BERT_PATH, max_seq_length=MAX_SEQ_LENGTH, n_fine_tune_encoders=10)
initialize_vars(sess)
print(bert_model.summary())

# TRAIN THE MODEL

filepath = "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=0, verbose=1, mode='auto', cooldown=0, min_lr=1e-6)
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
    update_freq='epoch', profile_batch=2, embeddings_freq=0,
    embeddings_metadata=None
)

bert_history = bert_model.fit(
    [train_input_ids, train_input_masks, train_segment_ids], 
    train_labels,
    validation_data=([val_input_ids, val_input_masks, val_segment_ids], val_labels),
    epochs=4,
    batch_size=8,
    verbose=1,

    # callbacks=[checkpoint, early_stopping, reduce_lr, tensorboard]
    callbacks=[checkpoint, reduce_lr]
)

result = bert_model.evaluate([test_input_ids, test_input_masks, test_segment_ids], test_labels)
print(result)

# MODEL EVALUATION

# test_predictions = bert_model.predict(x=[test_input_ids,
#                                          test_input_masks,
#                                          test_segment_ids],
#                                       batch_size=64,
#                                       verbose=1)

# from sklearn.metrics import confusion_matrix, classification_report

# test_pred_labels = []
# for prediction in test_predictions:
#     pred_label = [1 if pred == max(prediction) else 0 for pred in prediction]
#     test_pred_labels.append(pred_label)

# print('Classification Report:')
# print(classification_report(y_true=test_labels, y_pred=test_pred_labels))

# import seaborn as sns
# import matplotlib.pyplot as plt

# %matplotlib inline

# with tf.Session() as session:
#     cm = tf.confusion_matrix([np.argmax(label) for label in test_labels], [np.argmax(label) for label in test_pred_labels]).eval()

# LABELS = ['0', '1', '2', '3', '4']
# sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS, fmt='g')
# xl = plt.xlabel("Predicted")
# yl = plt.ylabel("Actuals")