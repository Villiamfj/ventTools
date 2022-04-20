from ast import literal_eval

import numpy as np
import pandas as pd
import tensorflow as tf
from .utils import _float_feature, _floatList_feature, _int64_feature, convert_to_fixed_length, get_train_val_filter, recover

# defining features of record
def map_fn(serialized_example):
    feature = {
        'label': tf.io.FixedLenFeature(1, tf.int64),
        'flow': tf.io.FixedLenFeature(200, tf.float32),
        'pressure' : tf.io.FixedLenFeature(200, tf.float32),
        'y' : tf.io.FixedLenFeature(1, tf.float32)
    }
    return tf.io.parse_single_example(serialized_example, feature)

def map_all(serialized):
  ex = map_fn(serialized)
  return (tf.stack([ex["flow"], ex["pressure"]], 1), ex["y"])


def _load_record_dataset(path, shuffle = True, buffer = 10000, reshuffle_each_iteration=True):
    dataset = tf.data.TFRecordDataset(path)
    all_ds = dataset.map(map_all)
    if shuffle:
        all_ds = all_ds.shuffle(buffer, reshuffle_each_iteration=reshuffle_each_iteration)
    return all_ds


def load_training_set(path, batch = 256, val_split = 5, *args, **kwargs):
    dataset = _load_record_dataset(path, *args, **kwargs)

    is_train, is_val = get_train_val_filter(val_split)

    train_ds = dataset.enumerate() \
        .filter(is_train) \
        .map(recover) \
        .batch(batch)

    val_ds = dataset.enumerate() \
        .filter(is_val) \
        .map(recover) \
        .batch(batch)

    return train_ds, val_ds



def load_test_set(path, batch = 256, *args, **kwargs):
    return _load_record_dataset(path, *args, **kwargs).batch(batch)


def write_as_record(file_link, record_name):
  #if file_link != None:
  #  file_csv = tf.keras.utils.get_file("data_raw", file_link, extract=True)
  data_csv = pd.read_csv(file_link)

  data_np = data_csv[["flow", "pressure", "y"]]
  data_np["flow"] = data_np["flow"].apply(literal_eval) # The csv reader from pandas reads them as strings
  data_np["pressure"] = data_np["pressure"].apply(literal_eval)
  data_np = convert_to_fixed_length(data_np, 200)
  #data_np = data_np.explode(["flow", "pressure"])
  data_np = data_np.to_numpy()

  # Writing rfRecord
  with tf.io.TFRecordWriter(record_name + '.tfrecord') as tfrecord:
   for idx in range(data_np.shape[0]):
     features = {
         'label'    : _int64_feature(idx),      #tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
         'flow'     : _floatList_feature(np.array(data_np[idx][0])),
         'pressure' : _floatList_feature(np.array(data_np[idx][1])),
         'y'        : _float_feature(data_np[idx][2]) #tf.train.Feature(int64_list= tf.train.FloatList(value = feature))
     }
     example = tf.train.Example(features = tf.train.Features(feature=features))
     tfrecord.write(example.SerializeToString())