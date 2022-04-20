from sklearn import datasets
import tensorflow as tf
from utils import get_train_val_filter, recover

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