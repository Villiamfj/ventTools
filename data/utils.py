import tensorflow as tf
import pandas as pd
import numpy as np
from ast import literal_eval

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Takes an array
def _floatList_feature(value):
  return tf.train.Feature(float_list = tf.train.FloatList(value = value.flatten()))


def writeAsRecord(file_name,recordName,file_link):
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
  with tf.io.TFRecordWriter(recordName + '.tfrecord') as tfrecord:
   for idx in range(data_np.shape[0]):
     feature = data_np[idx]
     features = {
         'label'    : _int64_feature(idx),      #tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
         'flow'     : _floatList_feature(np.array(data_np[idx][0])),
         'pressure' : _floatList_feature(np.array(data_np[idx][1])),
         'y'        : _float_feature(data_np[idx][2]) #tf.train.Feature(int64_list= tf.train.FloatList(value = feature))
     }
     example = tf.train.Example(features = tf.train.Features(feature=features))
     tfrecord.write(example.SerializeToString())

# converts flow and pressure to a fixed length n
# unexploded_df : Dataframe where flow and pressure are arrays
# n             : The desired length
def convert_to_fixed_length(unexploded_df,n):
  def fix_length(array):
    if len(array) < n:
      return np.pad(array, (0,n - len(array)))
    if len(array) > n:
      return array[0 : n]
    return array

  result = unexploded_df.copy()
  result.flow = result.flow.map(fix_length)
  result.pressure = result.pressure.map(fix_length)
  return result

def get_train_val_filter(split):
    
    val_filter = lambda x, _: x % split == 0
    train_filter = lambda x, _: not val_filter(x, _)
    
    return train_filter, val_filter

# # validation function
# def is_val(x, _):
#     return x % 5 == 0

# # Negation of v validation function
# def is_train(x, y):
#     return not is_val(x, y)

def recover(_, y):
    return y
