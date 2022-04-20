
from ventmode import datasets
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt
from ventmap.raw_utils import extract_raw
from ventmap.SAM import calc_inspiratory_plateau, findx02
import numpy as np
from collections import defaultdict
import re




def load_vwd_patients(fileset, plateau_patients):
  raw_wvd_df = []
  for patient, x_file in fileset['x']:
    if patient not in plateau_patients:
      continue
    data = None
    with open(x_file, encoding='ascii', errors='ignore') as file:
      data = extract_raw(file, True)
    raw_wvd_df.append(pd.DataFrame(data))
  return pd.concat(raw_wvd_df)

def get_plateau_df(raw_wvd_df, train_df):
  # Calculate plateau
  raw_wvd_df['plat_temp'] = raw_wvd_df.apply(lambda x: calc_inspiratory_plateau(x['flow'], x['pressure'], dt=0.02, min_time=.2), axis=1)

  # Merge raw and feature data
  plateau_df = train_df.merge(raw_wvd_df, on='vent_bn').explode(['flow', 'pressure'])
  plateau_df = plateau_df.reset_index(drop=True)

  # Unpack plat
  plateau_df = plateau_df.dropna()
  plateau_df['plat'] = plateau_df['plat_temp'].transform(lambda x: x[1])
  plateau_df = plateau_df.drop(['plat_temp'], axis=1)
  return plateau_df.dropna()

def load_full_plateau_dataset():
  #Creates the fileset and final featureset
  DERIVATION_COHORT_X_DIR = join("ventmode/anon_train_data", "raw_vwd")
  DERIVATION_COHORT_Y_DIR = join("ventmode/anon_train_data", "y_dir")

  fileset = datasets.get_cohort_fileset(DERIVATION_COHORT_X_DIR, DERIVATION_COHORT_Y_DIR)
  vfinal = datasets.VFinalFeatureSet(fileset, 10, 100)
  train_df = vfinal.create_learning_df()
  train_df = train_df[train_df['n_plats_past_20'] != 0]

  raw_vwd_df = load_vwd_patients(fileset, set(train_df['patient']))
  return get_plateau_df(raw_vwd_df, train_df)




def add_pip_min_plat(df):
  dfCopy = df.copy()
  dfCopy["pip_min_plat"] = dfCopy["pip"] - dfCopy["plat"]
  return dfCopy

def top_pip_plat_difference(df,count):
  if "pip_min_plat" not in df.columns:
    df = add_pip_min_plat(df)
  return df[["vent_bn", "pip_min_plat"]].drop_duplicates().nlargest(count,["pip_min_plat"])

def max_compliance(df, count):
  return df[["vent_bn", "c_dyn"]].drop_duplicates().nlargest(count,["c_dyn"])

def min_compliance(df, count):
  return df[["vent_bn", "c_dyn"]].drop_duplicates().nsmallest(count,["c_dyn"])

# df is a dataframe
# count is the amount of elements you want returned
# normal_compliance is the normal compliance
# slack is how much above or below normal_compliance is consided to also be normal
def n_least_normal_compliance(df, count = 5, normal_compliance = 50, slack = 0):
  max_df = max_compliance(df, count)
  min_df = min_compliance(df, count)

  if df['c_dyn'].min() > (normal_compliance - slack):
    if df['c_dyn'].max() < (normal_compliance + slack):
      return (False, None)
    else:
      return (True, max_df)

  if df['c_dyn'].max() < (normal_compliance + slack):
    if df['c_dyn'].min() > (normal_compliance - slack):
      return (False, None)
    else:
      return (True, min_df)

  return_df = pd.DataFrame(columns=['vent_bn', 'c_dyn'])
  i = 0
  j = 0
  run = True

  while run == True:
    if (((normal_compliance - slack) - min_df['c_dyn'].iloc[i]) > (max_df['c_dyn'].iloc[j] - (normal_compliance + slack))):
      temp_min_data = {'vent_bn': [min_df['vent_bn'].iloc[i]],'c_dyn': [min_df['c_dyn'].iloc[i]]}
      temp_min_df = pd.DataFrame(data=temp_min_data)
      return_df = pd.concat([return_df, temp_min_df], ignore_index=True)
      i += 1
    else:
      temp_max_data = {'vent_bn': [max_df['vent_bn'].iloc[j]],'c_dyn': [max_df['c_dyn'].iloc[j]]}
      temp_max_df = pd.DataFrame(data=temp_max_data)
      return_df = pd.concat([return_df, temp_max_df], ignore_index=True)
      j += 1
    if len(return_df) == count:
      return (True, return_df)

    if j >= (count-1) or i >= (count-1): #emergency stop. i or j should never go above 4
      print("Something went wrong")
      run = False

  return (False, None)

def return_abnormal_compliance(df, normal_compliance = 50, upper_bound = 50, lower_bound = 10, c_var = "c_dyn", group_var = "vent_bn"):

  df_out = df.copy()
  df_out["c_diff"] = df_out[c_var] - normal_compliance
  
  bn_group = df_out.groupby(group_var)
  df_high_list = []
  df_low_list = []

  for bn, group in bn_group:
    diff = df_out[df_out[group_var] == bn]["c_diff"].iloc[0]
    if(abs(diff) < upper_bound and abs(diff) > lower_bound):
      if(diff > 0):
        df_high_list.append(df_out[df_out[group_var] == bn])
      elif(diff <= 0):
        df_low_list.append(df_out[df_out[group_var] == bn])

  high_df = pd.concat(df_high_list, ignore_index=True)
  low_df = pd.concat(df_low_list, ignore_index=True)
  return (high_df, low_df)


units = defaultdict(lambda: "")
units["pressure"] = "cmH20"
units["flow"] = "ml/s"
units["pip_min_plat"] = "cmH20"
units["c_dyn"] = "ml/cmH20"
units["resistance"] = "cmH20/ml/s"

def add_unit_to_name(feature_name):
  return feature_name + " " + units[feature_name]

# Function for drawing features of a breach over timestep
# df             : dataFrame
# vent_bn        : The number of breath (not a list)
# viz_features   : The features to be shown in graphs (a list of strings)
# tabel_features : The features to be shown in a tabel(a optional list of strings)
# save_location  : The path of where pictures should be saved (if none no pictures will be saved)
# units          : If true the add_unit_to_name function will be used to add units to the labels
def plot_features_of_breath(df, vent_bn, viz_features, tabel_features = None, save_location = None, units = False):
  colormap = ['b', 'g', 'r', 'c', 'm', 'y']
  sub_plot_len = len(viz_features)
  if tabel_features is not None:
    height = 2 
    ratios = [5, 1]
  else:
    height = 1
    ratios = [1]

  # Setting up the subplots
  fig, axs = plt.subplots(nrows=height, ncols=sub_plot_len, figsize = (10 * sub_plot_len, 10), gridspec_kw={'height_ratios': ratios})
  
  # Getting group breath number
  grouped = df.groupby(["vent_bn"])
  data_to_plot = grouped.get_group(vent_bn)
  
  # Making a graph for each feature over timestep
  feature_axs = axs[0] if tabel_features is not None else axs
  for i, feature in enumerate(viz_features):
    data_to_plot[[feature, "timestep"]].plot(y = feature, x = "timestep", ax = feature_axs[i], title = str(vent_bn) + " : " + feature, c=colormap[i])
    
  if tabel_features != None:
    gs = axs[1][0].get_gridspec()

    for ax in axs[1]:
      ax.remove()

    axbig = fig.add_subplot(gs[-1, :])
    # Removes axis from the table subplot   
    axbig.axis("tight")
    axbig.axis("off")

    # Putting the first row of written features in table
    first_row_values = data_to_plot.head(1)[tabel_features].values.tolist()

    labels = list(map(add_unit_to_name, tabel_features)) if units else tabel_features

    axbig.table(cellText = first_row_values, colLabels = labels, loc = "top", cellLoc="center")

  plt.subplots_adjust(left=0.1,
                      bottom=0.1, 
                      right=0.9, 
                      top=1.3, 
                      wspace=0.1, 
                      hspace=0.4)
  
  if save_location != None:
    plt.tight_layout()
    plt.savefig(save_location + str(vent_bn) + ".png")
  
  plt.show()



# Loads all the raw data from the path given
def load_all_raw_data(path, explode = False):
  frames = []
  # Extracting the raw data one at a time
  for filename in os.listdir(path):
    with open(path + "/" + filename, encoding='ascii', errors='ignore') as file:
      data = extract_raw(file, True)
      data = pd.DataFrame(data)
      data["patient"] = re.match("^[^-]*", filename)[0]
    frames.append(data)
  
  # combining the raw data
  result = pd.concat(frames)

  # flow and pressure is contained as arrays, therefore we use the explode function
  if explode:
    result = result.explode(["flow","pressure"])
  return result

# Loads the y data from the path given
def load_y_data(path):
  frames = []
  # Extracting the raw data one at a time
  for filename in os.listdir(path):
    with open(path + "/" + filename, encoding='ascii', errors='ignore') as file:
      data = pd.read_csv(file)
      data["patient"] = re.match("^[^-]*", filename)[0]
    frames.append(pd.DataFrame(data))
  
  # combining the raw data
  return pd.concat(frames)
# explodes the raw data into columns
def explode_into_column(df, n):
  frames = []
  frames.append(pd.DataFrame(df["flow"].to_list(), columns = ["flow" + str(i) for i in range(0,188)], index = df.index ))
  frames.append(pd.DataFrame(df["pressure"].to_list(), columns = ["pressure" + str(i) for i in range(0,188)], index = df.index ))
  frames.append(df)
  return pd.concat(frames, axis = 1)



# A function to get a numpy array from groupings based on the group_key
# code from https://asifr.com/transform-grouped-dataframe-to-numpy
# df        : DataFrame
# values    : labels of the values wanted fx. ["pressure","flow"]
# group_key : the element(s) to groupby
def np_grouped(df, values, group_key = ["patient", "vent_bn", "rel_bn"]):
  xt = df.loc[:,values].values
  g = df.reset_index(drop=True).groupby(group_key)
  xtg = [xt[i.values,:] for k,i in g.groups.items()]
  return np.array(xtg)

# Gets a specific value of each breath
# df        : DataFrame
# y_label   : the label of the desired value
# group_key : the element(s) to groupby
def get_y_series(df, y_label, group_key = ["patient", "vent_bn", "rel_bn"]):
  grouped = df.groupby(group_key)
  return np.array([i[y_label].values[0] for k, i in grouped])


def get_name_of_mode(number):
  switcher = {
   0: "Ventilation Control",
   1: "Pressure Control",
   3: "Pressure Support",
   4: "Continuous positive airway pressure",
   6: "Proportional assist ventilation"
  }
  return switcher.get(number, "ERROR")


def add_mode_name(df):
  new_df = df.copy()
  new_df["mode_name"] = new_df["y"].apply(get_name_of_mode)
  return new_df

# A function to sort out breaths that does not match the given length
# df           : the dataFrame
# length       : The length of the breath in data points
# group_key    : key to group by
def only_specific_length(df, length, group_key = "vent_bn"):
  return df[df.groupby(group_key)[group_key].transform('size') == length]

# A function to combine breaths into a numpy array of inputs to a model that takes n breats as input
# df           : The dataFrame
# n            : The amount of breats in one input
# group_key    : key to group by
def group_in_n(df, n, group_key = ["patient", "vent_bn", "rel_bn"]):
  result = []
  grouped = df.groupby(group_key)
  count = 0
  tempList = []
  for name, group in grouped:
    tempList.append(df[df[group_key] == name])
    count += 1
    if count == n - 1:
      result.append(pd.concat(tempList).drop(columns = group_key).to_numpy())
      tempList = []
      count = 0
  return result

# Adds tvi to non exploded raw data
def add_tvi(dt):
  newdt = dt.copy()
  newdt["tvi"] = newdt.apply(lambda x : findx02(x["flow"], x["dt"])[2], axis = 1)
  return newdt

def only_specified_length_unexploded(df, n):
  return df[df['flow'].map(len) == n]


# Modified determine mode function
# Source : https://github.com/hahnicity/ventmode/blob/afa6ccb5eb9d64a591307a316de1e3e496c9231d/ventmode/datasets.py#L59
def add_mode(y_data):
  try:
    y_data.simv
  except AttributeError:
    y_data['simv'] = np.nan
  try:
    y_data.pav
  except AttributeError:
    y_data['pav'] = np.nan
  
  mode = []
  for i, row in y_data.iterrows():
                if row.vc == 1:
                    mode.append(0)
                elif row.pc == 1:
                    mode.append(1)
                elif row.prvc == 1:
                    mode.append(2)
                elif row.ps == 1:
                    mode.append(3)
                elif row.cpap_sbt == 1:
                    mode.append(4)
                elif row.simv == 1:
                    mode.append(5)
                elif row.pav == 1:
                    mode.append(6)
                else:
                    mode.append(7)
  new_y_data = y_data.copy()
  new_y_data.insert(0, "mode", mode)
  new_y_data = new_y_data.drop(["pav", "simv", "cpap_sbt", "ps", "prvc", "pc", "vc", "other"], axis = 1)
  return new_y_data
