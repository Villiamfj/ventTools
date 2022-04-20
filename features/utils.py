
import pandas as pd
from ventmap.SAM import calc_resistance
import numpy as np

def add_timestep(in_df, group_id):
  df = in_df.copy()
  df["temp_index"] = np.arange(len(df))  
  df["timestep"] = 0

  for id in df[group_id].unique():
    length = len(df[df[group_id] == id])
    df.loc[df[group_id] == id, "temp_index"] = np.arange(length)
    df["timestep"] = (df["temp_index"] + 1) / 50

  del df["temp_index"]
  return df

def add_compliance(df):
  df_out = df.copy()
  df_out["c_dyn"] = df_out["tvi"] / df_out["pip_min_peep"]
  return df_out

def add_volume(df):
  df_out = df.copy()
  df_out["volume"] = df_out["flow"] * df_out["timestep"]
  return df_out



def high_pressure_breaths(df_in, group_id, max_p = 30):
  
  bn_group = df_in.groupby(group_id)
  df_list = []
  i = 0
  j = 0

  for bn, group in bn_group:
    i += 1
    max = df_in[df_in[group_id] == bn]["pressure"].max()
    if  max >= max_p:
      j += 1
      df_list.append(df_in[df_in[group_id] == bn])

  new_df = pd.concat(df_list, ignore_index=True)
  print("breaths with p >= 30: ", j, " Total breaths: ", i)
  return new_df

# Resistance function that uess the function from ventMap.SAM
def add_resistance(df):
  grouped = df.groupby(["vent_bn"])
  df["resistance"] = df.apply( lambda x: calc_resistance(x["pif"], x["pip"], x["plat"]), axis = 1)

# Adds pip and pif for each breath
def add_pip_and_pif(df):
  grouped = df.groupby(["vent_bn"])
  df["pif"] = df.apply(lambda x: grouped.get_group(x["vent_bn"])["flow"].max(), axis = 1)
  df["pip"] = df.apply(lambda x: grouped.get_group(x["vent_bn"])["pressure"].max(), axis = 1)
