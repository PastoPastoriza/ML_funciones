import pandas as pd
import numpy as np



def window_before_array(table,split_window_column = "long_condition", condition=True, window=10, target = "target", split=0.85):
  """
  Agarra una tabla y: crea ventana, arma train-test, los cambia a arrays y reshape de tal forma que quede 1 orden con sus "window" anteriores.
  """
  data ={}

  print("Creating windows df:")
  df_f = window_before(table, split_window_column, condition=True, window = window)
  print()
  print(df_f.info())
  X = df_f.drop(columns=target)
  y = df_f[target]
  print()
  print("Creating train-test data:")
  X_train, X_test, y_train, y_test = train_test(X, y,window = window, split=split)
  print()
  print()
  y_train = y_train.astype(int)
  y_test = y_test.astype(int)

  X_train_a = X_train.to_numpy()
  y_train_a = y_train.to_numpy()
  X_test_a = X_test.to_numpy()
  y_test_a = y_test.to_numpy()
  print("Creating arrays from df:")
  print()
  print(f"X_train_a.shape : {X_train_a.shape}")
  print(f"y_train_a.shape : {y_train_a.shape}")
  print(f"X_test_a.shape : {X_test_a.shape}")
  print(f"y_test_a.shape : {X_test_a.shape}")
  print()
  print()
  X_train_a_len = len(X_train_a)/window
  X_test_a_len = len(X_test_a)/window
  col_len = X_train_a.shape[1] * window
  X_train_a_len = int(X_train_a_len)
  X_test_a_len = int(X_test_a_len)
  
  print(f"X_train_a_len: {X_train_a_len}")
  print(f"col_len: {col_len}")
  print(f"X_test_a_len : {X_test_a_len}")
  print()
  print()
  print(f"Creating window {window} arrays:")
  
  X_train_a = X_train_a.reshape(X_train_a_len,col_len)
  y_train_a = y_train_a[window - 1::window]
  X_test_a = X_test_a.reshape(X_test_a_len,col_len)
  y_test_a = y_test_a[window - 1::window]
  
  print(f"X_train_a.shape : {X_train_a.shape}")
  print(f"y_train_a.shape : {y_train_a.shape}")
  print(f"X_test_a.shape : {X_test_a.shape}")
  print(f"y_test_a.shape : {y_test_a.shape}")
  print()
  print(f"bincount(y_train_a) : {np.bincount(y_train_a)}")
  print(f"bincount(y_test_a) : {np.bincount(y_test_a)}")

  data['X_train'] = X_train
  data['y_train'] = y_train
  data['X_train_a'] = X_train_a
  data['y_train_a'] = y_train_a
  data['X_test'] = X_test
  data['y_test'] = y_test
  data['X_test_a'] = X_test_a
  data['y_test_a'] = y_test_a
  print()
  print("helpful print:")
  print()
  print("X_train = arrays['X_train']")
  print("y_train = arrays['y_train']")
  print("X_test = arrays['X_test']")
  print("y_test = arrays['y_test']")
  print("X_train_a = arrays['X_train_a']")
  print("y_train_a = arrays['y_train_a']")
  print("X_test_a = arrays['X_test_a']")
  print("y_test_a = arrays['y_test_a']")

  return data

def train_test(X,y,window=10,split=0.8):
  """
  Splits train-test with index for a window (default=10) data
  """
  length = len(X)

  split_index = int(split * length)
  split_index = (split_index // window) * window

  X_train = X[:split_index] 
  y_train =y[:split_index]
  X_test = X[split_index:]
  y_test = y[split_index:]
  print(f"Shape of X_train = {X_train.shape}")
  print(f"Shape of y_train = {y_train.shape}")
  print(f"Shape of X_test = {X_test.shape}")
  print(f"Shape of y_test = {y_test.shape}")
  return X_train, X_test, y_train, y_test

def window_before(table,split_window_column, condition=True, window=10):
  """
  Given a table and a reference column, it will creat a new dataframe with a window (default=10) rows starting the True condition of the reference column
  """
  print(f"Lenght of original Dataframe = {len(table)}\n")
  new_table = pd.DataFrame(columns=table.columns)
  for index, row in table.iterrows():
    if row[split_window_column] == condition:
      current_index = table.index.get_loc(index)
      new_table = pd.concat([new_table, table.iloc[current_index-window+1:current_index+1]])
  print(f"Lengh of New Dataframe = {len(new_table)}")
  return new_table


from sklearn.metrics import classification_report
from joblib import dump, load
import os

def grid_eval(model,model_name, X_test, y_test, path="/content/drive/MyDrive/Fendi Mio/EMA/modelos/"):
  """
  Guarda el mejor modelo de un GridSearch. Predice (y pred_proba) y devuelve un ClassReport y CM. Devuelve mejores parametros.
  """
  data={}

  model_best = model.best_estimator_

  path_join = os.path.join(path,f"{model_name}.joblib")
  dump(model_best, path_join)

  preds=model_best.predict(X_test)

  model_best_results = classification_report(y_test,preds)

  print(f"best results: {model_best_results}")

  model_bestparam = model.best_params_

  print(model_bestparam)

  cm=confusion_matrix(y_test, preds)
  display = ConfusionMatrixDisplay(cm)
  display.plot()

  preds_proba=model_best.predict_proba(X_test)
  preds_proba[:5]

  data["preds_proba"] = preds_proba
  data["preds"] = preds
  data["model_best"] = model_best
  data["model_best_results"] = model_best_results
  data["model_bestparam"] = model_bestparam

  return data

from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
def pred_round(threshold, preds_proba, y_test):
  """
  Definiendo un threshold y pred_proba, devuelve un CM y ClassReport
  """
  preds_round = (preds_proba[:, 1] > threshold).astype(int)

  cm_round=confusion_matrix(y_test, preds_round)
  display = ConfusionMatrixDisplay(cm_round)
  display.plot()
  model_round_results = classification_report(y_test,preds_round)
  return model_round_results


def date_window_before(data, train_date = "2021-01-01", valid_date = "2023-01-01",split_window_column = "Strategy Long Condition", condition=True, window=5, target = "Target"):
  
  table ={}

  train = data.loc[:train_date] #o con train_index_number
  valid = data.loc[train_date:valid_date] #o con valid_index_number
  test = data.loc[valid_date:]

  train_window = window_before(train, split_window_column=split_window_column, condition=True, window = window)
  X_train = train_window.drop(columns=target)
  y_train = train_window[target]
  valid_window = window_before(valid, split_window_column=split_window_column, condition=True, window = window)
  X_valid = valid_window.drop(columns=target)
  y_valid = valid_window[target]
  test_window = window_before(test, split_window_column=split_window_column, condition=True, window = window)
  X_test = test_window.drop(columns=target)
  y_test = test_window[target]

  y_train = y_train.astype(int)
  y_valid = y_valid.astype(int)
  y_test = y_test.astype(int)


  print("Creating arrays from df:")
  X_train_a = X_train.to_numpy()
  y_train_a = y_train.to_numpy()
  X_valid_a = X_valid.to_numpy()
  y_valid_a = y_valid.to_numpy()
  X_test_a = X_test.to_numpy()
  y_test_a = y_test.to_numpy()
  print()
  print(f"X_train_a.shape before flatten : {X_train_a.shape}")
  print(f"y_train_a.shape before flatten : {y_train_a.shape}")
  print(f"X_valid_a.shape before flatten : {X_valid_a.shape}")
  print(f"y_valid_a.shape before flatten: {y_valid_a.shape}")
  print(f"X_test_a.shape before flatten: {X_test_a.shape}")
  print(f"y_test_a.shape before flatten: {X_test_a.shape}")
  print()
  print()
  X_train_a_len = len(X_train_a)/window
  X_valid_a_len = len(X_valid_a)/window
  X_test_a_len = len(X_test_a)/window
  col_len = X_train_a.shape[1] * window
  X_train_a_len = int(X_train_a_len)
  X_valid_a_len = int(X_valid_a_len)
  X_test_a_len = int(X_test_a_len)

  print(f"Q of train orders: {X_train_a_len}")
  print(f"colums x window: {col_len}")
  print(f"Q of valid orders: {X_valid_a_len}")
  print(f"Q of test orders : {X_test_a_len}")
  print()
  print()
  print(f"Reshaping window {window} arrays:")

  X_train_a = X_train_a.reshape(X_train_a_len,col_len)
  y_train_a = y_train_a[window - 1::window]
  X_valid_a = X_valid_a.reshape(X_valid_a_len,col_len)
  y_valid_a = y_valid_a[window - 1::window]
  X_test_a = X_test_a.reshape(X_test_a_len,col_len)
  y_test_a = y_test_a[window - 1::window]

  print(f"X_train_a.shape : {X_train_a.shape}")
  print(f"y_train_a.shape : {y_train_a.shape}")
  print(f"X_valid_a.shape : {X_valid_a.shape}")
  print(f"y_valid_a.shape : {y_valid_a.shape}")
  print(f"X_test_a.shape : {X_test_a.shape}")
  print(f"y_test_a.shape : {y_test_a.shape}")


  print()
  print(f"bincount(y_train_a) : {np.bincount(y_train_a)}")
  print(f"bincount(y_valid_a) : {np.bincount(y_valid_a)}")
  print(f"bincount(y_test_a) : {np.bincount(y_test_a)}")

  table['X_train'] = X_train
  table['y_train'] = y_train
  table['X_train_a'] = X_train_a
  table['y_train_a'] = y_train_a
  table['X_valid'] = X_valid
  table['y_valid'] = y_valid
  table['X_valid_a'] = X_valid_a
  table['y_valid_a'] = y_valid_a
  table['X_test'] = X_test
  table['y_test'] = y_test
  table['X_test_a'] = X_test_a
  table['y_test_a'] = y_test_a
  print()
  print("helpful print:")
  print()
  print("X_train = table['X_train']")
  print("y_train = table['y_train']")
  print("X_valid = table['X_valid']")
  print("y_valid = table['y_valid']")
  print("X_test = table['X_test']")
  print("y_test = table['y_test']")
  print("X_train_a = table['X_train_a']")
  print("y_train_a = table['y_train_a']")
  print("X_valid_a = table['X_valid_a']")
  print("y_valid_a = table['y_valid_a']")
  print("X_test_a = table['X_test_a']")
  print("y_test_a = table['y_test_a']")

  return table
