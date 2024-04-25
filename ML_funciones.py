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
  print(f"y_test_a.shape : {X_test_a.shape}")
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

def model_eval(model, X_test=X_test_a, y_test = y_test_a, path="/content/drive/MyDrive/Fendi Mio/EMA/modelos/"):
  """
  Guarda el mejor modelo de un GridSearch. Predice (y pred_proba) y devuelve un ClassReport y CM. Devuelve mejores parametros.
  """
  data={}

  model_best = model.best_estimator_

  path_join = os.path.join(path,f"{model}.joblib")
  dump(model_best, path_join)

  preds=model_best.predict(X_test)

  model_best_results = classification_report(y_test,preds)

  print(f"best results: {model_best_results}")

  model_bestparam = lr_gs.best_params_

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


def pred_round(threshold, preds_proba, y_test=y_test_a):
  """
  Definiendo un threshold y pred_proba, devuelve un CM y ClassReport
  """
  preds_round = (preds_proba[:, 1] > threshold).astype(int)

  cm_round=confusion_matrix(y_test, preds_round)
  display = ConfusionMatrixDisplay(cm_round)
  display.plot()
  model_round_results = classification_report(y_test,preds_round)
  return model_round_results
