# Dataset Analysis on Diabetes

## About Dataset

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney
Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes,
based on certain diagnostic measurements included in the dataset. Several constraints were placed
on the selection of these instances from a larger database. In particular, all patients here are females
at least 21 years old of Pima Indian heritage.2


- Add Suggestion Information about dataset attributes -

- Pregnancies: To express the Number of pregnancies

- Glucose: To express the Glucose level in blood

- BloodPressure: To express the Blood pressure measurement

- SkinThickness: To express the thickness of the skin

- Insulin: To express the Insulin level in blood

- BMI: To express the Body mass index

- DiabetesPedigreeFunction: To express the Diabetes percentage

- Age: To express the age

- Outcome: To express the final result 1 is Yes and 0 is No


## Import Necesseary Modules

```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix

import joblib

import warnings
warnings.filterwarnings("ignore")
```

## Import Dataset
```
df = pd.read_csv("/kaggle/input/diabetes-dataset/diabetes.csv")
df.head()
```

## General Information About Dataset
```
def check_df(dataframe,head=5):
  print(20*"#", "Head", 20*"#")
  print(dataframe.head(head))
  print(20*"#", "Tail", 20*"#")
  print(dataframe.tail(head))
  print(20*"#", "Shape", 20*"#")
  print(dataframe.shape)
  print(20*"#", "Types", 20*"#")
  print(dataframe.dtypes)
  print(20*"#", "NA", 20*"#")
  print(dataframe.isnull().sum())
  print(20*"#", "Qurtiles", 20*"#")
  print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
```

`check_df(df)`

## Analysis of Categorical and Numerical Variables

```
def grab_col_names(dataframe, cat_th=10, car_th=20):
  #Catgeorical Variable Selection
  cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category","object","bool"]]
  num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ["uint8","int64","float64"]]
  cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category","object"]]
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car]

  #Numerical Variable Selection
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["uint8","int64","float64"]]
  num_cols = [col for col in num_cols if col not in cat_cols]

  return cat_cols, num_cols, cat_but_car, num_but_cat
```
```
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

#Print Categorical and Numerical Variables
print(f"Observations: {df.shape[0]}")
print(f"Variables: {df.shape[1]}")
print(f"Cat_cols: {len(cat_cols)}")
print(f"Num_cols: {len(num_cols)}")
print(f"Cat_but_car: {len(cat_but_car)}")
print(f"Num_but_cat: {len(num_but_cat)}")
```
```
def cat_summary_df(dataframe):
  cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
  for col in cat_cols:
    cat_summary(dataframe, col, plot=True)

cat_summary_df(df)
```
```
def num_summary(dataframe, num_col, plot=False):
  print(50*"#", num_col ,50*"#")
  quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
  print(dataframe[num_col].describe(quantiles).T)

  if plot:
    dataframe[num_col].hist(bins=20)
    plt.xlabel(num_col)
    plt.ylabel(num_col)
    plt.show(block=True)
```
```
def num_summary_df(dataframe):
  cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)
  for col in num_cols:
    num_summary(dataframe, col, plot=True)

num_summary_df(df)
```
```
def plot_num_summary(dataframe):
  cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)
  plt.figure(figsize=(12,4))
  for index, col in enumerate(num_cols):
    plt.subplot(2,4, index+1)
    plt.tight_layout()
    dataframe[col].hist(bins=20)
    plt.title(col)

plot_num_summary(df)
```

## Target Analysis
```
def target_summary_with_cat(dataframe, target, numerical_col):
  print(f"##################### {target} -> {numerical_col} #####################")
  print(pd.DataFrame({"Target Mean": dataframe.groupby(numerical_col)[target].mean()}))
```

```
def target_summary_with_num_df(dataframe, target):
    cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(dataframe)
    for col in num_cols:
        target_summary_with_cat(dataframe, target, col)
```
`target_summary_with_num_df(df, "Outcome")`

## Correlation Analysis

```
def high_correlated_cols(dataframe, corr_th = 0.90, plot=False):
  num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["uint8", "int64", "float64"]]
  corr = dataframe[num_cols].corr()
  corr_matrix = corr.abs()
  upper_triangular_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
  drop_list = [col for col in upper_triangular_matrix.columns if any(upper_triangular_matrix[col] > corr_th)]
  if drop_list == []:
    print("Aftre corelation analysis, we dont need to remove variables")

  if plot:
    sns.set(rc={'figure.figsize': (18,13)})
    sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".2f")
    plt.show()

  return drop_list

drop_list = high_correlated_cols(df, plot=True)
```

## Missing Value Analysis
```
zero_columns = [col for col in df.columns if (df[col].min() == 0) and col not in ["Pregnancies", "Outcome"]]
zero_columns
```
```
for col in zero_columns:
  df[col] = np.where(df[col] == 0, np.nan, df[col])

df.isnull().sum()
```
```
def missing_value_table(dataframe, na_names=False):
  n_miss = dataframe[zero_columns].isnull().sum().sort_values(ascending=False)
  ratio = (dataframe[zero_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
  missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
  print(missing_df)
  if na_names:
    print("######### Na Names ###########")
    return zero_columns

missing_value_table(df)
```
```
def fill_na_median(dataframe):
  dataframe = dataframe.apply(lambda x: x.fillna(x.median()) if x.dtype not in ["category", "object", "bool"] else x, axis=0)
  return dataframe

df = fill_na_median(df)
```

## Create a Base Model: Prediction Diabetes using Random Forest Classifier

```
X = df.drop(["Outcome"], axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
```
```
rf_model = RandomForestClassifier(random_state=1).fit(X_train, y_train)
rf_model.predict(X_train)
accuracy_score(y_train, rf_model.predict(X_train))
accuracy_score(y_test, rf_model.predict(X_test))
cv_results = cross_validate(rf_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
```
```
def RF_Model(dataframe, target, test_size=0.30, cv=10, results=False, plot_importance=False, save_results=False):
  X = dataframe.drop(target, axis=1)
  y = dataframe[target]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)
  rf_model = RandomForestClassifier(random_state=1).fit(X_train, y_train)
  if results:
    acc_train = accuracy_score(y_train, rf_model.predict(X_train))
    acc_test = accuracy_score(y_test, rf_model.predict(X_test))
    r2_train = rf_model.score(X_train, y_train)
    r2_test = rf_model.score(X_test, y_test)
    cv_results = cross_validate(rf_model, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

    print("Accuracy Train: ", "%.3f" % acc_train)
    print("Accuracy Test: ", "%.3f" % acc_test)
    print("R2 Train: ", "%.3f" % r2_train)
    print("R2 Test: ", "%.3f" % r2_test)
    print("Cross Validate Accuracy: ", "%.3f" % cv_results['test_accuracy'].mean())
    print("Cross Validate Precision: ", "%.3f" % cv_results['test_precision'].mean())
    print("Cross Validate Recall: ", "%.3f" % cv_results['test_recall'].mean())
    print("Cross Validate F1: ", "%.3f" % cv_results['test_f1'].mean())
    print("Cross Validate Roc Auc: ", "%.3f" % cv_results['test_roc_auc'].mean())

    if plot_importance:
      feature_imp = pd.DataFrame({'Value': rf_model.feature_importances_, 'Feature': X.columns})
      plt.figure(figsize=(8, 6))
      sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
      plt.title("Importance Features")
      plt.tight_layout()
      plt.savefig("importance.jpg")
      plt.show()

    if save_results:
      joblib.dump(rf_model, "rf_model.pkl")

    return rf_model
```
```
rf_model = RF_Model(df, "Outcome", results=True, plot_importance=True, save_results=True)

def load_model(pklfile):
  model_disc = joblib.load(pklfile)
  return model_disc

model_disc = load_model("rf_model.pkl")

patient = [5, 50.00, 65.00, 50.00, 160.00, 30.00, 0.574, 75]

model_disc.predict(pd.DataFrame(patient).T)[0]
```



























