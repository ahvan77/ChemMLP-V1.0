'''
Data X y
'''
import pandas as pd
from sklearn.model_selection import KFold
from math import sqrt
from sklearn.metrics import mean_squared_error
#from rdkit import rdBase
from rdkit import Chem
#import tensorflow as tf
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import RDKFingerprint
from rdkit.Chem.EState.Fingerprinter import FingerprintMol
from rdkit.Chem.AllChem import  GetMorganFingerprintAsBitVect, GetErGFingerprint
import rdkit.DataStructs.cDataStructs
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import Descriptors
import numpy as np
#from tensorflow.keras.preprocessing.text import Tokenizer
#from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ["KERAS_BACKEND"] = "jax"

# ✅ Modern, compatible imports for Keras 3.x or TensorFlow 2.20+
try:
    from keras_preprocessing.text import Tokenizer
    from keras_preprocessing.sequence import pad_sequences
except ModuleNotFoundError:
    # fallback if keras_preprocessing not installed
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    from keras.utils import to_categorical
except ImportError:
    from tensorflow.keras.utils import to_categorical
#from tkinter_script import GUI_final_input
import ast
import glob
from Err import error
from PySide6.QtWidgets import QMessageBox

def Check_index(input_file):
    pixmappath01 = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
    with open(pixmappath01 + 'Feature_Target.txt', "r+") as f:
        i = [line.strip() for line in f]

    index1 = i[:-1]
    target = i[-1]
    
    file_encoding = 'utf-8-sig'
    s = pd.read_csv(input_file, encoding=file_encoding)
    df = pd.DataFrame(s)
    
    # Debugging: Check columns
    print("Columns in the DataFrame:", df.columns)
    print("First few rows:\n", df.head())

    # Strip spaces from column names
    df.columns = df.columns.str.strip()

    # Drop missing rows
    df = df.dropna()
    index = index1[:]
    
    # Check for missing columns
    missing_columns = [col for col in index1 if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in input file: {missing_columns}")
    
    if 'SMILES' in index1:
        index.remove('SMILES')
        f_rdk = []
        indices_to_delete = []
        for k, i in enumerate(df['SMILES']):
            j = Chem.MolFromSmiles(i)
            if j is not None:
                fingerprint = np.array(RDKFingerprint(j))
                f_rdk.append(fingerprint)
            else:
                print(f"Invalid SMILES string: {i}")
                indices_to_delete.append(k)
        
        df = df.drop(indices_to_delete)
        X = df[index].values

        df['Mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
        df['Fingerprint'] = df['Mol'].apply(lambda mol: FingerprintMol(mol)[0] if mol else None)

        X1 = np.array(list(df['Fingerprint']))
        X = np.append(X1, X, axis=1)
        index1.append('Encoding Molecules IDs: SMILES')
    else:
        X = df[index1].values

    return target, index1, index, X, df


def data_xy(input_file):
    target, index1, index, X, df = Check_index(input_file)
    if 'SMILES' in target:
        smiles_strings = df['SMILES'].tolist()
        tokenizer = Tokenizer(char_level=True, filters='')
        tokenizer.fit_on_texts(smiles_strings)
        num_tokens = len(tokenizer.word_index) + 1
        sequences = tokenizer.texts_to_sequences(smiles_strings)
        max_sequence_length = max(len(seq) for seq in sequences)
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')
        target1 = to_categorical(padded_sequences, num_classes=num_tokens)
        y = np.array(target1)
    else:
        y = df[target].values
    return index1, index, target[0], df, X, y

def data_test(input_file):
    target, index1, index, X, df = Check_index(input_file)
  #  X = df[index].values
    
 #   y = df[target].values
    return  df, X


def input0():    
        pixmappath01 = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
        file_pattern = pixmappath01 + "_Output_*.csv*" 
        files = glob.glob(file_pattern)
        for file in files:
            os.remove(file) # delete each file
        
        f = open(pixmappath01 + 'input_temp.txt', "r+")
        #get_data = False
        i=[]
        for line in f:
            i.append(line.split())
        file_name = i[:len(i)] 
        file_name = [j for i in file_name for j in i]
        f.close()
        index1, index, target, df, X, y = data_xy(file_name[0])
        f1 = open(pixmappath01 + 'input_temp_pred.txt', "r+")
        #get_data = False
        i=[]
        for line in f1:
            i.append(line.split())
        file_name1 = i[:len(i)] 
        file_name1 = [j for i in file_name1 for j in i]
        f1.close()
       # os.remove('input_temp_pred.txt')   
      #  os.remove(f)   
        df1, X_test1 = data_test(file_name1[0])
        return index1, index, target, df, X, y, df1, X_test1
   
    
def input1():  
        pixmappath01 = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
        file_pattern = pixmappath01 + "_Output_*.csv*" 
        files = glob.glob(file_pattern)
        for file in files:
            os.remove(file) # delete each file
        
        f = open(pixmappath01 + 'input_temp.txt', "r+")
        #get_data = False
        i=[]
        for line in f:
            i.append(line.split())
        file_name = i[:len(i)] 
        file_name = [j for i in file_name for j in i]
        f.close()
        index1, index, target, df, X, y = data_xy(file_name[0])

        return index1, index, target, df, X, y
    
    
def model0(X,y,X_test1,KF_splits, model,TargetVarScalerFit, PredictorScalerFit):
    scores = []
    kf = KFold(n_splits=KF_splits, shuffle=True , random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
                    
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)  
        y_pred1 = model.predict(X_test1)
        scores.append(sqrt(mean_squared_error(y_test, y_pred)))
                    
    Predictions=TargetVarScalerFit.inverse_transform(y_pred.reshape(-1,1))
    Predictions1=TargetVarScalerFit.inverse_transform(y_pred1.reshape(-1,1))
    y_test_orig=TargetVarScalerFit.inverse_transform(y_test.reshape(-1,1))
    Test_Data=PredictorScalerFit.inverse_transform(X_test)
    Test_Data1=PredictorScalerFit.inverse_transform(X_test1)
 #   plt.scatter(y_test_orig,Predictions)
    return  Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1  