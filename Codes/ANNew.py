import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import KFold
from math import sqrt
from sklearn.metrics import mean_squared_error
import sklearn
import data_xy
from sklearn.preprocessing import StandardScaler
import sys
from PySide6.QtWidgets import QLineEdit, QCheckBox, QFrame, QTextEdit, QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QDialog,QGridLayout, QMessageBox
from PySide6 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtGui import QIcon, QFont
from icon import MyIcon
from Err import error
import keras
from keras import ops
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
import csv
from PySide6.QtCore import Qt
import os
import subprocess

class ANN_2L(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.init_ui()

    def  init_ui(self):
        self.pixmappath = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
        self.node_all1 =[]
        self.layer_all1 = []
        self.epochs_all = []
        self.batch_all = []
        self.label_all1 = []
        self.NL = 0
      #  self.setGeometry(200, 100, 50, 50)
        self.setWindowTitle('Deep Learning')
        MyIcon(self)
        font = QFont()
        font.setBold(True)
        self.grid_layout = QGridLayout()
        self.setLayout(self.grid_layout)
                
        self.label2 = QLabel('Parametes', self)
        self.label2.setFont(font)
        self.grid_layout.addWidget(self.label2, 0, 0)
        self.labelOpt = QLabel('Optimizer ' , self) 
        self.grid_layout.addWidget(self.labelOpt, 2, 0)
        self.combo_box1 = QComboBox()
        self.combo_box1.addItems(['adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'])
        self.grid_layout.addWidget(self.combo_box1, 2, 1)

        self.epochs_in = QLabel('Epochs ' , self) 
        self.grid_layout.addWidget(self.epochs_in, 4, 0)
        
        self.label1 = QLabel('From :', self)
        self.grid_layout.addWidget(self.label1, 4, 1)
        self.line_edit1 = QLineEdit(self)
        self.grid_layout.addWidget(self.line_edit1, 4, 2)
        self.label2 = QLabel('to :', self)
        self.grid_layout.addWidget(self.label2, 4, 3)
        self.line_edit2 = QLineEdit(self)
        self.grid_layout.addWidget(self.line_edit2, 4, 4)
        self.label3 = QLabel('step :', self)
        self.grid_layout.addWidget(self.label3, 4, 5)
        self.line_edit3 = QLineEdit(self)
        self.grid_layout.addWidget(self.line_edit3, 4, 6)
        
        
     #   self.ep = QLineEdit(self)
     #   self.grid_layout.addWidget(self.ep, 2, 1)
      #  self.epochs_all.append(self.ep)
        
        self.batchs_in = QLabel('Batch size ' , self) 
        self.grid_layout.addWidget(self.batchs_in, 5, 0)
        
        self.label_1 = QLabel('From :', self)
        self.grid_layout.addWidget(self.label_1, 5, 1)
        self.line_edit_1 = QLineEdit(self)
        self.grid_layout.addWidget(self.line_edit_1, 5, 2)
        self.label_2 = QLabel('to :', self)
        self.grid_layout.addWidget(self.label_2, 5, 3)
        self.line_edit_2 = QLineEdit(self)
        self.grid_layout.addWidget(self.line_edit_2, 5, 4)
        self.label_3 = QLabel('step :', self)
        self.grid_layout.addWidget(self.label_3, 5, 5)
        self.line_edit_3 = QLineEdit(self)
        self.grid_layout.addWidget(self.line_edit_3, 5, 6)
        
   #     self.ba = QLineEdit(self)
   #     self.grid_layout.addWidget(self.ba, 3, 1)
   #     self.batch_all.append(self.ba)
               # self.combo_box1_all.append(self.combo_box1.currentText())

              #  layout = QGridLayout()
               # self.setLayout(layout)
        self.DL_diagram = QCheckBox("Show Deep Learning Schematic Diagram")
        self.grid_layout.addWidget(self.DL_diagram, 6, 0)
                

        
           
        self.button1 = QPushButton('New Layer', self)
        self.button1.clicked.connect(self.run_svr)
        self.grid_layout.addWidget(self.button1, 8, 0)
        
        
        separator = QFrame(self)
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        
        separator1 = QFrame(self)
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        
        separator0 = QFrame(self)
        separator0.setFrameShape(QFrame.HLine)
        separator0.setFrameShadow(QFrame.Sunken)
        
        separator01 = QFrame(self)
        separator01.setFrameShape(QFrame.HLine)
        separator01.setFrameShadow(QFrame.Sunken)
        
        # add the separator to the grid layout
        self.grid_layout.addWidget(separator01, 1, 0, 1,8)
        self.grid_layout.addWidget(separator0, 3, 0, 1,8)
        self.grid_layout.addWidget(separator1, 7, 0, 1,8)
        self.grid_layout.addWidget(separator, 9, 0, 1,8)
        
        
                
        self.label00 = QLabel('Number of Nodes', self)
        self.label00.setFont(font)
        self.grid_layout.addWidget(self.label00, 10, 1)
        self.label01 = QLabel('Select the Function', self)
        self.label01.setFont(font)
        self.grid_layout.addWidget(self.label01, 10, 3)        
        
        # set the alignment of the labels
        self.label00.setAlignment(Qt.AlignCenter)
        self.label01.setAlignment(Qt.AlignCenter)
        
        # add the labels to the grid layout
        self.grid_layout.addWidget(self.label00, 10, 1)
        self.grid_layout.addWidget(self.label01, 10, 3)
        
        self.button1 = QPushButton('Run', self)
        self.button1.clicked.connect(self.run_svr2)
        self.grid_layout.addWidget(self.button1, 0, 5)     
        self.n = 10
        
        self.delete_button = QPushButton('Delete', self)
        self.grid_layout.addWidget(self.delete_button, self.n , 5)
        self.delete_button.clicked.connect(self.delete_layer)
        
    def run_svr(self):
        self.n += 1
        s = 'edit_' + str(self.n)
        try:
            self.NL += 1
            self.label1 = QLabel('Layer No. ' + str(self.NL), self)
            self.grid_layout.addWidget(self.label1, self.n+1, 0)
            self.label_all1.append(self.label1)
            
            self.s = QLineEdit(self)
            self.grid_layout.addWidget(self.s, self.n+1, 1)
            self.node_all1.append(self.s)
            
            self.combo_box = QComboBox()
           # self.combo_box.setMinimumSize(200, 30)
            self.combo_box.addItems(['relu', 'sigmoid', 'tanh', 'deserialize', 
                                     'elu', 'exponential', 'gelu', 'get', 
                                     'hard_sigmoid', 'linear', 'selu', 'serialize', 
                                     'softmax', 'softplus', 'softsign', 'swish'])
            self.grid_layout.addWidget(self.combo_box, self.n+1, 3)
            self.layer_all1.append(self.combo_box)
         #   self.numnode.textChanged.connect(self.update_node_all1)  # Connect textChanged signal
         #   self.grid_layout.addW3idget(self.numnode, self.n+row, 1)
          #  self.node_all1.append(self.numnode.text())
            self.n += 1  # Increment self.n in each iteration of the loop
            self.delete_button.setEnabled(True)
            
        except Exception as e:
            error(e)    

    def delete_layer(self, n):
        if len(self.node_all1) > 0:
            # Remove the last layer elements associated with the last row from the GUI
            last_label = self.label_all1.pop()  # Remove and get the last label
            self.grid_layout.removeWidget(last_label)  # Remove the label from the layout
            last_label.deleteLater()  # Delete the QLabel
            
            last_s = self.node_all1.pop()  # Remove and get the last QLineEdit
            self.grid_layout.removeWidget(last_s)  # Remove the QLineEdit from the layout
            last_s.deleteLater()  # Delete the QLineEdit
            
            last_combo_box = self.layer_all1.pop()  # Remove and get the last QComboBox
            self.grid_layout.removeWidget(last_combo_box)  # Remove the QComboBox from the layout
            last_combo_box.deleteLater()  # Delete the QComboBox
            
#            del self.label_all1[-1]
#            del self.node_all1[-1]
#            del self.layer_all1[-1]
            
            # Adjust the layout to fill the gap left by the deleted layer
            self.grid_layout.removeItem(self.grid_layout.itemAtPosition(self.n, 0))
            self.grid_layout.removeItem(self.grid_layout.itemAtPosition(self.n, 1))
            self.grid_layout.removeItem(self.grid_layout.itemAtPosition(self.n, 3))
            self.grid_layout.removeItem(self.grid_layout.itemAtPosition(self.n, 4))
            
            # Reduce the count of layers (self.n) by 1
            self.n -= 1
    
            # If there are no more layers, disable the delete button
        if len(self.node_all1) == 0:
                self.delete_button.setEnabled(False)
                
    def run_svr2(self):
        try:
                                self.node_all =[]
                                self.layer_all =[]
                                for i in range(len(self.node_all1)):
                                    self.node_all.append(self.node_all1[i].text())
                                    self.layer_all.append(self.layer_all1[i].currentText())
                                index1, index, target, df, X, y, df1, X_test1 = data_xy.input0()
                                
                                if "SMILES" in target:
                                    decode_target = True
                                    X = np.concatenate((X, y), axis=1)
                                else:
                                    decode_target = False 
                                    y = y.reshape(-1,1)
                            
                                PredictorScaler=StandardScaler()
                                TargetVarScaler=StandardScaler()
                            # Storing the fit object for later reference
                                PredictorScalerFit=PredictorScaler.fit(X)
                                PredictorScalerFit=PredictorScaler.fit(X_test1)
                                TargetVarScalerFit=TargetVarScaler.fit(y)
                            
                            # Generating the standardized values of X and y
                                X=PredictorScalerFit.transform(X)
                                X_test1=PredictorScalerFit.transform(X_test1)
                                y=TargetVarScalerFit.transform(y)
                                split=[]
                        
                        
                                from sklearn.model_selection import train_test_split
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        
                                # Defining a function to find the best parameters for ANN
                                def FunctionFindBestParams(X_train, y_train, X_test, y_test):
                                    
                                    # Defining the list of hyper parameters to try
                                    i_from = int(self.line_edit1.text())
                                    i_to = int(self.line_edit2.text())
                                    i_step = int(self.line_edit3.text())
                                    
                                    j_from = int(self.line_edit_1.text())
                                    j_to = int(self.line_edit_2.text())
                                    j_step = int(self.line_edit_3.text())
                                    
                                    
                                    batch_size_list = [i for i in range(i_from,i_to,i_step)] #[5, 10, 15, 20]
                                    epoch_list  = [j for j in range(j_from,j_to,j_step)] #  [5, 10, 50, 100]
                                    
                                    import pandas as pd
                                    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'Parameters', 'Accuracy'])
                                    
                                    # initializing the trials
                                    TrialNumber=0
                                    for batch_size_trial in batch_size_list:
                                        for epochs_trial in epoch_list:
                                            TrialNumber+=1
                                            # create ANN model
                                            model = tf.keras.Sequential()
                                            # Defining the first layer of the model
                                        #    model.add(tf.keras.layers.Dense(units= node_all[0], input_dim=X_train.shape[1], kernel_initializer='normal', activation= self.layer_all[0]))
                                             
                                            # Defining the Second layer of the model
                                            # after the first layer we don't have to specify input_dim as keras configure it automatically
                                            for i in range(0, len(self.node_all1)):
                                                model.add(tf.keras.layers.Dense(units=int(self.node_all[i]), kernel_initializer='normal', activation=self.layer_all[i]))
                                
                                            # The output neuron is a single fully connected node 
                                            # Since we will be predicting a single number
                                        ##    model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
                                
                                            # Compiling the model
                                            model.compile(loss='mean_squared_error', optimizer=self.combo_box1.currentText())
                                
                                            # Fitting the ANN to the Training set
                                            model.fit(X_train, y_train ,batch_size = batch_size_trial, epochs = epochs_trial, verbose=0)
                                
                                             
                                            # Scaling the y_test Price data back to original price scale
                                            y_test_trial=TargetVarScalerFit.inverse_transform(y_test)
                                             
                                            # Scaling the test data back to original scale
                                
                                
                                            MAPE = np.mean(100 * (np.abs(y_test_trial-TargetVarScalerFit.inverse_transform(model.predict(X_test)))/y_test_trial))
                                            
                                         ###   # printing the results of the current iteration
                                            print(TrialNumber, 'Parameters:','batch_size:', batch_size_trial,'-', 'epochs:',epochs_trial, 'Accuracy:', 100-MAPE)
                                            
                                            # Inside the loop, create a DataFrame with the new data
                                            new_data = pd.DataFrame(data=[[TrialNumber, str(batch_size_trial)+'-'+str(epochs_trial), batch_size_trial, epochs_trial, 100-MAPE]],
                                                                    columns=['TrialNumber', 'Parameters', 'batch', 'epochs', 'Accuracy'])
                                            
                                            # Append the new DataFrame to the existing one
                                            SearchResultsData = pd.concat([SearchResultsData, new_data], ignore_index=True)
                                    return(SearchResultsData)
                                
                                
                                ######################################################
                                
                                
                                # Calling the function
                                
                                 
                                
                                ResultsData=FunctionFindBestParams(X_train, y_train, X_test, y_test)
                                AcMax = ResultsData[['Accuracy']].idxmax()
                                batch_opt = int (ResultsData.iloc[AcMax]['batch'])
                                ep_opt = int (ResultsData.iloc[AcMax]['epochs'])
                                
                                # create ANN model
                                #print(node_all, self.layer_all)
                                model = tf.keras.Sequential()
                                #un = 5 
                                # Defining the Input layer and FIRST hidden layer, both are same!
                             #   model.add(tf.keras.layers.Dense(units= node_all[0], input_dim=X_train.shape[1], kernel_initializer='normal', activation= self.layer_all[0]))
                                 
                                # Defining the Second layer of the model
                                # after the first layer we don't have to specify input_dim as keras configure it automatically
                                for i in range(0, len(self.node_all1)):
                                    model.add(tf.keras.layers.Dense(units=int(self.node_all[i]), kernel_initializer='normal', activation=self.layer_all[i]))
                                 
                                # The output neuron is a single fully connected node 
                                # Since we will be predicting a single number
                               ## model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
                                 
                                # Compiling the model
                                model.compile(loss='mean_squared_error', optimizer=self.combo_box1.currentText())
                                 
                                # Fitting the ANN to the Training set
                           ##     model.fit(X_train, y_train ,batch_size = 20, epochs = int(self.ep.get()), verbose=1)
                            #    print (batch_opt, ep_opt)
                             #   print(ResultsData.iloc[AcMax]['batch'], ResultsData['Accuracy'])
                            #    ResultsData.plot(x='Parameters', y='Accuracy', figsize=(15,4), kind='line')
                                
                                model.fit(X_train, y_train ,batch_size = batch_opt, epochs = ep_opt, verbose=0)
                                 
                                # Generating Predictions on testing data
                                Predictions=model.predict(X_test)
                                 
                                # Scaling the predicted Price data back to original price scale
                                Predictions=TargetVarScalerFit.inverse_transform(Predictions)
                                 
                                # Scaling the y_test Price data back to original price scale
                                y_test_orig=TargetVarScalerFit.inverse_transform(y_test)
                                 
                                # Scaling the test data back to original scale
                                Test_Data=PredictorScalerFit.inverse_transform(X_test)
                                 
                            #    TestingData=pd.DataFrame(data=Test_Data, columns=index)
                            #    TestingData[target]=y_test_orig
                             #   TestingData['PredictedTarget']=Predictions
                             #   TestingData.head()
                                
                                Predictions1=model.predict(X_test1)
                                 
                                # Scaling the predicted Price data back to original price scale
                                Predictions1=TargetVarScalerFit.inverse_transform(Predictions1)
                                
                                if decode_target:
  								       # Decode the predictions back into SMILES strings
                                   decoded_sequences = []
                                   for prediction in predictions1[:,0]:
                                       predicted_tokens = [np.argmax(token) for token in prediction]
                                       decoded_sequence = ''.join(tokenizer.index_word[token] for token in predicted_tokens if token != 0)
                                       decoded_sequences.append(decoded_sequence)

                                   predictions[:,0] = decoded_sequences
                                     
                                 
                                
                                # Computing the absolute percent error
                                APE=100*(abs(y_test_orig-Predictions)/y_test_orig)
                              #  TestingData['APE']=APE
                                 
                                print('The Accuracy of ANN model is:', 100-np.mean(APE))
                              #  TestingData.head()
                    
                
                    
                    
                                def GSCV(X_train, y_train, X_test, y_test):
                                       
                                    # Function to generate Deep ANN model 
                                    def make_regression_ann(Optimizer_trial):
                                        
                                        model = tf.keras.Sequential()
                                        model.add(tf.keras.layers.Dense(units= int(self.node_all[0]), input_dim=X_train.shape[1], kernel_initializer='normal', activation= self.layer_all[0]))
                                         
                                        # Defining the Second layer of the model
                                        # after the first layer we don't have to specify input_dim as keras configure it automatically
                                        for i in range(1, self.val):
                                            model.add(tf.keras.layers.Dense(units=int(self.node_all[i]), kernel_initializer='normal', activation=self.layer_all[i]))
                                     ##   model.add(tf.keras.layers.Dense(1, kernel_initializer='normal'))
                                        model.compile(loss='mean_squared_error', optimizer=Optimizer_trial)
                                        return model
                                     
                                    ###########################################
                                    from sklearn.model_selection import GridSearchCV
                        #            from keras.wrappers.scikit_learn import KerasRegressor
                                     
                                    # Listing all the parameters to try
                                    Parameter_Trials={'batch_size':[10,20,30],
                                                          'epochs':[10,20],
                                                        'Optimizer_trial':['adam', 'rmsprop']
                                                     }
                                     
                                    # Creating the regression ANN model
                                    RegModel=tf.keras.wrappers.scikit_learn.KerasRegressor(make_regression_ann, verbose=0)
                                     
                                    ###########################################
                                    from sklearn.metrics import make_scorer
                                     
                                    # Defining a custom function to calculate accuracy
                                    def Accuracy_Score(orig,pred):
                                        MAPE = np.mean(100 * (np.abs(orig-pred)/orig))
                                        print('#'*70,'Accuracy:', 100-MAPE)
                                        return(100-MAPE)
                                     
                                    custom_Scoring=make_scorer(Accuracy_Score, greater_is_better=True)
                                     
                                    #########################################
                                    # Creating the Grid search space
                                    # See different scoring methods by using sklearn.metrics.SCORERS.keys()
                                    grid_search=GridSearchCV(estimator=RegModel, 
                                                             param_grid=Parameter_Trials, 
                                                             scoring=custom_Scoring, 
                                                             cv=10)
                                     
                                    #########################################
                                    # Measuring how much time it took to find the best params
                                    import time
                                    StartTime=time.time()
                                     
                                    # Running Grid Search for different paramenters
                                    grid_search.fit(X,y, verbose=1)
                                     
                                    EndTime=time.time()
                                    print("########## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes')
                                     
                                    print('### Printing Best parameters ###')
                                    grid_search.best_params_
                         #       return  grid_search.best_params_  
                
                              #  plt.show()
                              #  plt.scatter(y_test_orig,Predictions)
                                
                                dialog = QtWidgets.QDialog()
                                dialog.setWindowTitle('Deep learning: ' + str(self.NL) +  ' layers results')
                                MyIcon(dialog)
                                layout = QtWidgets.QVBoxLayout(dialog)

                                text_widget = QtWidgets.QLabel()
                                figure_widget = QtWidgets.QWidget()
                                        
                                layout.addWidget(text_widget)
                                layout.addWidget(figure_widget)
                                fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6, 6))
                                ax[0].plot(ResultsData['Parameters'], ResultsData['Accuracy'])
                                plt.subplots_adjust(hspace=0.5)
                                print(y_test_orig.shape, Predictions.shape)
                                ax[1].scatter(y_test_orig[:,0],Predictions[:,0], color = 'g')
                                
                                
                                ax[0].set_xlabel('Parameters')
                                ax[0].set_ylabel('Accuracy')
                                ax[0].set_title('Function Find Best Parameters', fontsize=10)
                                ax[0].tick_params(axis='both', which='major', labelsize=6, pad=5)
                                
                                ax[1].set_xlabel(target+'_test')
                                ax[1].set_ylabel(target+'_prediction')
                                ax[1].set_title('Deep ANN model', fontsize=10)
                                
                                canvas1 = FigureCanvas(fig)
                                layout.addWidget(canvas1)
                              #  canvas1.destroyed.connect(fig.clf)
                                
                                
                                
                                
                                
                                text_widget.setText("Features: \n")
                                layout.addWidget(text_widget)

                 
                                for x in index1:
                                    text_widget.setText(text_widget.text()+' '.join(x)+'\n')
                              
                                text_widget.setText(text_widget.text()+  '========================= '+ '\n')  
                                text_widget.setText(text_widget.text()+  'Number of nodes: '+ '\n')
                                i = 0
                                for x in self.node_all:
                                    if i == 0:
                                        text_widget.setText(text_widget.text()+ 'Input Layer: ' +' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                    elif i == len(self.node_all)-1 :
                                        text_widget.setText(text_widget.text()+ 'Output Layer: ' +  ' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                        
                                    else:
                                        text_widget.setText(text_widget.text()+ 'Hidden Layer #' + str(i+1) + ' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                    i += 1
                                    
                                text_widget.setText(text_widget.text()+ '========================= '+ '\n')      
                                text_widget.setText(text_widget.text()+  'The Accuracy of  model is: ' + str(100-np.mean(APE)) + '\n')    
                                text_widget.setText(text_widget.text()+  'RMSE = ' + str(sqrt(mean_squared_error(y_test_orig[:,0],Predictions[:,0]))))   
            
                                if self.DL_diagram.isChecked(): 
                                    import plot_ann
                                    plot_ann.visualize_nn(self, model, description=True, figsize=(10,8))                    
       
                                dialog.exec ()

                                
                                textfile = open(self.pixmappath  + 'Output.txt', 'w')
                             #   textfile.write('Total time = ' + str( round((EndTime-StartTime)/60))+ ' Minutes'+ '\n')   
                             #   textfile.write('========================= '+ '\n')  
                                textfile.write('Features: '+ '\n')
                        
                                for x in index1:
                                    textfile.write(x + '\n')
                                textfile.write('========================= '+ '\n')  
                                textfile.write('Number of nodes: '+ '\n')
                                i = 0
                                for x in self.node_all:
                                    if i == 0:
                                        textfile.write('Input Layer: ' +' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                    elif i == len(self.node_all)-1 :
                                        textfile.write('Output Layer: ' +  ' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                        
                                    else:
                                        textfile.write('Hidden Layer #' + str(i+1) + ' = ' +  str(x)  + '  ,  ' + ' Function= ' + self.layer_all[i] + '\n')
                                    i += 1
                                    
                                textfile.write('========================= '+ '\n')
                                textfile.write('The Accuracy of  Deep ANN model model is: ' + str(100-np.mean(APE)) + '\n')
                                textfile.write('RMSE = ' + str(sqrt(mean_squared_error(y_test_orig[:,0],Predictions[:,0])))+ '\n')  
                                textfile.write('y_test ' + 'y_predict'+ '\n') 
                                data = np.column_stack([y_test_orig[:,0],Predictions[:,0]])
                                np.savetxt(textfile , data, fmt=['%d','%-4d'])
                                textfile.close()     #   self.root.mainloop()
                    
                                
                                textfile2 = open(self.pixmappath + 'ANN_'+'Output_test.txt', 'w')
                                df1[target+'_prediction']=Predictions1[:,0]
                                df1.to_csv(self.pixmappath + 'ANN_Output_test.csv', index=False, lineterminator='\n')
                                textfile2.close()
                                return
        except Exception as e:
            error(e) 
                            
def ANN():
  #  app = QApplication(sys.argv)
    window = ANN_2L()
    window.show()
    window.exec_()   

#ANN()    