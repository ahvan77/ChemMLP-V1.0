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
from PySide6.QtWidgets import QLineEdit, QFrame, QTextEdit, QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QDialog,QGridLayout, QMessageBox
from PySide6 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtGui import QIcon
from icon import MyIcon
from Err import error
import csv
import os

class PLSR(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.init_ui()

    def  init_ui(self):  
        self.pixmappath = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
        self.setGeometry(100, 100, 50, 50)
        self.setWindowTitle('PLS parameters')
        MyIcon(self)
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        
        self.label1 = QLabel('n_components From (Min = 1):', self)
        grid_layout.addWidget(self.label1, 0, 0)
        self.line_edit1 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit1, 0, 1)
        self.label2 = QLabel('to :', self)
        grid_layout.addWidget(self.label2, 0, 2)
        self.line_edit2 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit2, 0, 3)
        self.label3 = QLabel('step :', self)
        grid_layout.addWidget(self.label3, 0, 4)
        self.line_edit3 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit3, 0, 5)
        
        self.label7 = QLabel('K-Fold :', self)
        grid_layout.addWidget(self.label7, 2, 0)
        self.line_edit7 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit7, 2, 1)
        
        self.button1 = QPushButton('Run', self)
        self.button1.clicked.connect(self.run_svr)
        grid_layout.addWidget(self.button1, 3, 2)
        
    def run_svr(self):
        try:
                i_from = int(self.line_edit1.text())
                i_to = int(self.line_edit2.text())
                i_step = int(self.line_edit3.text())
                k_fold = int(self.line_edit7.text())
         
                
                index1, index, target, df, X, y, df1, X_test1 = data_xy.input0()
                y = y.reshape(-1,1)
                
                PredictorScaler=StandardScaler()
                TargetVarScaler=StandardScaler()
                PredictorScalerFit=PredictorScaler.fit(X)
                PredictorScalerFit=PredictorScaler.fit(X_test1)
                TargetVarScalerFit=TargetVarScaler.fit(y)

                X=PredictorScalerFit.transform(X)
                X_test1=PredictorScalerFit.transform(X_test1)
                y=TargetVarScalerFit.transform(y)
                mscores = []
                split=[]
                C1=[]
                def model1(KF_splits,X,y,alfa, X_test1):
                    from sklearn.cross_decomposition import PLSRegression
                    model = PLSRegression(n_components=alfa) # alpha = 0.0199
                    Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1 = data_xy.model0(X,y,X_test1,KF_splits,model, TargetVarScalerFit, PredictorScalerFit)
                    return Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1
    
                j = 0
                textfile = open(self.pixmappath + 'Output.txt', 'w')
                al = []
                for i in range(i_from,i_to,i_step):
                        alfa = i
                        Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1 = model1(k_fold,X,y,alfa, X_test1)
     
                
                        APE=100*(abs(y_test_orig-Predictions)/y_test_orig)
                        
                        mscores.append(np.mean(scores))
                        al.append(alfa)
                        
                        
                        textfile.write('========================= '+ '\n')
                        textfile.write('PLS regression parameters: '+ '\n')
                        textfile.write('n_components = '+str(alfa)+'\n')
                        textfile.write('========================= '+ '\n')
                        textfile.write('The Accuracy of  PLS model is: ' + str(100-np.mean(APE)) + '\n')
                        textfile.write('RMSE = ' + str(mscores[j])+ '\n')  
                        textfile.write('y_test ' + 'y_predict'+ '\n') 
                        data = np.column_stack([y_test_orig,Predictions])
                        np.savetxt(textfile , data, fmt=['%d','%-4d'])
                        j += 1
                    
  
                ms = list(zip(mscores,al))
                argmax = min(enumerate(ms), key=lambda x: x[1])
                Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1= model1(k_fold,X,y,argmax[1][1], X_test1)
        
                        
                APE=100*(abs(y_test_orig-Predictions)/y_test_orig)

                dialog = QtWidgets.QDialog()
                dialog.setWindowTitle('PLS Results')
                MyIcon(dialog)
                layout = QtWidgets.QVBoxLayout(dialog)

                text_widget = QtWidgets.QLabel()
                figure_widget = QtWidgets.QWidget()
                        
                layout.addWidget(text_widget)
                layout.addWidget(figure_widget)
                
                text_widget.setText("Features: \n")
                layout.addWidget(text_widget)

 
                for x in index1:
                    text_widget.setText(text_widget.text()+' '.join(x)+'\n')
              
                text_widget.setText(text_widget.text()+'========================= \n')
                text_widget.setText(text_widget.text()+'PLS parameters: \n')
                text_widget.setText(text_widget.text()+'n_components = '+str(argmax[1][1])+'\n')
                text_widget.setText(text_widget.text()+'========================= \n')     
                text_widget.setText(text_widget.text()+'The Accuracy of PLS model is: ' + str(100-np.mean(APE)) + '\n')
                text_widget.setText(text_widget.text()+'RMSE = ' + str(np.mean(scores)))

                fig, ax = plt.subplots()


                ax.scatter(y_test_orig, Predictions, color='g')
                ax.set_xlabel(target+'_test')
                ax.set_ylabel(target+'_prediction')
                ax.set_title('PLS')
                
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                canvas.destroyed.connect(fig.clf)
                dialog.exec_()


                textfile = open(self.pixmappath + 'PLS_'+'Output.txt', 'w')
                df1[target+'_prediction']=Predictions1
                df1.to_csv(self.pixmappath + 'PLS_Output.csv', index=False, line_terminator='\n')
                textfile.close()                

        except Exception as e:
            error(e)          
            
def PLS():

  #  app = QApplication(sys.argv)
    window = PLSR()
    window.show()
    window.exec_()  