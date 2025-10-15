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
from PySide6.QtWidgets import QLineEdit, QTextEdit, QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QDialog,QGridLayout, QMessageBox
from PySide6 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtGui import QIcon
from icon import MyIcon
from Err import error
import csv
import os

class KNRM(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.init_ui()

    def  init_ui(self):   
        self.pixmappath = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
      #  self.setGeometry(100, 100, 50, 50)
        self.setWindowTitle('KNR parameters')
        MyIcon(self)
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        
        
        self.label7 = QLabel('K-Fold :', self)
        grid_layout.addWidget(self.label7, 0, 0)
        self.line_edit7 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit7, 0, 1)
        
        self.button1 = QPushButton('Run', self)
        self.button1.clicked.connect(self.run_svr)
        grid_layout.addWidget(self.button1, 1, 2)
        
        
    def run_svr(self):
        try:

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
                def model1(KF_splits,X,y,X_test1):
                    from sklearn.neighbors import KNeighborsRegressor
                    model = KNeighborsRegressor()
                    Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1 = data_xy.model0(X,y,X_test1,KF_splits,model, TargetVarScalerFit, PredictorScalerFit)
                    return Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1
    
 
                textfile = open(self.pixmappath + 'Output.txt', 'w')
                Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1 = model1(k_fold,X,y, X_test1)               
                APE=100*(abs(y_test_orig-Predictions)/y_test_orig)
                textfile.write('========================= '+ '\n')
                textfile.write('The Accuracy of  KNeighbor Regressor (KNR) model is: ' + str(100-np.mean(APE)) + '\n')
                textfile.write('RMSE = ' + str(np.mean(scores))+ '\n')  
                textfile.write('y_test ' + 'y_predict'+ '\n') 
                data = np.column_stack([y_test_orig,Predictions])
                np.savetxt(textfile , data, fmt=['%d','%-4d'])
                textfile.close()     #   self.root.mainloop()
                

                dialog = QtWidgets.QDialog()
                dialog.setWindowTitle('KNeighbor Regressor (KNR) model Results')
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
                text_widget.setText(text_widget.text()+'The Accuracy of KNR model is: ' + str(100-np.mean(APE)) + '\n')
                text_widget.setText(text_widget.text()+'RMSE = ' + str(np.mean(scores)))

                fig, ax = plt.subplots()

                ax.scatter(y_test_orig, Predictions, color='g')
                ax.set_xlabel(target+'_test')
                ax.set_ylabel(target+'_prediction')
                ax.set_title('KNR')
                
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                canvas.destroyed.connect(fig.clf)
                dialog.exec_()


                textfile = open(self.pixmappath + 'KNR_'+'Output.txt', 'w')
                df1[target+'_prediction']=Predictions1
                df1.to_csv(self.pixmappath + 'KNR_Output.csv' , index=False, lineterminator='\n')
                textfile.close()                

        except Exception as e:
            error(e)          

def KNR():

 #   app = QApplication(sys.argv)
    window = KNRM()
 #   window.show()
    window.exec_()   