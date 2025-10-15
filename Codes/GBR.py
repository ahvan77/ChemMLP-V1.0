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

class GBRM(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.init_ui()

    def  init_ui(self):   
        self.pixmappath = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
     #   self.setGeometry(100, 100, 50, 50)
        self.setWindowTitle('GBR parameters')
        MyIcon(self)
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        
        self.label1 = QLabel('n_estimators(100) ):', self)
        grid_layout.addWidget(self.label1, 0, 0)
        self.line_edit1 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit1, 0, 1)
        self.label2 = QLabel('learning_rate (0.1) :', self)
        grid_layout.addWidget(self.label2, 0, 2)
        self.line_edit2 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit2, 0, 3)
        self.label3 = QLabel('max_depth (1) :', self)
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
                ne = int(self.line_edit1.text())
                lr = float(self.line_edit2.text())
                md = float(self.line_edit3.text())
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
                def model1(ne,lr,md,KF_splits,X,y, X_test1):
                    from sklearn.ensemble import GradientBoostingRegressor
                    model =GradientBoostingRegressor(
                        n_estimators=int(ne), learning_rate=lr, max_depth=int(md), random_state=0)
                    Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1 = data_xy.model0(X,y,X_test1,int(KF_splits),model, TargetVarScalerFit, PredictorScalerFit)
                    return Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1   


                textfile = open(self.pixmappath + 'Output.txt', 'w')
                Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1 = model1(ne,lr,md,k_fold,X,y, X_test1)
                APE=100*(abs(y_test_orig-Predictions)/y_test_orig)
                textfile.write('========================= '+ '\n')
                textfile.write('The Accuracy of  Gradient Boosting Regressor (GBR) model is: ' + str(100-np.mean(APE)) + '\n')
                textfile.write('RMSE = ' + str(np.mean(scores))+ '\n')  
                textfile.write('y_test ' + 'y_predict'+ '\n') 
                data = np.column_stack([y_test_orig,Predictions])
                np.savetxt(textfile , data, fmt=['%d','%-4d'])
                textfile.close()     #   self.root.mainloop()
                import csv
                textfile2 = open('GB_'+'Output_test.csv', 'w')
                df1[target+'_prediction']=Predictions1
                df1.to_csv(textfile2, index=False, lineterminator='\n')
                textfile2.close()

                dialog = QtWidgets.QDialog()
                dialog.setWindowTitle('Gradient Boosting Regressor (GBR) model Results')
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
                text_widget.setText(text_widget.text()+'The Accuracy of GBR model is: ' + str(100-np.mean(APE)) + '\n')
                text_widget.setText(text_widget.text()+'RMSE = ' + str(np.mean(scores)))

                fig, ax = plt.subplots()


                ax.scatter(y_test_orig, Predictions, color='g')
                ax.set_xlabel(target+'_test')
                ax.set_ylabel(target+'_prediction')
                ax.set_title('GBR')
                
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                canvas.destroyed.connect(fig.clf)
                dialog.exec_()


                textfile = open(self.pixmappath + 'GBR_'+'Output.txt', 'w')
                df1[target+'_prediction']=Predictions1
                df1.to_csv(self.pixmappath + 'GBR_Output.csv' , index=False, lineterminator='\n')
                textfile.close()                

        except Exception as e:
            error(e)          
            
def GBR():

  #  app = QApplication(sys.argv)
    window = GBRM()
    window.show()
    window.exec_()  
