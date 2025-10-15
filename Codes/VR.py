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
from PySide6.QtWidgets import QLineEdit, QCheckBox, QTextEdit, QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QDialog,QGridLayout, QMessageBox
from PySide6 import QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtGui import QIcon
from icon import MyIcon
from Err import error
import csv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import os


class VRM(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.init_ui()

    def  init_ui(self):   
        self.pixmappath = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
      #  self.setGeometry(100, 100, 400, 200)
        self.setWindowTitle('VR Technique')
        MyIcon(self)
        layout = QGridLayout()
        self.setLayout(layout)
    #    layout.addWidget(QLabel("Select ML models of your choice:", font=14), 0, 0, 1, 2)
        
        # check boxes
        self.LR = QCheckBox("LinearRegression (LR)")
        layout.addWidget(self.LR, 2, 0)
        self.GBR = QCheckBox("Gradient Boosting Regressor (GBR)")
        layout.addWidget(self.GBR, 3, 0)
        self.RFR = QCheckBox("Random Forest Regressor (RFR)")
        layout.addWidget(self.RFR, 4, 0)
        self.KNR = QCheckBox("KNeighbors Regressor (KNR)")
        layout.addWidget(self.KNR, 5, 0)
        
        # K-Fold split
        
        self.label7 = QLabel('K-Fold :', self)
        layout.addWidget(self.label7, 6, 0)
        self.line_edit7 = QLineEdit(self)
        layout.addWidget(self.line_edit7, 6, 1)

        
        # submit button
        
        self.button = QPushButton("Run")
        self.button.clicked.connect(self.run_svr)
        layout.addWidget(self.button, 10, 0, 1, 2)
        
        self.setLayout(layout)
        
        
        
    def run_svr(self):
        try:
            k_fold = int(self.line_edit7.text())
        except Exception as e:
                error(e)     
        all_models = []
        try:    
           
                if self.LR.isChecked():
                    r = LinearRegression()
                    all_models.append(('lr', r))
                if self.GBR.isChecked():
                    r = GradientBoostingRegressor()
                    all_models.append(('gb', r))
                if self.RFR.isChecked():
                    r = RandomForestRegressor()
                    all_models.append(('rf', r))
                if self.KNR.isChecked():
                    r = KNeighborsRegressor()
                    all_models.append(('kn', r))
                
                
         
                
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
                def model1(KF_splits,X,y, all_models, X_test1):
                    model = VotingRegressor(all_models)
                    Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1 = data_xy.model0(X,y,X_test1,KF_splits,model, TargetVarScalerFit, PredictorScalerFit)
                    return Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1
    
              #  self.root= Toplevel()
                textfile = open(self.pixmappath + 'Output.txt', 'w')
                Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1 = model1(k_fold,X,y, all_models, X_test1)
                APE=100*(abs(y_test_orig-Predictions)/y_test_orig)
                textfile.write('========================= '+ '\n')
                textfile.write('The Accuracy of  Voting Regressor model is: ' + str(100-np.mean(APE)) + '\n')
                textfile.write('RMSE = ' + str(np.mean(scores))+ '\n')  
                textfile.write('y_test ' + 'y_predict'+ '\n') 
                data = np.column_stack([y_test_orig,Predictions])
                np.savetxt(textfile , data, fmt=['%d','%-4d'])
                textfile.close()     #   self.root.mainloop()
                

                dialog = QtWidgets.QDialog()
                dialog.setWindowTitle('Voting Regressor (VR) model Results')
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
                text_widget.setText(text_widget.text()+'The Accuracy of VR model is: ' + str(100-np.mean(APE)) + '\n')
                text_widget.setText(text_widget.text()+'RMSE = ' + str(np.mean(scores)))

                fig, ax = plt.subplots()

                ax.scatter(y_test_orig, Predictions, color='g')
                ax.set_xlabel(target+'_test')
                ax.set_ylabel(target+'_prediction')
                ax.set_title('VR')
                
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                canvas.destroyed.connect(fig.clf)
                dialog.exec_()


                textfile = open(self.pixmappath + 'VR_'+'Output.txt', 'w')
                df1[target+'_prediction']=Predictions1
                df1.to_csv(self.pixmappath + 'VR_Output.csv', index=False, lineterminator='\n')
                textfile.close()                

        except Exception as e:
            error(e)          

def VR():

 #   app = QApplication(sys.argv)
    window = VRM()
    window.show()
    window.exec_()   