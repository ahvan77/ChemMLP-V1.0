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
import os


class SVR1(QDialog):
    def __init__(self, parent=None):

        super().__init__(parent)
        self.init_ui()
        

      #  self.central_widget = QWidget()
       # self.setCentralWidget(self.central_widget)
    def  init_ui(self):   
        self.pixmappath = os.path.abspath(os.path.dirname(__file__)) + '/Data/'
#        self.setGeometry(100, 100, 50, 50)
        self.setWindowTitle('SVR parameters')
        MyIcon(self)
        grid_layout = QGridLayout()
        self.setLayout(grid_layout)
        
        self.label1 = QLabel('i (C=2\u00B0i) From :', self)
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
        
        self.label4 = QLabel('g (\u03B5 \u2215 14): From :', self)
        grid_layout.addWidget(self.label4, 1, 0)
        self.line_edit4 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit4, 1, 1)
        self.label5 = QLabel('to :', self)
        grid_layout.addWidget(self.label5, 1, 2)
        self.line_edit5 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit5, 1, 3)
        self.label6 = QLabel('step :', self)
        grid_layout.addWidget(self.label6, 1, 4)
        self.line_edit6 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit6, 1, 5)
        self.label7 = QLabel('K-Fold :', self)
        grid_layout.addWidget(self.label7, 2, 0)
        self.line_edit7 = QLineEdit(self)
        grid_layout.addWidget(self.line_edit7, 2, 1)
        
        self.button1 = QPushButton('Run', self)
        self.button1.clicked.connect(self.run_svr)
        grid_layout.addWidget(self.button1, 3, 2)
        
    #    self.text_edit = QTextEdit()
    #    self.text_edit.setReadOnly(True)
    #    grid_layout.addWidget(self.text_edit, 4, 0, 1, 6)
        
      #  self.central_widget.setLayout(grid_layout)
        
    def run_svr(self):
        try:
                i_from = int(self.line_edit1.text())
                i_to = int(self.line_edit2.text())
                i_step = int(self.line_edit3.text())
                g_from = int(self.line_edit4.text())
                g_to = int(self.line_edit5.text())
                g_step = int(self.line_edit6.text())
                k_fold = int(self.line_edit7.text())
         
                
                index1, index, target, df, X, y, df1, X_test1 = data_xy.input0()
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
                mscores = []
                split=[]
                C1=[]
                def model2(KF_splits,X,y,i,g0):
                            model = make_pipeline(StandardScaler(), SVR(C=2**i, epsilon=float(g0)/14))
                            Test_Data, y_test_orig, Predictions, scores, Test_Data1,Predictions1 = data_xy.model0(X,y,X_test1,KF_splits,model, TargetVarScalerFit, PredictorScalerFit)         
                            return Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1
                g=[i for i in range(g_from,g_to,g_step)]
                j = 0
                gl = []
                ci = []
                textfile = open(self.pixmappath + 'Output.txt', 'w')
                for i in range(i_from,i_to,i_step):
                    for jj in range(g_from,g_to,g_step):
                                
                    
                        Test_Data, y_test_orig, Predictions, scores, Test_Data2, Predictions2 = model2(k_fold,X,y,i, jj)
                        
                         
                     #   TestingData=pd.DataFrame(data=Test_Data, columns=index)
                     #   TestingData[target]=y_test_orig
                     #   TestingData['PredictedTarget']=Predictions
                     #   TestingData.head()
                
                        APE=100*(abs(y_test_orig-Predictions)/y_test_orig)
                   # for j in range(0,5):
                       
                      #  plt.scatter(y_test_orig,Predictions2)
                        mscores.append(np.mean(scores))
                        gl.append(jj)
                        ci.append(i)
                        textfile.write('========================= '+ '\n')
                        textfile.write('SVR parameters: '+ '\n')
                        textfile.write('C = 2^'+str(i)+ '   epsilon= '+ str((jj)/14) + '\n')
                        textfile.write('========================= '+ '\n')
                        textfile.write('The Accuracy of  SVR model is: ' + str(100-np.mean(APE)) + '\n')
                        textfile.write('RMSE = ' + str(mscores[j])+ '\n')  
                        textfile.write('y_test ' + 'y_predict'+ '\n') 
                        data = np.column_stack([y_test_orig,Predictions])
                        np.savetxt(textfile , data, fmt=['%d','%-4d'])
                        j += 1
                    
                    
                       #         TestingData.head()
         #       self.frame = QFrame(self)      
                ms = list(zip(mscores,ci,gl))
                argmax = min(enumerate(ms), key=lambda x: x[1])
                Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1= model2(k_fold,X,y,argmax[1][1], argmax[1][2])
            
                
                      #  Test_Data, y_test_orig, Predictions, scores, Test_Data1, Predictions1 = model2(ent,X,y, X_test1)
                    #    TestingData=pd.DataFrame(data=Test_Data, columns=index)
                    #    TestingData[target]=y_test_orig
                     #   TestingData['PredictedTarget']=Predictions
                     #   TestingData.head()
                        
                        
                        
            
                        
                APE=100*(abs(y_test_orig-Predictions)/y_test_orig)
                      #  TestingData['APE']=APE
                     #   app = QApplication(sys.argv)
         # Set up the main widget and layout
           #     main_widget = QtWidgets.QDialog(self)
                dialog = QtWidgets.QDialog()
                dialog.setWindowTitle('SVR Results')
                MyIcon(dialog)
              #  main_widget = QtWidgets.QWidget(dialog)
                layout = QtWidgets.QVBoxLayout(dialog)

                text_widget = QtWidgets.QLabel()
                figure_widget = QtWidgets.QWidget()
                        
                layout.addWidget(text_widget)
                layout.addWidget(figure_widget)
                
                text_widget.setText("Features: \n")
                layout.addWidget(text_widget)
                
           #     text_widget.setText("\n".join(text))
                
                # Add the text to the QTextEdit widget
             #   self.text_edit.append('Features: \n')
 
                for x in index1:
                    text_widget.setText(text_widget.text()+' '.join(x)+'\n')
                #    self.text_edit.append(x)
                
                text_widget.setText(text_widget.text()+'========================= \n')
           #     self.text_edit.append('========================= \n')
                argmax = [[0, 2, 5]]  # Replace with your argmax list
                text_widget.setText(text_widget.text()+'SVR parameters: \n')
            #    self.text_edit.append('SVR parameters: \n')
                text_widget.setText(text_widget.text()+'C = 2^'+str(argmax[0][1])+ '   epsilon= '+ str((argmax[0][2])/14) + '\n')
          #      self.text_edit.append('C = 2^'+str(argmax[0][1])+ '   epsilon= '+ str((argmax[0][2])/14) + '\n')
                text_widget.setText(text_widget.text()+'========================= \n')     
                #self.text_edit.append('========================= \n')
                text_widget.setText(text_widget.text()+'The Accuracy of SVR model is: ' + str(100-np.mean(APE)) + '\n')
           #     self.text_edit.append('The Accuracy of SVR model is: ' + str(100-np.mean(APE)) + '\n')
                text_widget.setText(text_widget.text()+'RMSE = ' + str(np.mean(scores)))
             #   self.text_edit.append('RMSE = ' + str(np.mean(scores)))
        
                # Create a matplotlib FigureCanvas widget
                
            #    main_layout = QHBoxLayout(self)
          #      self.setLayout(main_layout)

                fig, ax = plt.subplots()
               # self.figure = Figure(figsize=(5, 4), dpi=100)
               # self.ax = self.figure.add_subplot(111)

                ax.scatter(y_test_orig, Predictions, color='g')
                ax.set_xlabel(target+'_test')
                ax.set_ylabel(target+'_prediction')
                ax.set_title('SVR')
                
                canvas = FigureCanvas(fig)
                layout.addWidget(canvas)
                canvas.destroyed.connect(fig.clf)
                dialog.exec_()




            #    self.canvas = FigureCanvas(self.figure)
            #    main_layout.addWidget(self.canvas)
        
                # Set the size of the FigureCanvas widget
             #   self.canvas.setMinimumSize(400, 300)
        
                # Set the size of the QTextEdit widget
              #  self.text_edit.setMinimumSize(200, 300)
        
                # Set the layout of the dialog
               # self.canvas.draw()
                
               # self.setLayout(layout) 
                   #     sys.exit(app.exec_())                
                                    # Save scatter plot to file
                textfile = open(self.pixmappath + 'SVR_'+'Output.txt', 'w')
                df1[target+'_prediction'] = Predictions1
                df1.to_csv(self.pixmappath + 'SVR_Output.csv', index=False, lineterminator='\n')
                textfile.close()
        except Exception as e:
            error(e)          
                #    return
def SVR2():

  #  app = QApplication(sys.argv)
    window = SVR1()
    window.show()
    window.exec_()   # use app.exec_() without sys.exit()    
  