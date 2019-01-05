#Has not yet been applied with functions to the main, but is
#   a work in progress to future goals.


import sys
from PyQt5 import QtGui, QtCore

class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(0, 0, 800, 480)
        self.setWindowTitle("DistoPod")
        #self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))
        
        self.home()

    def home(self):
        btn = QtGui.QPushButton("Quit", self)
        btn.clicked.connect(self.close_application)
        btn.resize(btn.minimumSizeHint())
        btn.move(0,100)

        self.progress = QtGui.QProgressBar(self)
        self.progress.setGeometry(200, 80, 250, 20)

        self.btn = QtGui.QPushButton("Start Scan",self)
        self.btn.move(200,120)
        self.btn.clicked.connect(self.scan)
        
        self.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))
        
        # Input box used to csv file name
        flow = QtGui.QFormLayout()
        inputBox = QtGui.QLineEdit()
        flow.QtGui.addRow("Cave Name: ", inputBox)
        
        self.show()

    #Need to Find time associated with the resolution
    def scan(self):
        self.completed = 0

        while self.completed < 100:
            #Need to Find time associated with the resolution
            self.completed += 0.0001
            self.progress.setValue(self.completed)
        
    #def getCaveName(self):
       
    def close_application(self):
        choice = QtGui.QMessageBox.question(self, QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            print("Exiting DistoPod")
            sys.exit()
        else:
            pass
        
        

    
def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())


run()