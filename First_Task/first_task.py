import sys
import matplotlib
matplotlib.use('QtAgg')
import os
import csv
from sympy import sympify, symbols

import json
import numpy as np
from scipy.integrate import odeint

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QVBoxLayout, QPushButton, QFileDialog, QToolBar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(1, 1, 1)
        super(MplCanvas, self).__init__(fig)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)

        toolbar = QToolBar()
        self.addToolBar(toolbar)

        model_button_action = QAction("Open model", self)
        model_button_action.triggered.connect(self.openModelFileNameDialog)
        model_button_action.triggered.connect(self.areWeReadyToStart)
        toolbar.addAction(model_button_action)
        toolbar.addSeparator()

        coefficients_button_action = QAction("Read coefficients", self)
        coefficients_button_action.triggered.connect(self.openCoefficientsFileNameDialog)
        coefficients_button_action.triggered.connect(self.areWeReadyToStart)
        toolbar.addAction(coefficients_button_action)
        toolbar.addSeparator()

        initial_values_button_action = QAction("Read initial values", self)
        initial_values_button_action.triggered.connect(self.openInitialValuesFileNameDialog)
        initial_values_button_action.triggered.connect(self.areWeReadyToStart)
        toolbar.addAction(initial_values_button_action)
        toolbar.addSeparator()

        directory_button_action = QAction("Select directory for saving results", self)
        directory_button_action.triggered.connect(self.openDirectoryDialog)
        directory_button_action.triggered.connect(self.areWeReadyToStart)
        toolbar.addAction(directory_button_action)

        self.manageButton = QPushButton("Start/Stop")
        self.manageButton.clicked.connect(self.manage_process)
        self.manageButton.setEnabled(False)

        self.resetButton = QPushButton("Reset")
        self.resetButton.clicked.connect(self.reset)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.manageButton)
        layout.addWidget(self.resetButton)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)

        self.modelFilePath = None
        self.coefficientsFilePath = None
        self.initialValuesFilePath = None
        self.resultDirectoryPath = None

        self.time = 100
        self.points = 1001
        self.result_filename = 'results.csv'

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.setInterval(50)

        self.setCentralWidget(widget)
        self.show()

    def manage_process(self):
        if not self.timer.isActive():
            self.resetButton.setEnabled(False)
            self.timer.start()
        else:
            self.resetButton.setEnabled(True)
            self.timer.stop()

    def reset(self):
        self.process_initial_value_file()
        self.make_initial_plot()

    def openModelFileNameDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Select model file', '', 'Model Files (*.obj)')

        if fileName:
            self.modelFilePath = fileName
            print('Model file is selected:', self.modelFilePath)
            self.process_model_file()

    def process_model_file(self):
        self.squares = [36.1672 - 12.246120320000001, 99.4077 - 12.246120320000001 - 12.246120320000001, 36.1672 - 12.246120320000001 - 3.0615300800000003, 12.366 - 3.0615300800000003 - 3.0615300800000003, 160 - 3.0615300800000003]

        self.squares_matrix = np.zeros((5, 5))
        self.squares_matrix[0][1] = 12.246120320000001
        self.squares_matrix[1][2] = 12.246120320000001
        self.squares_matrix[2][3] = 3.0615300800000003
        self.squares_matrix[3][4] = 3.0615300800000003

        print('S_i:', self.squares)
        print('S_ij:', self.squares_matrix)

    def openCoefficientsFileNameDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Select coefficients file', '', 'Coefficients Files (*.json)')

        if fileName:
            self.coefficientsFilePath = fileName
            print('Opened coefficients file:', self.coefficientsFilePath)
            self.process_coefficients_file()

    def process_coefficients_file(self):
        with open(self.coefficientsFilePath, 'r') as json_file:
            coefficients = json.load(json_file)

        self.epsilons = []
        self.some_constants = []
        self.internal_heat_sources = []
        self.conductivities = np.zeros((5, 5))
        self.c_zero = 5.67

        for i in range(0, 5):
            self.epsilons.append(float(coefficients["eps" + str(i + 1)]))
            self.some_constants.append(float(coefficients["c" + str(i + 1)]))
            self.internal_heat_sources.append(coefficients["Q^R" + str(i + 1)])

            for j in range(0, 5):
                if "lambda" + str(i + 1) + str(j + 1) in coefficients:
                    self.conductivities[i][j] = coefficients["lambda" + str(i + 1) + str(j + 1)]

        print('Eps_i:', self.epsilons)
        print('c_i:', self.some_constants)
        print('Q^R_i:', self.internal_heat_sources)
        print('lambda_i:', self.conductivities)

    def openInitialValuesFileNameDialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, 'Select initial values file', '', 'Initial Values Files (*.csv)')

        if fileName:
            self.initialValuesFilePath = fileName
            print('Opened initial values file:', self.initialValuesFilePath)
            self.process_initial_value_file()

    def process_initial_value_file(self):
        with open(self.initialValuesFilePath, 'r') as initial_values_file:
            data = list(csv.reader(initial_values_file))
            self.initial_values = data[0]
            self.initial_values = [float(numeric_string) for numeric_string in self.initial_values]
        print('Initial temperatures:', self.initial_values)

    def openDirectoryDialog(self):
        directory_name = QFileDialog.getExistingDirectory(self, 'Select Directory for saving results')

        if directory_name:
            self.resultDirectoryPath = directory_name
            print('Selected directory for saving results:', self.resultDirectoryPath)

    def make_function_system(self, initial_values, time):
        self.k_matrix = np.zeros((5, 5))
        self.k_matrix[0][1] = self.conductivities[0, 1] * self.squares_matrix[0, 1]
        self.k_matrix[1][2] = self.conductivities[1, 2] * self.squares_matrix[1, 2]
        self.k_matrix[2][3] = self.conductivities[2, 3] * self.squares_matrix[2, 3]
        self.k_matrix[3][4] = self.conductivities[3, 4] * self.squares_matrix[3, 4]

        function = np.empty(5)

        A = symbols('A')
        t = symbols('t')
        QR2 = self.internal_heat_sources[1]
        QR2_function = sympify(QR2)

        function[0] = (-self.k_matrix[0, 1] * (initial_values[0] - initial_values[1]) - self.epsilons[0] * self.squares[0] * self.c_zero * (initial_values[0] / 100) ** 4) / self.some_constants[0]
        function[1] = (-self.k_matrix[0, 1] * (initial_values[1] - initial_values[0]) - self.k_matrix[1, 2] * (initial_values[1] - initial_values[2]) -self.epsilons[1] * self.squares[1] * self.c_zero * (initial_values[1] / 100) ** 4 + QR2_function.evalf(subs={A: 2, t: time})) / self.some_constants[1]
        function[2] = (-self.k_matrix[1, 2] * (initial_values[2] - initial_values[1]) - self.k_matrix[2, 3] * (initial_values[2] - initial_values[3]) -self.epsilons[2] * self.squares[2] * self.c_zero * (initial_values[2] / 100) ** 4) / self.some_constants[2]
        function[3] = (-self.k_matrix[2, 3] * (initial_values[3] - initial_values[2]) - self.k_matrix[3, 4] * (initial_values[3] - initial_values[4]) -self.epsilons[3] * self.squares[3] * self.c_zero * (initial_values[3] / 100) ** 4)  /self.some_constants[3]
        function[4] = (-self.k_matrix[3, 4] * (initial_values[4] - initial_values[3]) - self.epsilons[4] * self.squares[4] * self.c_zero * (initial_values[4] / 100) ** 4) / self.some_constants[4]

        return function

    def make_initial_plot(self):
        self.time = 100
        time = np.linspace(0, int(self.time), self.points)
        ode_system_solution = odeint(self.make_function_system, self.initial_values, time)
        np.savetxt(self.resultDirectoryPath + '/' + self.result_filename, np.column_stack([time, ode_system_solution]), delimiter=",", fmt='%10.5f')

        self.xdata = time
        self.ydata1 = ode_system_solution[:,0]
        self.ydata2 = ode_system_solution[:,1]
        self.ydata3 = ode_system_solution[:,2]
        self.ydata4 = ode_system_solution[:,3]
        self.ydata5 = ode_system_solution[:,4]

        self.canvas.axes.cla()
        self.canvas.axes.grid(True)
        self.canvas.axes.plot(self.xdata, self.ydata1, color='brown', label = "T_1(t)")
        self.canvas.axes.plot(self.xdata, self.ydata2, color='red', label = "T_2(t)")
        self.canvas.axes.plot(self.xdata, self.ydata3, color='orange', label = "T_3(t)")
        self.canvas.axes.plot(self.xdata, self.ydata4, color='green', label = "T_4(t)")
        self.canvas.axes.plot(self.xdata, self.ydata5, color='blue', label = "T_5(t)")
        self.canvas.axes.legend()
        self.canvas.draw()

    def update_plot(self):
            number_of_points = 11
            time = np.linspace(self.time, int(self.time) + 1, number_of_points)
            time_for_plot = np.linspace(int(self.time) - 90, int(self.time) + 1, 1001)
            self.xdata = time_for_plot

            with open(self.resultDirectoryPath + '/' + self.result_filename, 'rb') as initial_values_file:
                try:
                    initial_values_file.seek(-2, os.SEEK_END)
                    while initial_values_file.read(1) != b'\n':
                        initial_values_file.seek(-2, os.SEEK_CUR)
                except OSError:
                    initial_values_file.seek(0)

                last_line = initial_values_file.readline().decode()
                self.initial_values = last_line.split(",")
                self.initial_values = self.initial_values[1:]

            ode_system_solution = odeint(self.make_function_system, self.initial_values, time)
            self.time = self.time + 1

            with open(self.resultDirectoryPath + '/' + self.result_filename, "ab") as f:
                np.savetxt(f, np.column_stack([time[1:], ode_system_solution[1:]]), delimiter=",", fmt='%10.5f')

            self.ydata1 = np.concatenate((self.ydata1[10:], ode_system_solution[1:, 0]))
            self.ydata2 = np.concatenate((self.ydata2[10:], ode_system_solution[1:, 1]))
            self.ydata3 = np.concatenate((self.ydata3[10:], ode_system_solution[1:, 2]))
            self.ydata4 = np.concatenate((self.ydata4[10:], ode_system_solution[1:, 3]))
            self.ydata5 = np.concatenate((self.ydata5[10:], ode_system_solution[1:, 4]))

            self.canvas.axes.cla()
            self.canvas.axes.grid(True)
            self.canvas.axes.plot(self.xdata, self.ydata1, color='brown', label = "T_1(t)")
            self.canvas.axes.plot(self.xdata, self.ydata2, color='red', label = "T_2(t)")
            self.canvas.axes.plot(self.xdata, self.ydata3, color='orange', label = "T_3(t)")
            self.canvas.axes.plot(self.xdata, self.ydata4, color='green', label = "T_4(t)")
            self.canvas.axes.plot(self.xdata, self.ydata5, color='blue', label = "T_5(t)")
            self.canvas.axes.legend()
            self.canvas.draw()
    
    def areWeReadyToStart(self):
        if self.modelFilePath and self.coefficientsFilePath and self.initialValuesFilePath and self.resultDirectoryPath:
            self.manageButton.setEnabled(True)
            self.make_initial_plot()

app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec()
