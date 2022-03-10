from data_loader import *
path = "project_data/"
X_train_valid, X_test, y_train_valid, y_test = data_loader(path,True)
total_x_train,total_y_train,total_x_test,total_y_test = data_process(X_train_valid, 
        X_test, y_train_valid, y_test, 0.5, 2, 2, True, True)