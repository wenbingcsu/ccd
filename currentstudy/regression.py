import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score
import csv

Xenum = ["X11","X12","X13","X14","X15","X21","X22","X23","X24","X31","X32","X33"]
Yenum = ["Y11","Y12","Y13","Y14","Y15","Y21","Y22","Y23","Y24","Y31","Y32","Y33"]
regions = ['Beijing','Tianjin','Hebei','Shanxi','Neimenggu','Liaoning','Jilin','Heilongjiang','Shanghai','Jiangsu','Zhejiang','Anhui','Fujian','Jiangxi','Shandong','Henan','Hubei','Hunan','Guangdong','Guangxi','Hainan','Chongqing','Sichuan','Guizhou','Yunnan','Shaanxi','Gansu','Qinghai','Ningxia','Xinjiang']
years = ['2002','2003','2004','2005','2006', '2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020', '2021', '2022', '2023']


def plot_regression_result(region, run, X, y, y_pred, tag=""):
    plt.close()
    # Make predictions
    # X_plot = np.linspace(0, 13, 100).reshape(-1, 1)

    # Replace X labels
    custom_labels = years
    # for n in X:
    #    print("year: ", n[0])
    #    custom_labels.append(years[n[0]])

    plt.xticks(ticks=np.arange(len(custom_labels)), labels=custom_labels)

    # Visualize the results
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, y_pred, color='red', label='Fitted curve')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Years')
    plt.ylabel('$R^2$')
    title = region + "-"+str(run)
    if(run == -1):
        title = region+"-all"
    plt.legend(['Actual', 'Predicted'])
    plt.title(title)
    filename = tag+title+".pdf"
    # plt.show()
    plt.savefig(filename, bbox_inches='tight')

def regression_for_region(region, X, y):
    pipeline = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
    param_grid = {
        'polynomialfeatures__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }

    # Outer cross-validation for model evaluation
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    outer_scores = []
    predicted_values = []
    polydegrees = []

    run = 0
    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Inner cross-validation for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
        grid_search.fit(X_train, y_train)

        # Evaluate the best model on the outer test set
        best_model = grid_search.best_estimator_
        # print("+++++ best model parameters:", best_model)
        # Access the polynomial degree
        polynomial_degree = best_model.named_steps['polynomialfeatures'].degree
        print(f"Best polynomial degree: {polynomial_degree}")
        polydegrees.append(polynomial_degree)
        y_pred = best_model.predict(X_test)
        outer_scores.append(r2_score(y_test, y_pred))
        predicted_values.extend(y_pred)
        # print("++++", X_test, y_test, y_pred)
        plot_regression_result(region, run, X_test,y_test,y_pred, "poly-")
        run = run + 1

    # we want to show the regression model works on the entire dataset, for illustration purpose to be included in the paper
    # because for each fold, the parameter might not be the same, we simply use the last fold parameter

    y_pred = best_model.predict(X)
    plot_regression_result(region, -1, X,y,y_pred)

    print(f"Outer cross-validation R^2 scores: {outer_scores}")
    print(f"Mean R^2 score: {np.mean(outer_scores)}")
    print(f"Predicted values for each fold: {predicted_values}")
    return (polydegrees, outer_scores, np.mean(outer_scores))

def regression_for_region_gbm(region, X, y):
    pipeline = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    # Define the parameter grid
    # Define the parameter grid with correct prefixes
    param_grid = {
        'gradientboostingregressor__n_estimators': [50, 100, 200],
        'gradientboostingregressor__learning_rate': [0.01, 0.1, 0.2],
        'gradientboostingregressor__max_depth': [3, 4, 5],
        'gradientboostingregressor__min_samples_split': [2, 5, 10],
        'gradientboostingregressor__min_samples_leaf': [1, 2, 4]
    }

    # Outer cross-validation for model evaluation
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    outer_scores = []
    predicted_values = []

    run = 0
    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Inner cross-validation for hyperparameter tuning
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
        grid_search.fit(X_train, y_train)

        # Evaluate the best model on the outer test set
        best_model = grid_search.best_estimator_
        # print("+++++ best model parameters:", best_model)
        # Access the polynomial degree

        # Get the best parameters
        best_params = grid_search.best_params_
        print("Best parameters found: ", best_params)
        
        y_pred = best_model.predict(X_test)
        outer_scores.append(r2_score(y_test, y_pred))
        predicted_values.extend(y_pred)
        # print("++++", X_test, y_test, y_pred)
        plot_regression_result(region, run, X_test,y_test,y_pred, "gbm-")
        run = run + 1

    # we want to show the regression model works on the entire dataset, for illustration purpose to be included in the paper
    # because for each fold, the parameter might not be the same, we simply use the last fold parameter

    y_pred = best_model.predict(X)
    plot_regression_result(region, -1, X,y,y_pred, "gbm-")

    print(f"Outer cross-validation R^2 scores: {outer_scores}")
    print(f"Mean R^2 score: {np.mean(outer_scores)}")
    print(f"Predicted values for each fold: {predicted_values}")
    return (outer_scores, np.mean(outer_scores))

def regression_for_region2(region, X, y):
    pipeline = make_pipeline(PolynomialFeatures(degree=5), LinearRegression())
    param_grid = {
        'polynomialfeatures__degree': [1,2,3]
    }

    # We want to train the model using the oldest 2/3 of the data
    # and the latest 1/3 data for testing to see the MSE on regression accuracy
    # because this is more intuitive
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    outer_scores = []
    predicted_values = []
    polydegrees = []

    splitpoint = 17
    X_train = X[0:splitpoint]
    X_test = X[splitpoint:]
    y_train = y[0:splitpoint]
    y_test = y[splitpoint:]

    run = 0

    # Inner cross-validation for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Evaluate the best model on the outer test set
    best_model = grid_search.best_estimator_
        # print("+++++ best model parameters:", best_model)
        # Access the polynomial degree
    polynomial_degree = best_model.named_steps['polynomialfeatures'].degree
    print(f"Best polynomial degree: {polynomial_degree}")
    polydegrees.append(polynomial_degree)

    y_train_pred = best_model.predict(X_train)
    r2train = r2_score(y_train, y_train_pred)
    print("training R2: ", r2train)
    outer_scores.append(r2train)

    y_pred = best_model.predict(X_test)
    outer_scores.append(r2_score(y_test, y_pred))
    predicted_values.extend(y_pred)
    print("++++", X_test, y_test, y_pred)
    plot_regression_result(region, run, X_test,y_test,y_pred, "poly-")
    run = run + 1

    # we want to show the regression model works on the entire dataset, for illustration purpose to be included in the paper
    # because for each fold, the parameter might not be the same, we simply use the last fold parameter

    y_pred = best_model.predict(X)
    plot_regression_result(region, -1, X,y,y_pred)

    print(f"Outer cross-validation R^2 scores: {outer_scores}")
    print(f"Mean R^2 score: {np.mean(outer_scores)}")
    print(f"Predicted values for each fold: {predicted_values}")
    return (polydegrees, outer_scores, np.mean(outer_scores))

def regression_for_region2gbm(region, X, y):
    pipeline = make_pipeline(StandardScaler(), GradientBoostingRegressor())
    # Define the parameter grid
    # Define the parameter grid with correct prefixes
    param_grid = {
        'gradientboostingregressor__n_estimators': [50, 100, 200],
        'gradientboostingregressor__learning_rate': [0.01, 0.1, 0.2],
        'gradientboostingregressor__max_depth': [3, 4, 5],
        'gradientboostingregressor__min_samples_split': [2, 5, 10],
        'gradientboostingregressor__min_samples_leaf': [1, 2, 4]
    }


    # We want to train the model using the oldest 2/3 of the data
    # and the latest 1/3 data for testing to see the MSE on regression accuracy
    # because this is more intuitive
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    outer_scores = []
    predicted_values = []
    polydegrees = []

    splitpoint = 16
    X_train = X[0:splitpoint]
    X_test = X[splitpoint:]
    y_train = y[0:splitpoint]
    y_test = y[splitpoint:]

    run = 0

    # Inner cross-validation for hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='r2')
    grid_search.fit(X_train, y_train)

    # Evaluate the best model on the outer test set
    best_model = grid_search.best_estimator_
        # print("+++++ best model parameters:", best_model)
    y_pred = best_model.predict(X_test)
    outer_scores.append(r2_score(y_test, y_pred))
    predicted_values.extend(y_pred)
    print("++++", X_test, y_test, y_pred)
    plot_regression_result(region, run, X_test,y_test,y_pred, "poly-")
    run = run + 1

    # we want to show the regression model works on the entire dataset, for illustration purpose to be included in the paper
    # because for each fold, the parameter might not be the same, we simply use the last fold parameter

    y_pred = best_model.predict(X)
    plot_regression_result(region, -1, X,y,y_pred)

    print(f"Outer cross-validation R^2 scores: {outer_scores}")
    print(f"Mean R^2 score: {np.mean(outer_scores)}")
    print(f"Predicted values for each fold: {predicted_values}")
    return (polydegrees, outer_scores, np.mean(outer_scores))

def regression_for_D(D):
    matrix = D.to_numpy()
    num_rows = matrix.shape[0]
    num_columns = matrix.shape[1]

    ll = []
    for i in range(0,num_columns):
        ll.append(i)
    X = np.array(ll).reshape(-1, 1)

    results = []
    meanR2s = []
    captionrow = ["Region", "mR2", "R2-1", "R2-2", "R2-3", "d1", "d2", "d3"]
    results.append(captionrow)
    # we want to compute for each region. there are 30 of them.
    for i in range(num_rows):
        y = np.array(matrix[i,])
        print("========= i =========",i, X, y)
        region = regions[i]
        polydegrees, outer_scores, meanscore = regression_for_region2(region,X,y)
        meanR2s.append(meanscore)
        row = []
        row.append(region)
        row.append(meanscore)
        for s in outer_scores:
            row.append(s)

        for d in polydegrees:
            row.append(d)
            
        results.append(row)

    df = pd.DataFrame(results)
    # Reset the index
    # df.reset_index(drop=True, inplace=True)
    # df = df.drop(df.index[0])
    # df.iloc[0] = captionrow
    filename = 'regression-results.csv'
    df.to_csv(filename, index=False, header=False)
    print("Matrix saved to "+filename)  

    plt.close()
    plt.rcParams['figure.figsize'] = [17, 7]
    plt.rcParams.update({'font.size': 18})
    plt.bar(regions, meanR2s)
    # Rotate the x-axis labels
    plt.xticks(rotation=45, ha='right')
    # plt.xlabel("Regions")
    plt.ylabel("Mean $R^2$")
    plt.savefig("RegressionResult.pdf", bbox_inches='tight')

#    filename = "regperf.csv"
#    with open(filename, mode='w', newline='') as file:
#        writer = csv.writer(file)
#        writer.writerows(results)
#        print("Matrix saved to "+filename)  


def regression_for_D_gbm(D):
    matrix = D.to_numpy()
    num_rows = matrix.shape[0]
    num_columns = matrix.shape[1]

    ll = []
    for i in range(0,num_columns):
        ll.append(i)
    X = np.array(ll).reshape(-1, 1)

    results = []
    meanR2s = []
    captionrow = ["Region", "mR2", "R2-1", "R2-2", "R2-3"]
    results.append(captionrow)
    # we want to compute for each region. there are 30 of them.
    for i in range(num_rows):
        y = np.array(matrix[i,])
        print("========= i =========",i, X, y)
        region = regions[i]
        outer_scores, meanscore = regression_for_region_gbm(region,X,y)
        meanR2s.append(meanscore)
        row = []
        row.append(region)
        row.append(meanscore)
        for s in outer_scores:
            row.append(s)
            
        results.append(row)

    df = pd.DataFrame(results)
    # Reset the index
    # df.reset_index(drop=True, inplace=True)
    # df = df.drop(df.index[0])
    # df.iloc[0] = captionrow
    filename = 'gbm-regression-results.csv'
    df.to_csv(filename, index=False, header=False)
    print("Matrix saved to "+filename)  

    plt.close()
    plt.rcParams['figure.figsize'] = [17, 7]
    plt.rcParams.update({'font.size': 18})
    plt.bar(regions, meanR2s)
    # Rotate the x-axis labels
    plt.xticks(rotation=45, ha='right')
    # plt.xlabel("Regions")
    plt.ylabel("Mean $R^2$")
    plt.savefig("gbm-RegressionResult.pdf", bbox_inches='tight')

    # filename = "regperf.csv"
    # with open(filename, mode='w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerows(results)
    #    print("Matrix saved to "+filename)  

D = pd.read_csv("D.csv")
# remove first column
D.drop(columns=D.columns[0], inplace=True)
regression_for_D(D)
# regression_for_D_gbm(D)