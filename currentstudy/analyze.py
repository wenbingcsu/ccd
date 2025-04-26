# import pandas lib as pd
import math
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import itertools
import fnmatch

# from objective_weighting.mcda_methods import VIKOR
from objective_weighting import weighting_methods as mcda_weights
from objective_weighting import normalizations as norms
# from objective_weighting.additions import rank_preferences

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


Xenum = ["X11","X12","X13","X14","X15","X21","X22","X23","X24","X31","X32","X33"]
Xncol = [18,18,18,16,14,18,17,17,17,18,17,16]
Yenum = ["Y11","Y12","Y13","Y14","Y15","Y21","Y22","Y23","Y24","Y31","Y32","Y33"]
Yncol = [18,16,16,16,16,16,18,16,16,16,16,16]
maxcol = 18

def replace_column_keys(keyrow):
    newkeys = []
    for col in keyrow:
        # print(f"Column: {col}, Type: {type(col)}")
        if isinstance(col, float) and col.is_integer():
            col = int(col)
        newcol = str(col)
        # print(f"Column pre: {newcol}, Type: {type(newcol)}")
        newkeys.append(newcol)

    # print(name, newkeys)
    # sheet.columns = newkeys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
fillmissingdatarecords = []

# Function to perform regression for each row
def fill_missing_data_rowwise(sheet, df):
    # newrow.append(sheet)
    df.columns = df.iloc[0]
    # df.reset_index(drop=True, inplace=True)
    df = df[1:].reset_index(drop=True)
    # Use the first row (years) as the independent variable (x)
    x = df['地区/年份'].values.reshape(-1, 1)
    
    for col in df.columns[1:]:  # Skip the 'Year' column
        print("pre-processing ", col)
        # newrow.append(col)
        for i in range(len(df)):
            if pd.isnull(df.at[i, col]):
                # Prepare the target values (y) excluding NaN values
                y = df[col].dropna().values
                # print("======y", y)
                t = df[col].notna()
                # print("t", t)
                x_valid = x[t]  # Match x to the non-missing values in y
                # print("x_valid", x_valid)
                
                # Ensure there are enough points to fit the model
                if len(y) > 1:
                    # Fit the regression model
                    # model = LinearRegression().fit(x_valid, y)
                    newrow = []

                    # we first try linearregression, then trey gradientboosting
                    model = LinearRegression()
                    model.fit(x_valid, y)
                    y_pred = model.predict(x_valid)
                    # Calculate the Mean Squared Error (MSE) for the training data
                    training_error = mean_squared_error(y, y_pred)
                    print("LR: fill missing data regression error:",training_error)
                    newrow.append(training_error)

                    # Create a pipeline with PolynomialFeatures and LinearRegression
                    pipeline = Pipeline([
                        ('poly', PolynomialFeatures()),
                        ('linear', LinearRegression())
                    ])

                    # Define the parameter grid
                    param_grid = {
                        'poly__degree': [1, 2, 3, 4, 5]  # Example range of degrees
                    }

                    # Perform grid search with cross-validation
                    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
                    grid_search.fit(x_valid, y)

                    # Get the best parameters
                    best_degree = grid_search.best_params_['poly__degree']
                    print(f'Best degree: {best_degree}')

                    # Evaluate the model
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(x_valid)
                    training_error = mean_squared_error(y, y_pred)

                    print(f'Test Error (MSE): {training_error}')
                    newrow.append(training_error)

                    model = GradientBoostingRegressor()
                    model.fit(x_valid, y)
                    # Predict the target values for the training data
                    y_pred = model.predict(x_valid)
                    # Calculate the Mean Squared Error (MSE) for the training data
                    training_error = mean_squared_error(y, y_pred)
                    print("GBR: fill missing data regression error:",training_error)
                    newrow.append(training_error)
                    
                    # Predict the missing value for the current row
                    row = x[i].reshape(1, -1)
                    print("@@@@@@@@@ col", col, "row", i, row)
                    df.at[i, col] = model.predict(row)[0]
                    fillmissingdatarecords.append(newrow)    
    
    return df.T


def getDataNew(pattern):
    excel_file = 'currentdata.xlsx'
    xls = pd.ExcelFile(excel_file)
    sheet_names = xls.sheet_names
    # print("sheet names:", sheet_names)
    
    # Filter sheet names using a wildcard-like pattern
    pattern = pattern+'*'
    filtered_sheets = fnmatch.filter(sheet_names, pattern)

    # Read the filtered sheets into a dictionary of DataFrames
    dfs = {sheet: pd.read_excel(excel_file, sheet_name=sheet, header=None) for sheet in filtered_sheets}
    # Now you can work with the DataFrames
    dataframes = []

    for sheet, df in dfs.items():
        print("sheet: ", sheet)
        # Display the DataFrame before dropping the first row
        # print("Before dropping the first row:")
        # print(df)

        # Drop the first row using the drop method
        df = df.drop(index=df.index[0])

        # Alternatively, drop the first row using slicing
        # df = df.iloc[1:]

        # Display the DataFrame after dropping the first row
        # print("After dropping the first row:")
        # print(df)

        # num_columns = 23 # df.shape[1]
        # df = df.iloc[:,1:num_columns]
        # df = df.iloc[1:,0:num_columns]

        # drop extra data beyond the main information
        # 30 regions + 1 columns caption
        df = df.drop(df.index[31:])

        # print(sheet, "df", df)
        # offset = 4
        # df = df.iloc[:,1:maxcol - offset]
        # print("###", sheet, offset, " dataframe shape: ", np.shape(df))
        # df = df.astype(float)
        df = fill_missing_data_rowwise(sheet, df.T)
        print("filled missing data: ", df)

        # we do not need the caption row anymore after filled the missing data
        df = df.iloc[1:]

        # df = df.drop(index=0)
        l = df.values.flatten().tolist()
        print(sheet, "column: ", len(l))
        dataframes.append(l)

    return dataframes

# Linear normalization
def linear_normalization(matrix, types):
    """
    Normalize decision matrix using linear normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = linear_normalization(matrix, types)
    """
    x_norm = np.zeros(np.shape(matrix))
    x_norm[:, types == 1] = matrix[:, types == 1] / (np.amax(matrix[:, types == 1], axis = 0))
    x_norm[:, types == -1] = np.amin(matrix[:, types == -1], axis = 0) / matrix[:, types == -1]
    return x_norm

# Vector normalization
def vector_normalization(matrix, types):
    """
    Normalize decision matrix using vector normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    -----------
    >>> nmatrix = vector_normalization(matrix, types)
    """
    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    x_norm[:, types == 1] = matrix[:, types == 1] / (np.sum(matrix[:, types == 1] ** 2, axis = 0))**(0.5)
    x_norm[:, types == -1] = 1 - (matrix[:, types == -1] / (np.sum(matrix[:, types == -1] ** 2, axis = 0))**(0.5))

    return x_norm

# Sum normalization
def sum_normalization(matrix, types):
    """
    Normalize decision matrix using sum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = sum_normalization(matrix, types)
    """
    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    x_norm[:, types == 1] = matrix[:, types == 1] / np.sum(matrix[:, types == 1], axis = 0)
    x_norm[:, types == -1] = (1 / matrix[:, types == -1]) / np.sum((1 / matrix[:, types == -1]), axis = 0)

    return x_norm

def minmax_normalization(matrix, types):
    """
    Normalize decision matrix using minimum-maximum normalization method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with m alternatives in rows and n criteria in columns
        types : ndarray
            Criteria types. Profit criteria are represented by 1 and cost by -1.

    Returns
    --------
        ndarray
            Normalized decision matrix

    Examples
    ----------
    >>> nmatrix = minmax_normalization(matrix, types)
    """
    x_norm = np.zeros((matrix.shape[0], matrix.shape[1]))
    x_norm[:, types == 1] = (matrix[:, types == 1] - np.amin(matrix[:, types == 1], axis = 0)
                             ) / (np.amax(matrix[:, types == 1], axis = 0) - np.amin(matrix[:, types == 1], axis = 0))

    x_norm[:, types == -1] = (np.amax(matrix[:, types == -1], axis = 0) - matrix[:, types == -1]
                           ) / (np.amax(matrix[:, types == -1], axis = 0) - np.amin(matrix[:, types == -1], axis = 0))

    return x_norm


# Entropy weighting
def entropy_weighting(matrix):
    """
    Calculate criteria weights using objective Entropy weighting method.

    Parameters
    -----------
        matrix : ndarray
            Decision matrix with performance values of m alternatives and n criteria.

    Returns
    --------
        ndarray
            Vector of criteria weights.

    Examples
    ----------
    >>> weights = entropy_weighting(matrix)
    """
    # normalize the decision matrix with `sum_normalization` method from `normalizations` as for profit criteria
    types = np.ones(np.shape(matrix)[1])
    pij = minmax_normalization(matrix, types)
    # pij = sum_normalization(matrix, types)
    # Transform negative values in decision matrix `matrix` to positive values
    pij = np.abs(pij)
    m, n = np.shape(pij)
    H = np.zeros((m, n))

    # Calculate entropy
    for j, i in itertools.product(range(n), range(m)):
        if pij[i, j]:
            H[i, j] = pij[i, j] * np.log(pij[i, j])

    h = np.sum(H, axis = 0) * (-1 * ((np.log(m)) ** (-1)))

    # Calculate degree of diversification
    d = 1 - h

    # Set w as the degree of importance of each criterion
    w = d / (np.sum(d))
    return w

def compute_comprehensive_evaluation_index(matrix, weights, t=1):  
    df = pd.DataFrame(matrix)
    print("in cei func:", np.shape(matrix))
    # Save the DataFrame to a CSV file
    filename = str(t)+'matrix.csv'
    df.to_csv(filename, index=False)
    print("Matrix saved to "+filename)  

    cei = []
    # we must first normalize the matrix
    types = np.ones(np.shape(matrix)[1])
    normalized_matrix = minmax_normalization(matrix, types)
    print("normalized matrix: ", normalized_matrix)
    # Convert the matrix to a DataFrame
    sdf = pd.DataFrame(normalized_matrix)
    filename = str(t)+'-normalized-matrix.csv'
    sdf.to_csv(filename, index=False)
    print("Matrix saved to "+filename)  

    # must use normalized matrix
    df = pd.DataFrame(normalized_matrix)
    # we want to compute the comprehensive evaluation index for each region, i.e., each row
    for index, row in df.iterrows():
        # print(index, row)
        s = 0
        for column, value in row.items():
            # print("in row: ", column, "value", value, "w", weights[column])
            s = s + value*weights[column]
        # print(index, s)
        cei.append(s)

    S_i = np.dot(normalized_matrix, weights)
    return (cei, S_i)

years = ['2002','2003','2004','2005','2006', '2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020', '2021', '2022', '2023']

captions = ['region',2002, 2003, 2004, 2005, 2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]
regions = ['Beijing','Tianjin','Hebei','Shanxi','Neimenggu','Liaoning','Jilin','Heilongjiang','Shanghai','Jiangsu','Zhejiang','Anhui','Fujian','Jiangxi','Shandong','Henan','Hubei','Hunan','Guangdong','Guangxi','Hainan','Chongqing','Sichuan','Guizhou','Yunnan','Shaanxi','Gansu','Qinghai','Ningxia','Xinjiang']
def save_to_csv(cei, filename):
    # Reshape the array
    array = np.array(cei)
    num_columns = len(captions) - 1
    num_rows = len(cei) // num_columns
    matrix = array.reshape(num_rows, num_columns)
    new_column = regions
    column_captions = captions
    new_matrix = np.column_stack((new_column,matrix))
    df = pd.DataFrame(new_matrix)
    df.columns = column_captions
    # df.iloc[0] = column_captions
    df.to_csv(filename+'.csv', index=False)
    print("Matrix saved to "+filename)  


def CCD(U1, U2):
    # U1 and U2 have the same length, each element corresponds to the same city and same year
    C = []
    D = []
    for u1, u2 in zip(U1, U2):
        c = 2*math.sqrt(u1*u2)/(u1+u2)
        t = 0.5*u1+0.5*u2
        d = math.sqrt(c*t)
        # this corresponds to each row of the samples (390 of rows)
        # print("CCD: ", c, d)
        C.append(c)
        D.append(d)

    return (C,D)


# newrow = ["LR", "Poly", "GBR"]
#   fillmissingdatarecords.append(newrow)
dataframes = getDataNew("X")
# print("list0", dataframes[0])
matrix = np.column_stack(dataframes)
print("X matrix shape: ", np.shape(matrix))
# print(matrix)

# test to see if the weights are the same if we use the standardized matrix
# types = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
types = np.ones(np.shape(matrix)[1])
normalizedmatrix = sum_normalization(matrix,types)
print(normalizedmatrix[0:10])

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

def test(matrix):
    df = pd.DataFrame(matrix)
    # we want to compute the comprehensive evaluation index for each region, i.e., each row
    for index, col in df.items():
        # print(index, col.array.reshape(-1, 1))
        # Fit and transform the data
        normalized_col = scaler.fit_transform(col.array.reshape(-1, 1))
        print(index, normalized_col[:10])

weights = entropy_weighting(matrix)
print("X weights: ", weights, sum(weights))

weights = entropy_weighting(normalizedmatrix)
print("X weights again: ", weights, sum(weights))

plt.bar(Xenum, weights)
plt.xlabel("Predictors")
plt.ylabel("Weights")
plt.savefig("Xweights.pdf", bbox_inches='tight')

cei, cei2 = compute_comprehensive_evaluation_index(matrix, weights, 1)
# print("cei: ", cei)
# print("cei2", cei2)
save_to_csv(cei2, "Xcei")
U1 = cei2

dataframes = getDataNew("Y")
# print("list0", dataframes[0])
matrix = np.column_stack(dataframes)
print("Y matrix shape: ", np.shape(matrix))
# print(matrix)
weights = entropy_weighting(matrix)
print("Y weights: ", weights)
plt.close()
plt.bar(Yenum, weights)
plt.xlabel("Predictors")
plt.ylabel("Weights")
plt.savefig("Yweights.pdf", bbox_inches='tight')

cei, cei2 = compute_comprehensive_evaluation_index(matrix, weights, 2)
save_to_csv(cei2, "Ycei")
U2 = cei2


C, D = CCD(U1, U2)
save_to_csv(C, "C")
save_to_csv(D, "D")

df = pd.DataFrame(fillmissingdatarecords)
columns = ["LR", "Poly", "GBR"]
df.columns = columns
df.to_csv('fillerror.csv', index=False)





