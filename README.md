This project contains two sets of data and python code to compute and combine the coupling and coordination degrees between the technological innovation subsystem and the financial development subsystem for 30 regions of China between the years 2002-2023. One set contains the dataset using our proposed 24 predictors. The other set contains the dataset using 14 predictors compiled according to a previous study published in 2015. The goal of the study is demonstrate the importance of using a proper set of predictors that model accurately each subsystem. We propose an objective evaluation method using the regression performance and projection performance. A proper set of predictors would lead to small regression errors, and better projection accuracy. 

The dataset for our study with our set of predictors is named currentdata.xlsx. The dataset for the reference study is named referencedata.xlsx. The readme for each dataset is also provided, but in Chinese.

The python code in each folder (currentstudy, and referencestudy) is rather similar. Only analyze.py is slightly different to accommondate the difference file name and predictor names. 

Additional csv files are generated files for comprehensive evaluation index for each subsystem, the coupling degree, and coordination degree. 

analyze.py: perform pre-processing to fill the missing data using regression; compute the comprehensive evaluation index, the coupling degree (C), and the coordination degree (D).
plotFilldataerror.py: plot the errors for the regression model using available data
plotC.py: plot the coupling degree
plotD.py: plot the coordination degree
plotU.py: plot the comprehensive evaluation index
regressionD.py: train the regression model using data from 2002-2018, test the model using data from 2019-2023 for projection. Also plot the regression error R^2. 

To use our datasets and our code, please cite the following paper:
Zhou, J.; Jia, Y.; Yang, Y.; Zhao, W. Coordinated Evaluation of Technological Innovation and Financial Development in China: An Engineering Perspective. Appl. Syst. Innov. 2025, 8, 77. https://doi.org/10.3390/asi8030077
https://www.mdpi.com/2571-5577/8/3/77

