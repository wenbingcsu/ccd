import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 18})
plt.rcParams['figure.figsize'] = [17, 5]

yearssample = ['2002', '2012', '2023']
columnscap = ['year','Beijing','Tianjin','Hebei','Shanxi','Neimenggu','Liaoning','Jilin','Heilongjiang','Shanghai','Jiangsu','Zhejiang','Anhui','Fujian','Jiangxi','Shandong','Henan','Hubei','Hunan','Guangdong','Guangxi','Hainan','Chongqing','Sichuan','Guizhou','Yunnan','Shaanxi','Gansu','Qinghai','Ningxia','Xinjiang']
years = ['2002','2003','2004','2005','2006', '2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020', '2021', '2022', '2023']


def plot(t, year):
    if(t==0):
        title = "Technological Innovation Subsystem"+ " ("+year+ ")"
        filename = "Xcei"
    else:
        title = "Financial Development Subsystem"+ " ("+year+ ")"
        filename = "Ycei"

    filename = filename + "-"+year+".pdf"

    plt.bar(df['region'], df[year])
    plt.xlabel("Regions")   
    plt.title(title)
    plt.ylabel("Comprehensive Evaluation Index")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

def plottemporoal(df, t, region):
    if(t==0):
        title = "Technological Innovation Subsystem"+ " ("+region+ ")"
        filename = "Xcei"
    else:
        title = "Financial Development Subsystem"+ " ("+region+ ")"
        filename = "Ycei"

    filename = filename + "-"+region+".pdf"

    plt.bar(df['region'], df[region])
    plt.xlabel("Years")   
    plt.title(title)
    plt.ylabel("Comprehensive Evaluation Index")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

df = pd.read_csv('Xcei.csv')
print("before transpose", df)
for year in yearssample:
    plot(0, year)

regionsample = ["Beijing", "Qinghai"] 

df = df.T
# print(df)
df = df.reset_index()
# print(df)
df.columns = df.iloc[0]
df = df[1:]

plt.rcParams['figure.figsize'] = [11, 5]
# 
for region in regionsample:
    plottemporoal(df, 0, region)

plt.rcParams['figure.figsize'] = [17, 5]    
df = pd.read_csv('Ycei.csv')
for year in yearssample:
    plot(1, year)

df = df.T
# print(df)
df = df.reset_index()
# print(df)
df.columns = df.iloc[0]
df = df[1:]

# 
plt.rcParams['figure.figsize'] = [11, 5]
for region in regionsample:
    plottemporoal(df, 1, region)

