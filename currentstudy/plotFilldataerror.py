import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

df = pd.read_csv('fillerror.csv')

fig,axs = plt.subplots(1,2, figsize=(12,6))
# ax=axs[0],

def lrplot():
    # Create the histogram
    df['Poly'].plot(kind='hist', ax=axs[0], bins=100, edgecolor='black', title='Linear Regression with Polynomical Features', log=True)
    # plt.xscale("log")
    # Customize the plot
    # plt.title('Linear Regression MSE')
    axs[0].set_xlabel('MSE Value')
    plt.ylabel('Frequency')
    # plt.tight_layout()
    # plt.show()

def gbrplot():
    # ax=axs[1],
    # Create the histogram
    df['GBR'].plot(kind='hist', ax=axs[1], bins=100, edgecolor='black',  title="Gradient Boosting Regression", log=True)
    # plt.xscale("log")
    # Customize the plot
    # plt.title('GradientBoostingRegression MSE')
    # plt.xlabel('Value')
    axs[1].set_xlabel('MSE Value')
    plt.ylabel('Frequency')
    # Get the current axis
    ax = plt.gca()

    # Set x-axis to scientific notation
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(True)
    ax.xaxis.get_major_formatter().set_powerlimits((-1, 1))  # Defines when to switch to scientific notation
    # Show the plot
    
lrplot()
gbrplot()
plt.tight_layout()
# plt.show()
plt.savefig("fillerror.pdf", bbox_inches='tight')
