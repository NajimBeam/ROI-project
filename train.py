
import json
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf


json_file = open("configs.json","r").read()
CF = json.loads(json_file)


def get_data_stats(df):

    df.info()

    df['3D ROAS'].mean()

    df[['3D ROAS','30D ROAS']].corr()

    df[['3D ROAS','30D ROAS']][df['Spend in USD']>100].corr()

    plt.style.use('bmh')
    df.hist(column = 'Spend in USD', bins = np.arange(0,16), color='darkblue')
    df.corr()['30D ROAS']


def load_data():

    df_train = pd.read_csv(CF['TRAIN'], encoding='ISO-8859-1', sep=";")

    get_data_stats(df_train)

    df_test = pd.read_csv(CF['TEST'], encoding='ISO-8859-1', sep=";")

    x = df_train[['3D ROAS','7D ROAS']]
    y = df_train['30D ROAS']

    return x,y,df_train,df_test


def train_model(x,y):

    model = sm.OLS(y,x).fit()

    return model


def get_prediction(model,x):

    return model.predict(x)


def main():

    x,y,df_train,df_test = load_data()

    model = train_model(x,y)
    predictions = get_prediction(model,x)

    z = df_test[['3D ROAS', '7D ROAS']]
    predictions2 = model.predict(z)


if __name__ == "__main__":
    main()



