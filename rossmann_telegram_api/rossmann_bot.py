import pandas as pd
import json
import requests
import os


def load_dataset( store_id):
    #loading test dataset
    df10 = pd.read_csv(os.path.abspath(r'Ds_em_producao/dataset/test.csv'))

    df_store_raw = pd.read_csv(os.path.abspath(r'Ds_em_producao/dataset/store.csv'))

    # df10 = pd.read_csv('Ds_em_producao\dataset\test.csv', low_memory=False)
    # df_store_raw = pd.read_csv('Ds_em_producao\dataset\store.csv')

    #merge test dataset + store
    df_test = pd.merge(df10, df_store_raw, how='left', on='Store' )

    #choose store for prediction

    df_test = df_test[df_test['Store'] == store_id]

    #remove closed days
    df_test = df_test[df_test['Open'] !=0]
    df_test = df_test[~df_test['Open'].isnull()]
    df_test = df_test.drop('Id', axis=1)

    #convert Dataframe to json
    data = json.dumps(df_test.to_dict(orient='records'))
    
    return data

def predict(data):
    # API Call
    url = 'https://rossmann-sales-prediction-gck6.onrender.com/rossmann/predict'
    header = {'Content-type': 'application/json' }
    data = data

    r = requests.post( url, data=data, headers=header )
    print( 'Status Code {}'.format( r.status_code ) )

    d1 = pd.DataFrame( r.json(), columns=r.json()[0].keys() )
    return d1

d2 = d1[['store', 'prediction']].groupby( 'store' ).sum().reset_index()
for i in range( len( d2 ) ):
    print( 'Store Number {} will sell R${:,.2f} in the next 6 weeks'.format(
    d2.loc[i, 'store'],
    d2.loc[i, 'prediction'] ) )
