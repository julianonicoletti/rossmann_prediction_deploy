import pandas as pd
import json
import requests
import os
from flask import Flask, request, Response

#constants
TOKEN = '6713345458:AAGlvOV4DRuyxg21mai2z5HZUGykC1w8Hww'

# # Info about the bot
# https://api.telegram.org/bot6713345458:AAGlvOV4DRuyxg21mai2z5HZUGykC1w8Hww/getMe

# # Get Updates
# https://api.telegram.org/bot6713345458:AAGlvOV4DRuyxg21mai2z5HZUGykC1w8Hww/getUpdates

# # Webhook
# https://api.telegram.org/bot6713345458:AAGlvOV4DRuyxg21mai2z5HZUGykC1w8Hww/setWebhook?url=https://https://b14f14c66feb16.lhr.life/

# # send messages
# https://api.telegram.org/bot6713345458:AAGlvOV4DRuyxg21mai2z5HZUGykC1w8Hww/sendMessage?chat_id=1399408599&text=hi, juliano, iam good.


def send_message (chat_id, text):
    url = 'https://api.telegram.org/bot{}/sendMessage'.format(TOKEN)
    
    payload = {'chat_id': chat_id, 'text': text}
    
    r = requests.post(url, json=payload)
    print('Status Code {}'.format(r.status_code))
    
    return None
    
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
    
    if not df_test.empty:
        #remove closed days
        df_test = df_test[df_test['Open'] !=0]
        df_test = df_test[~df_test['Open'].isnull()]
        df_test = df_test.drop('Id', axis=1)

        #convert Dataframe to json
        data = json.dumps(df_test.to_dict(orient='records'))
    else:
        data = 'error'
    
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


def parse_message (message):
    
    chat_id = message['message']['chat']['id']
    store_id = message['message']['text']
    store_id = store_id.replace('/', '')
    
    try:
        store_id = int(store_id)
    
    except ValueError:
        send_message(chat_id, 'Store ID is Wrong!')
        store_id = 'error'
    
    return chat_id, store_id

#API Iniatilizer
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':
        message = request.get_json()
        chat_id, store_id = parse_message(message)
        
        if store_id !='error':
            #loading data
            data = load_dataset(store_id)
            
            if data != 'error':
                
                #prediction
                d1 = predict(data)
                #calculation
                d2 = d1[['store', 'prediction']].groupby( 'store' ).sum().reset_index()
                msg = ( 'Store Number {} will sell R${:,.2f} in the next 6 weeks'.format(
                    d2.loc['store'].values[0],
                    d2.loc['prediction'].values[0] ) )
                
                #send message
                send_message(chat_id, msg)
                return Response('Ok', status=200)
            else:
                send_message(chat_id, 'Store not Available')
                return Response('Ok', status=200)
        else:        
            send_message(chat_id, 'Store ID is Wrong!')
            return Response('Ok', status=200)
    
    else:
        return '<h1> Rossmann Telegram BOT'

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)







