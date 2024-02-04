import os
import pandas as pd
import pickle
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

#loadin model 
model = pickle.load(open(os.path.abspath(r'model/model_rossmann.pkl'), 'rb'))

app = Flask(__name__)

#iniatilize API
@app.route('/rossmann/predict', methods=['Post'])

def rossmann_predict():
    test_json = request.get_json()
    if test_json: # there is data
        if isinstance( test_json, dict ): # unique example
            est_raw = pd.DataFrame( test_json, index=[0] )
        else: # multiple example
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            # Instantiate Rossmann class
            pipeline = Rossmann()
            # data cleaning
            df1 = pipeline.data_cleaning( test_raw )
            # feature engineering
            df2 = pipeline.feature_engineering( df1 )
            # data preparation
            df3 = pipeline.data_preparation( df2 )
            # prediction
            df_response = pipeline.get_prediction( model, test_raw, df3 )
            return df_response
    else:
        return Response( '{}', status=200, mimetype='application/json' )
    
if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run( '0.0.0.0', port=port )