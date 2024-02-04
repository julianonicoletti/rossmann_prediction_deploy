import pickle
import os
import inflection
import pandas as pd
import numpy as np 
import math
import datetime


class Rossmann (object):
    def __init__(self):
        
        self.home_path =  ''
        self.parameter_path = os.path.join(self.home_path, 'parameter')

        self.competition_distance_scaler = pickle.load(open(os.path.join(self.parameter_path, 'competition_distance_scaler.pkl'), 'rb'))
        self.competition_time_month_scaler = pickle.load(open(os.path.join(self.parameter_path, 'competition_time_month_scaler.pkl'), 'rb'))
        self.promo_time_week_scaler = pickle.load(open(os.path.join(self.parameter_path, 'promo_time_week_scaler.pkl'), 'rb'))
        self.year_scaler = pickle.load(open(os.path.join(self.parameter_path, 'year_scaler.pkl'), 'rb'))
        self.store_type_scaler = pickle.load(open(os.path.join(self.parameter_path, 'store_type_scaler.pkl'), 'rb'))

            
    def data_cleaning(self, df1):
        #Rename Collummns
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
                    'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
                    'CompetitionDistance', 'CompetitionOpenSinceMonth',
                    'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
                    'Promo2SinceYear', 'PromoInterval']
        snakecase = lambda x: inflection.underscore(x)

        cols_new = list(map( snakecase, cols_old))

        df1.columns = cols_new

        df1['date'] = pd.to_datetime(df1['date'])

        #preencher NA
        # competition_distance              2642

        df1['competition_distance'] = (df1['competition_distance']
                                        .apply(lambda x: 200000 if math.isnan(x) else x))

            # competition_open_since_month    323348
        df1['competition_open_since_month'] = (df1.apply( lambda x: x['date'].month
                                                        if math.isnan(x['competition_open_since_month'])
                                                        else x['competition_open_since_month'], axis=1))
        # competition_open_since_year     323348

        df1['competition_open_since_year'] = (df1.apply( lambda x: x['date'].year
                                                        if math.isnan(x['competition_open_since_year'])
                                                        else x['competition_open_since_year'], axis=1))

        # promo2_since_week               508031
        df1['promo2_since_week'] = (df1.apply( lambda x: x['date'].week
                                                        if math.isnan(x['promo2_since_week'])
                                                        else x['promo2_since_week'], axis=1))

        # promo2_since_year               508031
        df1['promo2_since_year'] = (df1.apply( lambda x: x['date'].year
                                                if math.isnan(x['promo2_since_year'])
                                                else x['promo2_since_year'], axis=1))
        # promo_interval                  508031


        month_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May',
                        6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'} 

        df1['promo_interval'].fillna(0, inplace=True)

        df1['month_map'] = df1['date'].dt.month.map(month_map)

        df1['is_promo'] = (df1[['promo_interval', 'month_map']].apply(lambda x :0 if x['promo_interval'] == 0 
                else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1))


        #Change data Types
        #competition
        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

        #promo2
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        
        return df1
    
    
    def feature_engineering(self, df2):
        
        # year
        df2 ['year'] = df2['date'].dt.year

        # month
        df2 ['month'] = df2['date'].dt.month
        # day
        df2 ['day'] = df2['date'].dt.day
        # week of year
        df2 ['week_of_year'] = df2['date'].dt.isocalendar().week
        # year week
        df2 ['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # competition since
        df2['competition_since'] = pd.to_datetime(df2['competition_open_since_year']
                                                .astype(str) + '-' + df2['competition_open_since_month']
                                                .astype(str) + '-1', format='%Y-%m-%d')
        # Convertendo para o formato datetime
        df2['competition_since'] = pd.to_datetime(df2['competition_since'])
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since']).dt.days / 30).astype(int)

        #df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)


        # promo since
            # Criando a coluna 'promo_since' a partir de 'promo2_since_year' e 'promo2_since_week'
        df2['promo_since'] = df2['promo2_since_year'].astype(str) + '-' + df2['promo2_since_week'].astype(str) + '-1'

            # Convertendo a coluna 'promo_since' para datetime
        df2['promo_since'] = pd.to_datetime(df2['promo_since'], format='%Y-%W-%w')

            # Subtraindo 7 dias para ajustar a data para o início da semana anterior
        df2['promo_since'] = df2['promo_since'] - pd.to_timedelta(7, unit='d')
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since'])/7).apply(lambda x: x.days).astype(int)


        # assortment
        #alterando os valores de a, b e c para a notação basic, extra e extended
        df2['assortment'] = df2['assortment'].replace({'^a$': 'basic', '^b$': 'extra', '^c$': 'extended'}, regex=True)



        # state holiday
        df2['state_holiday'] = (df2['state_holiday']
                                .replace({'^a$': 'public_holiday', '^b$': 'easter_holiday', '^c$': 'christimas', '0':'regular_day'}, regex=True))
        
        #filtragem das linhas
        df2 = df2[(df2['open'] != 0)]
        
        #filtragem das colunas
        cols_drop = [ 'open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, errors='ignore', axis=1)
        
        return df2
    
    
    def data_preparation(self, df5):
        
        # df5['week_of_year'] = df5['week_of_year'].astype('float64')
        # var_numeric = df5.select_dtypes(include=['int32', 'int64', 'float64'])
        
        # competition distance
        df5['competition_distance'] = self.competition_distance_scaler.fit_transform(df5[['competition_distance']].values)

        #year
        df5['year'] = self.year_scaler.fit_transform(df5[['year']].values)

        # competition time month
        df5['competition_time_month'] = self.competition_time_month_scaler.fit_transform(df5[['competition_time_month']].values)
        #promo time week
        df5['promo_time_week'] = self.promo_time_week_scaler.fit_transform(df5[['promo_time_week']].values)

        #state holiday - one hot encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'], dtype=int)
        # store_type - labelEncoder
        df5['store_type'] = self.store_type_scaler.fit_transform(df5[['store_type']])

        #assortment
        assortment_dict = {'a': 1, 'b': 2, 'c': 3}
        df5['assortment'] = df5['assortment'].map(assortment_dict)
        
        # day of week
        df5['day_of_week_sin'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi/7))).round(4)
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi/7))).round(4)


        # month
        df5['month_sin'] = df5['month'].apply(lambda x: np.sin(x * (2. * np.pi/12))).round(4)
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2. * np.pi/12))).round(4)


        # day

        df5['day_sin'] = df5['day'].apply(lambda x: np.sin(x * (2. * np.pi/30))).round(4)
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2. * np.pi/30))).round(4)

        # week of year

        df5['week_of_year_sin'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi/52))).round(4)
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi/52))).round(4)
        
        #Variáveis relevantes
        cols_selected = ['store','promo', 'store_type', 'assortment', 'competition_distance',
                'competition_open_since_month', 'competition_open_since_year', 'promo2',
                'promo2_since_week', 'promo2_since_year', 'competition_time_month', 'promo_time_week',
                'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                'week_of_year_cos', 'week_of_year_sin',
                ]
        
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        #prediction
        pred = model.predict(test_data)
                        
        #join pred into the original data
        
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records', date_format='iso')