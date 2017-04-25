import numpy as np
import re
import pandas as pd
from scipy import sparse
import xgboost as xgb
from sklearn import preprocessing
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bs4 import BeautifulSoup
#input data


features_use = [
    'latitude', 'longitude', 'num_rho', 'num_phi', 'density',
    'bathrooms', 'bedrooms', 'rooms', 
    'price', 'price_per_bedroom', 'price_per_room', #'price_per_bathroom', 
    'log_price',  #'log_price_per_bedroom', 'log_price_per_bathroom', 'log_price_per_room',
    #'sqrt_price', #'sqrt_price_per_bedroom', 'sqrt_price_per_bathroom', 'sqrt_price_per_room',
    'photos_num', 'features_num', #'description_num',
    'created_month', 'created_hour', 'created_dayofyear', 'created_month_early_late', #'created_day', 
    'img_date_month', 'img_date_hour', 'img_date_dayofyear', 'img_date_month_early_late', #'img_date_day', 
]

features_ratios = [
    'building_id', 'manager_id', 'display_address', 
    #'pos',
    #'photos_num', 'features_num', 'description_num',
    #'created_month', 'created_hour', 'created_dayofyear', 
    #'img_date_month', 'img_date_hour', 'img_date_dayofyear', 
]

features_lables = [
    'building_id', 'manager_id', 'display_address', 
    'listing_id',
]

level_map = {
    'high': 0,
    'medium': 1,
    'low': 2,
}

interest_level_ratios = {
    'high': 0.077353663787644689,
    'medium': 0.22674197715356753,
    'low': 0.69590435905878778,
}

angles = [15,30,45,60]


def add_features(train_df, test_df):
    price = pd.concat([train_df['price'], test_df['price']])
    price_999 = np.percentile(price, 99.9)
    for df in [train_df, test_df]:
        df['rooms'] = df['bedrooms'] + df['bathrooms']

        df['price'] = df['price'].clip(upper=price_999)
        df['log_price'] = np.log(df['price'])
        df['sqrt_price'] = np.sqrt(df['price'])
        df['price_per_bedroom'] = df['price'] / df['bedrooms']
        df['price_per_bathroom'] = df['price'] / df['bathrooms']
        df['price_per_room'] = df['price'] / df['rooms']
        df['log_price_per_bedroom'] = np.log(df['price_per_bedroom'])
        df['log_price_per_bathroom'] = np.log(df['price_per_bathroom'])
        df['log_price_per_room'] = np.log(df['price_per_room'])
        df['sqrt_price_per_bedroom'] = np.sqrt(df['price_per_bedroom'])
        df['sqrt_price_per_bathroom'] = np.sqrt(df['price_per_bathroom'])
        df['sqrt_price_per_room'] = np.sqrt(df['price_per_room'])
    

    
    for df in [train_df, test_df]:
        df['photos_num'] = df['photos'].apply(len)
        df['features_words'] = df['features'].apply(lambda x: ' '.join(y.lower().replace(' ', '_') for y in x))
        df['features_num'] = df['features'].apply(len)

        regex = re.compile(r'[^a-zA-Z\s]')
        df['description_words'] = df['description'].apply(lambda x: str.lower(regex.sub('', x)) if len(x) else '')
        df['description_num'] = df['description_words'].apply(lambda x: len(x.split()))

        dt = pd.to_datetime(df['created']).dt
        df['created_month'] = dt.month
        df['created_day'] = dt.day
        df['created_dayofweek'] = dt.dayofweek
        df['created_dayofyear'] = dt.dayofyear
        df['created_hour'] = dt.hour
        df['created_month_early_late'] = dt.day.apply(lambda x: min(x // 10, 2))

        regex = re.compile(r'[\W+]')
        df['display_address'] = df['display_address'].apply(lambda x: str.lower(regex.sub('',x)))

        df['pos'] = df['longitude'].round(3).astype(str) + '_' + df['latitude'].round(3).astype(str)

    '''
    photos_num = pd.concat([train_df['photos_num'], test_df['photos_num']])
    photos_num_999 = np.percentile(photos_num, 99.9)
    features_num = pd.concat([train_df['features_num'], test_df['features_num']])
    features_num_999 = np.percentile(features_num, 99.9)
    description_num = pd.concat([train_df['description_num'], test_df['description_num']])
    description_num_999 = np.percentile(description_num, 99.9)
    '''
    pos = pd.concat([train_df['pos'], test_df['pos']])
    density_dict = pos.value_counts().to_dict()

    for df in [train_df, test_df]:
        '''
        df['photos_num'] = df['photos_num'].clip(upper=photos_num_999)
        df['features_num'] = df['features_num'].clip(upper=features_num_999)
        df['description_num'] = df['description_num'].clip(upper=description_num_999)
        '''
        df['density'] = df['pos'].apply(lambda x: density_dict.get(x, 1))

    for feature in features_ratios:
        features_use.append(feature + '_high_ratios')
        features_use.append(feature + '_medium_ratios')
        features_use.append(feature + '_low_ratios')
        for df in [train_df, test_df]:
            df[feature + '_high_ratios'] = interest_level_ratios['high']
            df[feature + '_medium_ratios'] = interest_level_ratios['medium']
            df[feature + '_low_ratios'] = interest_level_ratios['low']

    for feature in features_lables:
        features_use.append(feature + '_label')
        for df in [train_df, test_df]:
            df[feature + '_label'] = 0

    for angle in angles:
        features_use.append('num_rot' + str(angle) + '_x')
        features_use.append('num_rot' + str(angle) + '_y')

def add_id_features(train_df, test_df, features):
    for feature in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[feature].values) + list(test_df[feature].values))
        train_df[feature + '_label'] = lbl.transform(list(train_df[feature].values))
        test_df[feature+'_label'] = lbl.transform(list(test_df[feature].values))


def add_image_date(train_df, test_df):
    image_date = pd.read_csv('../input/listing_image_time.csv')
    image_date.columns = ['listing_id', 'time_stamp']
    image_date.loc[80240,'time_stamp'] = 1478129766 
    image_date['img_date'] = pd.to_datetime(image_date['time_stamp'], unit='s')
    dt = image_date['img_date'].dt
    image_date['img_date_month'] = dt.month
    image_date['img_date_day'] = dt.day
    image_date['img_date_week'] = dt.week
    image_date['img_date_dayofweek'] = dt.dayofweek
    image_date['img_date_dayofyear'] = dt.dayofyear
    image_date['img_date_month_early_late'] = dt.day.apply(lambda x: min(x // 10, 2))
    image_date['img_date_hour'] = dt.hour
    return pd.merge(train_df, image_date, on='listing_id', how='left'), pd.merge(test_df, image_date, on='listing_id', how='left')

def add_rotation(df, angle):
    rotation_x = lambda row, alpha: row['latitude'] * np.cos(alpha) + row['longitude'] * np.sin(alpha)
    rotation_y = lambda row, alpha: row['latitude'] * np.cos(alpha) - row['longitude'] * np.sin(alpha)
    df['num_rot' + str(angle) + '_x'] = df.apply(lambda row: rotation_x(row, np.pi/(180/angle)), axis=1)
    df['num_rot' + str(angle) + '_y'] = df.apply(lambda row: rotation_y(row, np.pi/(180/angle)), axis=1)
    return df

def add_polar(train_df, test_df):
    cart2rho = lambda x, y: np.sqrt(x * x + y * y)
    cart2phi = lambda x, y: np.arctan2(y, x)
    for df in [train_df, test_df]:
        df["num_rho"] = df.apply(lambda x: cart2rho(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        df["num_phi"] = df.apply(lambda x: cart2phi(x["latitude"] - 40.78222222, x["longitude"]+73.96527777), axis=1)
        for angle in angles:
            df = add_rotation(df, angle)
    return train_df, test_df


def get_feature_ratios_dict(df, feature, d=3):
    h = df.groupby([feature])['high'].sum()
    m = df.groupby([feature])['medium'].sum()
    l = df.groupby([feature])['low'].sum()
    s = h + m + l
    h = (h + interest_level_ratios['high'] * d) / (s + d)
    m = (m + interest_level_ratios['medium'] * d) / (s + d)
    l = (l + interest_level_ratios['low'] * d) / (s + d)
    feature_dict = {
        'high': h.to_dict(),
        'medium': m.to_dict(),
        'low': l.to_dict(),
    }
    return feature_dict

def set_feature_ratios(train_df, test_df, feature):
    feature_dict = get_feature_ratios_dict(train_df, feature)
    test_df.loc[:, [feature + '_high_ratios']] = test_df[feature].apply(lambda x: feature_dict['high'].get(x, interest_level_ratios['high']))
    test_df.loc[:, [feature + '_medium_ratios']] = test_df[feature].apply(lambda x: feature_dict['medium'].get(x, interest_level_ratios['medium']))
    test_df.loc[:, [feature + '_low_ratios']] = test_df[feature].apply(lambda x: feature_dict['low'].get(x, interest_level_ratios['low']))
    return test_df

def set_features_ratios(train_df, test_df, features):
    for feature in features:
        test_df = set_feature_ratios(train_df, test_df, feature)
    return test_df

def get_train_y(train_df):
    return train_df['interest_level'].apply(lambda x: level_map[x])

def run_xgb(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=321, num_rounds=2000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.02
    param['max_depth'] = 6
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = 'mlogloss'
    param['min_child_weight'] = 1
    param['subsample'] = 0.7
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model

train_df=pd.read_json('../input/train.json')
test_df=pd.read_json('../input/test.json')
test_df['bathrooms'].iloc[19671] = 1.5
test_df['bathrooms'].iloc[22977] = 2.0
test_df['bathrooms'].iloc[63719] = 2.0

interest_level = pd.get_dummies(train_df['interest_level'])
train_df = pd.concat([train_df, interest_level], axis=1)
interest_level_ratios = {
    'high': train_df['high'].sum() / len(train_df),
    'medium': train_df['medium'].sum() / len(train_df),
    'low': train_df['low'].sum() / len(train_df),
}

add_features(train_df, test_df)
add_id_features(train_df, test_df, features_lables)
train_df, test_df = add_image_date(train_df, test_df)
train_df, test_df = add_polar(train_df, test_df)
y = get_train_y(train_df)

features_words = pd.concat([train_df['features_words'], test_df['features_words']])
tfidf_features_words = CountVectorizer(stop_words='english', binary=True, max_features=int(features_words.apply(len).mean()) * 2)
tfidf_features_words.fit(features_words)
description_words = pd.concat([train_df['description_words'], test_df['description_words']])
tfidf_description_words = CountVectorizer(stop_words='english', binary=True, max_features=int(description_words.apply(len).mean()) * 2)
tfidf_description_words.fit(description_words)

kf = KFold(n_splits=5, shuffle=False)
for train_index, test_index in kf.split(train_df):
    train = train_df.loc[train_index]
    test = train_df.loc[test_index]
    train_df.loc[test_index] = set_features_ratios(train, test, features_ratios)

models = []
logloss = []

train_X = sparse.hstack([
    train_df[features_use], 
    tfidf_features_words.transform(train_df['features_words']).tocsr(), 
    #tfidf_description_words.transform(train_df['description_words']).tocsr()
]).tocsr()


kf = KFold(n_splits=5, shuffle=True, random_state=2016)
for train_index, test_index in kf.split(train_X):
    train_x = train_X[train_index, :]
    test_x = train_X[test_index, :]
    train_y = y.loc[train_index]
    test_y = y.loc[test_index]
    y_pred, model = run_xgb(train_x, train_y, test_x, test_y)
    logloss.append(log_loss(test_y.values, y_pred))
    models.append(model)
    break

test_df = set_features_ratios(train_df, test_df, features_ratios)
test_X = sparse.hstack([
    test_df[features_use], 
    tfidf_features_words.transform(test_df['features_words']).tocsr(), 
    #tfidf_description_words.transform(test_df['description_words']).tocsr()
]).tocsr()

y_pred, model = run_xgb(train_X, y, test_X, num_rounds=1500)
out_df = pd.DataFrame(y_pred)
out_df.columns = ['high', 'medium', 'low']
out_df['listing_id'] = test_df.listing_id.values
out_df.to_csv('../results/xgb.csv', index=False)
