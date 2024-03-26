from PKG_0321.PKGS_0321_PY import *
from PKG_0321.INPUT_0321_PY import *
from PKG_0321.UTILS_0321_PY import *
from PKG_0321.ENSEMBLE_LIST_0321_PY import *

def raw_shifted_df(selected_feature_set, cutoff_date, i):
    
    df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y, cutoff_date)
    
    future_date_series = pd.date_range(pd.date_range(cutoff_date,periods=2,freq='M')[1],periods=i,freq='M')
    future_date = pd.DataFrame(future_date_series,columns=['ds'])
    shifted_df = pd.merge(future_date, df_scaler_total, on='ds',how='outer').reset_index(drop=True)
    
    futr_list_grouped_lists = make_futr_list_grouped_lists(selected_feature_set)
    futr_list = futr_list_grouped_lists[i]
    
    for column in futr_list:
        shifted_df[column] = shifted_df[column].shift(i)
    
    shifted_train_df = shifted_df.loc[(shifted_df.ds <= cutoff_date) & (shifted_df.ds >= training_start)]
    shifted_futr_df = shifted_df.loc[shifted_df.ds > cutoff_date]
    
    training_X = shifted_train_df.drop(['ds', 'y'], axis=1)
    training_y = shifted_train_df['y']
    testing_X = shifted_futr_df.drop(['ds', 'y'], axis=1)
    testing_y = shifted_futr_df['y']
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    training_X_scaled = scaler_X.fit_transform(training_X)
    testing_X_scaled = scaler_X.transform(testing_X)
    
    training_X_scaled_tmp = pd.DataFrame(training_X_scaled, columns=shifted_train_df.drop(['y','ds'], axis=1).columns)
    testing_X_scaled_tmp = pd.DataFrame(testing_X_scaled, columns=shifted_futr_df.drop(['y','ds'], axis=1).columns)
    
    training_y_scaled_arry = scaler_y.fit_transform(training_y.values.reshape(-1, 1))
    testing_y_scaled_arry = scaler_y.transform(testing_y.values.reshape(-1, 1))
    
    # training_y_scaled_tmp = pd.DataFrame(training_y_scaled_arry, columns=['y'])
    # testing_y_scaled_tmp = pd.DataFrame(testing_y_scaled_arry, columns=['y'])
    
    # X_train_orig = scaler_X.inverse_transform(training_X_scaled_tmp)
    # X_test_orig = scaler_X.inverse_transform(testing_X_scaled_tmp)
    
    # y_train_orig = scaler_y.inverse_transform(training_y_scaled_arry)
    # y_test_orig = scaler_y.inverse_transform(testing_y_scaled_arry)
    
    training_X_scaled_df = pd.concat([shifted_train_df[shifted_train_df['ds'] <= cutoff_date].ds.reset_index(drop=True) ,training_X_scaled_tmp.reset_index(drop=True)], axis= 1)
    training_X_scaled_df = training_X_scaled_df.bfill().ffill()
    testing_X_scaled_df = pd.concat([shifted_futr_df[shifted_futr_df['ds'] > cutoff_date].ds.reset_index(drop=True) ,testing_X_scaled_tmp.reset_index(drop=True)], axis= 1)
    
    df_with_index = list(['ds']) + futr_list
    testing_X_scaled_df = testing_X_scaled_df[df_with_index]
    
    ## y Scaled (True)
    if choose_y_train_scaler == True:
        training_X_scaled_df['y'] = training_y_scaled_arry
        testing_X_scaled_df['y'] = testing_y_scaled_arry
    ## y Not Scaled (False)
    elif choose_y_train_scaler == False:
        training_X_scaled_df['y'] = shifted_train_df['y'].reset_index(drop=True)
        testing_X_scaled_df['y'] = shifted_futr_df['y'].reset_index(drop=True)
    else: ## (None)
        pass
        
    columns = training_X_scaled_df.columns.tolist()
    columns.insert(1, columns.pop(-1))
    training_X_scaled_df = training_X_scaled_df[columns]
    
    training_X_scaled_df.insert(1, 'unique_id', 'A')
    testing_X_scaled_df.insert(1, 'unique_id', 'A')
        
    return training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists
    
def raw_y_scaled_df(cutoff_date):
    
    df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
    
    df_temp_scaled = df_temp.copy()
    scaler_y = StandardScaler()
    df_temp_scaled['y'] = scaler_y.fit_transform(df_temp_scaled['y'].values.reshape(-1, 1))
    df_temp_scaled = df_temp_scaled[['ds','y']].loc[(df_temp_scaled.ds >= training_start)]

    return df_temp_scaled