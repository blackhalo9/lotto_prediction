from PKG_0321.PKGS_0321_PY import *
from PKG_0321.INPUT_0321_PY import *

def raw_data_loader(target_y, cutoff_date):
    # df_temp = pd.read_csv('./DATA/data101114/INTEGR_METAL_NON_METAL_MONTHLY_231115_V2.csv')
    if target_y == 'LI2CO3(EXW)':
        target_y = 'PRICE_LI2CO3_EXW_FAST_KG'
    elif target_y == 'LI2CO3(CIF)':
        target_y = 'PRICE_LI2CO3_CIF_FAST_KG_SPOT_EXCHNG'
    elif target_y == 'LIOH(EXW)':
        target_y = 'PRICE_LIOH_EXW_FAST_KG'
    elif target_y == 'LIOH(CIF)':
        target_y = 'PRICE_LIOH_CIF_FAST_KG_SPOT_EXCHNG'
    elif target_y == 'PRICE_AG_HIGH_ICC_RMB':
        target_y_name = 'AG_HIGH'
    elif target_y == 'PRICE_COKE_GNR_GRN_ICC_RMB':
        target_y_name ='COKE_GNR'
    else:
        pass
    df_temp = pd.read_csv(RAW_PATH)
    df_temp.rename(columns = {"Unnamed: 0" : "ds"}, inplace = True)
    df_temp.rename(columns={target_y:'y'},inplace=True)
    df_temp['ds'] = pd.to_datetime(df_temp['ds'], format='%Y-%m-%d')
    df_temp = df_temp.loc[:, ~df_temp.columns.str.contains('|'.join(keywords_excluded))]
    df_temp.insert(1, 'y', df_temp.pop('y'))
        
    ## df_total: 가공 되지 않은 2035년까지 data
    df_temp = df_temp.set_index('ds')
    df_temp = df_temp.resample('M').mean()
    df_temp = df_temp.reset_index()
    
    ## 최신데이터까지 (df_total)
    df_temp = df_temp.loc[(df_temp.ds >= raw_start) & (df_temp.ds <= last_month_updated)]

    # df_temp = df_temp.dropna(axis=1, how='all') ## remove columns contain NaN only
    columns_to_drop = [col for col in df_temp.columns if not df_temp[col].replace(np.nan, 0).any()] ## remove columns contain NaN and zero
    df_temp = df_temp.drop(columns=columns_to_drop)

    ## 다채움
    df_temp = df_temp.bfill().ffill()

    ## 2016~데이터 마지막 시점까지
    df_scaler = df_temp.copy()
    
     ## df_scaler: feature 변수 검색용
    df_scaler_fi = df_scaler.loc[df_scaler.ds >= training_start]
    df_scaler_fi = df_scaler_fi.loc[df_scaler_fi.ds <= feature_cutoff_date]

     ## df_scaler_total: 분리용
    df_scaler_total = df_scaler.loc[df_scaler.ds >= raw_start]
    df_scaler_total = df_scaler_total.loc[df_scaler_total.ds <= cutoff_date]
    df_scaler_total = df_scaler_total.ffill()
    df_scaler_total.reset_index(drop=True, inplace=True)

    return df_scaler_total, df_scaler_fi, df_temp

def raw_fi_df(df_scaler_fi):
    
    fi_training_X = df_scaler_fi.drop(['ds', 'y'], axis=1)
    fi_training_y = df_scaler_fi['y']
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    fi_training_X_scaled = scaler_X.fit_transform(fi_training_X)
    fi_training_X_scaled_tmp = pd.DataFrame(fi_training_X_scaled, columns=df_scaler_fi.drop(['y','ds'], axis=1).columns)
    fi_training_X_scaled_df = pd.concat([df_scaler_fi[['ds','y']].reset_index(drop=True) ,fi_training_X_scaled_tmp.reset_index(drop=True)], axis= 1)
    fi_training_X_scaled_df = fi_training_X_scaled_df.bfill().ffill()
    
    fi_training_y_scaled_arry = scaler_y.fit_transform(fi_training_y.values.reshape(-1, 1))
    
    # ## y Scaled (True)
    # if choose_y_train_scaler == True:
    #     fi_training_X_scaled_df['y'] = fi_training_y_scaled_arry
    # elif choose_y_train_scaler == False:
    #     fi_training_X_scaled_df['y'] = fi_training_X_scaled_df['y'].reset_index(drop=True)
    # else: ## (None)
    #     pass
        
    return fi_training_X_scaled_df