from PKG_0321.PKGS_0321_PY import *
from PKG_0321.INPUT_0321_PY import *
from PKG_0321.UTILS_0321_PY import *
from PKG_0321.PREPROCESSING_0321_PY import *

def mutual_information(fi_training_X_scaled_df):
    ## Mutual Information - User Input Parameters ###############################
    max_lags = 12 # Total number of lags to be checked (from 1 to 12 only)
    n_neighbors_num = 5 # k_nearnest neighbors options (KNN)
    # Exceptional variables that are not belong to filrtering options
    exceptional_keywords = [] # 'PRICE' is tentatively crucial 
    mi_score_threshold = 0.3 # Filereting threshold of Mutual Information score
    top_n_num = 50 # Number of variables displayed for every plots
    top_limit_num = 5 # Limit number of variables by a category per a lag plot  
    top_limit_ratio = 0.2 # Limit ratio of variables by a category per a lag plot  
    #############################################################################

    raw_y_set_resampled_copy = fi_training_X_scaled_df.copy()
    
    df = raw_y_set_resampled_copy
    df = df[[c for c in df if c != 'y'] + ['y']]
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df.index.name = None
    columns = df.columns.drop(['y'])
    
    mutual_info_selected_features_by_lag = {}
    
    for lag in range(1, max_lags + 1):
        lagged_features = df[columns].shift(lag)
        lagged_df = pd.concat([lagged_features, df['y']], axis=1).dropna()
        
        if lagged_df.shape[0] == 0:
            continue
    
        mi_scores = mutual_info_regression(lagged_df[columns[:-1]], lagged_df['y'], n_neighbors=n_neighbors_num, random_state=42)
        mutual_info_selected_features_by_lag[lag] = sorted(zip(columns[:-1], mi_scores), key=lambda x: x[1], reverse=True)
    
        def extract_group(name, exceptional_keywords):
            first_part = name.split('_')[0]
            if first_part in exceptional_keywords:
                return name
            return first_part
      
        grouped = defaultdict(list)
        for name, value in mutual_info_selected_features_by_lag[lag]:
            group = extract_group(name, exceptional_keywords)
            grouped[group].append((name, value))
        
        result = []
        for group, name_values in grouped.items():
            sorted_names = sorted(name_values, key=lambda x: x[1], reverse=True)
            filtered_names = [pair for pair in sorted_names if pair[1] >= mi_score_threshold]
            top_limit = min(top_limit_num, math.ceil(top_limit_ratio * len(filtered_names)))
            if group == 'CPI' or 'PRICE':
                top_limit = 5
            top_names = sorted_names[:top_limit]
            result.extend(top_names)
            result = sorted(result, key=lambda x: x[1], reverse=True)
            mutual_info_selected_features_by_lag[lag] = result
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 30))
    
    for i, ax in enumerate(axes.flatten()):
        lag = i + 1
        if lag in mutual_info_selected_features_by_lag:
            scores = mutual_info_selected_features_by_lag[lag]
            top_n = top_n_num
            top_scores = scores[:top_n]
            top_scores = sorted(top_scores, key=lambda x: x[1], reverse=True)
            features, mi_scores = zip(*top_scores)
            ax.barh(features, mi_scores, color='green', alpha=0.5)
            ax.set_title(f'Features for Lag{lag}')
            # ax.set_xlabel('Mutual Information Feature Selection')
            ax.invert_yaxis()
        else:
            ax.set_title(f'Lag {lag} (not enough data)')
            ax.axis('off')
    
    fig.text(0.5, 0.02, 'Mutual Information Feature Selection of every 12 Lags', ha='center', va='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    # plt.show()

    mutual_info_max_length = max(len(items) for items in mutual_info_selected_features_by_lag.values())
    mutual_info_padded_data = {key: values + [(None, None, None)] * (mutual_info_max_length - len(values)) 
                   for key, values in mutual_info_selected_features_by_lag.items()}
    mutual_info_feature_data = {key: [item[0] for item in value] for key, value in mutual_info_padded_data.items()}
    mutual_info_feature_data_df = pd.DataFrame(mutual_info_feature_data)
    # mutual_info_feature_data_df.to_csv(f'./2024_02/mutual_info_feature_data_df.csv', index=False)
    
    return mutual_info_selected_features_by_lag

def boruta_fi(fi_training_X_scaled_df):
    ## Boruta Feature Importance - User Input Parameters ########################
    max_lags = 12 # Total number of lags to be checked (from 1 to 12 only)
    # Exceptional variables that are not belong to filrtering options
    exceptional_keywords = [] # 'PRICE' is tentatively crucial 
    boruta_rank_threshold = 1 # Filereting threshold of Boruta rank
    top_n_num = 50 # Number of variables displayed for every plots
    top_limit_num = 5 # Limit number of variables by a category per a lag plot  
    top_limit_ratio = 0.2 # Limit ratio of variables by a category per a lag plot 
    #############################################################################
    
    
    raw_y_set_resampled_copy = fi_training_X_scaled_df.copy()
    df = raw_y_set_resampled_copy
    df = df[[c for c in df if c != 'y'] + ['y']]
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df.index.name = None
    columns = df.columns.drop(['y'])
    
    boruta_selected_features_by_lag = {}
    
    for lag in range(1, max_lags + 1):
        lagged_features = df[columns].shift(lag)
        lagged_df = pd.concat([lagged_features, df['y']], axis=1).dropna()
        
        if lagged_df.shape[0] == 0:
            continue
    
        X = lagged_df.drop('y', axis=1)
        y = lagged_df['y']
    
        rf = RandomForestRegressor(n_jobs=-1, n_estimators=1000, max_depth=5, random_state=42)
        feature_selector = BorutaPy(rf, n_estimators='auto', max_iter=10, random_state=42)
        feature_selector.fit(X.values, y.values)
    
        features = X.columns
        importance_scores = rf.feature_importances_
        boruta_selected_features_by_lag[lag] = sorted(zip(features, feature_selector.ranking_, importance_scores), key=lambda x: x[1], reverse=False)
        
        def extract_group(name, exceptional_keywords):
            first_part = name.split('_')[0]
            if first_part in exceptional_keywords:
                return name
            return first_part
      
        grouped = defaultdict(list)
        for name, rank, rf_fi in boruta_selected_features_by_lag[lag]:
            if rank <= boruta_rank_threshold:
                group = extract_group(name, exceptional_keywords)
                grouped[group].append((name, rank, rf_fi))
        
        result = []
        for group, name_values_fi in grouped.items():
            if group == 'CPI' or 'PRICE':
                num_to_select = 5
            else:
                num_to_select = min(top_limit_num, math.ceil(top_limit_ratio * len(name_values_fi)))
                if num_to_select <= 0:
                    num_to_select = 1
            top_names = sorted(name_values_fi, key=lambda x: x[2], reverse=False)[:num_to_select]
            result.extend(top_names)
            boruta_selected_features_by_lag[lag] = result
            
    fig, axes = plt.subplots(3, 4, figsize=(20, 30))
    
    for i, ax in enumerate(axes.flatten()):
        lag = i + 1
        if lag in boruta_selected_features_by_lag:
            scores = boruta_selected_features_by_lag[lag]
            top_n = top_n_num
            top_scores = scores[:top_n]
            top_scores = sorted(top_scores, key=lambda x: x[2], reverse=True)
            features, boruta_rank, rf_fi = zip(*top_scores)
            ax.barh(features, rf_fi, color='purple', alpha=0.5)
            ax.set_title(f'Features for Lag{lag}')
            # ax.set_xlabel('Boruta Feature Selection')
            ax.invert_yaxis()
        else:
            ax.set_title(f'Lag {lag} (not enough data)')
            ax.axis('off')
    
    fig.text(0.5, 0.02, 'Boruta Feature Selection of every 12 Lags', ha='center', va='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    # plt.show()

    boruta_max_length = max(len(items) for items in boruta_selected_features_by_lag.values())
    boruta_padded_data = {key: values + [(None, None, None)] * (boruta_max_length - len(values)) 
                   for key, values in boruta_selected_features_by_lag.items()}
    boruta_feature_data = {key: [item[0] for item in value] for key, value in boruta_padded_data.items()}
    boruta_feature_data_df = pd.DataFrame(boruta_feature_data)
    # boruta_feature_data_df.to_csv(f'./2024_02/boruta_feature_data_df.csv', index=False)

    return boruta_selected_features_by_lag

def elastic_net_fi(fi_training_X_scaled_df):
    ## Elastic Net Feature Importance - User Input Parameters ###################
    max_lags = 12 # Total number of lags to be checked (from 1 to 12 only)
    # Exceptional variables that are not belong to filrtering options
    exceptional_keywords = [] # 'PRICE' is considerably crucial 
    coef_threshold = 0.1
    top_n_num = 50 # Number of variables displayed for every plots
    top_limit_num = 5 # Limit number of variables by a category per a lag plot  
    top_limit_ratio = 0.2 # Limit ratio of variables by a category per a lag plot 
    #############################################################################
    
    
    raw_y_set_resampled_copy = fi_training_X_scaled_df.copy()
    df = raw_y_set_resampled_copy
    df = df[[c for c in df if c != 'y'] + ['y']]
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df.index.name = None
    columns = df.columns.drop(['y'])
    
    elastic_net_selected_features_by_lag = {}
    
    for lag in range(1, max_lags + 1):
        lagged_features = df[columns].shift(lag)
        lagged_df = pd.concat([lagged_features, df['y']], axis=1).dropna()
        
        if lagged_df.shape[0] == 0:
            continue
    
        X = lagged_df.drop('y', axis=1)
        y = lagged_df['y']
    
        elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elastic_net.fit(X, y)
    
        coefficients = elastic_net.coef_
        elastic_net_selected_features_by_lag[lag] = [(feature, coef) for feature, coef in zip(X.columns, coefficients)]
        
        def extract_group(name, exceptional_keywords):
            first_part = name.split('_')[0]
            if first_part in exceptional_keywords:
                return name
            return first_part
      
        grouped = defaultdict(list)
        for name, coef in elastic_net_selected_features_by_lag[lag]:
            if abs(coef) >= coef_threshold:
                group = extract_group(name, exceptional_keywords)
                grouped[group].append((name, abs(coef)))
        
        result = []
        for group, name_values_fi in grouped.items():
            if group == 'CPI' or 'PRICE':
                num_to_select = 5
            else:
                num_to_select = min(top_limit_num, math.ceil(top_limit_ratio * len(name_values_fi)))
                if num_to_select <= 0:
                    num_to_select = 1
            top_names = sorted(name_values_fi, key=lambda x: x[1], reverse=False)[:num_to_select]
            result.extend(top_names)
            elastic_net_selected_features_by_lag[lag] = result
            
    fig, axes = plt.subplots(3, 4, figsize=(20, 30))
    
    for i, ax in enumerate(axes.flatten()):
        lag = i + 1
        if lag in elastic_net_selected_features_by_lag:
            scores = elastic_net_selected_features_by_lag[lag]
            top_n = top_n_num
            top_scores = scores[:top_n]
            top_scores = sorted(top_scores, key=lambda x: x[1], reverse=True)
            features, coef = zip(*top_scores)
            ax.barh(features, coef, color='orange', alpha=0.5)
            ax.set_title(f'Features for Lag{lag}')
            # ax.set_xlabel('ElasticNet Feature Selection')
            ax.invert_yaxis()
        else:
            ax.set_title(f'Lag {lag} (not enough data)')
            ax.axis('off')
    
    fig.text(0.5, 0.02, 'ElasticNet Feature Selection of every 12 Lags', ha='center', va='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

    elasticnet_max_length = max(len(items) for items in elastic_net_selected_features_by_lag.values())
    elasticnet_padded_data = {key: values + [(None, None, None)] * (elasticnet_max_length - len(values)) 
                   for key, values in elastic_net_selected_features_by_lag.items()}
    elasticnet_feature_data = {key: [item[0] for item in value] for key, value in elasticnet_padded_data.items()}
    elasticnet_feature_data_df = pd.DataFrame(elasticnet_feature_data)
    # elasticnet_feature_data_df.to_csv(f'./2024_02/elasticnet_feature_data_df.csv', index=False)

    return elastic_net_selected_features_by_lag

def pearson_corr_fi(fi_training_X_scaled_df):
    ## Lagged Correlation Feature Importance - User Input Parameters ###################
    max_lags = 12 # Total number of lags to be checked (from 1 to 12 only)
    # Exceptional variables that are not belong to filrtering options
    exceptional_keywords = [] # 'PRICE' is considerably crucial 
    corr_threshold = 0.8
    top_n_num = 50 # Number of variables displayed for every plots
    top_limit_num = 5 # Limit number of variables by a category per a lag plot  
    top_limit_ratio = 0.2 # Limit ratio of variables by a category per a lag plot 
    ####################################################################################
    
    
    raw_y_set_resampled_copy = fi_training_X_scaled_df.copy()
    df = raw_y_set_resampled_copy
    df = df[[c for c in df if c != 'y'] + ['y']]
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df.index.name = None
    columns = df.columns.drop(['y'])
    
    pearson_corr_selected_features_by_lag = {}
    
    for lag in range(1, max_lags + 1):
        lagged_features = df[columns].shift(lag)
        lagged_df = pd.concat([lagged_features, df['y']], axis=1).dropna()
        
        if lagged_df.shape[0] == 0:
            continue
    
        correlations = {}
        for col in lagged_df.columns.drop('y'):
            corr, _ = pearsonr(lagged_df[col], lagged_df['y'])
            correlations[col] = corr
            
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        pearson_corr_selected_features_by_lag[lag] = sorted_correlations
        
        def extract_group(name, exceptional_keywords):
            first_part = name.split('_')[0]
            if first_part in exceptional_keywords:
                return name
            return first_part
      
        grouped = defaultdict(list)
        for name, corr in pearson_corr_selected_features_by_lag[lag]:
            if abs(corr) >= corr_threshold:
                group = extract_group(name, exceptional_keywords)
                grouped[group].append((name, abs(corr)))
        
        result = []
        for group, name_values_fi in grouped.items():
            if group == 'CPI' or 'PRICE':
                num_to_select = 5
            else:
                num_to_select = min(top_limit_num, math.ceil(top_limit_ratio * len(name_values_fi)))
                if num_to_select <= 0:
                    num_to_select = 1
            top_names = sorted(name_values_fi, key=lambda x: x[1], reverse=False)[:num_to_select]
            result.extend(top_names)
            pearson_corr_selected_features_by_lag[lag] = result
            
    fig, axes = plt.subplots(3, 4, figsize=(20, 30))
    
    for i, ax in enumerate(axes.flatten()):
        lag = i + 1
        if lag in pearson_corr_selected_features_by_lag:
            scores = pearson_corr_selected_features_by_lag[lag]
            top_n = top_n_num
            top_scores = scores[:top_n]
            top_scores = sorted(top_scores, key=lambda x: x[1], reverse=True)
            features, corr = zip(*top_scores)
            ax.barh(features, corr)
            ax.set_title(f'Features for Lag{lag}')
            # ax.set_xlabel('Lagged Correlation(Pearson) Feature Selection')
            ax.invert_yaxis()
        else:
            ax.set_title(f'Lag {lag} (not enough data)')
            ax.axis('off')
    
    fig.text(0.5, 0.02, 'Lagged Correlation(Pearson) Feature Selection of every 12 Lags', ha='center', va='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    # plt.show()

    pearson_corr_max_length = max(len(items) for items in pearson_corr_selected_features_by_lag.values())
    pearson_corr_padded_data = {key: values + [(None, None, None)] * (pearson_corr_max_length - len(values)) 
                   for key, values in pearson_corr_selected_features_by_lag.items()}
    pearson_corr_feature_data = {key: [item[0] for item in value] for key, value in pearson_corr_padded_data.items()}
    pearson_corr_feature_data_df = pd.DataFrame(pearson_corr_feature_data)
    # pearson_corr_feature_data_df.to_csv(f'./2024_02/pearson_corr_feature_data_df.csv', index=False)

    return pearson_corr_selected_features_by_lag

def kendall_corr_fi(fi_training_X_scaled_df):
    ## Lagged Correlation Feature Importance - User Input Parameters ###################
    max_lags = 12 # Total number of lags to be checked (from 1 to 12 only)
    # Exceptional variables that are not belong to filrtering options
    exceptional_keywords = [] # 'PRICE' is considerably crucial 
    corr_threshold = 0.5
    top_n_num = 50 # Number of variables displayed for every plots
    top_limit_num = 5 # Limit number of variables by a category per a lag plot  
    top_limit_ratio = 0.2 # Limit ratio of variables by a category per a lag plot 
    ####################################################################################
    
    
    raw_y_set_resampled_copy = fi_training_X_scaled_df.copy()
    df = raw_y_set_resampled_copy
    df = df[[c for c in df if c != 'y'] + ['y']]
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df.index.name = None
    columns = df.columns.drop(['y'])
    
    kendall_corr_selected_features_by_lag = {}
    
    for lag in range(1, max_lags + 1):
        lagged_features = df[columns].shift(lag)
        lagged_df = pd.concat([lagged_features, df['y']], axis=1).dropna()
        
        if lagged_df.shape[0] == 0:
            continue
    
        correlations = {}
        for col in lagged_df.columns.drop('y'):
            corr, _ = kendalltau(lagged_df[col], lagged_df['y'])
            correlations[col] = abs(corr)
            
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        kendall_corr_selected_features_by_lag[lag] = sorted_correlations
        
        def extract_group(name, exceptional_keywords):
            first_part = name.split('_')[0]
            if first_part in exceptional_keywords:
                return name
            return first_part
      
        grouped = defaultdict(list)
        for name, corr in kendall_corr_selected_features_by_lag[lag]:
            if abs(corr) >= corr_threshold:
                group = extract_group(name, exceptional_keywords)
                grouped[group].append((name, abs(corr)))
        
        result = []
        for group, name_values_fi in grouped.items():
            if group == 'CPI' or 'PRICE':
                num_to_select = 5
            else:
                num_to_select = min(top_limit_num, math.ceil(top_limit_ratio * len(name_values_fi)))
                if num_to_select <= 0:
                    num_to_select = 1
            top_names = sorted(name_values_fi, key=lambda x: x[1], reverse=False)[:num_to_select]
            result.extend(top_names)
            kendall_corr_selected_features_by_lag[lag] = result
            
    fig, axes = plt.subplots(3, 4, figsize=(20, 30))
    
    for i, ax in enumerate(axes.flatten()):
        lag = i + 1
        if lag in kendall_corr_selected_features_by_lag:
            scores = kendall_corr_selected_features_by_lag[lag]
            top_n = top_n_num
            top_scores = scores[:top_n]
            top_scores = sorted(top_scores, key=lambda x: x[1], reverse=True)
            features, corr = zip(*top_scores)
            ax.barh(features, corr)
            ax.set_title(f'Features for Lag{lag}')
            # ax.set_xlabel('Lagged Correlation(Kendall) Feature Selection')
            ax.invert_yaxis()
        else:
            ax.set_title(f'Lag {lag} (not enough data)')
            ax.axis('off')

    kendall_corr_max_length = max(len(items) for items in kendall_corr_selected_features_by_lag.values())
    kendall_corr_padded_data = {key: values + [(None, None, None)] * (kendall_corr_max_length - len(values)) 
                   for key, values in kendall_corr_selected_features_by_lag.items()}
    kendall_corr_feature_data = {key: [item[0] for item in value] for key, value in kendall_corr_padded_data.items()}
    kendall_corr_feature_data_df = pd.DataFrame(kendall_corr_feature_data)
    # kendall_corr_feature_data_df.to_csv(f'./2024_02/kendall_corr_feature_data_df.csv', index=False)
    
    fig.text(0.5, 0.02, 'Lagged Correlation(Kendall) Feature Selection of every 12 Lags', ha='center', va='center', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    # plt.show()

def intersection_features_by_lags():
    top_n = 50
    intersection_features_by_lag = {}
    
    for lag in range(1, max_lags+1):
        top_features_method1 = [feature for feature, _ in mutual_info_selected_features_by_lag[lag][:top_n]]
        top_features_method2 = [feature for feature, _ ,_ in boruta_selected_features_by_lag[lag][:top_n]]
        top_features_method3 = [feature for feature, _ in elastic_net_selected_features_by_lag[lag][:top_n]]
        top_features_method4 = [feature for feature, _ in pearson_corr_selected_features_by_lag[lag][:top_n]]
        # top_features_method5 = [feature for feature, _ in kendall_corr_selected_features_by_lag[lag][:top_n]]
        
        intersection = set(top_features_method1) & \
                       set(top_features_method2) & \
                       set(top_features_method3) & \
                       set(top_features_method4)
                       # set(top_features_method5)
        intersection_features_by_lag[lag] = list(intersection)

    # intersection_features_by_lag_df = pd.DataFrame({k: pd.Series(v) for k, v in intersection_features_by_lag.items()})
    # intersection_features_by_lag.to_csv(f'./2024_02/intersection_features_by_lag.csv', index=False)
    return intersection_features_by_lag

def partial_intersection_by_lags():
    top_n = 50
    partial_union_features_by_lag = {}

    for lag in range(1, max_lags +1):
        top_features_method1 = set([feature for feature, _ in mutual_info_selected_features_by_lag[lag][:top_n]])
        top_features_method2 = set([feature for feature, _, _ in boruta_selected_features_by_lag[lag][:top_n]])
        top_features_method3 = set([feature for feature, _ in elastic_net_selected_features_by_lag[lag][:top_n]])
        top_features_method4 = set([feature for feature, _ in pearson_corr_selected_features_by_lag[lag][:top_n]])
        # top_features_method5 = set([feature for feature, _ in kendall_corr_selected_features_by_lag[lag][:top_n]])
    
        partial_union = top_features_method1 & top_features_method2 | \
                        top_features_method1 & top_features_method3 | \
                        top_features_method1 & top_features_method4 | \
                        top_features_method2 & top_features_method3 | \
                        top_features_method2 & top_features_method4 | \
                        top_features_method3 & top_features_method4
        
        partial_union_features_by_lag[lag] = list(partial_union)
    
    # partial_union_features_by_lag_df = pd.DataFrame({k: pd.Series(v) for k, v in partial_union_features_by_lag.items()})
    # partial_union_features_by_lag_df.to_csv(f'./2024_02/partial_union_features_by_lag_df.csv', index=False)
    return partial_union_features_by_lag

def complete_union_by_lags():
    
    union_features_by_lag = {}

    for lag in range(1, max_lags +1):
        top_features_method1 = set([feature for feature, _ in mutual_info_selected_features_by_lag[lag][:top_n]])
        top_features_method2 = set([feature for feature, _, _ in boruta_selected_features_by_lag[lag][:top_n]])
        top_features_method3 = set([feature for feature, _ in elastic_net_selected_features_by_lag[lag][:top_n]])
        top_features_method4 = set([feature for feature, _ in pearson_corr_selected_features_by_lag[lag][:top_n]])
        # top_features_method5 = set([feature for feature, _ in kendall_corr_selected_features_by_lag[lag][:top_n]])
    
        union = top_features_method1 \
                | top_features_method2 \
                | top_features_method3 \
                | top_features_method4 \
                # | top_features_method5
        union_features_by_lag[lag] = list(union)
    
    # union_features_by_lag_df = pd.DataFrame({k: pd.Series(v) for k, v in union_features_by_lag.items()})
    # union_features_by_lag_df.to_csv(f'./2024_02/union_features_by_lag_df.csv', index=False)

    return union_features_by_lag

def n_most_frequent_by_features():
    count_threshold = 10
    top_n = 100
    emphasized_features = [
                           'PRICE_CU_LME_KG',
                           'KOMIS_MOI_CU',
                           'KOMIS_MOI_LI',
                           'PRICE_AL_LME_KG',
                           'LGES_LIOH_RQMTS_KG',
                           'SHIP_SHANGHAI_CONT_FREIGHT_IDX',
                           'SHIP_HOWE_ROBINSON_CONTAINER_IDX',
                           'CHN_Q245R_30MM_STEEL_PLATE',
                           'EVV_LFP_APAC',
                           'EVV_NMC_EUR',
                           'EVV_NMC_AMER',
                           'STOCK_LI_TIANQI_CLOSE',
                           'STOCK_AG_NINGBO_SHANSHAN_CLOSE',
                           'STOCK_LI_GANFENG_CLOSE',
                           'AM_CHN_LI2CO3_SLS',
                           'AM_CHN_LIOH_SLS',
                           'AM_CHN_NCM_PRD',
                           'DRV_LIOH_NI',
                          ]
    
    def count_feature_appearances(selected_features_by_lag):
        all_features = []
        for lag in range(1, max_lags+1):
            feature_names = [feature_tuple[0] for feature_tuple in selected_features_by_lag[lag]]
            all_features.extend(feature_names)
        return Counter(all_features)
    
    feature_counts_method1 = count_feature_appearances(mutual_info_selected_features_by_lag)
    feature_counts_method2 = count_feature_appearances(boruta_selected_features_by_lag)
    feature_counts_method3 = count_feature_appearances(elastic_net_selected_features_by_lag)
    feature_counts_method4 = count_feature_appearances(pearson_corr_selected_features_by_lag)
    # feature_counts_method5 = count_feature_appearances(kendall_corr_selected_features_by_lag)
    
    combined_feature_counts = Counter()
    combined_feature_counts.update(feature_counts_method1)
    combined_feature_counts.update(feature_counts_method2)
    combined_feature_counts.update(feature_counts_method3)
    combined_feature_counts.update(feature_counts_method4)
    # combined_feature_counts.update(feature_counts_method5)
    
    filtered_sorted_features = [(feature, count) for feature, count in combined_feature_counts.items() if count >= count_threshold]
    filtered_sorted_features = sorted(filtered_sorted_features, key=lambda x: x[1], reverse=True)[:top_n]
    
    features, counts = zip(*filtered_sorted_features)
    
    plt.figure(figsize=(20, 10))
    plt.bar(features, counts) 
    
    for feature, count in filtered_sorted_features:
        if feature in emphasized_features:
            plt.bar(feature, count, color='red', alpha=0.7)
        else:
            plt.bar(feature, count, color='blue', alpha=0.1)  
    
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Total Count Across All Methods', fontsize=14)
    plt.title(f'Top Features with Frequency in All Lags and Methods - shown more than {count_threshold}', fontsize=14)
    plt.xticks(rotation=90) 
    
    for tick in plt.gca().get_xticklabels():
        if tick.get_text() in emphasized_features:
            tick.set_weight('bold')
            tick.set_color('red')
    
    plt.show()