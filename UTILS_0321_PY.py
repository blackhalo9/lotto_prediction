from PKG_0321.PKGS_0321_PY import *
from PKG_0321.INPUT_0321_PY import *
from PKG_0321.PREPROCESSING_0321_PY import *

def check_missing_func():
    columns_with_nan = df_scaler_fi.columns[df_scaler_fi.isna().any()].tolist()
    missing_percentage = df_scaler_fi[columns_with_nan].isna().mean() * 100
    
    print("Percentage of missing values in these columns:\n")
    return print(round((missing_percentage),1))

def drop_missing_ratio_func():
    percent_missing = df_scaler_total.isnull().sum() * 100 / len(df_scaler_total)
    columns_to_drop = percent_missing[percent_missing > 80].index
    training_X_scaled_df_cleaned = df_scaler_total.drop(columns=columns_to_drop)
    return columns_to_drop

def make_futr_list_grouped_lists(partial_2union_features_by_lag): 
    partial_features = pd.DataFrame({k: pd.Series(v) for k, v in partial_2union_features_by_lag.items()})
    futr_list_grouped_lists = partial_features.T.reset_index(drop=True)
    futr_list_grouped_lists.index = range(1, 13)
    futr_list_grouped_lists = futr_list_grouped_lists.apply(lambda row: [x for x in row if pd.notna(x)], axis=1)
    futr_list_grouped_lists[13] = futr_list_grouped_lists[12]
    return futr_list_grouped_lists

def make_forecasting_comparison_table(model_keyword, target_y_name, quantile):
    ## usage : make_forecasting_comparison_table('FEDformer', 'LI2CO3(EXW)', 0.25)
    df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
    
    comparison_data_path = f'/home/eunsuk.ko/METAL_FCST/RESULTS/{daterecord}'
    model_keyword = model_keyword
    
    comparison_column_name = f'{model_keyword}_{quantile}_Median'
    comparison_filetype = sorted(glob.glob(f'{comparison_data_path}/*.csv*'))
    comparison_quantile = sorted(glob.glob(f'{comparison_data_path}/*{quantile}*'))
    comparison_model_name = sorted(glob.glob(f'{comparison_data_path}/*{model_keyword}*.csv')) 
    comparison_target_y = sorted(glob.glob(f'{comparison_data_path}/*{target_y_name}*.csv'))
    
    comparison_files_tmp = sorted(set(comparison_filetype).intersection(set(comparison_quantile)))
    comparison_files_tmp = sorted(set(comparison_files_tmp).intersection(set(comparison_model_name)))
    comparison_files = sorted(set(comparison_files_tmp).intersection(set(comparison_target_y)))
    comparison_df_list = []
    
    for file in comparison_files:
        comparison_df = pd.read_csv(file, parse_dates=[0]) 
        comparison_match = re.search(r'\d{4}-\d{2}-\d{2}', file)
        if comparison_match:
            date_str = comparison_match.group()
        else:
            date_str = 'unknown_date'
        
        comparison_df_renamed = comparison_df.rename(columns={comparison_column_name: date_str})
        comparison_df_renamed = comparison_df_renamed.set_index(comparison_df.columns[0])[[date_str]]
        
        comparison_df_list.append(comparison_df_renamed)
    
    comparison_concatenated_df = pd.concat(comparison_df_list, axis=1)
    comparison_sorted_columns = comparison_concatenated_df.apply(lambda x: x.first_valid_index()).sort_values().index
    comparison_concatenated_df = comparison_concatenated_df[comparison_sorted_columns]
    comparison_concatenated_df.reset_index(inplace=True)
    
    another_df = df_temp
    another_column_name = 'y'
    
    comparison_concatenated_df['ds'] = pd.to_datetime(comparison_concatenated_df['ds'])
    comparison_concatenated_df.set_index('ds', inplace=True)
    
    another_df['ds'] = pd.to_datetime(another_df['ds'])
    another_df.set_index('ds', inplace=True)
    
    comparison_start_date = '2020-01-01'
    comparison_end_date = comparison_concatenated_df.index[-1].strftime('%Y-%m-%d')
    
    comparison_date_range = pd.date_range(start=comparison_start_date, end=comparison_end_date, freq='M')
    comparison_concatenated_df_reindexed = comparison_concatenated_df.reindex(comparison_date_range)
    comparison_another_df_reindexed = another_df.rename(columns={another_column_name: f'{target_y_name} Actual Price'}).reindex(comparison_date_range)
    
    comparison_final_df = pd.concat([comparison_another_df_reindexed[[f'{target_y_name} Actual Price']], comparison_concatenated_df_reindexed], axis=1)
    comparison_final_df.index = comparison_final_df.index.to_period('M').to_timestamp()

    return model_keyword,comparison_final_df, quantile

def save_features_by_lag(selected_feature_set, target_y_name, feature_selection_name):
    FILEPATH = os.getcwd()
    with open(f'{FILEPATH}/PKG_0321/FEATURES/saved_feature_set_{target_y_name}_{feature_selection_name}_{feature_cutoff_date}.json', 'w') as json_file:
        json.dump(selected_feature_set, json_file)

def load_features_by_lag(target_y_name, feature_selection_name):
    FILEPATH = os.getcwd()
    with open(f'{FILEPATH}/PKG_0321/FEATURES/saved_feature_set_{target_y_name}_{feature_selection_name}_{feature_cutoff_date}.json', 'r') as json_file:
        loaded_feature_set = json.load(json_file)
    return loaded_feature_set

def make_txt_file_from_forecasted_values(comparison_final_df, target_y_name, quantile):
    processed_rows = []
    for column in comparison_final_df.columns[1:]:
        non_nan_values = comparison_final_df[column].dropna().values[:forecasting_period]
        row_text = ' '.join(map(str, non_nan_values)) + ' '
        processed_rows.append(row_text)
    
    corrected_text_file_path = f'{RESULTS_DIR}/{daterecord}/{target_y_name}_predict_13m_{quantile}.txt'
    with open(corrected_text_file_path, 'w') as file:
        for row in processed_rows:
            file.write(row + "\n")

def make_minimal_matrix(par_directory, par_containing_keywords, target_y_name):
    ## usage : minimal_matrix = make_minimal_matrix(f"{RESULTS_DIR}/{daterecord}/", ['LIOH(EXW)','predict','13m], 'LIOH(EXW)')
    keywords_lower = [keyword.lower() for keyword in par_containing_keywords]    
    pattern = os.path.join(par_directory, '*')
    matching_files = []
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath).lower()
        if all(keyword in filename for keyword in keywords_lower):
            matching_files.append(filepath)
            
    minimal_matrix = np.full((forecasting_period, forecasting_period), np.inf)   
    for matching_file in matching_files:
        matrix = []
        with open(matching_file, 'r') as file:
            for line in file:
                row = list(map(float, line.strip().split(' ')))
                matrix.append(row)
        minimal_matrix = np.minimum(minimal_matrix, matrix)

    file_path = f'{RESULTS_DIR}/{daterecord}/{target_y_name}_TXT_MIN.txt'
    
    with open(file_path, 'w') as file:
        for row in minimal_matrix:
            row_text = ' '.join(map(str, row)) + ' '
            file.write(row_text + "\n")
    print(f"The minimal bound forcasted values has been saved to {file_path}.")

def make_maximal_matrix(par_directory, par_containing_keywords, target_y_name):
    ## usage : maximal_matrix = make_maximal_matrix(f"{RESULTS_DIR}/{daterecord}/", ['LIOH(EXW)','predict','13m], 'LIOH(EXW)')
    keywords_lower = [keyword.lower() for keyword in par_containing_keywords]    
    pattern = os.path.join(par_directory, '*')
    matching_files = []
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath).lower()
        if all(keyword in filename for keyword in keywords_lower):
            matching_files.append(filepath)
            
    maximal_matrix = np.full((forecasting_period, forecasting_period), -np.inf)   
    for matching_file in matching_files:
        matrix = []
        with open(matching_file, 'r') as file:
            for line in file:
                row = list(map(float, line.strip().split(' ')))
                matrix.append(row)
        maximal_matrix = np.maximum(maximal_matrix, matrix)

    file_path = f'{RESULTS_DIR}/{daterecord}/{target_y_name}_TXT_MAX.txt'
    
    with open(file_path, 'w') as file:
        for row in maximal_matrix:
            row_text = ' '.join(map(str, row)) + ' '
            file.write(row_text + "\n")
    print(f"The maximal bound forcasted values has been saved to {file_path}.")

def count_feature_frequency(load_features_by_lag):
    item_frequencies = Counter()
    for key, tuples_list in load_features_by_lag.items():
        for item, _ in tuples_list:
            item_frequencies[item] += 1
    
    sorted_frequencies = sorted(item_frequencies.items(), key=lambda x: x[1], reverse=True)
    
    for item, freq in sorted_frequencies:
        print(f"{item}: {freq}")

def remove_lightning_log_folder():
    if os.path.exists('./lightning_logs'):
        shutil.rmtree('./lightning_logs')
        print(f"Folder '{'./lightning_logs'}' removed successfully.")
    else:
        pass
def remove_chekpoint_tmp_trainer_folder():
    if os.path.exists(f"{BASE_DIR}/{str(fulldaterecord)[:4]+str('_')+str(fulldaterecord)[5:7]}/lightning_logs"):
        shutil.rmtree(f"{BASE_DIR}/{str(fulldaterecord)[:4]+str('_')+str(fulldaterecord)[5:7]}/lightning_logs")
        print(f"Folder {BASE_DIR}/{str(fulldaterecord)[:4]+str('_')+str(fulldaterecord)[5:7]}/lightning_logs removed successfully.")
    else:
        pass
    if os.path.exists(f"{BASE_DIR}/{str(fulldaterecord)[:4]+str('_')+str(fulldaterecord)[5:7]}/checkpoint"):
        shutil.rmtree(f"{BASE_DIR}/{str(fulldaterecord)[:4]+str('_')+str(fulldaterecord)[5:7]}/checkpoint")
        print(f"Folder {BASE_DIR}/{str(fulldaterecord)[:4]+str('_')+str(fulldaterecord)[5:7]}/checkpoint removed successfully.")
    else:
        pass
    if os.path.exists(f"{BASE_DIR}/{str(fulldaterecord)[:4]+str('_')+str(fulldaterecord)[5:7]}/tmp_trainer"):
        shutil.rmtree(f"{BASE_DIR}/{str(fulldaterecord)[:4]+str('_')+str(fulldaterecord)[5:7]}/tmp_trainer")
        print(f"Folder {BASE_DIR}/{str(fulldaterecord)[:4]+str('_')+str(fulldaterecord)[5:7]}/tmp_trainer removed successfully.")
    else:
        pass
def clear_cache():
    gc.collect()
