from PKG_0321.PKGS_0321_PY import *
from PKG_0321.INPUT_0321_PY import *
from PKG_0321.UTILS_0321_PY import *
from PKG_0321.PREPROCESSING_0321_PY import *
from PKG_0321.FEATURE_IMPORTANCE_0321_PY import *
from PKG_0321.ENSEMBLE_LIST_0321_PY import *
from PKG_0321.FUTURE_FRAME_0321_PY import *
from PKG_0321.DATE_DICT_0321_PY import *
from PKG_0321.DLINEAR_MODEL_0321_PY import *
from PKG_0321.NLINEAR_MODEL_0321_PY import *
from PKG_0321.TSMIXERX_MODEL_0321_PY import *

def timesnet_optuna(selected_feature_set, cutoff_date, optuna_iterations):
    
    model_name = 'TimesNet'
    quantile = 0.5
    iter = optuna_iterations   
    
    def objective(trial: Trial) -> float:
        Y_hat_df_appended = pd.DataFrame()
        remove_lightning_log_folder()
        clear_cache()
        df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
        for i in range(forecasting_start, forecasting_period+1): 
            training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date, i)
            futr_list = futr_list_grouped_lists[i]
            params_optuna = {
            "h": i,
            "input_size": len(training_X_scaled_df)-1,
            # "batch_size" : trial.suggest_categorical('batch_size',[2, 4, 8, 16, 32, 64]),
            # "conv_hidden_size" : trial.suggest_categorical("conv_hidden_size",[4,8, 16,32,64]),
            # "hidden_size" : trial.suggest_categorical("hidden_size", [16,32,64,128]),
            'learning_rate': trial.suggest_categorical("learning_rate", [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, \
                                                                         0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, \
                                                                         0.001, 0.002, 0.003]),
            "max_steps" : trial.suggest_categorical("max_steps",[50,100,150,200,250,300]),
            # # "max_steps" : trial.suggest_categorical("max_steps",[5,10]),
            "num_kernels" : trial.suggest_int("num_kernels",2,12),
            'windows_batch_size' : trial.suggest_categorical("windows_batch_size", [2,16]),
            'inference_windows_batch_size' : trial.suggest_categorical("inference_windows_batch_size", [2,16]),
            "scaler_type" : trial.suggest_categorical("scaler_type",['robust','standard']),
            "top_k" : trial.suggest_int("top_k",1,5),
            "encoder_layers" : trial.suggest_int("encoder_layers",2,8),
            "dropout" : trial.suggest_categorical("dropout",[0.1, 0.2, 0.3]),
            "loss": QuantileLoss(0.5),
            "futr_exog_list" : futr_list,
            "random_seed": 42,
            # "start_padding_enabled" : trial.suggest_categorical("start_padding_enabled",[True,False]),    
            }
           
        # futr_list = None
            Y_hat_df_appended_vertical_mean_median_list=[]
            for cutoff_list in date_dict[i]:
                futr_df = testing_X_scaled_df.ffill().bfill()
        
                models = [TimesNet(**params_optuna)]
               
                nforecast = NeuralForecast(models=models, freq='M')
                nforecast.fit(df=training_X_scaled_df)
                
                Y_hat_df = nforecast.predict(futr_df=testing_X_scaled_df).reset_index()
                Y_hat_df.insert(1, 'LAG', i)
                Y_hat_df_appended = pd.concat([Y_hat_df_appended,Y_hat_df])    
                print('predicted by ','LAG ', i)
                Y_hat_df_appended_vertical = Y_hat_df_appended.pivot(index='ds', columns='LAG', values=f'{model_name}_ql{quantile}')
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical\
                                                         .assign(Median=Y_hat_df_appended_vertical.median(axis=1))
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'] = Y_hat_df_appended_vertical_mean_median['Median']
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Last_Lag'] = Y_hat_df_appended_vertical[Y_hat_df_appended_vertical.columns[-1]]
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical_mean_median.drop(['Median'], axis=1)
            
            df_temp_normal = df_temp[['ds','y']].set_index('ds')
            df_temp_normal = df_temp_normal.loc[(df_temp_normal.index > cutoff_date)][:i]
            y_valid = df_temp_normal['y'][:i].values
            y_pred = Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'][:i].values
            score = mean_absolute_error(np.array(y_pred), y_valid)
            if iter % 5 == 0:
                remove_lightning_log_folder()
                clear_cache()
                 
        return score
        
    torch.cuda.empty_cache()
    sampler = TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name='parameter_opt', direction='minimize',sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=iter, show_progress_bar=True)
    optuna_results = study.trials_dataframe()
    optuna_results.drop(columns=['datetime_start', 'duration', 'datetime_complete', 'state'], inplace=True)
    optuna_results = optuna_results.sort_values(by='value', ascending=True)
    optuna_best_parmas = study.best_params
    if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
        os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    optuna_results.to_csv(Path(f'{OPTUNA_DIR}/{daterecord}/{model_name}_{target_y_name}_LAG{forecasting_period}_CUTOFF{cutoff_list}_OPT_ITER{iter}_DATE{daterecord}.csv', index=False))

    return optuna_results

def tcn_optuna(selected_feature_set, cutoff_date, optuna_iterations):
    
    model_name = 'TCN'
    quantile = 0.5
    iter = optuna_iterations   
    
    def objective(trial: Trial) -> float:
        Y_hat_df_appended = pd.DataFrame()
        remove_lightning_log_folder()
        clear_cache()
        df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
        for i in range(forecasting_start, forecasting_period+1): 
            training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date, i)
            futr_list = futr_list_grouped_lists[i]
            params_optuna = {
            'h' : i,
            'input_size' : len(training_X_scaled_df)-1,
            'futr_exog_list': futr_list,
            'hist_exog_list' : None,
            'encoder_hidden_size' :  trial.suggest_int('encoder_hidden_size', 16,512),
            'decoder_hidden_size' :  trial.suggest_int('decoder_hidden_size', 16,512),
            'kernel_size' : trial.suggest_int('kernel_size',2,16),
            'decoder_layers' : trial.suggest_int('decoder_layers',2,16),
            'context_size' : trial.suggest_int('context_size',2,16),
            'dilations' : trial.suggest_categorical('dilations',[[1,2,4],[1,2,4,8],[1,2,4,8,16],[1,2,4,8,16,32]]),
            'encoder_activation' : trial.suggest_categorical("encoder_activation",['Tanh','LeakyReLU','ReLU','SELU']),
            'scaler_type' : trial.suggest_categorical("scaler_type",['robust','standard']),
            'loss': QuantileLoss(0.5),
            'learning_rate': trial.suggest_categorical("learning_rate", [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, \
                                                                         0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, \
                                                                         0.001, 0.002, 0.003]),
            'max_steps' : trial.suggest_categorical("max_steps",[50,100,150,200,250,300,350,400]),
            # 'val_check_steps' : trial.suggest_int('val_check_steps',4,32),     
            'random_seed': 42,
            }
           
        # futr_list = None
            Y_hat_df_appended_vertical_mean_median_list=[]
            for cutoff_list in date_dict[i]:
                futr_df = testing_X_scaled_df.ffill().bfill()
        
                models = [TCN(**params_optuna)]
               
                nforecast = NeuralForecast(models=models, freq='M')
                nforecast.fit(df=training_X_scaled_df)
                
                Y_hat_df = nforecast.predict(futr_df=testing_X_scaled_df).reset_index()
                Y_hat_df.insert(1, 'LAG', i)
                Y_hat_df_appended = pd.concat([Y_hat_df_appended,Y_hat_df])    
                print('predicted by ','LAG ', i)
                Y_hat_df_appended_vertical = Y_hat_df_appended.pivot(index='ds', columns='LAG', values=f'{model_name}_ql{quantile}')
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical\
                                                         .assign(Median=Y_hat_df_appended_vertical.median(axis=1))
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'] = Y_hat_df_appended_vertical_mean_median['Median']
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Last_Lag'] = Y_hat_df_appended_vertical[Y_hat_df_appended_vertical.columns[-1]]
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical_mean_median.drop(['Median'], axis=1)
            
            df_temp_normal = df_temp[['ds','y']].set_index('ds')
            df_temp_normal = df_temp_normal.loc[(df_temp_normal.index > cutoff_date)][:i]
            y_valid = df_temp_normal['y'][:i].values
            y_pred = Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'][:i].values
            score = mean_absolute_error(np.array(y_pred), y_valid)
            if iter % 5 == 0:
                remove_lightning_log_folder()
                clear_cache()
            else:
                pass
                 
        return score
        
    torch.cuda.empty_cache()
    sampler = TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name='parameter_opt', direction='minimize',sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=iter, show_progress_bar=True)
    optuna_results = study.trials_dataframe()
    optuna_results.drop(columns=['datetime_start', 'duration', 'datetime_complete', 'state'], inplace=True)
    optuna_results = optuna_results.sort_values(by='value', ascending=True)
    optuna_best_parmas = study.best_params
    if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
        os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    optuna_results.to_csv(Path(f'{OPTUNA_DIR}/{daterecord}/{model_name}_{target_y_name}_LAG{forecasting_period}_CUTOFF{cutoff_list}_OPT_ITER{iter}_DATE{daterecord}.csv', index=False))

    return optuna_results

def nhits_optuna(selected_feature_set, cutoff_date, optuna_iterations):
    
    model_name = 'NHITS'
    quantile = 0.5
    iter = optuna_iterations   
    
    def objective(trial: Trial) -> float:
        Y_hat_df_appended = pd.DataFrame()
        remove_lightning_log_folder()
        clear_cache()
        df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
        for i in range(forecasting_period, forecasting_period+1): 
            training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date, i)
            futr_list = futr_list_grouped_lists[i]
            params_optuna = {
            "h": i,
            "input_size": len(training_X_scaled_df)-1,
            "hist_exog_list" : None,
            "futr_exog_list" : futr_list,
            "n_blocks" : trial.suggest_categorical('n_blocks', [[1, 1, 1], [1, 1, 2], [2, 1, 1] , [1, 2, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 2]]),
            "mlp_units" : trial.suggest_categorical('mlp_units', [[[8, 8], [8, 8], [8, 8]],
                                                                  [[16, 16], [16, 16], [16, 16]],
                                                                  [[32, 32], [32, 32], [32, 32]],
                                                                  [[64, 64], [64, 64], [64, 64]], 
                                                                  [[128, 128], [128, 128], [128, 128]],
                                                                  [[256, 256], [256, 256], [256, 256]]]),
            "n_pool_kernel_size" : trial.suggest_categorical('n_pool_kernerl_size', [[1, 1, 1], [1, 1, 2], [2, 1, 1] , [1, 2, 1], [1, 2, 2], [2, 1, 2], [2, 2, 1], [2, 2, 2]]),
            "n_freq_downsample" : trial.suggest_categorical('n_freq_downsample', [[4, 2, 2], [2, 1, 1], [4, 2, 1], [2, 2, 2], [2, 2, 1]]),
            'scaler_type' : trial.suggest_categorical("scaler_type",['robust','standard']),
            "pooling_mode" : 'MaxPool1d',
            "activation": trial.suggest_categorical("activation", ['LeakyReLU','SELU','ReLU']),
            # "activation": 'ReLU',    
            "batch_size" : trial.suggest_categorical('batch_size',[4, 8, 16, 32]),
            "max_steps" : trial.suggest_categorical("max_steps",[50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]),
            "random_seed": 42,
            "loss": QuantileLoss(0.5),
            'learning_rate': trial.suggest_categorical("learning_rate", [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, \
                                                                         0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, \
                                                                         0.001, 0.002, 0.003]),
            }     
        # futr_list = None
            Y_hat_df_appended_vertical_mean_median_list=[]
            for cutoff_list in date_dict[i]:
                futr_df = testing_X_scaled_df.ffill().bfill()
        
                models = [NHITS(**params_optuna)]
               
                nforecast = NeuralForecast(models=models, freq='M')
                nforecast.fit(df=training_X_scaled_df)
                
                Y_hat_df = nforecast.predict(futr_df=testing_X_scaled_df).reset_index()
                Y_hat_df.insert(1, 'LAG', i)
                Y_hat_df_appended = pd.concat([Y_hat_df_appended,Y_hat_df])    
                print('predicted by ','LAG ', i)
                Y_hat_df_appended_vertical = Y_hat_df_appended.pivot(index='ds', columns='LAG', values=f'{model_name}_ql{quantile}')
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical\
                                                         .assign(Median=Y_hat_df_appended_vertical.median(axis=1))
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'] = Y_hat_df_appended_vertical_mean_median['Median']
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Last_Lag'] = Y_hat_df_appended_vertical[Y_hat_df_appended_vertical.columns[-1]]
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical_mean_median.drop(['Median'], axis=1)
            
            df_temp_normal = df_temp[['ds','y']].set_index('ds')
            df_temp_normal = df_temp_normal.loc[(df_temp_normal.index > cutoff_date)][:i]
            y_valid = df_temp_normal['y'][:i].values
            y_pred = Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'][:i].values
            score = mean_absolute_error(np.array(y_pred), y_valid)
                 
        return score
        
    torch.cuda.empty_cache()
    sampler = TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name='parameter_opt', direction='minimize',sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=iter, show_progress_bar=True)
    optuna_results = study.trials_dataframe()
    optuna_results.drop(columns=['datetime_start', 'duration', 'datetime_complete', 'state'], inplace=True)
    optuna_results = optuna_results.sort_values(by='value', ascending=True)
    optuna_best_parmas = study.best_params
    if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
        os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    optuna_results.to_csv(Path(f'{OPTUNA_DIR}/{daterecord}/{model_name}_{target_y_name}_LAG{forecasting_period}_CUTOFF{cutoff_list}_OPT_ITER{iter}_DATE{daterecord}.csv', index=False))

    return optuna_results

def fedformer_optuna(selected_feature_set, cutoff_date, optuna_iterations):
    
    model_name = 'FEDformer'
    quantile = 0.5
    iter = optuna_iterations   
    
    def objective(trial: Trial) -> float:
        Y_hat_df_appended = pd.DataFrame()
        remove_lightning_log_folder()
        clear_cache()
        df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
        for i in range(forecasting_start, forecasting_period+1): 
            training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date, i)
            futr_list = futr_list_grouped_lists[i]
            params_optuna = {
            'h' : i,
            'input_size' : len(training_X_scaled_df)-1,
            'futr_exog_list': futr_list,
            'hist_exog_list' : None,
            'modes' : trial.suggest_categorical("modes",[4, 8, 16, 32, 64]),
            'hidden_size' : trial.suggest_categorical('hidden_size',[2,4,8,16,32,64,128]),
            'encoder_layers' :  trial.suggest_categorical('encoder_hidden_size', [2,4,8,16,32,64,128]),
            'decoder_layers' :  trial.suggest_categorical('decoder_hidden_size', [2,4,8,16,32,64,128]),
            'dropout' : trial.suggest_categorical("dropout",[0.1, 0.2, 0.3, 0.4, 0.5]),
            'conv_hidden_size' : trial.suggest_categorical("conv_hidden_size",[4, 8, 16, 32, 64, 128]),
            'inference_windows_batch_size' : trial.suggest_categorical("inference_windows_batch_size",[2, 4, 8, 16, 25]),
            'activation' : trial.suggest_categorical("encoder_activation",['relu','gelu']),
            'scaler_type' : trial.suggest_categorical("scaler_type",['robust','standard']),
            'loss': QuantileLoss(0.5),
            'learning_rate': trial.suggest_categorical("learning_rate", [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, \
                                                                         0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, \
                                                                         0.001, 0.002, 0.003]),
            'max_steps' : trial.suggest_categorical("max_steps",[50,100,150,200,250,300,350,400]),
            'val_check_steps' : trial.suggest_int('val_check_steps',4,32),     
            'random_seed': 42,
            }
           
        # futr_list = None
            Y_hat_df_appended_vertical_mean_median_list=[]
            for cutoff_list in date_dict[i]:
                futr_df = testing_X_scaled_df.ffill().bfill()
        
                models = [FEDformer(**params_optuna)]
               
                nforecast = NeuralForecast(models=models, freq='M')
                nforecast.fit(df=training_X_scaled_df)
                
                Y_hat_df = nforecast.predict(futr_df=testing_X_scaled_df).reset_index()
                Y_hat_df.insert(1, 'LAG', i)
                Y_hat_df_appended = pd.concat([Y_hat_df_appended,Y_hat_df])    
                print('predicted by ','LAG ', i)
                Y_hat_df_appended_vertical = Y_hat_df_appended.pivot(index='ds', columns='LAG', values=f'{model_name}_ql{quantile}')
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical\
                                                         .assign(Median=Y_hat_df_appended_vertical.median(axis=1))
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'] = Y_hat_df_appended_vertical_mean_median['Median']
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Last_Lag'] = Y_hat_df_appended_vertical[Y_hat_df_appended_vertical.columns[-1]]
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical_mean_median.drop(['Median'], axis=1)
            
            df_temp_normal = df_temp[['ds','y']].set_index('ds')
            df_temp_normal = df_temp_normal.loc[(df_temp_normal.index > cutoff_date)][:i]
            y_valid = df_temp_normal['y'][:i].values
            y_pred = Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'][:i].values
            score = mean_absolute_error(np.array(y_pred), y_valid)
                 
        return score
        
    torch.cuda.empty_cache()
    sampler = TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name='parameter_opt', direction='minimize',sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=iter, show_progress_bar=True)
    optuna_results = study.trials_dataframe()
    optuna_results.drop(columns=['datetime_start', 'duration', 'datetime_complete', 'state'], inplace=True)
    optuna_results = optuna_results.sort_values(by='value', ascending=True)
    optuna_best_parmas = study.best_params
    if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
        os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    optuna_results.to_csv(Path(f'{OPTUNA_DIR}/{daterecord}/{model_name}_{target_y_name}_LAG{forecasting_period}_CUTOFF{cutoff_list}_OPT_ITER{iter}_DATE{daterecord}.csv', index=False))

    return optuna_results

def stemgnn_optuna(selected_feature_set, cutoff_date, optuna_iterations):
    
    model_name = 'StemGNN'
    quantile = 0.5
    iter = optuna_iterations   
    
    def objective(trial: Trial) -> float:
        Y_hat_df_appended = pd.DataFrame()
        remove_lightning_log_folder()
        clear_cache()
        df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
        for i in range(forecasting_start, forecasting_period+1): 
            training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date, i)
            futr_list = futr_list_grouped_lists[i]
            params_optuna = {
            'h' : i,
            'input_size' : len(training_X_scaled_df)-1,
            'futr_exog_list': futr_list,
            'hist_exog_list' : None,
            'n_series' : 1,
            'loss': QuantileLoss(0.5),
            'learning_rate': trial.suggest_categorical("learning_rate", [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, \
                                                                         0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, \
                                                                         0.001, 0.002, 0.003]),
            'max_steps' : trial.suggest_categorical("max_steps",[50,100,150,200,250,300,350,400]),
            'val_check_steps' : trial.suggest_int('val_check_steps',4,32),     
            'random_seed': 42,
            }
           
        # futr_list = None
            Y_hat_df_appended_vertical_mean_median_list=[]
            for cutoff_list in date_dict[i]:
                futr_df = testing_X_scaled_df.ffill().bfill()
        
                models = [StemGNN(**params_optuna)]
               
                nforecast = NeuralForecast(models=models, freq='M')
                nforecast.fit(df=training_X_scaled_df)
                
                Y_hat_df = nforecast.predict(futr_df=testing_X_scaled_df).reset_index()
                Y_hat_df.insert(1, 'LAG', i)
                Y_hat_df_appended = pd.concat([Y_hat_df_appended,Y_hat_df])    
                print('predicted by ','LAG ', i)
                Y_hat_df_appended_vertical = Y_hat_df_appended.pivot(index='ds', columns='LAG', values=f'{model_name}_ql{quantile}')
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical\
                                                         .assign(Median=Y_hat_df_appended_vertical.median(axis=1))
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'] = Y_hat_df_appended_vertical_mean_median['Median']
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Last_Lag'] = Y_hat_df_appended_vertical[Y_hat_df_appended_vertical.columns[-1]]
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical_mean_median.drop(['Median'], axis=1)
            
            df_temp_normal = df_temp[['ds','y']].set_index('ds')
            df_temp_normal = df_temp_normal.loc[(df_temp_normal.index > cutoff_date)][:i]
            y_valid = df_temp_normal['y'][:i].values
            y_pred = Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'][:i].values
            score = mean_absolute_error(np.array(y_pred), y_valid)
                 
        return score
        
    torch.cuda.empty_cache()
    sampler = TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name='parameter_opt', direction='minimize',sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=iter, show_progress_bar=True)
    optuna_results = study.trials_dataframe()
    optuna_results.drop(columns=['datetime_start', 'duration', 'datetime_complete', 'state'], inplace=True)
    optuna_results = optuna_results.sort_values(by='value', ascending=True)
    optuna_best_parmas = study.best_params
    if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
        os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    optuna_results.to_csv(Path(f'{OPTUNA_DIR}/{daterecord}/{model_name}_{target_y_name}_LAG{forecasting_period}_CUTOFF{cutoff_list}_OPT_ITER{iter}_DATE{daterecord}.csv', index=False))

    return optuna_results

def dlinear_optuna(selected_feature_set, cutoff_date, optuna_iterations):
    
    model_name = 'DLinear'
    quantile = 0.5
    iter = optuna_iterations   
    
    def objective(trial: Trial) -> float:
        Y_hat_df_appended = pd.DataFrame()
        remove_lightning_log_folder()
        clear_cache()
        df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
        for i in range(forecasting_start, forecasting_period+1): 
            training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date, i)
            futr_list = futr_list_grouped_lists[i]
            params_optuna = {
            'h' : i,
            'input_size' : len(training_X_scaled_df)-1,
            'batch_size' : trial.suggest_categorical('batch_size',[2,4,8,16,32,64]),
            # 'futr_exog_list': futr_list,
            'hist_exog_list' : None,
            'loss': QuantileLoss(0.5),
            'learning_rate': trial.suggest_categorical("learning_rate", [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, \
                                                                         0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, \
                                                                         0.001, 0.002, 0.003]),
            'max_steps' : trial.suggest_categorical("max_steps",[50,100,150,200,250,300,350,400]),
            'val_check_steps' : trial.suggest_int('val_check_steps',4,16),     
            'random_seed': 42,
            }
           
        # futr_list = None
            Y_hat_df_appended_vertical_mean_median_list=[]
            for cutoff_list in date_dict[i]:
                futr_df = testing_X_scaled_df.ffill().bfill()
        
                models = [DLinear(**params_optuna)]
               
                nforecast = NeuralForecast(models=models, freq='M')
                nforecast.fit(df=training_X_scaled_df)
                
                Y_hat_df = nforecast.predict(futr_df=testing_X_scaled_df).reset_index()
                Y_hat_df.insert(1, 'LAG', i)
                Y_hat_df_appended = pd.concat([Y_hat_df_appended,Y_hat_df])    
                print('predicted by ','LAG ', i)
                Y_hat_df_appended_vertical = Y_hat_df_appended.pivot(index='ds', columns='LAG', values=f'{model_name}_ql{quantile}')
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical\
                                                         .assign(Median=Y_hat_df_appended_vertical.median(axis=1))
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'] = Y_hat_df_appended_vertical_mean_median['Median']
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Last_Lag'] = Y_hat_df_appended_vertical[Y_hat_df_appended_vertical.columns[-1]]
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical_mean_median.drop(['Median'], axis=1)
            
            df_temp_normal = df_temp[['ds','y']].set_index('ds')
            df_temp_normal = df_temp_normal.loc[(df_temp_normal.index > cutoff_date)][:i]
            y_valid = df_temp_normal['y'][:i].values
            y_pred = Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'][:i].values
            score = mean_absolute_error(np.array(y_pred), y_valid)
                 
        return score
        
    torch.cuda.empty_cache()
    sampler = TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name='parameter_opt', direction='minimize',sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=iter, show_progress_bar=True)
    optuna_results = study.trials_dataframe()
    optuna_results.drop(columns=['datetime_start', 'duration', 'datetime_complete', 'state'], inplace=True)
    optuna_results = optuna_results.sort_values(by='value', ascending=True)
    optuna_best_parmas = study.best_params
    if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
        os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    optuna_results.to_csv(Path(f'{OPTUNA_DIR}/{daterecord}/{model_name}_{target_y_name}_LAG{forecasting_period}_CUTOFF{cutoff_list}_OPT_ITER{iter}_DATE{daterecord}.csv', index=False))

    return optuna_results

def tsmixerx_optuna(selected_feature_set, cutoff_date, optuna_iterations):
    
    model_name = 'TSMixerx'
    quantile = 0.5
    iter = optuna_iterations   
    
    def objective(trial: Trial) -> float:
        Y_hat_df_appended = pd.DataFrame()
        remove_lightning_log_folder()
        remove_chekpoint_tmp_trainer_folder()
        clear_cache()
        df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
        for i in range(forecasting_period, forecasting_period+1): 
            training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date, i)
            futr_list = futr_list_grouped_lists[i]
            params_optuna = {
            'h' : i,
            'input_size' : len(training_X_scaled_df)-1,
            'batch_size' : trial.suggest_categorical('batch_size',[2,4,8,16,32,64]),
            'n_series' : 1,
            'futr_exog_list': futr_list,
            'revin' : trial.suggest_categorical("revin",[True, False]),
            'hist_exog_list' : None,
            'loss': QuantileLoss(0.5),
            'valid_loss' : QuantileLoss(0.5),
            'learning_rate': trial.suggest_categorical("learning_rate", [0.00001, 0.00003, 0.00005, 0.00007, 0.00009, \
                                                                         0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, \
                                                                         0.001, 0.002, 0.003]),
            'max_steps' : trial.suggest_categorical("max_steps",[50,100,150,200,250,300,350,400]),
            'n_block' : trial.suggest_int('n_block',2,8),
            'ff_dim' : trial.suggest_int('ff_dim',2,8),
            'val_check_steps' : trial.suggest_int('val_check_steps',3,13),
            'scaler_type' : trial.suggest_categorical("scaler_type",['robust','standard','identity']),
            "dropout" : trial.suggest_categorical("dropout",[0.1, 0.2, 0.3, 0.4, 0.5]),
            'random_seed': 42,
            }
           
        # futr_list = None
            Y_hat_df_appended_vertical_mean_median_list=[]
            for cutoff_list in date_dict[i]:
                futr_df = testing_X_scaled_df.ffill().bfill()
        
                models = [TSMixerx(**params_optuna)]
               
                nforecast = NeuralForecast(models=models, freq='M')
                nforecast.fit(df=training_X_scaled_df)
                
                Y_hat_df = nforecast.predict(futr_df=testing_X_scaled_df).reset_index()
                Y_hat_df.insert(1, 'LAG', i)
                Y_hat_df_appended = pd.concat([Y_hat_df_appended,Y_hat_df])    
                print('predicted by ','LAG ', i)
                Y_hat_df_appended_vertical = Y_hat_df_appended.pivot(index='ds', columns='LAG', values=f'{model_name}_ql{quantile}')
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical\
                                                         .assign(Median=Y_hat_df_appended_vertical.median(axis=1))
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'] = Y_hat_df_appended_vertical_mean_median['Median']
                Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Last_Lag'] = Y_hat_df_appended_vertical[Y_hat_df_appended_vertical.columns[-1]]
                Y_hat_df_appended_vertical_mean_median = Y_hat_df_appended_vertical_mean_median.drop(['Median'], axis=1)
            
            df_temp_normal = df_temp[['ds','y']].set_index('ds')
            df_temp_normal = df_temp_normal.loc[(df_temp_normal.index > cutoff_date)][:i]
            y_valid = df_temp_normal['y'][:i].values
            y_pred = Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'][:i].values
            score = mean_absolute_error(np.array(y_pred), y_valid)
                 
        return score
        
    torch.cuda.empty_cache()
    sampler = TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(study_name='parameter_opt', direction='minimize',sampler=sampler, pruner=pruner)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=iter, show_progress_bar=True)
    optuna_results = study.trials_dataframe()
    optuna_results.drop(columns=['datetime_start', 'duration', 'datetime_complete', 'state'], inplace=True)
    optuna_results = optuna_results.sort_values(by='value', ascending=True)
    optuna_best_parmas = study.best_params
    if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
        os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    optuna_results.to_csv(Path(f'{OPTUNA_DIR}/{daterecord}/{model_name}_{target_y_name}_LAG{forecasting_period}_CUTOFF{cutoff_list}_OPT_ITER{iter}_DATE{daterecord}.csv', index=False))

    return optuna_results