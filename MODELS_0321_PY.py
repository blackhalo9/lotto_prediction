from PKG_0321.PKGS_0321_PY import *
from PKG_0321.INPUT_0321_PY import *
from PKG_0321.UTILS_0321_PY import *
from PKG_0321.PREPROCESSING_0321_PY import *
from PKG_0321.FEATURE_IMPORTANCE_0321_PY import *
from PKG_0321.ENSEMBLE_LIST_0321_PY import *
from PKG_0321.FUTURE_FRAME_0321_PY import *
from PKG_0321.DATE_DICT_0321_PY import *
from PKG_0321.REPORT_0321_PY import *
from PKG_0321.MODEL_DLINEAR_0321_PY import *
from PKG_0321.MODEL_NLINEAR_0321_PY import *
from PKG_0321.MODEL_TSMIXERX_0321_PY import *
from PKG_0321.MODEL_TIMELLM_0321_PY import *

def autoformer_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'Autoformer'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set,cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [Autoformer(h=i,
                     input_size=len(training_X_scaled_df)-1,
                     hidden_size = 16,
                     conv_hidden_size = 32,
                     n_head=2,
                     loss=QuantileLoss(quantile),
                     futr_exog_list=futr_list,
                     scaler_type='robust',
                     activation='relu',
                     learning_rate=1e-4,
                     max_steps=200,
                     random_seed=42
                                )
                     ]
    
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def timesnet_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'TimesNet'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [
                     TimesNet(h=i,
                     input_size=len(training_X_scaled_df)-1,
                     batch_size = 32,
                     conv_hidden_size = 128,
                     dropout = 0.1,
                     encoder_layers = 4,
                     hidden_size = 128,
                     inference_windows_batch_size = 16,
                     learning_rate=0.001,
                     max_steps= 200,
                     num_kernels = 4,
                     scaler_type='standard',
                     top_k = 2,
                     windows_batch_size = 16,
                     loss = QuantileLoss(quantile),
                     random_seed=42,
                     futr_exog_list=futr_list,
                    ),
                     ]
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def tcn_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'TCN'
    quantile = quantile

    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set,cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models=[TCN(h=i,
                        input_size=len(training_X_scaled_df)-1,
                        # loss=DistributionLoss(distribution='Normal', level=[80, 90]),
                        # loss=GMM(n_components=7, return_params=True, level=[80,90]),
                        loss=QuantileLoss(quantile),
                        context_size=3,
                        # decoder_hidden_size=443,
                        # dilations=[1,2,4,8,16],
                        encoder_activation='ReLU',
                        # encoder_hidden_size=195,
                        kernel_size=2,
                        learning_rate=0.0001,
                        decoder_layers=2,
                        max_steps=200,
                        scaler_type='robust',
                        futr_exog_list=futr_list,
                        random_seed=42,
                        # hist_exog_list=None,
                        # stat_exog_list=None
                        )]  
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def nhits_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'NHITS'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_period,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set,cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [
                     NHITS(input_size=len(training_X_scaled_df)-1,
                     h=i,
                     stat_exog_list = None,
                     hist_exog_list = None,
                     futr_exog_list = futr_list,
                     # batch_size=64,
                     # learning_rate=0.0001,
                     loss=QuantileLoss(quantile),
                     max_steps=200,
                     # mlp_units = [[256, 256], [256, 256], [256, 256]],
                     # # mlp_units = [[128, 128], [128, 128], [128, 128]],
                     # mlp_units = [[512, 512], [512, 512], [512, 512]],
                     # n_blocks = [1, 1, 1],
                     # n_freq_downsample=[2, 2, 2],
                     # n_pool_kernel_size = [2, 1, 2],
                     # scaler_type = 'robust',
                     # # learning_rate=1e-4,
                     # pooling_mode = 'MaxPool1d',
                     # activation='ReLU',
                     random_seed=42
                           ),
                         ]  
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def informer_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'Informer'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set,cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [
                      Informer(h=i,
                      input_size=len(training_X_scaled_df)-1,
                      hidden_size = 16,
                      conv_hidden_size = 32,
                      n_head = 2,
                      #loss=DistributionLoss(distribution='StudentT', level=[80, 90]),
                      loss=QuantileLoss(quantile),
                      futr_exog_list=futr_list,
                      scaler_type='robust',
                      learning_rate=1e-3,
                      max_steps=200,
                      random_seed=42
                              ),
                         ]  
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def vanillatransformer_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'VanillaTransformer'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date,i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [
                      VanillaTransformer(h=i,
                      input_size=len(training_X_scaled_df)-1,
                      hidden_size=16,
                      conv_hidden_size=32,
                      n_head=2,
                      loss=QuantileLoss(quantile),
                      futr_exog_list=futr_list,
                      scaler_type='robust',
                      learning_rate=1e-3,
                      max_steps=200,
                      random_seed=42
                              ),
                         ]  
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def tft_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'TFT'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set,cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [
                     TFT(h=i, 
                     input_size=len(training_X_scaled_df)-1,
                     hidden_size=20,
                     loss=QuantileLoss(quantile),
                     learning_rate=0.005,
                     stat_exog_list=None,
                     hist_exog_list=None,
                     max_steps=200,
                     scaler_type='robust',
                     windows_batch_size=None,
                     enable_progress_bar=True,
                     random_seed=42
                              ),
                         ]   
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def patchtst_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'PatchTST'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set,cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [PatchTST(h=i,
                             input_size=len(training_X_scaled_df)-1,
                             patch_len=24,
                             stride=24,
                             revin=False,
                             hidden_size=16,
                             n_heads=4,
                             scaler_type='standard',
                             loss=QuantileLoss(quantile),
                             learning_rate=1e-4,
                             activation = 'ReLU',
                             max_steps=100,
                             random_seed=42
                              ),
                         ] 
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            # print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def fedformer_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'FEDformer'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_period, forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [FEDformer(h=i,
                             input_size=len(training_X_scaled_df)-1,
                             futr_exog_list = futr_list,
                             loss=QuantileLoss(quantile),
                             learning_rate=1e-4,
                             activation = 'relu',
                             max_steps=100,
                             random_seed=42
                              ),
                         ]
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def stemgnn_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'StemGNN'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set, cutoff_date,i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [StemGNN(h=i,
                     n_series=1,             
                     input_size=len(training_X_scaled_df)-1,
                     futr_exog_list = futr_list,
                     scaler_type='robust',
                     max_steps=200,
                     learning_rate=1e-3,
                     loss=QuantileLoss(quantile),
                     batch_size=32,
                     random_seed=42
                     ),]
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def dlinear_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'DLinear'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set,cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
            models = [DLinear(h=i,
                     input_size=len(training_X_scaled_df)-1,
                     # futr_exog_list = futr_list,
                     loss=QuantileLoss(quantile),
                     #loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
                     scaler_type='robust',
                     learning_rate=1e-3,
                     max_steps=200,
                     val_check_steps=5,
                     random_seed=42
                     )]
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def nlinear_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'NLinear'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set,cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
            models = [NLinear(h=i,
                     input_size=len(training_X_scaled_df)-1,
                     # futr_exog_list = futr_list,
                     loss=QuantileLoss(quantile),
                     #loss=DistributionLoss(distribution='StudentT', level=[80, 90], return_params=True),
                     scaler_type='robust',
                     learning_rate=1e-3,
                     max_steps=200,
                     val_check_steps=5,
                     random_seed=42
                     )]
            
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median

def tsmixerx_model(selected_feature_set, cutoff_date, forecasting_period, quantile):
    model_name = 'TSMixerx'
    quantile = quantile
    quantile_appended = pd.DataFrame()
    Y_hat_df_appended = pd.DataFrame()
    
    selected_feature_set = selected_feature_set
    remove_lightning_log_folder()
    remove_chekpoint_tmp_trainer_folder()
    
    for i in range(forecasting_start,forecasting_period+1):
        
        training_X_scaled_df, testing_X_scaled_df, futr_list_grouped_lists  = raw_shifted_df(selected_feature_set,cutoff_date, i)
        futr_list = futr_list_grouped_lists[i]
        # futr_list = None
        
        for cutoff_list in date_dict[i]:
            futr_df = testing_X_scaled_df.ffill().bfill()
    
            models = [TSMixerx(h=i,
                            input_size=len(training_X_scaled_df)-1,
                            n_series=1,
                            futr_exog_list=futr_list,
                            n_block=4,
                            ff_dim=4,
                            revin=True,
                            scaler_type='standard',
                            max_steps=200,
                            # # early_stop_patience_steps=-1,
                            learning_rate=1e-3,
                            loss=QuantileLoss(quantile),
                            # valid_loss=MAE(),
                            batch_size=32,
                            random_seed=42
                            )]
    
            nforecast = NeuralForecast(models=models, freq='M')
            nforecast.fit(df=training_X_scaled_df)
    
            print(futr_list)
            
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
            
            if not os.path.exists(Path(f"{OPTUNA_DIR}/{daterecord}")):
                os.mkdir(Path(f"{OPTUNA_DIR}/{daterecord}"))
    
            if not os.path.exists(Path(f"{RESULTS_DIR}/{daterecord}")):
                os.mkdir(Path(f"{RESULTS_DIR}/{daterecord}"))
                
            Y_hat_df_appended_vertical_mean_median.to_csv(Path(f"{RESULTS_DIR}/{daterecord}/{model_name}_{target_y_name}_{cutoff_date}_{quantile}.csv", index=True))
    return png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile), Y_hat_df_appended_vertical_mean_median