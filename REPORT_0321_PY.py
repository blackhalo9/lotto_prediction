from PKG_0321.PKGS_0321_PY import *
from PKG_0321.INPUT_0321_PY import *
from PKG_0321.UTILS_0321_PY import *
from PKG_0321.PREPROCESSING_0321_PY import *
from PKG_0321.FEATURE_IMPORTANCE_0321_PY import *
from PKG_0321.ENSEMBLE_LIST_0321_PY import *
from PKG_0321.FUTURE_FRAME_0321_PY import *

def plot_comparison_forecasting(model_keyword, target_y_name, quantile):
    model_keyword, comparison_final_df, quantile = make_forecasting_comparison_table(model_keyword, target_y_name, quantile)
    new_column_names = {}
    for col in comparison_final_df.columns:
        try:
            col_date = pd.to_datetime(col)
            new_column_names[col] = col_date.replace(day=1).strftime('%Y-%m-%d')
        except ValueError:
            new_column_names[col] = col
    comparison_final_df.rename(columns=new_column_names, inplace=True)
    columns_with_date = sorted([col for col in comparison_final_df.columns if '-' in col])
    
    df_old = comparison_final_df
    df_old = df_old[df_old.index >= '2020-01-01']
    
    fig, ax = plt.subplots(1, 1, figsize = (20, 7))
    
    df_old[[f'{target_y_name} Actual Price',f'{columns_with_date[0]}',f'{columns_with_date[-1]}']].plot(ax=ax, linewidth=2.5, marker='o', markersize=6)
    for date_column in columns_with_date[1:-1]:
        df_old[date_column].plot(ax=ax, linewidth=1, marker='o', markersize=3)
    
    plt.axvline(columns_with_date[-1], color='red',linewidth=3, linestyle='--')
    ax.set_title(f"{target_y_name} Forecasting Comparison by Cutoff ({model_keyword} Median List w/ Feature Ensemble)", fontsize=18)
    ax.set_ylabel(f'{target_y_name} PRICE', fontsize=16)
    ax.set_xlabel('Date', fontsize=16)
    ax.legend(prop={'size': 16}, loc=2)
    ax.grid()

    if not os.path.exists(Path(f"{PNG_DIR}/{daterecord}")):
        os.mkdir(Path(f"{PNG_DIR}/{daterecord}"))
    if cutoff_date >= last_month_updated:
        fig.savefig(Path(f"{PNG_DIR}/{daterecord}/COMPARISON_{model_keyword}_{target_y_name}_{quantile}_{cutoff_date}.png", dpi=100))
    else:
        fig.savefig(Path(f"{PNG_DIR}/{daterecord}/COMPARISON_{model_keyword}_{target_y_name}_{quantile}_{cutoff_date}.png", dpi=100))

    return comparison_final_df.tail(26)

def plot_forecasting(model_keyword, target_y_name, quantile):
    model_keyword, comparison_final_df, quantile = make_forecasting_comparison_table(model_keyword, target_y_name, quantile)

    new_column_names = {}
    for col in comparison_final_df.columns:
        try:
            col_date = pd.to_datetime(col)
            new_column_names[col] = col_date.replace(day=1).strftime('%Y-%m-%d')
        except ValueError:
            new_column_names[col] = col
    comparison_final_df.rename(columns=new_column_names, inplace=True)
    columns_with_date = sorted([col for col in comparison_final_df.columns if '-' in col])

    df_old = comparison_final_df
    df_old = df_old[df_old.index >= '2020-01-01']
     
    fig, ax = plt.subplots(1, 1, figsize = (20, 7))
    
    df_old[[f'{target_y_name} Actual Price',f'{columns_with_date[0]}',f'{columns_with_date[-1]}']].plot(ax=ax, linewidth=2.5, marker='o', markersize=6)
    
    plt.axvline(columns_with_date[-1], color='red',linewidth=3, linestyle='--')
    ax.set_title(f"{target_y_name} Forecasting ({model_keyword} Median List w/ Feature Ensemble)", fontsize=18)
    ax.set_ylabel(f'{target_y_name} PRICE', fontsize=16)
    ax.set_xlabel('Date', fontsize=16)
    ax.legend(prop={'size': 16}, loc=2)
    ax.grid()

    if not os.path.exists(Path(f"{PNG_DIR}/{daterecord}")):
        os.mkdir(Path(f"{PNG_DIR}/{daterecord}"))
    if cutoff_date >= last_month_updated:
        fig.savefig(Path(f"{PNG_DIR}/{daterecord}/FORCASTING_{model_keyword}_{target_y_name}_{quantile}_{cutoff_date}.png", dpi=100))
    else:
        fig.savefig(Path(f"{PNG_DIR}/{daterecord}/FORCASTING_{model_keyword}_{target_y_name}_{quantile}_{cutoff_date}.png", dpi=100))

def png_func(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile):

    df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
    fi_training_X_scaled_df = raw_fi_df(df_scaler_fi)
    df_temp_scaled = raw_y_scaled_df(cutoff_date)
    
    fig, ax = plt.subplots(1, 1, figsize = (20, 7))

    if choose_y_train_scaler == False:
        if len(df_scaler_total.loc[(df_scaler_total.ds > cutoff_date)]) > 1:
            plot_df = pd.concat([df_temp.set_index('ds'), Y_hat_df_appended_vertical_mean_median], join='inner')
        else:
            plot_df = pd.merge(df_temp.set_index('ds'), Y_hat_df_appended_vertical_mean_median, left_index=True, right_index=True, how='outer')
    if choose_y_train_scaler == True:
        if len(df_scaler_total.loc[(df_scaler_total.ds > cutoff_date)]) > 1:
            plot_df = pd.concat([df_temp_scaled.set_index('ds')['y'], Y_hat_df_appended_vertical_mean_median], join='inner')
        else:
            plot_df = pd.merge(df_temp_scaled.set_index('ds')['y'], Y_hat_df_appended_vertical_mean_median, left_index=True, right_index=True, how='outer')
    
    plot_df = plot_df[plot_df.index >= '2020-01-01']
    
    # plot_df = plot_df.rename(columns={f'{model_name}_0.5_Last_Lag' : f'{model_name} Last Lag - {i} Month Forecast'})
    plot_df = plot_df.rename(columns={'y' : f'{target_y_name} Actual Price',
                                      f'{model_name}_{quantile}_Median' : f'{model_name} Median List - {forecasting_period} Month Forecast',
                                      # f'{model_name}_0.5_Mean' : f'{model_name} Mean List - {i} Month Forecast', 
                                      f'{model_name}_{quantile}_Last_Lag' : f'{model_name} Last Lag - {forecasting_period} Month Forecast'},
                            )
                                          # f'{model_name}_0.5_Mean' : f'{model_name} Mean List - {i} Month Forecast',)
    # plot_df = plot_df.rename(columns={f'{model_name}_Median' : f'{model_name} Forecast - Median'})
    # plot_df = plot_df.rename(columns={f'{model_name}_Combined' : f'{model_name} Forecast - Combined'})
    plot_df_clear = plot_df[[f'{target_y_name} Actual Price',
                             f'{model_name} Median List - {forecasting_period} Month Forecast'
                            ]]
    plot_df[[f'{target_y_name} Actual Price', f'{model_name} Last Lag - {forecasting_period} Month Forecast', f'{model_name} Median List - {forecasting_period} Month Forecast',
             # f'{model_name} Mean List - {i} Month Forecast',
            ]]\
    .plot(ax=ax, linewidth=2.5, marker='o', markersize=8)
    # plot_df[[f'{target_y_name} Actual Price',f'{model_name} Forecast - Mean', f'{model_name} Forecast - Median']] \
    # .plot(ax=ax, linewidth=2, marker='o')
    
    plt.axvline(cutoff_date, color='red',linewidth=3, linestyle='--')
    ax.set_title(f'{target_y_name} Actual and Predicted Plot - Cutoff: {cutoff_date}', fontsize=22)
    ax.set_ylabel(f'{target_y_name}PRICE', fontsize=20)
    ax.set_xlabel('Date', fontsize=20)
    ax.legend(prop={'size': 15})
    ax.grid()
    
    if not os.path.exists(Path(f"{PNG_DIR}/{daterecord}")):
        os.mkdir(Path(f"{PNG_DIR}/{daterecord}"))
    if cutoff_date >= last_month_updated:
        fig.savefig(Path(f"{PNG_DIR}/{daterecord}/{model_name}_{cutoff_date}_{target_y_name}.png", dpi=100))
    else:
        fig.savefig(Path(f"{PNG_DIR}/{daterecord}/{model_name}_{cutoff_date}_{target_y_name}.png", dpi=100))

def mape_calculator(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period, quantile):
    
    df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
    df_temp_scaled = raw_y_scaled_df(cutoff_date)
    
    df_temp_normal = df_temp[['ds','y']].set_index('ds')
    df_temp_scaled = df_temp_scaled[['ds','y']].set_index('ds')

    if choose_y_train_scaler == False:
        if len(df_temp_normal.loc[(df_temp_normal.index > cutoff_date)&(df_temp_normal.index <= last_month_updated)])> 1:
            df_temp_normal = df_temp_normal.loc[(df_temp_normal.index > cutoff_date)][:forecasting_period]
            median_based = 1 - mean_absolute_percentage_error(df_temp_normal['y'], Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'][:len(df_temp_normal)].values)
            last_based = 1 - mean_absolute_percentage_error(df_temp_normal['y'], Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Last_Lag'][:len(df_temp_normal)].values)
            print(f'{target_y_name} - {len(df_temp_normal.loc[(df_temp_normal.index > cutoff_date)&(df_temp_normal.index <= last_month_updated)])} lag (last) error based 1-MAPE: ', round(last_based*100,2), '%')
            print(f'{target_y_name} - Median List based 1-MAPE: ', round(median_based*100,2), '%')
        else:
            print( "Accuracy of predictions is not made available in forecasting mode")
    if choose_y_train_scaler == True:
        if len(df_temp_normal.loc[(df_temp_normal.index > cutoff_date)&(df_temp_normal.index <= last_month_updated)]) > 1:
            df_temp_scaled = df_temp_scaled.loc[(df_temp_scaled.index > cutoff_date)][:forecasting_period]
            median_based = 1 - mean_absolute_percentage_error(df_temp_scaled['y'], Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Median'][:len(df_temp_scaled)].values)
            last_based = 1 - mean_absolute_percentage_error(df_temp_scaled['y'], Y_hat_df_appended_vertical_mean_median[f'{model_name}_{quantile}_Last_Lag'][:len(df_temp_scaled)].values)
            print(f'{target_y_name} - {len(df_temp_scaled.loc[(df_temp_scaled.index > cutoff_date)&(df_temp_scaled.index <= last_month_updated)])} lag (last) error based 1-MAPE: ', round(last_based*100,2), '%')
            print(f'{target_y_name} - Median List based 1-MAPE: ', round(median_based*100,2), '%')
        else:
            print( "Accuracy of predictions is not made available in forecasting mode")

def png_func_no_quantile(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period):

    df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
    fi_training_X_scaled_df = raw_fi_df(df_scaler_fi)
    df_temp_scaled = raw_y_scaled_df()
    
    fig, ax = plt.subplots(1, 1, figsize = (20, 7))

    if choose_y_train_scaler == False:
        if len(df_scaler_total.loc[(df_scaler_total.ds > cutoff_date)]) > 1:
            plot_df = pd.concat([df_temp.set_index('ds'), Y_hat_df_appended_vertical_mean_median], join='inner')
        else:
            plot_df = pd.merge(df_temp.set_index('ds'), Y_hat_df_appended_vertical_mean_median, left_index=True, right_index=True, how='outer')
    if choose_y_train_scaler == True:
        if len(df_scaler_total.loc[(df_scaler_total.ds > cutoff_date)]) > 1:
            plot_df = pd.concat([df_temp_scaled.set_index('ds')['y'], Y_hat_df_appended_vertical_mean_median], join='inner')
        else:
            plot_df = pd.merge(df_temp_scaled.set_index('ds')['y'], Y_hat_df_appended_vertical_mean_median, left_index=True, right_index=True, how='outer')
    
    plot_df = plot_df[plot_df.index >= '2020-01-01']
    
    # plot_df = plot_df.rename(columns={f'{model_name}_0.5_Last_Lag' : f'{model_name} Last Lag - {i} Month Forecast'})
    plot_df = plot_df.rename(columns={'y' : f'{target_y_name} Actual Price',
                                      f'{model_name}_Median' : f'{model_name} Median List - {forecasting_period} Month Forecast',
                                      # f'{model_name}_0.5_Mean' : f'{model_name} Mean List - {i} Month Forecast', 
                                      f'{model_name}_Last_Lag' : f'{model_name} Last Lag - {forecasting_period} Month Forecast'},
                            )
                                          # f'{model_name}_0.5_Mean' : f'{model_name} Mean List - {i} Month Forecast',)
    # plot_df = plot_df.rename(columns={f'{model_name}_Median' : f'{model_name} Forecast - Median'})
    # plot_df = plot_df.rename(columns={f'{model_name}_Combined' : f'{model_name} Forecast - Combined'})
    plot_df_clear = plot_df[[f'{target_y_name} Actual Price',
                             f'{model_name} Median List - {forecasting_period} Month Forecast'
                            ]]
    plot_df[[f'{target_y_name} Actual Price', f'{model_name} Last Lag - {forecasting_period} Month Forecast', f'{model_name} Median List - {forecasting_period} Month Forecast',
             # f'{model_name} Mean List - {i} Month Forecast',
            ]]\
    .plot(ax=ax, linewidth=2.5, marker='o', markersize=8)
    # plot_df[[f'{target_y_name} Actual Price',f'{model_name} Forecast - Mean', f'{model_name} Forecast - Median']] \
    # .plot(ax=ax, linewidth=2, marker='o')
    
    plt.axvline(cutoff_date, color='red',linewidth=3, linestyle='--')
    ax.set_title(f'{target_y_name} Actual and Predicted Plot - Cutoff: {cutoff_date}', fontsize=22)
    ax.set_ylabel(f'{target_y_name}PRICE', fontsize=20)
    ax.set_xlabel('Date', fontsize=20)
    ax.legend(prop={'size': 15})
    ax.grid()
    
    if not os.path.exists(Path(f"{PNG_DIR}/{daterecord}")):
        os.mkdir(Path(f"{PNG_DIR}/{daterecord}"))
    if cutoff_date >= last_month_updated:
        fig.savefig(Path(f"{PNG_DIR}/{daterecord}/{model_name}_{cutoff_date}_{target_y_name}.png", dpi=100))
    else:
        fig.savefig(Path(f"{PNG_DIR}/{daterecord}/{model_name}_{cutoff_date}_{target_y_name}.png", dpi=100))

def mape_calculator_no_quantile(model_name, Y_hat_df_appended_vertical_mean_median, cutoff_date, forecasting_period):
    
    df_scaler_total, df_scaler_fi, df_temp = raw_data_loader(target_y_name, cutoff_date)
    df_temp_scaled = raw_y_scaled_df()
    
    df_temp_normal = df_temp[['ds','y']].set_index('ds')
    df_temp_scaled = df_temp_scaled[['ds','y']].set_index('ds')

    if choose_y_train_scaler == False:
        if len(df_temp_normal.loc[(df_temp_normal.index > cutoff_date)&(df_temp_normal.index <= last_month_updated)])> 1:
            df_temp_normal = df_temp_normal.loc[(df_temp_normal.index > cutoff_date)][:forecasting_period]
            median_based = 1 - mean_absolute_percentage_error(df_temp_normal['y'], Y_hat_df_appended_vertical_mean_median[f'{model_name}_Median'][:len(df_temp_normal)].values)
            last_based = 1 - mean_absolute_percentage_error(df_temp_normal['y'], Y_hat_df_appended_vertical_mean_median[f'{model_name}_Last_Lag'][:len(df_temp_normal)].values)
            print(f'{target_y_name} - {len(df_temp_normal.loc[(df_temp_normal.index > cutoff_date)&(df_temp_normal.index <= last_month_updated)])} lag (last) error based 1-MAPE: ', round(last_based*100,2), '%')
            print(f'{target_y_name} - Median List based 1-MAPE: ', round(median_based*100,2), '%')
        else:
            print( "Accuracy of predictions is not made available in forecasting mode")
    if choose_y_train_scaler == True:
        if len(df_temp_normal.loc[(df_temp_normal.index > cutoff_date)&(df_temp_normal.index <= last_month_updated)]) > 1:
            df_temp_scaled = df_temp_scaled.loc[(df_temp_scaled.index > cutoff_date)][:forecasting_period]
            median_based = 1 - mean_absolute_percentage_error(df_temp_scaled['y'], Y_hat_df_appended_vertical_mean_median[f'{model_name}_Median'][:len(df_temp_scaled)].values)
            last_based = 1 - mean_absolute_percentage_error(df_temp_scaled['y'], Y_hat_df_appended_vertical_mean_median[f'{model_name}_Last_Lag'][:len(df_temp_scaled)].values)
            print(f'{target_y_name} - {len(df_temp_scaled.loc[(df_temp_scaled.index > cutoff_date)&(df_temp_scaled.index <= last_month_updated)])} lag (last) error based 1-MAPE: ', round(last_based*100,2), '%')
            print(f'{target_y_name} - Median List based 1-MAPE: ', round(median_based*100,2), '%')
        else:
            print( "Accuracy of predictions is not made available in forecasting mode")