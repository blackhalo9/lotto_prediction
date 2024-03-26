import os
import datetime as dt
from pathlib import Path

fulldaterecord = dt.datetime.now().strftime('%Y-%m-%d')
daterecord = dt.datetime.now().strftime('%m%d')

############ Keywords excluded in Dataset ###############
RAW_PATH = '/home/eunsuk.ko/METAL_FCST/DATA/INTEGR_METAL_NON_METAL_MONTHLY_2024-01.csv' ## '23년 11월 cutoff
# RAW_PATH = '/home/eunsuk.ko/METAL_FCST/DATA/INTEGR_METAL_NON_METAL_MONTHLY_2024-02.csv' ## '23년 12월 cutoff
# RAW_PATH = '/home/eunsuk.ko/METAL_FCST/DATA/INTEGR_METAL_NON_METAL_MONTHLY_2024-03.csv' ## '24년 1월 cutoff
# RAW_PATH = '/home/eunsuk.ko/METAL_FCST/DATA/INTEGR_METAL_NON_METAL_MONTHLY_2024-04.csv' ## '24년 2월 cutoff

keywords_excluded = [
                     'PRICE_9311SPRT_ICC_RMB',
                     'PRICE_SSG_SML_BALL6_RMB',
                     'PRICE_CTHD_FEPO4_WF_SMM_RMB',
                     'PRICE_CO3O4_SMM_RMB',
                     # 'LGES_LI2CO3_RQMTS_OLK_KG', ## All NaN
                     # 'LGES_LIOH_RQMTS_OLK_KG', ## All NaN
                     # 'LGES_NI_RQMTS_OLK', ## All NaNx 
                     'HOUSECOST', 'PPP', 'GDP','_OPEN','_HIGH','_LOW', '_RES'
                    ]

############ Base Location for Results ##################

BASE_DIR =  Path(f"/home/eunsuk.ko/METAL_FCST/")
DATA_DIR = Path(f"/{BASE_DIR}/DATA/")
OPTUNA_DIR = Path(f"/{BASE_DIR}/OPTUNA/")
RESULTS_DIR = Path(f"/{BASE_DIR}/RESULTS/")
PNG_DIR = Path(f"/{BASE_DIR}/PNG/")

############ Input Period & Options #####################

choose_y_train_scaler = False
past_validation_cutoff = True

cutoff_date = '2024-02-28'
validation_cutoff = '2022-02-28'

if past_validation_cutoff == True:
    cutoff_date = validation_cutoff
else:
    cutoff_date = cutoff_date

raw_start = '2016-12-31'
training_start = '2018-01-31'
feature_cutoff_date = '2023-02-28'
last_month_updated = '2024-03-31'

forecasting_start = 1
forecasting_period = 13

############ Target y ##################################

target_y = 'PRICE_LI2CO3_EXW_FAST_KG'
target_y_name = 'LI2CO3(EXW)'

# target_y = 'PRICE_LI2CO3_CIF_FAST_KG_SPOT_EXCHNG'
# target_y_name = 'LI2CO3(CIF)'

# target_y = 'PRICE_LIOH_EXW_FAST_KG'
# target_y_name = 'LIOH(EXW)'

# target_y = 'PRICE_LIOH_CIF_FAST_KG_SPOT_EXCHNG'
# target_y_name = 'LIOH(CIF)'

############ Exceptional Target y ######################

# target_y = 'PRICE_AG_HIGH_ICC_RMB'
# target_y_name = 'AG_HIGH'

# target_y = 'PRICE_COKE_GNR_GRN_ICC_RMB'
# target_y_name ='COKE_GNR'