import numpy as np
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import joblib, pickle, shutil, warnings, time, os, gc, math, glob, plotly, torch, re, json, csv
from dateutil.relativedelta import relativedelta
from pprint import pprint
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr, kendalltau
from boruta import BorutaPy
from collections import Counter, defaultdict
from openfe import OpenFE, transform, tree_to_formula

import xgboost as xgb
from nixtlats import TimeGPT
from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NBEATSx, NHITS, TFT, LSTM, MLP, Autoformer, Informer, StemGNN, TCN, PatchTST, TimesNet, VanillaTransformer, TFT, FEDformer
from neuralforecast.losses.pytorch import QuantileLoss, MQLoss, DistributionLoss, GMM, PMM, HuberMQLoss, sCRPS, MAE
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.losses.numpy import mae, mse, mape
from openpyxl import Workbook, load_workbook

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
