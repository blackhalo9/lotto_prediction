from PKG_0321.PKGS_0321_PY import *
from PKG_0321.INPUT_0321_PY import *
from PKG_0321.UTILS_0321_PY import *
from PKG_0321.PREPROCESSING_0321_PY import *
from PKG_0321.FEATURE_IMPORTANCE_0321_PY import *
from PKG_0321.ENSEMBLE_LIST_0321_PY import *
from PKG_0321.FUTURE_FRAME_0321_PY import *
from PKG_0321.DATE_DICT_0321_PY import *

from typing import Optional
import torch.nn as nn
from neuralforecast.common._base_windows import BaseWindows
from neuralforecast.losses.pytorch import MAE
from fastcore.test import test_eq
from nbdev.showdoc import show_doc

#| export
class NLinear(BaseWindows):
    """ NLinear

    *Parameters:*
    `h`: int, forecast horizon.
    `input_size`: int, maximum sequence length for truncated train backpropagation. Default -1 uses all history.
    `futr_exog_list`: str list, future exogenous columns.
    `hist_exog_list`: str list, historic exogenous columns.
    `stat_exog_list`: str list, static exogenous columns.
    `exclude_insample_y`: bool=False, the model skips the autoregressive features y[t-input_size:t] if True.
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).
    `max_steps`: int=1000, maximum number of training steps.
    `learning_rate`: float=1e-3, Learning rate between (0, 1).
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.
    `val_check_steps`: int=100, Number of training steps between every validation loss check.
    `batch_size`: int=32, number of different series in each batch.
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.
    `inference_windows_batch_size`: int=1024, number of windows to sample in each inference batch.
    `start_padding_enabled`: bool=False, if True, the model will pad the time series with zeros at the beginning, by input size.
    `scaler_type`: str='robust', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).
    `random_seed`: int=1, random_seed for pytorch initializer and numpy generators.
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.
    `alias`: str, optional,  Custom name of the model.
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

	*References*
	- Zeng, Ailing, et al. "Are transformers effective for time series forecasting?." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 9. 2023."
    """
    # Class attributes
    SAMPLING_TYPE = 'windows'

    def __init__(self,
                 h: int, 
                 input_size: int,
                 stat_exog_list = None,
                 hist_exog_list = None,
                 futr_exog_list = None,
                 exclude_insample_y = False,
                 loss = MAE(),
                 valid_loss = None,
                 max_steps: int = 5000,
                 learning_rate: float = 1e-4,
                 num_lr_decays: int = -1,
                 early_stop_patience_steps: int =-1,
                 val_check_steps: int = 100,
                 batch_size: int = 32,
                 valid_batch_size: Optional[int] = None,
                 windows_batch_size = 1024,
                 inference_windows_batch_size = 1024,
                 start_padding_enabled = False,
                 step_size: int = 1,
                 scaler_type: str = 'identity',
                 random_seed: int = 1,
                 num_workers_loader: int = 0,
                 drop_last_loader: bool = False,
                 **trainer_kwargs):
        super(NLinear, self).__init__(h=h,
                                       input_size=input_size,
                                       hist_exog_list=hist_exog_list,
                                       stat_exog_list=stat_exog_list,
                                       futr_exog_list = futr_exog_list,
                                       exclude_insample_y = exclude_insample_y,
                                       loss=loss,
                                       valid_loss=valid_loss,
                                       max_steps=max_steps,
                                       learning_rate=learning_rate,
                                       num_lr_decays=num_lr_decays,
                                       early_stop_patience_steps=early_stop_patience_steps,
                                       val_check_steps=val_check_steps,
                                       batch_size=batch_size,
                                       windows_batch_size=windows_batch_size,
                                       valid_batch_size=valid_batch_size,
                                       inference_windows_batch_size=inference_windows_batch_size,
                                       start_padding_enabled = start_padding_enabled,
                                       step_size=step_size,
                                       scaler_type=scaler_type,
                                       num_workers_loader=num_workers_loader,
                                       drop_last_loader=drop_last_loader,
                                       random_seed=random_seed,
                                       **trainer_kwargs)

        # Architecture
        self.futr_input_size = len(self.futr_exog_list)
        self.hist_input_size = len(self.hist_exog_list)
        self.stat_input_size = len(self.stat_exog_list)

        if self.stat_input_size > 0:
            raise Exception('NLinear does not support static variables yet')
        
        if self.hist_input_size > 0:
            raise Exception('NLinear does not support historical variables yet')
        
        if self.futr_input_size > 0:
            raise Exception('NLinear does not support future variables yet')

        self.c_out = self.loss.outputsize_multiplier
        self.output_attention = False
        self.enc_in = 1 
        self.dec_in = 1

        self.linear = nn.Linear(self.input_size, self.loss.outputsize_multiplier * h, bias=True)

    def forward(self, windows_batch):
        # Parse windows_batch
        insample_y    = windows_batch['insample_y']
        #insample_mask = windows_batch['insample_mask']
        #hist_exog     = windows_batch['hist_exog']
        #stat_exog     = windows_batch['stat_exog']
        #futr_exog     = windows_batch['futr_exog']

        # Parse inputs
        batch_size = len(insample_y)
        
        # Input normalization
        last_value = insample_y[:, -1:]
        norm_insample_y = insample_y - last_value
        
        # Final
        forecast = self.linear(norm_insample_y) + last_value
        forecast = forecast.reshape(batch_size, self.h, self.loss.outputsize_multiplier)
        forecast =  self.loss.domain_map(forecast)
        return forecast