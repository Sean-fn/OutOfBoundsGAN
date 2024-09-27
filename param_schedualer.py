# from carbs import CARBS, CARBSParams, Param, LogSpace, ObservationInParam
from carbs import CARBS
from carbs import CARBSParams
from carbs import LogSpace
from carbs import LogitSpace
from carbs import ObservationInParam
from carbs import ParamDictType
from carbs import Param

def param_schedualer():
    param_spaces = [
        Param(name="lr", space=LogSpace(scale=0.5), search_center=0.00009),
        # Param(name="batch_size", space=LogSpace(is_integer=True, min=32, max=512), search_center=320),
        # Param(name="n_epochs", space=LogSpace(is_integer=True, min=10, max=200), search_center=200),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=-1,  # Assuming lower loss is better
        is_wandb_logging_enabled=False,
        resample_frequency=0,  # Adjust as needed
    )

    return CARBS(carbs_params, param_spaces)