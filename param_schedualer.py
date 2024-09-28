# from carbs import CARBS, CARBSParams, Param, LogSpace, ObservationInParam
from carbs import CARBS
from carbs import CARBSParams
from carbs import LogSpace
from carbs import LogitSpace
from carbs import ObservationInParam
from carbs import ParamDictType
from carbs import Param

def param_schedualer(config):
    param_spaces = [
        Param(name="lr", space=LogSpace(scale=0.5), search_center=config.opt.lr),
        Param(name="lr_min", space=LogSpace(scale=0.5, min=1e-5, max=1e-3), search_center=config.opt.lr_min),
        Param(name="lr_max", space=LogSpace(scale=0.5, min=1e-3, max=1e-1), search_center=config.opt.lr_max),
        Param(name="step_size_up", space=LogSpace(scale=0.5, min=100, max=1000), search_center=config.opt.step_size_up),
        Param(name="plateau_factor", space=LogSpace(scale=0.5, min=0.1, max=0.9), search_center=config.opt.plateau_factor),
        Param(name="plateau_patience", space=LogSpace(scale=1, min=2, max=10), search_center=config.opt.plateau_patience),
    ]

    carbs_params = CARBSParams(
        better_direction_sign=-1,
        is_wandb_logging_enabled=False,
        resample_frequency=0,
    )

    return CARBS(carbs_params, param_spaces)