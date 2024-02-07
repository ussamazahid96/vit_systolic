import torch
from brevitas.inject.enum import *
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
from omegaconf import DictConfig

brevitas_dict = {
    "QuantType.BINARY" : QuantType.BINARY,
    "QuantType.TERNARY" : QuantType.TERNARY,
    "QuantType.INT" : QuantType.INT,
    "QuantType.FP" : QuantType.FP,

    "RestrictValueType.POWER_OF_TWO": RestrictValueType.POWER_OF_TWO,
    "RestrictValueType.FP": RestrictValueType.FP,
    "RestrictValueType.LOG_FP": RestrictValueType.LOG_FP,
    "RestrictValueType.INT": RestrictValueType.INT,

    "ScalingImplType.HE": ScalingImplType.HE,
    "ScalingImplType.CONST": ScalingImplType.CONST,
    "ScalingImplType.STATS": ScalingImplType.STATS,
    "ScalingImplType.PARAMETER": ScalingImplType.PARAMETER,
    "ScalingImplType.PARAMETER_FROM_STATS": ScalingImplType.PARAMETER_FROM_STATS,

    "StatsOp.MAX" : StatsOp.MAX,
    "StatsOp.AVE" : StatsOp.AVE,
    "StatsOp.PERCENTILE" : StatsOp.PERCENTILE,
    
    "FloatToIntImplType.ROUND": FloatToIntImplType.ROUND,

    "BitWidthImplType.CONST": BitWidthImplType.CONST,
    "BitWidthImplType.PARAMETER": BitWidthImplType.PARAMETER,
    "None": None    
    
}

def get_input_quantizer(cfg: DictConfig):
    class InputQuantizer(ActQuantSolver):
        bit_width = cfg.bit_width  # bit width is 8
        signed = cfg.signed # quantization range is signed
        narrow_range = cfg.narrow_range # quantization range is [-128, 127] rather than [-127, 127]
        min_val = cfg.min_val
        max_val = cfg.max_val
        quant_type = brevitas_dict[cfg.quant_type] # integer quantization
        float_to_int_impl_type = brevitas_dict[cfg.float_to_int_impl_type] # round to nearest
        
        scaling_impl_type = brevitas_dict[cfg.scaling_impl_type] # scale is a parameter initialized from statistics
        restrict_scaling_type = brevitas_dict[cfg.restrict_scaling_type] # scale is a floating-point value
        scaling_per_output_channel = cfg.scaling_per_output_channel  # scale is per tensor

        bit_width_impl_type = brevitas_dict[cfg.bit_width_impl_type] # constant bit width
        zero_point_impl = ZeroZeroPoint # zero point is 0.
    return InputQuantizer

def get_weight_quantizer(cfg: DictConfig):
    class WeightQuantizer(WeightQuantSolver):
        bit_width = cfg.bit_width  # bit width is 8
        signed = cfg.signed # quantization range is signed
        narrow_range = cfg.narrow_range # quantization range is [-128, 127] rather than [-127, 127]
        
        quant_type = brevitas_dict[cfg.quant_type] # integer quantization
        float_to_int_impl_type = brevitas_dict[cfg.float_to_int_impl_type] # round to nearest

        scaling_const = cfg.scaling_const
        scaling_impl_type = brevitas_dict[cfg.scaling_impl_type] # scale based on statistics
        scaling_stats_op = brevitas_dict[cfg.scaling_stats_op]
        # scaling_stats_op = StatsOp.MAX # scale statistics is the absmax value
        restrict_scaling_type = brevitas_dict[cfg.restrict_scaling_type] # scale factor is a floating point value
        scaling_per_output_channel = cfg.scaling_per_output_channel  # scale is per tensor

        bit_width_impl_type = brevitas_dict[cfg.bit_width_impl_type] # constant bit width
        zero_point_impl = ZeroZeroPoint # zero point is 0.
    return WeightQuantizer

def get_activation_quantizer(cfg: DictConfig):
    class ActivationQuantizer(ActQuantSolver):
        bit_width = cfg.bit_width  # bit width is 8
        signed = cfg.signed # quantization range is signed
        narrow_range = cfg.narrow_range # quantization range is [-128, 127] rather than [-127, 127]
        min_val = cfg.min_val
        max_val = cfg.max_val
        quant_type = brevitas_dict[cfg.quant_type] # integer quantization
        float_to_int_impl_type = brevitas_dict[cfg.float_to_int_impl_type] # round to nearest

        scaling_const = cfg.scaling_const
        scaling_impl_type = brevitas_dict[cfg.scaling_impl_type] # scale is a parameter initialized from statistics
        scaling_stats_op = brevitas_dict[cfg.scaling_stats_op]    
        # scaling_stats_op = StatsOp.PERCENTILE # scale statistics is a percentile of the abs value
        # percentile_q = 99.999 # percentile is 99.999
        collect_stats_steps = 300  # statistics are collected for 300 forward steps before switching to a learned parameter
        restrict_scaling_type = brevitas_dict[cfg.restrict_scaling_type] # scale is a floating-point value
        scaling_per_output_channel = cfg.scaling_per_output_channel  # scale is per tensor
        scaling_stats_permute_dims = tuple(int(x) for x in cfg.scaling_stats_permute_dims.split(','))
        per_channel_broadcastable_shape = tuple(int(x) for x in cfg.per_channel_broadcastable_shape.split(','))
        bit_width_impl_type = brevitas_dict[cfg.bit_width_impl_type] # constant bit width
        zero_point_impl = ZeroZeroPoint # zero point is 0.
    return ActivationQuantizer
