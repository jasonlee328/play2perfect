"""Depth-vision distillation: teacher labeling, student network, DAgger loop."""

from isaacsimenvs.distillation.a2c_aux_cnn import (
    CNN_OUT_FEATURES,
    A2CAuxCNNBuilder,
    CustomCNN,
    register_student_networks,
)
from isaacsimenvs.distillation.dagger import Dagger
from isaacsimenvs.distillation.teacher import Teacher, teacher_env_info_from_dims

__all__ = [
    "Dagger",
    "Teacher",
    "teacher_env_info_from_dims",
    "A2CAuxCNNBuilder",
    "CustomCNN",
    "CNN_OUT_FEATURES",
    "register_student_networks",
]
