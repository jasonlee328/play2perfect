"""Depth-vision distillation: teacher labeling, student network, DAgger loop."""

from isaacsimenvs.distillation.a2c_aux_cnn import A2CBuilder, CustomCNN
from isaacsimenvs.distillation.dagger import Dagger
from isaacsimenvs.distillation.student_policy import StudentPolicy
from isaacsimenvs.distillation.teacher import Teacher, teacher_env_info_from_dims

__all__ = [
    "Dagger",
    "StudentPolicy",
    "Teacher",
    "teacher_env_info_from_dims",
    # Upstream's names, from the verbatim a2c_with_aux_cnn.py copy. Register with
    # model_builder.register_network("a2c_aux_cnn_net", A2CBuilder) at the call
    # site, as run_distillation.py:187 does.
    "A2CBuilder",
    "CustomCNN",
]
