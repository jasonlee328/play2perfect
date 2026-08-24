"""Depth-vision distillation: teacher labeling, student network, DAgger loop."""

from isaacsimenvs.distillation.teacher import Teacher, teacher_env_info_from_dims

__all__ = ["Teacher", "teacher_env_info_from_dims"]
