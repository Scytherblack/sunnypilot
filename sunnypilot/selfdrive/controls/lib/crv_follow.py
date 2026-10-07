"""
CR-V "follow like you" (10-07): with CrvFollowLikeYou on, the planner's following time comes from the
owner's own measured gaps instead of the personality setting.

Measured when he drove (steady following, lead within 2 mph of his speed): 1.6 s at 10-25 mph, 2.05 s at
25-35, 2.5 s at 35-45 (10-07); 2.0 s at 55 and 1.6 s at 65 mph (10-02, 500 min). Never below 1.75 s, the
stock "relaxed" value, so this only ever follows further than today.
"""
import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

FOLLOW_BP = [4.5, 13.4, 17.9, 24.6, 29.0]  # m/s: 10, 30, 40, 55, 65 mph
FOLLOW_T = [1.75, 2.0, 2.5, 2.0, 1.75]     # s


def follow_time(v_ego: float) -> float:
  return float(np.interp(v_ego, FOLLOW_BP, FOLLOW_T))


class CrvFollow:
  def __init__(self):
    self.params = Params()
    self.frame = 0
    self._read_params()

  def _read_params(self) -> None:
    self.enabled = self.params.get_bool("CrvFollowLikeYou")

  def update(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self._read_params()
    self.frame += 1

  def t_follow(self, v_ego: float):
    """Following time to use, or None to keep the personality's value."""
    return follow_time(v_ego) if self.enabled else None
