"""
CR-V acceleration limit: caps the planned positive acceleration when the CrvAccelLimit param is on.
Braking limits are never touched.
"""
import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

# m/s^2 at speeds in m/s: 0.67 (1.5 mph/s) up to ~22 mph, easing to 0.45 (1 mph/s) by ~90 mph
CRV_ACCEL_MAX_BP = [0., 10., 25., 40.]
CRV_ACCEL_MAX_VALS = [0.67, 0.67, 0.55, 0.45]


class CrvAccelLimit:
  def __init__(self):
    self.params = Params()
    self.frame = 0
    self.enabled = self.params.get_bool("CrvAccelLimit")

  def update(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("CrvAccelLimit")
    self.frame += 1

  def max_accel(self, v_ego: float, stock_max: float) -> float:
    if not self.enabled:
      return stock_max
    return min(stock_max, float(np.interp(v_ego, CRV_ACCEL_MAX_BP, CRV_ACCEL_MAX_VALS)))
