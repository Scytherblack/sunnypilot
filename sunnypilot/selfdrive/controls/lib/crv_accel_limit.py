"""
CR-V acceleration limit, "virtual pedal": when CrvAccelLimit is on, the planner never asks for more
acceleration than this car gives with the accelerator held still at CrvVirtualPedal percent of its
travel, and never limits below CrvAccelFloor. Braking limits are never touched.

CRV_PEDAL_ACCEL is measured from the owner's own driving (09-23 to 10-01, 500 min with the driver
controlling speed): median acceleration at a held pedal position, per speed. Smoothed so it never
rises with speed past the launch peak and never falls with more pedal.
"""
import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD

CRV_PEDAL_BP = [17.5, 22.5, 27.5, 32.5, 37.5, 45.0]  # % of pedal travel
CRV_SPEED_BP = [0.56, 1.68, 2.79, 3.91, 5.59, 7.82, 10.06, 12.29, 15.65, 20.12, 24.59, 30.18]  # m/s
CRV_PEDAL_ACCEL = [  # m/s^2, rows follow CRV_PEDAL_BP, columns follow CRV_SPEED_BP
  [1.41, 1.41, 1.05, 0.88, 0.58, 0.40, 0.17, 0.10, 0.04, 0.01, 0.00, 0.00],
  [1.61, 1.57, 1.34, 1.17, 0.89, 0.63, 0.42, 0.29, 0.18, 0.09, 0.06, 0.02],
  [1.82, 1.82, 1.62, 1.40, 1.13, 0.85, 0.63, 0.48, 0.32, 0.20, 0.12, 0.05],
  [2.02, 2.02, 1.86, 1.65, 1.36, 1.08, 0.88, 0.69, 0.53, 0.34, 0.22, 0.10],
  [2.27, 2.27, 2.17, 1.96, 1.67, 1.39, 1.15, 0.96, 0.74, 0.50, 0.28, 0.19],
  [2.49, 2.49, 2.38, 2.24, 1.99, 1.75, 1.54, 1.32, 1.08, 0.76, 0.47, 0.31],
]

DEFAULT_PEDAL = 28.0  # % of travel, the owner's median during launches
DEFAULT_FLOOR = 0.45  # m/s^2, 1 mph/s
PEDAL_RANGE = (CRV_PEDAL_BP[0], CRV_PEDAL_BP[-1])
FLOOR_RANGE = (0.0, 1.0)
DEFAULT_SCALE = 1.0  # CrvAccelScale: the whole cap (pedal curve and floor) times this; 0.95 = 5 % less everywhere
SCALE_RANGE = (0.5, 1.0)
# CrvEvPowerKw (0 = off): also cap acceleration so that wheel power stays below this many kW, the way the
# owner launches (he holds about 14 kW, which keeps the hybrid in EV mode at 75-80 % charge; the engine
# started at 14-18 kW on 10-07). Road load fitted from 128 min of steady driving: kW = 0.244 v + 0.00030 v^3.
DEFAULT_EV_KW = 0.0
EV_KW_RANGE = (0.0, 60.0)
CRV_MASS = 1839.0  # kg, from the car's fingerprint
ROAD_LOAD_A = 0.244  # kW per m/s
ROAD_LOAD_B = 0.00030  # kW per (m/s)^3


def pedal_accel(v_ego: float, pedal: float) -> float:
  """Acceleration this car gives at speed v_ego with the pedal held at `pedal` percent."""
  per_row = [np.interp(v_ego, CRV_SPEED_BP, row) for row in CRV_PEDAL_ACCEL]
  return float(np.interp(pedal, CRV_PEDAL_BP, per_row))


def _read_float(params: Params, key: str, default: float, bounds: tuple[float, float]) -> float:
  try:
    value = params.get(key)
    value = default if value is None else float(value)
  except (TypeError, ValueError):
    value = default
  return float(np.clip(value, *bounds))


def road_load_kw(v_ego: float) -> float:
  return ROAD_LOAD_A * v_ego + ROAD_LOAD_B * v_ego ** 3


def ev_accel(v_ego: float, power_kw: float) -> float:
  """Acceleration that keeps wheel power at `power_kw` (unbounded at a standstill)."""
  return (power_kw - road_load_kw(v_ego)) * 1000.0 / (CRV_MASS * max(v_ego, 1.0))


class CrvAccelLimit:
  def __init__(self):
    self.params = Params()
    self.frame = 0
    self._read_params()

  def _read_params(self) -> None:
    self.enabled = self.params.get_bool("CrvAccelLimit")
    self.pedal = _read_float(self.params, "CrvVirtualPedal", DEFAULT_PEDAL, PEDAL_RANGE)
    self.floor = _read_float(self.params, "CrvAccelFloor", DEFAULT_FLOOR, FLOOR_RANGE)
    self.scale = _read_float(self.params, "CrvAccelScale", DEFAULT_SCALE, SCALE_RANGE)
    self.ev_kw = _read_float(self.params, "CrvEvPowerKw", DEFAULT_EV_KW, EV_KW_RANGE)

  def update(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self._read_params()
    self.frame += 1

  def max_accel(self, v_ego: float, stock_max: float) -> float:
    if not self.enabled:
      return stock_max
    cap = pedal_accel(v_ego, self.pedal)
    if self.ev_kw > 0.0:
      cap = min(cap, ev_accel(v_ego, self.ev_kw))
    return min(stock_max, self.scale * max(self.floor, cap))
