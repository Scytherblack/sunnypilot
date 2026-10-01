"""CR-V acceleration limit (CrvAccelLimit): the planner never asks for more than the eco table,
launches included, and braking is untouched.

Regenerate the stock reference ONLY on code without the limit:
  python selfdrive/test/longitudinal_maneuvers/test_crv_accel_limit.py
"""
import re
from pathlib import Path

import numpy as np
import pytest

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from opendbc.car.interfaces import ACCEL_MIN
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, CarControllerParams
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import create_maneuvers

HARD_LIMIT = 0.67  # m/s^2, the owner's limit
REF = Path(__file__).with_name("crv_stock_ref.npz")
T, V, A = 0, 3, 5  # columns of a maneuver log row: t, x, x_lead, v, v_lead, a, d_rel


def crv_maneuvers():
  stock = {m.title: m for m in create_maneuvers({"e2e": False, "force_decel": False})}
  return {
    "resume": stock["resume from a stop"],
    "decel3plus": stock["steady state following a car at 20m/s, then lead decel to 0mph at 3+m/s^2"],
    "zero_to_cruise": Maneuver("0 to 45 mph, no lead", duration=60., initial_speed=0., lead_relevancy=False,
                               cruise_values=[20., 20.], breakpoints=[0., 60.]),
    "fast_lead_launch": Maneuver("launch behind a lead pulling away at 2 m/s^2", duration=30., initial_speed=0.,
                                 lead_relevancy=True, initial_distance_lead=STOP_DISTANCE,
                                 speed_lead_values=[0., 0., 20.], breakpoints=[0., 2., 12.]),
  }


def run(name, enabled=None):
  if enabled is not None:
    Params().put_bool("CrvAccelLimit", enabled)
  valid, logs = crv_maneuvers()[name].evaluate()
  assert valid
  return logs


@pytest.mark.parametrize("name", ["resume", "zero_to_cruise", "fast_lead_launch"])
def test_accel_never_above_cap(name):
  from openpilot.sunnypilot.selfdrive.controls.lib.crv_accel_limit import CRV_ACCEL_MAX_BP, CRV_ACCEL_MAX_VALS
  logs = run(name, True)
  assert logs[:, A].max() <= HARD_LIMIT + 1e-6
  # each step was planned at the speed of the previous row
  assert np.all(logs[1:, A] <= np.interp(logs[:-1, V], CRV_ACCEL_MAX_BP, CRV_ACCEL_MAX_VALS) + 1e-6)


def test_still_reaches_cruise():
  logs = run("zero_to_cruise", True)
  assert logs[-1, V] > 19.0
  assert logs[:, A].max() > 0.40


def test_stock_exceeds_cap():
  # proves the cap tests can fail: without the limit a launch goes well above it
  assert run("fast_lead_launch", False)[:, A].max() > HARD_LIMIT + 0.2


def test_param_refresh():
  from openpilot.sunnypilot.selfdrive.controls.lib.crv_accel_limit import CrvAccelLimit
  Params().put_bool("CrvAccelLimit", False)
  lim = CrvAccelLimit()
  assert lim.max_accel(0., 1.6) == 1.6
  Params().put_bool("CrvAccelLimit", True)
  for _ in range(61):  # 3 s at 20 Hz
    lim.update()
  assert lim.max_accel(0., 1.6) == pytest.approx(HARD_LIMIT)
  assert lim.max_accel(30., 1.6) < HARD_LIMIT


def test_floor_constants_unchanged():
  assert ACCEL_MIN == -3.5
  assert CarControllerParams.BOSCH_ACCEL_MIN == -3.5
  CP = CarInterface.get_non_essential_params(CAR.HONDA_CRV_6G)
  CP_SP = CarInterface.get_non_essential_params_sp(CP, CAR.HONDA_CRV_6G)
  assert CarInterface.get_pid_accel_limits(CP, CP_SP, 10., 20.)[0] == -3.5
  safety = (Path(BASEDIR) / "opendbc_repo/opendbc/safety/modes/honda.h").read_text()
  assert re.search(r"\.min_accel = -350,", safety)


def test_floor_in_planner_and_mpc():
  Params().put_bool("CrvAccelLimit", True)
  plant = Plant(lead_relevancy=True, speed=20., distance_lead=35.)
  for _ in range(100):
    plant.step(v_lead=10.)
  assert plant.planner.prev_accel_clip[0] == ACCEL_MIN
  assert np.all(plant.planner.mpc.params[:, 0] == ACCEL_MIN)


def test_decel_3plus_unchanged():
  on = run("decel3plus", True)[:, A].min()
  ref = np.load(REF)["decel3plus"][:, A].min()
  assert on == pytest.approx(ref, abs=0.05)
  assert on < -3.0


@pytest.mark.parametrize("name", ["resume", "zero_to_cruise", "fast_lead_launch", "decel3plus"])
def test_param_off_is_stock(name):
  np.testing.assert_allclose(run(name, False), np.load(REF)[name], rtol=0, atol=1e-6)


if __name__ == "__main__":
  with OpenpilotPrefix():
    np.savez_compressed(REF, **{n: run(n) for n in crv_maneuvers()})
  ref = np.load(REF)
  for n in ref.files:
    print(f"{n}: max a {ref[n][:, A].max():.3f}, min a {ref[n][:, A].min():.3f}, end v {ref[n][-1, V]:.2f}")
