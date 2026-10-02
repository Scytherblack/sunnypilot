"""CR-V acceleration limit (CrvAccelLimit, "virtual pedal"): the planner never asks for more than the
car gives at the chosen held pedal position (or the floor), launches included, and braking is untouched.

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

STOCK_LAUNCH_MAX = 1.6  # m/s^2, stock planner ceiling at standstill
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
  from openpilot.sunnypilot.selfdrive.controls.lib.crv_accel_limit import DEFAULT_FLOOR, DEFAULT_PEDAL, pedal_accel
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_max_accel
  logs = run(name, True)
  # each step was planned at the speed of the previous row
  v_prev = logs[:-1, V]
  cap = np.array([min(get_max_accel(v), max(DEFAULT_FLOOR, pedal_accel(v, DEFAULT_PEDAL))) for v in v_prev])
  assert np.all(logs[1:, A] <= cap + 1e-6)
  assert logs[:, A].max() <= STOCK_LAUNCH_MAX + 1e-6


def test_cap_follows_the_pedal_curve():
  from openpilot.sunnypilot.selfdrive.controls.lib.crv_accel_limit import CrvAccelLimit
  Params().put_bool("CrvAccelLimit", True)
  lim = CrvAccelLimit()
  launch, mid, cruise = lim.max_accel(2.0, 1.6), lim.max_accel(10.0, 1.2), lim.max_accel(25.0, 0.8)
  assert launch > 1.4            # strong off the line, like the owner's launches
  assert 0.5 < mid < 0.75        # fading by 22 mph
  assert cruise == pytest.approx(0.45)  # the 1 mph/s floor at highway speed
  assert launch > mid > cruise


def test_still_reaches_cruise():
  logs = run("zero_to_cruise", True)
  assert logs[-1, V] > 19.0
  assert logs[:, A].max() > 1.0


def test_slower_than_stock_above_launch():
  def t_to(logs, v):
    return logs[np.argmax(logs[:, V] >= v), T]
  on, ref = run("zero_to_cruise", True), np.load(REF)["zero_to_cruise"]
  assert t_to(on, 5.0) < t_to(ref, 5.0) + 1.0     # launch close to stock
  assert t_to(on, 19.0) > t_to(ref, 19.0) + 3.0   # but takes clearly longer to reach 42 mph


def test_stock_exceeds_cap():
  # proves the cap tests can fail: without the limit the mid-speed pull is well above the curve
  logs = run("zero_to_cruise", False)
  mid = (logs[:, V] > 9.0) & (logs[:, V] < 12.0)
  assert logs[mid, A].max() > 0.9


def test_param_refresh():
  from openpilot.sunnypilot.selfdrive.controls.lib.crv_accel_limit import CrvAccelLimit
  params = Params()
  params.put_bool("CrvAccelLimit", False)
  lim = CrvAccelLimit()
  assert lim.max_accel(10., 1.2) == 1.2
  params.put_bool("CrvAccelLimit", True)
  params.put("CrvVirtualPedal", 22.5)
  for _ in range(61):  # 3 s at 20 Hz
    lim.update()
  gentle = lim.max_accel(10., 1.2)
  params.put("CrvVirtualPedal", 32.5)
  for _ in range(61):
    lim.update()
  brisk = lim.max_accel(10., 1.2)
  assert gentle == pytest.approx(0.45)   # 22.5 % gives 0.42, the floor holds it at 0.45
  assert brisk == pytest.approx(0.88, abs=0.01)
  params.put("CrvAccelFloor", 0.6)
  for _ in range(61):
    lim.update()
  assert lim.max_accel(25., 0.8) == pytest.approx(0.6)


def test_bad_param_values_fall_back():
  from openpilot.sunnypilot.selfdrive.controls.lib.crv_accel_limit import CrvAccelLimit
  params = Params()
  params.put_bool("CrvAccelLimit", True)
  params.put("CrvVirtualPedal", 500.0)
  params.put("CrvAccelFloor", -3.0)
  lim = CrvAccelLimit()
  assert lim.pedal == 45.0 and lim.floor == 0.0
  assert lim.max_accel(0., 1.6) <= 1.6


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
