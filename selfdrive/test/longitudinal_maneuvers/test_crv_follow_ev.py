"""CR-V follow-like-you (CrvFollowLikeYou) and the EV power cap (CrvEvPowerKw)."""
import numpy as np
import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE, get_T_FOLLOW
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver
from openpilot.sunnypilot.selfdrive.controls.lib.crv_accel_limit import CrvAccelLimit, ev_accel, road_load_kw, pedal_accel, DEFAULT_PEDAL
from openpilot.sunnypilot.selfdrive.controls.lib.crv_follow import CrvFollow, follow_time

T, V, A, D = 0, 3, 5, 6


def test_params_registered():
  p = Params()
  assert p.get_bool("CrvFollowLikeYou") is False
  assert float(p.get("CrvEvPowerKw", return_default=True)) == 0.0


def test_follow_time_table():
  assert follow_time(0.0) == 1.75
  assert follow_time(4.5) == 1.75                      # never below stock relaxed
  assert follow_time(13.4) == pytest.approx(2.0)       # 30 mph
  assert follow_time(17.9) == pytest.approx(2.5)       # 40 mph, his measured 2.5 s
  assert follow_time(24.6) == pytest.approx(2.0)       # 55 mph
  assert follow_time(40.0) == 1.75
  assert all(follow_time(v) >= 1.75 for v in np.linspace(0, 40, 81))
  Params().put_bool("CrvFollowLikeYou", False)
  assert CrvFollow().t_follow(17.9) is None
  Params().put_bool("CrvFollowLikeYou", True)
  assert CrvFollow().t_follow(17.9) == pytest.approx(2.5)


def steady_gap(on):
  Params().put_bool("CrvFollowLikeYou", on)
  m = Maneuver("steady following at 20 m/s", duration=60., initial_speed=20., lead_relevancy=True, initial_distance_lead=60.,
               speed_lead_values=[20., 20.], breakpoints=[0., 60.], cruise_values=[25., 25.], personality=2)  # relaxed, his setting
  ok, logs = m.evaluate()
  assert ok
  return logs[-200:, D].mean()


def test_follow_like_you_opens_the_gap():
  off, on = steady_gap(False), steady_gap(True)
  assert off == pytest.approx(get_T_FOLLOW(2) * 20. + STOP_DISTANCE, abs=4.0)   # relaxed, about 41 m
  assert on == pytest.approx(follow_time(20.) * 20. + STOP_DISTANCE, abs=4.0)   # about 53 m
  assert on > off + 8.0


def test_ev_cap_numbers():
  assert road_load_kw(20.) == pytest.approx(0.244 * 20 + 0.0003 * 8000, rel=1e-6)
  # 14 kW: about 1.0 m/s^2 at 15 mph, 0.41 at 30 mph, 0.24 at 40 mph (10-07 fit)
  assert ev_accel(6.7, 14.) == pytest.approx(1.0, abs=0.05)
  assert ev_accel(13.4, 14.) == pytest.approx(0.41, abs=0.03)
  assert ev_accel(17.9, 14.) == pytest.approx(0.24, abs=0.03)
  assert ev_accel(0.0, 14.) > 5.0


def test_ev_cap_applies_only_when_set():
  p = Params(); p.put_bool("CrvAccelLimit", True)
  p.put("CrvEvPowerKw", 0.0)
  base = CrvAccelLimit()
  p.put("CrvEvPowerKw", 14.0)
  ev = CrvAccelLimit()
  for v in (2.0, 6.7, 13.4, 17.9, 25.0):
    assert ev.max_accel(v, 9.9) == pytest.approx(min(base.max_accel(v, 9.9), max(base.floor, ev_accel(v, 14.0))), rel=1e-6)
    assert ev.max_accel(v, 9.9) <= base.max_accel(v, 9.9) + 1e-9
  assert ev.max_accel(2.0, 9.9) == pytest.approx(base.max_accel(2.0, 9.9))       # launch unchanged (EV allows more than the pedal curve)
  assert ev.max_accel(17.9, 9.9) == pytest.approx(base.floor)                   # 40 mph: the floor still wins over 0.24
  p.put("CrvAccelFloor", 0.2)
  assert CrvAccelLimit().max_accel(17.9, 9.9) == pytest.approx(ev_accel(17.9, 14.0), abs=1e-6)
  p.put("CrvAccelFloor", 0.45); p.put("CrvEvPowerKw", 0.0)


def test_ev_cap_closed_loop_power():
  p = Params(); p.put_bool("CrvAccelLimit", True); p.put("CrvEvPowerKw", 14.0); p.put("CrvAccelFloor", 0.2)
  try:
    m = Maneuver("0 to 45 mph, EV cap", duration=60., initial_speed=0., lead_relevancy=False, cruise_values=[20., 20.], breakpoints=[0., 60.])
    ok, logs = m.evaluate()
    assert ok
    v, a = logs[1:, V], logs[1:, A]
    kw = (1839. * a * v) / 1000. + np.array([road_load_kw(x) for x in v])
    assert kw[v > 3.0].max() <= 14.0 * 1.05
    assert logs[-1, V] > 19.0
  finally:
    p.put("CrvEvPowerKw", 0.0); p.put("CrvAccelFloor", 0.45)
