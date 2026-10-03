"""CR-V coast first (CrvCoastFirst): with Experimental / Dynamic Experimental Control blending in the driving
model, a model braking request is held at coasting for its first CrvCoastFirstTime seconds. Braking for a lead,
late stops and hard requests are never held.

Closed-loop cases use the maneuver simulator (real planner, ideal car) with a fake driving model:
  phantom   the model asks for -1.2 m/s^2 for 1.2 s with nothing there (the 10-02 drives had ~1 every 13 min)
  red_far   a red light with no lead, the model starts stopping 160 m out (DEC went blended ~140 m out on 10-02)
  red_late  the same light first seen 70 m out at 38 mph: needs ~2 m/s^2, so no hold
  lead      the stock "lead brakes to 0 at 3+ m/s^2" maneuver in experimental mode
"""
import numpy as np
import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant
from openpilot.selfdrive.test.longitudinal_maneuvers.test_longitudinal import create_maneuvers
from openpilot.sunnypilot.selfdrive.controls.lib.crv_coast_first import (CrvCoastFirst, HARD_REQUEST, MIN_SPEED,
                                                                         RELEASE_TIME)

DT = 0.05
COAST = -0.3
V0 = 17.0  # m/s, 38 mph


def set_params(on: bool, hold: float | None = None):
  p = Params()
  p.put_bool("CrvCoastFirst", on)
  if hold is not None:
    p.put("CrvCoastFirstTime", hold)


# ---------- the hold itself ----------

def feed(cf, a_model, n, v=V0, end_x=200., urgent=False):
  return [cf.limit(a_model, v, COAST, end_x, urgent) for _ in range(n)]


def test_off_passes_through():
  set_params(False)
  cf = CrvCoastFirst()
  assert feed(cf, -1.0, 10) == [-1.0] * 10


def test_holds_then_applies():
  set_params(True)
  cf = CrvCoastFirst()
  out = feed(cf, -1.0, 60)  # 3 s
  held = int(round(2.0 / DT))
  assert all(a == pytest.approx(COAST) for a in out[:held - 1])
  assert all(a == -1.0 for a in out[held + 1:])


def test_hold_time_setting():
  set_params(True, 0.0)
  assert feed(CrvCoastFirst(), -1.0, 5) == [-1.0] * 5
  set_params(True, 1.0)
  out = feed(CrvCoastFirst(), -1.0, 40)
  assert out[15] == pytest.approx(COAST) and out[25] == -1.0


def test_no_hold_for_coasting_or_positive_requests():
  set_params(True)
  cf = CrvCoastFirst()
  assert feed(cf, -0.3, 5) == [-0.3] * 5
  assert feed(cf, 0.5, 5) == [0.5] * 5
  assert not cf.holding


@pytest.mark.parametrize("case", ["hard", "urgent", "slow", "close_stop"])
def test_never_held(case):
  set_params(True)
  cf = CrvCoastFirst()
  if case == "hard":
    out = feed(cf, HARD_REQUEST - 0.01, 10)
  elif case == "urgent":
    out = feed(cf, -1.0, 10, urgent=True)
  elif case == "slow":
    out = feed(cf, -1.0, 10, v=MIN_SPEED - 0.1)
  else:  # model plans to stop 70 m ahead at 38 mph: after a 2 s coast that needs about 2.3 m/s^2
    out = feed(cf, -1.0, 10, end_x=70.)
  assert out[0] != pytest.approx(COAST) and min(out) < -0.9


def test_brief_release_does_not_restart_the_hold():
  set_params(True)
  cf = CrvCoastFirst()
  feed(cf, -1.0, 50)                       # 2.5 s: hold over, braking
  feed(cf, -0.2, int(RELEASE_TIME / DT) - 4)  # let go for less than RELEASE_TIME
  assert feed(cf, -1.0, 1) == [-1.0]       # same request continues: no second hold
  feed(cf, -0.2, int(RELEASE_TIME / DT) + 2)
  assert feed(cf, -1.0, 1)[0] == pytest.approx(COAST)  # a new request gets a new hold


def test_reset_starts_a_new_request():
  set_params(True)
  cf = CrvCoastFirst()
  feed(cf, -1.0, 50)
  cf.reset()
  assert feed(cf, -1.0, 1)[0] == pytest.approx(COAST)


def test_params_registered_with_defaults():
  p = Params()
  assert p.get_bool("CrvCoastFirst") is False
  assert float(p.get("CrvCoastFirstTime", return_default=True)) == pytest.approx(2.0)


# ---------- closed loop ----------

def drive(scenario: str, on: bool, duration=30.):
  set_params(on)
  plant = Plant(lead_relevancy=False, speed=V0, distance_lead=200., enabled=True, e2e=True)
  real_update = plant.planner.update
  line = 160.0 if scenario == "red_far" else 70.0
  seen = {"t": None}

  def update(sm):
    md = sm['modelV2']
    t = plant.current_time
    if scenario == "phantom" and 5.0 <= t < 6.2:
      md.action.desiredAcceleration = -1.2
    elif scenario in ("red_far", "red_late"):
      d = line - plant.distance
      if d < 165.0 if scenario == "red_far" else d < 72.0:
        seen["t"] = seen["t"] if seen["t"] is not None else t
        md.action.desiredAcceleration = float(-np.clip(plant.speed ** 2 / (2 * max(d - 2.0, 0.5)), 0., 3.5))
        md.position.x = [float(min(x, max(d - 2.0, 0.))) for x in md.position.x]
    real_update(sm)

  plant.planner.update = update
  rows = []
  while plant.current_time < duration:
    log = plant.step(v_cruise=V0)
    rows.append((plant.current_time, log["distance"], log["speed"], log["acceleration"]))
  return np.array(rows), seen["t"], line


def test_phantom_brake_is_absorbed():
  off, _, _ = drive("phantom", False, 12.)
  on, _, _ = drive("phantom", True, 12.)
  w = lambda r: r[(r[:, 0] > 5.0) & (r[:, 0] < 8.0)]
  assert w(off)[:, 3].min() < -1.0                # stock: brakes for nothing
  assert w(on)[:, 3].min() > -0.45                 # coast first: only lifts off
  assert on[-1, 2] > off[-1, 2]                    # and keeps more speed


def test_red_light_far_still_stops_gently_before_the_line():
  # the fake model aims 2 m short of the line and both runs end about 2 m past that aim point (stock 159.97 m,
  # coast first 160.13 m on 2026-10-03), so "at the line" is judged with 0.5 m and the two must end within 0.3 m
  ends = {}
  for on in (False, True):
    r, t_seen, line = drive("red_far", on)
    assert r[-1, 2] < 0.1, "stopped"
    assert r[:, 1].max() <= line + 0.5, "at the line"
    assert r[:, 3].min() > -2.0, "gentle"
    ends[on] = r[:, 1].max()
  assert abs(ends[True] - ends[False]) < 0.3
  r, t_seen, _ = drive("red_far", True)
  first = r[(r[:, 0] > t_seen + 0.2) & (r[:, 0] < t_seen + 1.5)]
  assert first[:, 3].min() > -0.45, "lifts off first, like the owner"


def test_red_light_late_is_not_held():
  off, _, line = drive("red_late", False, 15.)
  on, _, _ = drive("red_late", True, 15.)
  assert np.allclose(off[:, 1:], on[:, 1:], atol=1e-6)
  assert on[:, 1].max() <= line + 0.5


def test_lead_braking_never_delayed():
  m = {x.title: x for x in create_maneuvers({"e2e": True, "force_decel": False})}[
    "steady state following a car at 20m/s, then lead decel to 0mph at 3+m/s^2"]
  set_params(False)
  ok_off, off = m.evaluate()
  set_params(True)
  ok_on, on = m.evaluate()
  assert ok_off and ok_on
  assert on[:, 5].min() == pytest.approx(off[:, 5].min(), abs=0.05)   # same hardest braking
  assert on[:, 6].min() == pytest.approx(off[:, 6].min(), abs=0.5)    # same closest gap
