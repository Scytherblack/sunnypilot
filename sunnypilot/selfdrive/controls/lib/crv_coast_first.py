"""
CR-V coast first (change 2a, 2026-10-03).

With Dynamic Experimental Control in "blended" mode the planner commands min(model request, plan), where the
model request is modelV2.action.desiredAcceleration. On the 10-02 drives the model asked to brake while the
driver kept going about once every 13 minutes, mostly for 1 to 1.5 s (often approaching a light that was green).
The driver himself lifts off and coasts 3 to 4 s before braking for a red light.

When CrvCoastFirst is on, each continuous model braking request above MIN_SPEED is held at coasting for its first
CrvCoastFirstTime seconds, and applies in full after that. Only the model's term is held: the plan's own braking
(lead car, cruise speed) is unchanged, because the planner still takes the minimum of the two.

The hold is skipped, so the model's braking applies at once, when
  - the model asks for HARD_REQUEST or more (something urgent),
  - stopping where the model plans to stop would need more than MAX_STOP_DECEL after the rest of the hold,
  - Dynamic Experimental Control rates the stop urgent.
"""
import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.crv_accel_limit import _read_float

MIN_SPEED = 5.0          # m/s (11 mph); below this the model's stop-and-go handling is left alone
HARD_REQUEST = -1.5      # m/s^2; a request this strong is never held
MAX_STOP_DECEL = 1.5     # m/s^2; the hold never leaves a stop that needs more than this
COAST_RANGE = (-0.6, -0.1)  # m/s^2; the coasting value used for the hold, from the planner's pitch-corrected fit
REQUEST_MARGIN = 0.05    # m/s^2; a request counts as braking once it is this far below coasting
RELEASE_TIME = 1.0       # s; a request must stop for this long before a new one gets a new hold
DEFAULT_TIME = 2.0
TIME_RANGE = (0.0, 4.0)
PARAMS_UPDATE_PERIOD = 3.0  # s


class CrvCoastFirst:
  def __init__(self, params: Params | None = None, dt: float = DT_MDL):
    self.params = params or Params()
    self.dt = dt
    self.frame = 0
    self.request_t = 0.0   # how long the current model braking request has lasted
    self.release_t = 0.0   # how long the model has not been asking to brake
    self.holding = False   # true while the model's request is being held at coasting
    self._read_params()

  def _read_params(self) -> None:
    self.enabled = self.params.get_bool("CrvCoastFirst")
    self.hold_time = _read_float(self.params, "CrvCoastFirstTime", DEFAULT_TIME, TIME_RANGE)

  def update(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / self.dt) == 0:
      self._read_params()
    self.frame += 1

  def reset(self) -> None:
    """The model is not in control (Dynamic Experimental Control in acc mode): forget any request."""
    self.request_t = 0.0
    self.release_t = RELEASE_TIME
    self.holding = False

  def limit(self, a_model: float, v_ego: float, a_coast: float, plan_end_x: float, urgent: bool) -> float:
    """The model's acceleration request to use this cycle.

    a_model     modelV2.action.desiredAcceleration
    a_coast     the car's coasting acceleration (planner's pitch-corrected fit)
    plan_end_x  distance to the end of the model's planned path, where it plans to be in 10 s (its stop point when stopping)
    urgent      Dynamic Experimental Control rates the slowdown urgent
    """
    coast = float(np.clip(a_coast, *COAST_RANGE))
    if a_model < coast - REQUEST_MARGIN:
      if self.release_t >= RELEASE_TIME:
        self.request_t = 0.0  # a new request
      self.release_t = 0.0
      self.request_t += self.dt
    else:
      self.release_t += self.dt
      self.holding = False
      return a_model

    self.holding = False
    if not self.enabled or v_ego < MIN_SPEED or urgent or a_model <= HARD_REQUEST:
      return a_model
    t_left = self.hold_time - self.request_t
    if t_left <= 0.0:
      return a_model
    d_after = plan_end_x - v_ego * t_left
    v_after = max(v_ego + coast * t_left, 0.0)
    if d_after <= 1.0 or v_after ** 2 / (2.0 * d_after) > MAX_STOP_DECEL:
      return a_model
    self.holding = True
    return max(a_model, coast)
