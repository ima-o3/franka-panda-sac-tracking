"""Shared position references; SI units, smooth assembly segments, no contact task."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Trajectory:
    home: np.ndarray
    kind: str = "circle"
    duration: float = 20.0
    radius: float = 0.08
    omega: float = 0.5
    pick_offset: tuple = (-0.08, -0.10, -0.10)
    place_offset: tuple = (-0.12, 0.10, -0.08)
    clearance: float = 0.08
    phase_weights: tuple = (1, 3, 2, 1, 2, 4, 2, 4, 1, 2, 3)
    names: tuple = field(init=False, default=("home", "approach_pick", "descend_pick",
        "dwell_pick", "lift", "transfer", "approach_insertion", "insert",
        "dwell_place", "retract", "return_home"))

    def __post_init__(self):
        self.home = np.asarray(self.home, dtype=float)
        if self.kind not in ("circle", "assembly") or self.duration <= 0:
            raise ValueError("Expected circle/assembly and a positive duration")
        if self.radius <= 0 or self.clearance <= 0 or self.omega <= 0:
            raise ValueError("Radius, clearance and omega must be positive")
        weights = np.asarray(self.phase_weights, dtype=float)
        if weights.shape != (11,) or np.any(weights <= 0):
            raise ValueError("Provide 11 positive phase weights")
        pick = self.home + np.asarray(self.pick_offset)
        place = self.home + np.asarray(self.place_offset)
        up = np.array([0., 0., self.clearance])
        self.points = np.array([self.home, self.home, pick + up, pick, pick,
            pick + up, place + up + [0, 0, 0.03], place + up, place, place,
            place + up, self.home])
        self.times = np.r_[0., np.cumsum(weights / weights.sum() * self.duration)]

    def sample(self, t):
        if self.kind == "circle":
            p = self.omega * t
            return (self.home + self.radius * np.array([np.cos(p)-1, np.sin(p), 0]),
                    self.radius * self.omega * np.array([-np.sin(p), np.cos(p), 0]), "circle")
        i = min(np.searchsorted(self.times[1:], t, side="right"), 10)
        length = self.times[i+1] - self.times[i]
        u = np.clip((t-self.times[i])/length, 0., 1.)
        blend = 10*u**3 - 15*u**4 + 6*u**5
        rate = (30*u**2 - 60*u**3 + 30*u**4)/length
        delta = self.points[i+1]-self.points[i]
        return self.points[i] + blend*delta, rate*delta, self.names[i]
