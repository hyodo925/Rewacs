# from collections import namedtuple

# ActionXY = namedtuple('ActionXY', ['vx', 'vy'])
# ActionRot = namedtuple('ActionRot', ['v', 'r'])
# ActionVW = namedtuple('ActionVW', ['v', 'w'])

from typing import NamedTuple


class ActionXY(NamedTuple):
    vx: float
    vy: float


class ActionXYW(NamedTuple):
    vx: float
    vy: float
    vw: float


class ActionRot(NamedTuple):
    v: float
    r: float


class ActionVW(NamedTuple):
    v: float
    w: float


class ActionMecanumTorque(NamedTuple):
    t1: float
    t2: float
    t3: float
    t4: float
