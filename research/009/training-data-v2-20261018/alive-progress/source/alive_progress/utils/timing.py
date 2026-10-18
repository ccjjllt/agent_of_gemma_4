from collections import namedtuple
from math import floor

from typing import Callable

TimeDisplay = namedtuple('TimeDisplay',
                         'round, sec_prec, min_prec, hour_prec, prefix, '
                         'round_sec_on_min, clear_sec_on_hour')
RUN = TimeDisplay(0, .0, 2.0, 2.0, '', False, False)
END = TimeDisplay(1, .1, 4.1, 4.1, '', False, False)
ETA = RUN._replace(prefix='~', round_sec_on_min=True, clear_sec_on_hour=True)


def time_display(seconds: float, conf: TimeDisplay) -> str:
    seconds = round(seconds, conf.round)
    if seconds < 60.:
        return '{}{:{}f}s'.format(conf.prefix, seconds, conf.sec_prec)

    minutes, seconds = divmod(seconds, 60.)
    if minutes < 60.:
        if conf.round_sec_on_min:
            seconds = floor(seconds / 10) * 10
        return '{}{:.0f}:{:0{}f}'.format(conf.prefix, minutes, seconds, conf.min_prec)

    hours, minutes = divmod(minutes, 60.)
    if conf.clear_sec_on_hour:
        seconds = 0
    return '{}{:.0f}:{:02.0f}:{:0{}f}'.format(conf.prefix, hours, minutes, seconds, conf.hour_prec)


def eta_text(eta):
    if eta is None or eta < 0.:
        return '-'
    return elapsed_text(eta, False)


def simple_eta(logic_total, pos, rate):
    return (logic_total - pos) / rate


def gen_simple_exponential_smoothing_eta(alfa, logic_total):
    """Implements a generator with a simple exponential smoothing of the
    eta time series.
    Given alfa and y_hat (t-1), we can calculate the next y_hat:
        y_hat = alfa * y + (1 - alfa) * y_hat
        y_hat = alfa * y + y_hat - alfa * y_hat
        y_hat = y_hat + alfa * (y - y_hat)

    Args:
        alfa (float): the smoothing coefficient
        logic_total (float):

    Returns:

    """
    pos = rate = None
    while not rate:
        pos, rate = yield
    y_hat = simple_eta(logic_total, pos, rate)
    while True:
        temp, rate = yield y_hat
        if temp == pos:  # reduce numbers bouncing around.
            continue
        pos = temp
        y = simple_eta(logic_total, pos, rate)
        y_hat += alfa * (y - y_hat)
