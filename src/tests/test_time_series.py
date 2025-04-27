# ---------------------------------------------------------------------------- #
#                                                                              #
#     Setup                                                                 ####
#                                                                              #
# ---------------------------------------------------------------------------- #


## --------------------------------------------------------------------------- #
##  Imports                                                                 ####
## --------------------------------------------------------------------------- #


# ## Python StdLib Imports ----
from datetime import datetime
from unittest import TestCase

# ## Python Third Party Imports ----
import numpy as np
from numpy.typing import NDArray
from pandas import Timestamp
from toolbox_python.classes import class_property
from toolbox_python.collection_types import datetime_list

# ## Local First Party Imports ----
from synthetic_data_generators.time_series import (
    TimeSeriesGenerator,
)


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Test Suite                                                            ####
#                                                                              #
# ---------------------------------------------------------------------------- #


class Dates_Mixin(TestCase):

    @class_property
    def dates_apr_2025(cls) -> datetime_list:
        return TimeSeriesGenerator._generate_dates(
            start_date="2025-04-01",
            end_date="2025-04-30",
        )


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Seasonalities                                      ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Seasonalidies(Dates_Mixin):

    def setUp(self) -> None:
        self.tsg = TimeSeriesGenerator()
        self.seed = 123

    def test_holidays_in_april_2025(self) -> None:
        _input = self.dates_apr_2025
        _output: NDArray[np.int_] = self.tsg.generate_season_index(
            dates=_input,
            style="holiday",
            season_dates=[
                (datetime(2025, 4, 18), 4),  # <-- Easter
                (datetime(2025, 4, 25), 1),  # <-- ANZAC
            ],
        ).tolist()
        _expected: NDArray[np.int_] = np.array(
            [
                0.0,  # 1st: Tue
                0.0,  # 2nd: Wed
                0.0,  # 3rd: Thu
                0.0,  # 4th: Fri
                0.0,  # 5th: Sat
                0.0,  # 6th: Sun
                0.0,  # 7th: Mon
                0.0,  # 8th: Tue
                0.0,  # 9th: Wed
                0.0,  # 10th: Thu
                0.0,  # 11th: Fri
                0.0,  # 12th: Sat
                0.0,  # 13th: Sun
                0.0,  # 14th: Mon
                0.0,  # 15th: Tue
                0.0,  # 16th: Wed
                0.0,  # 17th: Thu
                1.0,  # 18th: Easter Fri
                1.0,  # 19th: Easter Sat
                1.0,  # 20th: Easter Sun
                1.0,  # 21st: Easter Mon
                0.0,  # 22nd: Tue
                0.0,  # 23rd: Wed
                0.0,  # 24th: Thu
                1.0,  # 25th: ANZAC Day
                0.0,  # 26th: Sat
                0.0,  # 27th: Sun
                0.0,  # 28th: Mon
                0.0,  # 29th: Tue
                0.0,  # 30th: Wed
            ]
        ).tolist()
        assert _output == _expected

    def test_markov_for_april_2025(self) -> None:
        _input = self.dates_apr_2025
        _output: NDArray[np.int_] = self.tsg.generate_season_index(
            dates=_input,
            style="semi-markov",
            period_length=7,
            period_sd=2,
            start_index=4,
            seed=self.seed,
        ).tolist()
        output = list(zip(_input, _output))
        _expected: NDArray[np.int_] = np.array(
            [
                0.0,  # 1st: Tue
                0.0,  # 2nd: Wed
                0.0,  # 3rd: Thu
                0.0,  # 4th: Fri
                1.0,  # 5th: Sat
                0.0,  # 6th: Sun
                0.0,  # 7th: Mon
                0.0,  # 8th: Tue
                1.0,  # 9th: Wed
                0.0,  # 10th: Thu
                0.0,  # 11th: Fri
                0.0,  # 12th: Sat
                0.0,  # 13th: Sun
                0.0,  # 14th: Mon
                0.0,  # 15th: Tue
                0.0,  # 16th: Wed
                0.0,  # 17th: Thu
                0.0,  # 18th: Easter Fri
                1.0,  # 19th: Easter Sat
                0.0,  # 20th: Easter Sun
                0.0,  # 21st: Easter Mon
                0.0,  # 22nd: Tue
                0.0,  # 23rd: Wed
                1.0,  # 24th: Thu
                0.0,  # 25th: ANZAC Day
                0.0,  # 27th: Sun
                0.0,  # 26th: Sat
                0.0,  # 28th: Mon
                1.0,  # 29th: Tue
                0.0,  # 30th: Wed
            ]
        ).tolist()
        expected = list(zip(_input, _expected))
        assert output == expected


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Creation                                           ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Creation(Dates_Mixin):

    def setUp(self) -> None:
        self.tsg = TimeSeriesGenerator()
        self.seed = 123

    def test_linear_trend(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[int]] = [[n_periods * i / 4, 100 * i] for i in range(4)]
        _output = self.tsg.create_time_series(
            start_date=datetime(2019, 1, 1),
            n_periods=n_periods,
            interpolation_nodes=interpolation_nodes,
            level_breaks=[],
            AR=[],
            MA=[],
            randomwalk_scale=2,
            exogenous=[],
            season_eff=1,
            man_outliers=[],
            # noise_scale=10,
            noise_scale=0,
            seed=self.seed,
            season_conf=None,
        ).values.tolist()
        _expected = [
            [Timestamp("2019-01-01 00:00:00"), -11.869456204174211],
            [Timestamp("2019-01-02 00:00:00"), 13.608317481689703],
            [Timestamp("2019-01-03 00:00:00"), 52.74128713183952],
            [Timestamp("2019-01-04 00:00:00"), 62.18972754853841],
            [Timestamp("2019-01-05 00:00:00"), 91.29275415289055],
            [Timestamp("2019-01-06 00:00:00"), 109.01569065157898],
            [Timestamp("2019-01-07 00:00:00"), 115.60708898255473],
            [Timestamp("2019-01-08 00:00:00"), 148.47515209118805],
            [Timestamp("2019-01-09 00:00:00"), 159.2564844730953],
            [Timestamp("2019-01-10 00:00:00"), 178.55376959084595],
            [Timestamp("2019-01-11 00:00:00"), 202.94366857648103],
            [Timestamp("2019-01-12 00:00:00"), 203.66083051154902],
            [Timestamp("2019-01-13 00:00:00"), 253.22612782595849],
            [Timestamp("2019-01-14 00:00:00"), 253.25139068285253],
            [Timestamp("2019-01-15 00:00:00"), 291.96552047050716],
            [Timestamp("2019-01-16 00:00:00"), 303.59867976015],
            [Timestamp("2019-01-17 00:00:00"), 340.61986547716435],
            [Timestamp("2019-01-18 00:00:00"), 337.37990171537456],
            [Timestamp("2019-01-19 00:00:00"), 360.2380575756537],
            [Timestamp("2019-01-20 00:00:00"), 387.4092356590589],
        ]
        assert _output == _expected
