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
import pandas as pd
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


class Dates_Mixin:

    @class_property
    def dates_apr_2025(cls) -> datetime_list:
        return TimeSeriesGenerator._generate_dates(
            start_date="2025-04-01",
            end_date="2025-04-30",
        )


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Generics                                           ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Generics(TestCase, Dates_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator()
        cls.seed = 123

    def test_random_generator_1(self) -> None:
        _input: list[float] = self.tsg._random_generator().normal(loc=0, scale=1, size=10).tolist()
        _expected: list[float] = np.random.default_rng().normal(loc=0, scale=1, size=10).tolist()
        assert _input != _expected

    def test_random_generator_2(self) -> None:
        _input: list[float] = self.tsg._random_generator(seed=self.seed).normal(loc=0, scale=1, size=10).tolist()
        _expected: list[float] = np.random.default_rng(self.seed).normal(loc=0, scale=1, size=10).tolist()
        assert _input == _expected

    def test_generate_dates(self) -> None:
        _input = self.dates_apr_2025
        _output = pd.date_range("2025-04-01", "2025-04-30").to_pydatetime().tolist()
        assert _input == _output


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Seasonalities                                      ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Seasonalities(TestCase, Dates_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator()
        cls.seed = 123

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


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Creation                                           ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Creation(TestCase, Dates_Mixin):

    def setUp(self) -> None:
        self.tsg = TimeSeriesGenerator()
        self.seed = 123

    def test_linear_trend(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[float | int]] = [[n_periods * i / 4, 100 * i] for i in range(4)]
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
            [Timestamp("2019-01-01 00:00:00"), -1.9782427006957017],
            [Timestamp("2019-01-02 00:00:00"), 17.286183996368536],
            [Timestamp("2019-01-03 00:00:00"), 39.862034518947034],
            [Timestamp("2019-01-04 00:00:00"), 60.24998335721227],
            [Timestamp("2019-01-05 00:00:00"), 82.09044515649198],
            [Timestamp("2019-01-06 00:00:00"), 103.24465273900647],
            [Timestamp("2019-01-07 00:00:00"), 121.97172544626453],
            [Timestamp("2019-01-08 00:00:00"), 143.0556298870851],
            [Timestamp("2019-01-09 00:00:00"), 162.42243898475346],
            [Timestamp("2019-01-10 00:00:00"), 181.77766075243554],
            [Timestamp("2019-01-11 00:00:00"), 201.97199538977645],
            [Timestamp("2019-01-12 00:00:00"), 218.92013457673855],
            [Timestamp("2019-01-13 00:00:00"), 241.3044667849419],
            [Timestamp("2019-01-14 00:00:00"), 259.96228743459363],
            [Timestamp("2019-01-15 00:00:00"), 281.9628262739126],
            [Timestamp("2019-01-16 00:00:00"), 302.2354685216188],
            [Timestamp("2019-01-17 00:00:00"), 325.2995346808764],
            [Timestamp("2019-01-18 00:00:00"), 343.9795958532928],
            [Timestamp("2019-01-19 00:00:00"), 363.3560061403529],
            [Timestamp("2019-01-20 00:00:00"), 384.0315443934706],
        ]
        assert _output == _expected
