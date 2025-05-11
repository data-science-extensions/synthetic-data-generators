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
from pytest import raises
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
        _output: list[datetime] = pd.date_range("2025-04-01", "2025-04-30").to_pydatetime().tolist()
        assert _input == _output


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Linear                                             ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Linear(TestCase, Dates_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator()
        cls.seed = 123

    def test_straight_line(self) -> None:
        _input: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=20,
                interpolation_nodes=[(0, 0), (5, 100), (10, 200), (15, 300)],
                level_breaks=[],
                season_eff=0,
                season_conf=None,
                noise_scale=0,
                seed=self.seed,
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.9782427006957017],
            ["2019-01-02", 17.286183996368536],
            ["2019-01-03", 39.862034518947034],
            ["2019-01-04", 60.24998335721227],
            ["2019-01-05", 82.09044515649198],
            ["2019-01-06", 103.24465273900647],
            ["2019-01-07", 121.97172544626453],
            ["2019-01-08", 143.0556298870851],
            ["2019-01-09", 162.42243898475346],
            ["2019-01-10", 181.77766075243554],
            ["2019-01-11", 201.97199538977645],
            ["2019-01-12", 218.92013457673855],
            ["2019-01-13", 241.3044667849419],
            ["2019-01-14", 259.96228743459363],
            ["2019-01-15", 281.9628262739126],
            ["2019-01-16", 302.2354685216188],
            ["2019-01-17", 325.2995346808764],
            ["2019-01-18", 343.9795958532928],
            ["2019-01-19", 363.3560061403529],
            ["2019-01-20", 384.0315443934706],
        ]
        assert _input == _expected

    def test_smooth_curve(self) -> None:
        _input: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=20,
                interpolation_nodes=[(0, 0), (5, 300), (10, 200), (15, 100)],
                level_breaks=[],
                season_eff=0,
                season_conf=None,
                noise_scale=0,
                seed=self.seed,
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.9782427006957017],
            ["2019-01-02", 108.48618399636854],
            ["2019-01-03", 193.46203451894706],
            ["2019-01-04", 250.64998335721228],
            ["2019-01-05", 286.89044515649204],
            ["2019-01-06", 303.2446527390065],
            ["2019-01-07", 301.17172544626465],
            ["2019-01-08", 288.65562988708535],
            ["2019-01-09", 264.82243898475355],
            ["2019-01-10", 234.57766075243572],
            ["2019-01-11", 201.97199538977623],
            ["2019-01-12", 166.1201345767386],
            ["2019-01-13", 138.904466784942],
            ["2019-01-14", 114.3622874345933],
            ["2019-01-15", 102.76282627391299],
            ["2019-01-16", 102.23546852161859],
            ["2019-01-17", 120.49953468087622],
            ["2019-01-18", 153.5795958532922],
            ["2019-01-19", 209.75600614035255],
            ["2019-01-20", 292.83154439346896],
        ]
        assert _input == _expected

    def test_smooth_curve_with_level_breaks(self) -> None:
        _input: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=20,
                interpolation_nodes=[(0, 0), (5, 100), (10, 200), (15, 300)],
                level_breaks=[(8, 500), (12, -500)],
                season_eff=0,
                season_conf=None,
                noise_scale=0,
                seed=self.seed,
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.9782427006957017],
            ["2019-01-02", 17.286183996368536],
            ["2019-01-03", 39.862034518947034],
            ["2019-01-04", 60.24998335721227],
            ["2019-01-05", 82.09044515649198],
            ["2019-01-06", 103.24465273900647],
            ["2019-01-07", 121.97172544626453],
            ["2019-01-08", 143.0556298870851],
            ["2019-01-09", 662.4224389847535],
            ["2019-01-10", 681.7776607524355],
            ["2019-01-11", 701.9719953897765],
            ["2019-01-12", 718.9201345767385],
            ["2019-01-13", 241.3044667849419],
            ["2019-01-14", 259.96228743459363],
            ["2019-01-15", 281.9628262739126],
            ["2019-01-16", 302.2354685216188],
            ["2019-01-17", 325.2995346808764],
            ["2019-01-18", 343.9795958532928],
            ["2019-01-19", 363.3560061403529],
            ["2019-01-20", 384.0315443934706],
        ]
        assert _input == _expected

    def test_smooth_curve_with_outliers(self) -> None:
        _input: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=20,
                interpolation_nodes=[(0, 0), (5, 100), (10, 200), (15, 300)],
                level_breaks=[],
                season_eff=0,
                season_conf=None,
                noise_scale=0,
                seed=self.seed,
                manual_outliers=[(8, 500), (12, -500)],
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.9782427006957017],
            ["2019-01-02", 17.286183996368536],
            ["2019-01-03", 39.862034518947034],
            ["2019-01-04", 60.24998335721227],
            ["2019-01-05", 82.09044515649198],
            ["2019-01-06", 103.24465273900647],
            ["2019-01-07", 121.97172544626453],
            ["2019-01-08", 143.0556298870851],
            ["2019-01-09", 500.0],
            ["2019-01-10", 181.77766075243554],
            ["2019-01-11", 201.97199538977645],
            ["2019-01-12", 218.92013457673855],
            ["2019-01-13", -500.0],
            ["2019-01-14", 259.96228743459363],
            ["2019-01-15", 281.9628262739126],
            ["2019-01-16", 302.2354685216188],
            ["2019-01-17", 325.2995346808764],
            ["2019-01-18", 343.9795958532928],
            ["2019-01-19", 363.3560061403529],
            ["2019-01-20", 384.0315443934706],
        ]
        assert _input == _expected


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Fixed Errors                                       ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_FixedErrors(TestCase, Dates_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator()
        cls.seed = 123
        cls.interpolation_nodes = [[len(cls.dates_apr_2025) * i / 4, 100 * i] for i in range(4)]

    def test_errors_one_week(self) -> None:
        _input: list[float] = (
            self.tsg.create_time_series(
                start_date=min(self.dates_apr_2025),
                n_periods=len(self.dates_apr_2025),
                interpolation_nodes=self.interpolation_nodes,
                season_conf={
                    "style": "fixed+error",
                    "period_length": 7,
                    "period_sd": 20,
                    "start_index": 10,
                },
                noise_scale=10,
                seed=self.seed,
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2025-04-01", -11.869456204174211],
            ["2025-04-02", 6.941650815023032],
            ["2025-04-03", 39.407953798506185],
            ["2025-04-04", 42.18972754853839],
            ["2025-04-05", 64.62608748622387],
            ["2025-04-06", 75.68235731824564],
            ["2025-04-07", 75.60708898255471],
            ["2025-04-08", 101.80848542452136],
            ["2025-04-09", 105.92315113976197],
            ["2025-04-10", 118.55376959084593],
            ["2025-04-11", 136.27700190981435],
            ["2025-04-12", 130.32749717821568],
            ["2025-04-13", 173.22612782595846],
            ["2025-04-14", 166.58472401618587],
            ["2025-04-15", 198.63218713717384],
            ["2025-04-16", 203.59867976014993],
            ["2025-04-17", 233.9531988104977],
            ["2025-04-18", 224.0465683820412],
            ["2025-04-19", 240.23805757565373],
            ["2025-04-20", 260.7425689923922],
            ["2025-04-21", 244.20855788173955],
            ["2025-04-22", 289.5516594957758],
            ["2025-04-23", 313.1053431498092],
            ["2025-04-24", 324.54405405541667],
            ["2025-04-25", 335.66655518784677],
            ["2025-04-26", 339.7004573606749],
            ["2025-04-27", 369.876396349877],
            ["2025-04-28", 383.2790748762765],
            ["2025-04-29", 390.5835521248177],
            ["2025-04-30", 400.05204876632604],
        ]
        assert _input == _expected


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
        _expected: list[float] = [
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
        assert _output == _expected

    def test_seasonal_sine(self) -> None:
        _input = self.dates_apr_2025
        _output: list[float] = self.tsg.generate_season_index(
            dates=_input,
            style="sin",
            period_length=7,
            start_index=0,
        ).tolist()
        _expected: list[float] = [
            0.716941869558779,
            0.28305813044122086,
            0.01253604390908819,
            0.1090842587659851,
            0.5,
            0.890915741234015,
            0.9874639560909118,
            0.7169418695587791,
            0.28305813044122097,
            0.01253604390908819,
            0.10908425876598504,
            0.4999999999999999,
            0.8909157412340147,
            0.9874639560909119,
            0.7169418695587793,
            0.2830581304412211,
            0.012536043909088246,
            0.10908425876598493,
            0.4999999999999998,
            0.8909157412340147,
            0.9874639560909119,
            0.7169418695587801,
            0.2830581304412212,
            0.012536043909088246,
            0.10908425876598488,
            0.4999999999999996,
            0.8909157412340147,
            0.987463956090912,
            0.7169418695587795,
            0.2830581304412213,
        ]
        assert _output == _expected

    def test_seasonal_sine_covar(self) -> None:
        _input = self.dates_apr_2025
        _output: list[float] = self.tsg.generate_season_index(
            dates=_input,
            style="sin_covar",
            period_length=7,
            start_index=0,
        ).tolist()
        _expected: list[float] = [
            0.33447582524571406,
            0.641571147682851,
            0.8756288658782773,
            0.9929271368785473,
            0.9607802717256118,
            0.7674514013336318,
            0.43050109253078467,
            0.0005436884451878313,
            -0.44252044329485324,
            -0.8025012025995614,
            -0.988387703899999,
            -0.9402420135781213,
            -0.6523465345428824,
            -0.18490813295942413,
            0.3429505668354759,
            0.7812504489963998,
            0.9930738900741214,
            0.9015385102276254,
            0.5227784307640564,
            -0.03008225256236298,
            -0.5779884563006392,
            -0.9346864950802858,
            -0.973308335789122,
            -0.6753688991753649,
            -0.14154706824406452,
            0.4421335490960394,
            0.8710896772483772,
            0.9964415671788943,
            0.7779673617983672,
            0.2958159927306796,
        ]
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
        _output: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=n_periods,
                interpolation_nodes=interpolation_nodes,
                level_breaks=[],
                AR=[],
                MA=[],
                randomwalk_scale=2,
                exogenous=[],
                season_eff=1,
                manual_outliers=[],
                # noise_scale=10,
                noise_scale=0,
                seed=self.seed,
                season_conf=None,
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.9782427006957017],
            ["2019-01-02", 17.286183996368536],
            ["2019-01-03", 39.862034518947034],
            ["2019-01-04", 60.24998335721227],
            ["2019-01-05", 82.09044515649198],
            ["2019-01-06", 103.24465273900647],
            ["2019-01-07", 121.97172544626453],
            ["2019-01-08", 143.0556298870851],
            ["2019-01-09", 162.42243898475346],
            ["2019-01-10", 181.77766075243554],
            ["2019-01-11", 201.97199538977645],
            ["2019-01-12", 218.92013457673855],
            ["2019-01-13", 241.3044667849419],
            ["2019-01-14", 259.96228743459363],
            ["2019-01-15", 281.9628262739126],
            ["2019-01-16", 302.2354685216188],
            ["2019-01-17", 325.2995346808764],
            ["2019-01-18", 343.9795958532928],
            ["2019-01-19", 363.3560061403529],
            ["2019-01-20", 384.0315443934706],
        ]
        assert _output == _expected

    def test_linear_trend_with_level_breaks(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[float | int]] = [[n_periods * i / 4, 100 * i] for i in range(4)]
        _output: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=n_periods,
                interpolation_nodes=interpolation_nodes,
                level_breaks=[(5, 100), (10, -200)],
                AR=[],
                MA=[],
                randomwalk_scale=2,
                exogenous=[],
                season_eff=1,
                manual_outliers=[],
                # noise_scale=10,
                noise_scale=0,
                seed=self.seed,
                season_conf=None,
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.9782427006957017],
            ["2019-01-02", 17.286183996368536],
            ["2019-01-03", 39.862034518947034],
            ["2019-01-04", 60.24998335721227],
            ["2019-01-05", 82.09044515649198],
            ["2019-01-06", 203.24465273900648],
            ["2019-01-07", 221.97172544626451],
            ["2019-01-08", 243.0556298870851],
            ["2019-01-09", 262.42243898475346],
            ["2019-01-10", 281.77766075243557],
            ["2019-01-11", 101.97199538977647],
            ["2019-01-12", 118.92013457673856],
            ["2019-01-13", 141.3044667849419],
            ["2019-01-14", 159.96228743459366],
            ["2019-01-15", 181.96282627391258],
            ["2019-01-16", 202.2354685216188],
            ["2019-01-17", 225.2995346808764],
            ["2019-01-18", 243.97959585329275],
            ["2019-01-19", 263.3560061403529],
            ["2019-01-20", 284.0315443934706],
        ]
        assert _output == _expected

    def test_linear_trend_with_outliers(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[float | int]] = [[n_periods * i / 4, 100 * i] for i in range(4)]
        _output: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=n_periods,
                interpolation_nodes=interpolation_nodes,
                level_breaks=[],
                AR=[],
                MA=[],
                randomwalk_scale=2,
                exogenous=[],
                season_eff=1,
                manual_outliers=[(5, 5000), (10, 2000)],
                # noise_scale=10,
                noise_scale=0,
                seed=self.seed,
                season_conf=None,
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.9782427006957017],
            ["2019-01-02", 17.286183996368536],
            ["2019-01-03", 39.862034518947034],
            ["2019-01-04", 60.24998335721227],
            ["2019-01-05", 82.09044515649198],
            ["2019-01-06", 5000.0],
            ["2019-01-07", 121.97172544626453],
            ["2019-01-08", 143.0556298870851],
            ["2019-01-09", 162.42243898475346],
            ["2019-01-10", 181.77766075243554],
            ["2019-01-11", 2000.0],
            ["2019-01-12", 218.92013457673855],
            ["2019-01-13", 241.3044667849419],
            ["2019-01-14", 259.96228743459363],
            ["2019-01-15", 281.9628262739126],
            ["2019-01-16", 302.2354685216188],
            ["2019-01-17", 325.2995346808764],
            ["2019-01-18", 343.9795958532928],
            ["2019-01-19", 363.3560061403529],
            ["2019-01-20", 384.0315443934706],
        ]
        assert _output == _expected

    def test_linear_trend_with_seasonality(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[float | int]] = [[n_periods * i / 4, 100 * i] for i in range(4)]
        _output: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=n_periods,
                interpolation_nodes=interpolation_nodes,
                level_breaks=[],
                AR=[],
                MA=[],
                randomwalk_scale=2,
                exogenous=[],
                manual_outliers=[],
                # noise_scale=10,
                noise_scale=0,
                seed=self.seed,
                season_eff=0.5,
                season_conf={"style": "sin", "period_length": 7, "start_index": 0},
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.2691001905568096],
            ["2019-01-02", 14.83968653413102],
            ["2019-01-03", 39.612178411429475],
            ["2019-01-04", 56.96382096962005],
            ["2019-01-05", 61.56783386736899],
            ["2019-01-06", 57.253509577296256],
            ["2019-01-07", 61.75038417606308],
            ["2019-01-08", 91.77434451600733],
            ["2019-01-09", 139.43494302438967],
            ["2019-01-10", 180.6382743839936],
            ["2019-01-11", 190.9560126854861],
            ["2019-01-12", 164.19010093255392],
            ["2019-01-13", 133.8134928405493],
            ["2019-01-14", 131.61059304229036],
            ["2019-01-15", 180.8873483664645],
            ["2019-01-16", 259.46036518524085],
            ["2019-01-17", 323.26055005569367],
            ["2019-01-18", 325.21821623115306],
            ["2019-01-19", 272.51700460526473],
            ["2019-01-20", 212.96167037819444],
        ]
        assert _output == _expected

    def test_sine_trend(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[float | int]] = [[n_periods * i / 4, 100 * i] for i in range(4)]
        _output: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=n_periods,
                interpolation_nodes=interpolation_nodes,
                level_breaks=[],
                AR=[],
                MA=[],
                randomwalk_scale=2,
                exogenous=[],
                season_eff=0,
                manual_outliers=[],
                # noise_scale=10,
                noise_scale=0,
                seed=self.seed,
                season_conf={"style": "sin", "period_length": 7, "start_index": 0},
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -0.5599576804179172],
            ["2019-01-02", 12.393189071893506],
            ["2019-01-03", 39.362322303911924],
            ["2019-01-04", 53.67765858202784],
            ["2019-01-05", 41.04522257824599],
            ["2019-01-06", 11.262366415586047],
            ["2019-01-07", 1.5290429058616282],
            ["2019-01-08", 40.493059144929546],
            ["2019-01-09", 116.44744706402585],
            ["2019-01-10", 179.49888801555167],
            ["2019-01-11", 179.94002998119575],
            ["2019-01-12", 109.4600672883693],
            ["2019-01-13", 26.322518896156698],
            ["2019-01-14", 3.2588986499870565],
            ["2019-01-15", 79.81187045901642],
            ["2019-01-16", 216.68526184886287],
            ["2019-01-17", 321.22156543051096],
            ["2019-01-18", 306.4568366090133],
            ["2019-01-19", 181.67800307017654],
            ["2019-01-20", 41.891796362918306],
        ]
        assert _output == _expected

    def test_sine_covar_trend(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[float | int]] = [[n_periods * i / 4, 100 * i] for i in range(4)]
        _output: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=n_periods,
                interpolation_nodes=interpolation_nodes,
                level_breaks=[],
                AR=[],
                MA=[],
                randomwalk_scale=2,
                exogenous=[],
                season_eff=0,
                manual_outliers=[],
                # noise_scale=10,
                noise_scale=0,
                seed=self.seed,
                season_conf={"style": "sin_covar", "period_length": 7, "start_index": 0},
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.3165683408441968],
            ["2019-01-02", 6.1958670907614435],
            ["2019-01-03", 4.957686441520703],
            ["2019-01-04", 0.4261398853553669],
            ["2019-01-05", 3.219564952961186],
            ["2019-01-06", 24.00939931425177],
            ["2019-01-07", 69.46276438378274],
            ["2019-01-08", 142.97785219409644],
            ["2019-01-09", 234.2976886853178],
            ["2019-01-10", 327.6544521120002],
            ["2019-01-11", 401.5986321651788],
            ["2019-01-12", 424.7580427239645],
            ["2019-01-13", 398.71859946181684],
            ["2019-01-14", 308.0314286439855],
            ["2019-01-15", 185.26351517674144],
            ["2019-01-16", 66.11387303646684],
            ["2019-01-17", 2.253060336036911],
            ["2019-01-18", 33.868743459014524],
            ["2019-01-19", 173.40132344160438],
            ["2019-01-20", 395.5840783038293],
        ]
        assert _output == _expected


## --------------------------------------------------------------------------- #
##  Validations                                                             ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Validations(TestCase, Dates_Mixin):

    def setUp(self) -> None:
        self.tsg = TimeSeriesGenerator()
        self.seed = 123

    def test_value_is_between(self) -> None:
        _input: bool = self.tsg._value_is_between(0.7, 0.5, 1)
        _expected = True
        assert _input == _expected

    def test_value_is_not_between(self) -> None:
        _input: bool = self.tsg._value_is_between(0.7, 0.8, 1)
        _expected = False
        assert _input == _expected

    def test_assert_value_is_between(self) -> None:
        assert self.tsg._assert_value_is_between(0.7, 0.5, 1) is None

    def test_assert_value_is_not_between(self) -> None:
        with raises(AssertionError):
            self.tsg._assert_value_is_between(0.7, 0.8, 1)

    def test_all_values_are_between(self) -> None:
        _input: bool = self.tsg._all_values_are_between([0.7, 0.8], 0.5, 1)
        _expected = True
        assert _input == _expected

    def test_not_all_values_are_between(self) -> None:
        _input: bool = self.tsg._all_values_are_between([0.3, 0.8], 0.5, 1)
        _expected = False
        assert _input == _expected

    def test_assert_all_values_are_between(self) -> None:
        assert self.tsg._assert_all_values_are_between([0.7, 0.8], 0.5, 1) is None

    def test_assert_not_all_values_are_between(self) -> None:
        with raises(AssertionError):
            self.tsg._assert_all_values_are_between([0.3, 0.8], 0.5, 1)
