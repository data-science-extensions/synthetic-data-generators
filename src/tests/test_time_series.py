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
from functools import lru_cache, partial
from unittest import TestCase

# ## Python Third Party Imports ----
import numpy as np
from numpy.random import Generator as RandomGenerator
from numpy.typing import NDArray
from pytest import raises

# ## Local First Party Imports ----
from synthetic_data_generators.time_series import TimeSeriesGenerator


## --------------------------------------------------------------------------- #
##  Constants                                                               ####
## --------------------------------------------------------------------------- #


RELATIVE_TOLERANCE = 1e-7
ABSOLUTE_TOLERANCE = 1e-10


## --------------------------------------------------------------------------- #
##  Partials                                                                ####
## --------------------------------------------------------------------------- #


assert_all_close = partial(
    np.testing.assert_allclose,
    rtol=RELATIVE_TOLERANCE,
    atol=ABSOLUTE_TOLERANCE,
    err_msg="Numeric values do not match between actual and expected time series.",
)


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Helper Functions                                                      ####
#                                                                              #
# ---------------------------------------------------------------------------- #


def assert_timeseries_equal(
    actual: list[list[str | float]],
    expected: list[list[str | float]],
) -> None:
    """
    !!! note "Summary"
        Compare two time series with tolerance for floating-point differences.

    ???+ abstract "Details"
        - This helper function ensures tests pass consistently across different platforms (Ubuntu, macOS, Windows) and CPU architectures (x86-64, ARM64).
        - Different operating systems and CPU architectures use different math library implementations, which can produce slightly different floating-point results in the last decimal places.
        - The function compares date strings exactly and numeric values with configurable relative and absolute tolerances.
        - Uses `numpy.testing.assert_allclose()` for robust floating-point comparison.
        - Provides clear error messages indicating which row failed and the expected vs actual values.

    Params:
        actual (list[list[str | float]]):
            The actual time series data from the test.<br>
            Each inner list contains [date_string, numeric_value].
        expected (list[list[str | float]]):
            The expected time series data.<br>
            Each inner list contains [date_string, numeric_value].

    Raises:
        (AssertionError):
            If the time series lengths don't match.
        (AssertionError):
            If any date strings don't match exactly.
        (AssertionError):
            If any numeric values differ beyond the specified tolerances.
    """

    # Initial length checks
    assert len(actual) == len(expected), f"Length mismatch: {len(actual)} != {len(expected)}"
    assert all(len(a) == 2 for a in actual), "Actual time series rows must each have exactly 2 elements."
    assert all(len(e) == 2 for e in expected), "Expected time series rows must each have exactly 2 elements."

    # Extract dates and values
    actual_dates: list[str] = [row[0] for row in actual]  # type:ignore
    expected_dates: list[str] = [row[0] for row in expected]  # type:ignore
    actual_values: list[float] = [row[1] for row in actual]  # type:ignore
    expected_values: list[float] = [row[1] for row in expected]  # type:ignore

    # Compare dates exactly
    assert actual_dates == expected_dates, "Date strings do not match between actual and expected time series."

    # Compare numeric values with tolerance
    assert_all_close(actual=actual_values, desired=expected_values)


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Test Suite                                                            ####
#                                                                              #
# ---------------------------------------------------------------------------- #


class Default_Mixin:

    seed: int = 123

    @classmethod
    @lru_cache
    def dates_apr_2025(cls) -> list[datetime]:
        return TimeSeriesGenerator._get_dates(start_date=datetime(2025, 4, 1), end_date=datetime(2025, 4, 30))


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Generics                                           ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Generics(TestCase, Default_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator(cls.seed)

    def test_random_generator_1(self) -> None:
        _input: list[float] = self.tsg._get_random_generator().normal(loc=0, scale=1, size=10).tolist()
        _expected: list[float] = np.random.default_rng().normal(loc=0, scale=1, size=10).tolist()
        assert _input != _expected

    def test_random_generator_2(self) -> None:
        _input: list[float] = self.tsg._get_random_generator(seed=self.seed).normal(loc=0, scale=1, size=10).tolist()
        _expected: list[float] = np.random.default_rng(self.seed).normal(loc=0, scale=1, size=10).tolist()
        assert_all_close(_input, _expected)

    def test_generate_dates(self) -> None:
        _input: list[datetime] = self.dates_apr_2025()
        _output: list[datetime] = self.tsg._get_dates(start_date=datetime(2025, 4, 1), end_date=datetime(2025, 4, 30))
        assert _input == _output


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Linear                                             ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Linear(TestCase, Default_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator(cls.seed)

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
            ["2019-01-01", -0.7126267682641084],
            ["2019-01-02", 18.775748880855602],
            ["2019-01-03", 40.39371228586089],
            ["2019-01-04", 60.901829655589246],
            ["2019-01-05", 80.33125611336143],
            ["2019-01-06", 100.9602630978712],
            ["2019-01-07", 121.11250611417192],
            ["2019-01-08", 141.62177196116986],
            ["2019-01-09", 165.6242343247752],
            ["2019-01-10", 185.0138184774244],
            ["2019-01-11", 203.93429276153148],
            ["2019-01-12", 226.761554349751],
            ["2019-01-13", 245.35015306159386],
            ["2019-01-14", 268.7899320129483],
            ["2019-01-15", 288.4015364005244],
            ["2019-01-16", 308.54921543661294],
            ["2019-01-17", 330.0164719988947],
            ["2019-01-18", 352.4426913897209],
            ["2019-01-19", 374.4366886930567],
            ["2019-01-20", 394.02941798254517],
        ]
        assert_timeseries_equal(_input, _expected)

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
        assert_timeseries_equal(_input, _expected)

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
            ["2019-01-01", 1.4562551157782855],
            ["2019-01-02", 18.933054481938896],
            ["2019-01-03", 41.79293153531631],
            ["2019-01-04", 61.47998088565751],
            ["2019-01-05", 80.1324625856834],
            ["2019-01-06", 98.85434238481896],
            ["2019-01-07", 118.73161972957824],
            ["2019-01-08", 137.94604988443834],
            ["2019-01-09", 662.5258697790675],
            ["2019-01-10", 681.0895074833063],
            ["2019-01-11", 701.1547229696203],
            ["2019-01-12", 721.2108227607915],
            ["2019-01-13", 241.26736700627103],
            ["2019-01-14", 261.3780587301764],
            ["2019-01-15", 280.41493301379654],
            ["2019-01-16", 299.2481180128683],
            ["2019-01-17", 317.5237970087257],
            ["2019-01-18", 334.54744778222255],
            ["2019-01-19", 354.98006144844095],
            ["2019-01-20", 376.9488141498327],
        ]
        assert_timeseries_equal(_input, _expected)

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
            ["2019-01-01", -1.240950576469526],
            ["2019-01-02", 20.385596864372175],
            ["2019-01-03", 43.669198891853696],
            ["2019-01-04", 63.21619719509536],
            ["2019-01-05", 81.92026677310469],
            ["2019-01-06", 101.3535243598563],
            ["2019-01-07", 119.36326163991622],
            ["2019-01-08", 138.8175181003376],
            ["2019-01-09", 500.0],
            ["2019-01-10", 179.49972046009438],
            ["2019-01-11", 201.96887565421758],
            ["2019-01-12", 222.27065171865846],
            ["2019-01-13", -500.0],
            ["2019-01-14", 262.93537570850793],
            ["2019-01-15", 285.5667071216286],
            ["2019-01-16", 303.1220159172155],
            ["2019-01-17", 322.51483323661506],
            ["2019-01-18", 340.16745588525464],
            ["2019-01-19", 361.82000289927805],
            ["2019-01-20", 383.5206474785241],
        ]
        assert_timeseries_equal(_input, _expected)


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Fixed Errors                                       ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_FixedErrors(TestCase, Default_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator(cls.seed)
        cls.interpolation_nodes: list[list[int]] = [[len(cls.dates_apr_2025()) * i // 4, 100 * i] for i in range(4)]
        # [[0, 0], [7, 100], [15, 200], [22, 300]]

    def test_errors_one_week(self) -> None:
        _input: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=min(self.dates_apr_2025()),
                n_periods=len(self.dates_apr_2025()),
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
            ["2025-04-01", -0.33444200394902257],
            ["2025-04-02", 21.79408675158681],
            ["2025-04-03", 48.647478052896304],
            ["2025-04-04", 56.027376960886215],
            ["2025-04-05", 59.17523089550758],
            ["2025-04-06", 84.22722701919659],
            ["2025-04-07", 76.36870929005457],
            ["2025-04-08", 117.35501515397219],
            ["2025-04-09", 113.58495846373215],
            ["2025-04-10", 120.29980951230523],
            ["2025-04-11", 133.24373204779207],
            ["2025-04-12", 148.30652130053485],
            ["2025-04-13", 159.71427989690477],
            ["2025-04-14", 197.6016466479992],
            ["2025-04-15", 182.05374206783392],
            ["2025-04-16", 202.56154595318853],
            ["2025-04-17", 218.5670466237458],
            ["2025-04-18", 230.5610183793915],
            ["2025-04-19", 243.90946475988],
            ["2025-04-20", 253.3717599674153],
            ["2025-04-21", 262.6136960235986],
            ["2025-04-22", 276.74174915038475],
            ["2025-04-23", 289.4739597370538],
            ["2025-04-24", 325.39576441056846],
            ["2025-04-25", 351.988596277657],
            ["2025-04-26", 354.6687888210948],
            ["2025-04-27", 376.2331284507467],
            ["2025-04-28", 400.93238192177216],
            ["2025-04-29", 421.4448856605953],
            ["2025-04-30", 433.97154148111997],
        ]
        assert_timeseries_equal(_input, _expected)

    def test_generate_fixed_error_index_directly(self) -> None:
        """Test generate_fixed_error_index method directly - this exercises the method"""
        _input: NDArray = self.tsg.generate_fixed_error_index(
            dates=self.dates_apr_2025(),
            period_length=7,
            period_sd=1,
            start_index=5,
            seed=111,
        )
        # Verify it returns the correct length and is binary (0s and 1s)
        assert len(_input) == len(self.dates_apr_2025())
        assert set(_input).issubset({0, 1})


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Seasonalities                                      ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Seasonalities(TestCase, Default_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator(cls.seed)

    def test_holidays_in_april_2025(self) -> None:
        _input: list[datetime] = self.dates_apr_2025()
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
        assert_all_close(_output, _expected)

    def test_seasonal_sine(self) -> None:
        _input: list[datetime] = self.dates_apr_2025()
        _output: list[float] = self.tsg.generate_season_index(
            dates=_input,
            style="sin",
            period_length=7,
            start_index=0,
        ).tolist()
        _expected: list[float] = [
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
            0.012536043909088301,
            0.10908425876598482,
            0.4999999999999995,
            0.8909157412340156,
        ]
        assert_all_close(_output, _expected)

    def test_seasonal_sine_covar(self) -> None:
        _input: list[datetime] = self.dates_apr_2025()
        _output: list[float] = self.tsg.generate_season_index(
            dates=_input,
            style="sin_covar",
            period_length=7,
            start_index=0,
        ).tolist()
        _expected: list[float] = [
            0.3894183423086505,
            0.7276885734836275,
            0.9473175387510923,
            0.9929915397082303,
            0.8378773869800502,
            0.4971532637305591,
            0.03353172037375691,
            -0.4497290823791353,
            -0.830010218073391,
            -0.9977581894790544,
            -0.8919926194035165,
            -0.527048485754282,
            0.0012425292216746869,
            0.5366738981996544,
            0.9085062401218937,
            0.9889062395314718,
            0.7416513034277096,
            0.24198798060210708,
            -0.3434344888005015,
            -0.8124473584868283,
            -0.9997350969903036,
            -0.8381127871571578,
            -0.3842299399919373,
            0.20199312975240916,
            0.7162251988170882,
            0.9836094723292308,
            0.9191531624870717,
            0.5528090219630393,
            0.012526004102890335,
            -0.5250053705158881,
        ]
        assert_all_close(_output, _expected)


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Creation                                           ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Creation(TestCase, Default_Mixin):

    def setUp(self) -> None:
        self.tsg = TimeSeriesGenerator(self.seed)

    def test_linear_trend(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[int]] = [[n_periods * i // 4, 100 * i] for i in range(4)]
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
        assert_timeseries_equal(_output, _expected)

    def test_linear_trend_with_level_breaks(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[int]] = [[n_periods * i // 4, 100 * i] for i in range(4)]
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
        assert_timeseries_equal(_output, _expected)

    def test_linear_trend_with_outliers(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[int]] = [[n_periods * i // 4, 100 * i] for i in range(4)]
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
        assert_timeseries_equal(_output, _expected)

    def test_linear_trend_with_seasonality(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[int]] = [[n_periods * i // 4, 100 * i] for i in range(4)]
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
                noise_scale=0,
                seed=self.seed,
                season_eff=0.5,
                season_conf={"style": "sin", "period_length": 7, "start_index": 0},
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.4836820255217762],
            ["2019-01-02", 9.585917282252415],
            ["2019-01-03", 20.180873366991072],
            ["2019-01-04", 38.65211550270973],
            ["2019-01-05", 70.47226118994988],
            ["2019-01-06", 102.59751298894909],
            ["2019-01-07", 115.31912781590752],
            ["2019-01-08", 107.29172241531383],
            ["2019-01-09", 90.07008517418437],
            ["2019-01-10", 92.0282167446597],
            ["2019-01-11", 129.5709054031447],
            ["2019-01-12", 187.9365725721225],
            ["2019-01-13", 239.79196508940433],
            ["2019-01-14", 245.78339071863738],
            ["2019-01-15", 211.47211970543447],
            ["2019-01-16", 167.60230028904493],
            ["2019-01-17", 164.6887519656209],
            ["2019-01-18", 220.6729085827261],
            ["2019-01-19", 311.93057024901435],
            ["2019-01-20", 381.62442624197485],
        ]
        assert_timeseries_equal(_output, _expected)

    def test_sine_trend(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[int]] = [[n_periods * i // 4, 100 * i] for i in range(4)]
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
                noise_scale=0,
                seed=self.seed,
                season_conf={"style": "sin", "period_length": 7, "start_index": 0},
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -0.9891213503478509],
            ["2019-01-02", 1.8856505681362947],
            ["2019-01-03", 0.4997122150351114],
            ["2019-01-04", 17.054247648207177],
            ["2019-01-05", 58.85407722340778],
            ["2019-01-06", 101.95037323889171],
            ["2019-01-07", 108.66653018555053],
            ["2019-01-08", 71.52781494354257],
            ["2019-01-09", 17.7177313636153],
            ["2019-01-10", 2.278772736883859],
            ["2019-01-11", 57.169815416512975],
            ["2019-01-12", 156.95301056750642],
            ["2019-01-13", 238.27946339386673],
            ["2019-01-14", 231.60449400268107],
            ["2019-01-15", 140.98141313695635],
            ["2019-01-16", 32.96913205647106],
            ["2019-01-17", 4.077969250365404],
            ["2019-01-18", 97.36622131215944],
            ["2019-01-19", 260.50513435767573],
            ["2019-01-20", 379.2173080904791],
        ]
        assert_timeseries_equal(_output, _expected)

    def test_sine_covar_trend(self) -> None:
        n_periods = 20
        interpolation_nodes: list[list[int]] = [[n_periods * i // 4, 100 * i] for i in range(4)]
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
                noise_scale=0,
                seed=self.seed,
                season_conf={"style": "sin_covar", "period_length": 7, "start_index": 0},
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        _expected: list[list[str | float]] = [
            ["2019-01-01", -1.2078787075065935],
            ["2019-01-02", 4.707225423075605],
            ["2019-01-03", 2.10003008884705],
            ["2019-01-04", 0.42225961593880834],
            ["2019-01-05", 13.30871747274136],
            ["2019-01-06", 51.91623666708119],
            ["2019-01-07", 117.88180365509574],
            ["2019-01-08", 207.3919070453731],
            ["2019-01-09", 297.23472298650074],
            ["2019-01-10", 363.1478104325234],
            ["2019-01-11", 382.1295246036581],
            ["2019-01-12", 334.3016600065322],
            ["2019-01-13", 241.004638933641],
            ["2019-01-14", 120.44731325217124],
            ["2019-01-15", 25.79783912165755],
            ["2019-01-16", 3.352927892872225],
            ["2019-01-17", 84.040710780377],
            ["2019-01-18", 260.7406680844256],
            ["2019-01-19", 488.14499036175687],
            ["2019-01-20", 696.0369582115629],
        ]
        assert_timeseries_equal(_output, _expected)


## --------------------------------------------------------------------------- #
##  Validations                                                             ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Validations(TestCase, Default_Mixin):

    def setUp(self) -> None:
        self.tsg = TimeSeriesGenerator(self.seed)

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

    def test_value_is_between_invalid_range(self) -> None:
        with raises(ValueError, match="Invalid range"):
            self.tsg._assert_value_is_between(0.7, 1.0, 0.5)


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Polynomial Trends                                  ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_PolynomialTrends(TestCase, Default_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator(cls.seed)

    def test_polynomial_trend_zero_nodes(self) -> None:
        """Test generate_polynom_trend with 0 interpolation nodes (no trend)"""
        _input: NDArray = self.tsg.generate_polynom_trend(n_periods=10, interpolation_nodes=[])
        _expected: NDArray = np.zeros(10)
        assert np.array_equal(_input, _expected)

    def test_polynomial_trend_one_node(self) -> None:
        """Test generate_polynom_trend with 1 interpolation node (constant value)"""
        _input: NDArray = self.tsg.generate_polynom_trend(n_periods=10, interpolation_nodes=[(0, 100)])
        _expected: NDArray = np.zeros(10) + 100
        assert np.array_equal(_input, _expected)

    def test_polynomial_trend_two_nodes(self) -> None:
        """Test generate_polynom_trend with 2 interpolation nodes (linear)"""
        # Test a simple linear trend from (0, 0) to (9, 90) - slope of 10
        _input: NDArray = self.tsg.generate_polynom_trend(n_periods=10, interpolation_nodes=[(0, 0), (9, 90)])
        # Verify length
        assert len(_input) == 10
        # Verify it passes through the interpolation nodes
        assert np.isclose(_input[0], 0, atol=0.01)
        assert np.isclose(_input[9], 90, atol=0.01)
        # Verify linearity - should increase by 10 each step
        for i in range(1, 10):
            assert np.isclose(_input[i], i * 10, atol=0.01)
        # Verify it's actually linear (constant slope)
        diffs = np.diff(_input)
        assert np.allclose(diffs, 10, atol=0.01)

    def test_polynomial_trend_three_nodes(self) -> None:
        """Test generate_polynom_trend with 3 interpolation nodes (quadratic)"""
        _input: NDArray = self.tsg.generate_polynom_trend(n_periods=10, interpolation_nodes=[(0, 0), (5, 100), (9, 50)])
        # Verify it returns the correct length
        assert len(_input) == 10
        # Verify it contains expected values at the interpolation nodes
        assert np.isclose(_input[0], 0, atol=1)
        assert np.isclose(_input[5], 100, atol=1)
        assert np.isclose(_input[9], 50, atol=1)
        # Verify that it's actually a quadratic (middle values should be calculated)
        # For a quadratic passing through (0,0), (5,100), (9,50), intermediate values should exist
        assert _input[1] > 0  # Should be positive between 0 and 100
        assert _input[7] > 50  # Should be > 50 between peak and end
        # Force evaluation of all values to ensure the quadratic calculation is executed
        for i in range(len(_input)):
            _ = float(_input[i])  # Access each element to ensure computation

    def test_polynomial_trend_three_nodes_simple(self) -> None:
        """Test generate_polynom_trend with 3 nodes using simple parabola y=x^2"""
        # Test with simple values that form a parabola: (0,0), (1,1), (2,4)
        _input: NDArray = self.tsg.generate_polynom_trend(n_periods=5, interpolation_nodes=[(0, 0), (1, 1), (2, 4)])
        # Verify length
        assert len(_input) == 5
        # Verify the interpolation nodes match
        assert np.isclose(_input[0], 0, atol=0.01)
        assert np.isclose(_input[1], 1, atol=0.01)
        assert np.isclose(_input[2], 4, atol=0.01)
        # Verify the quadratic continues: 3^2=9, 4^2=16
        assert np.isclose(_input[3], 9, atol=0.01)
        assert np.isclose(_input[4], 16, atol=0.01)
        # Explicitly verify it's a proper quadratic by checking shape
        assert _input[3] > _input[2]  # Increasing
        assert _input[4] > _input[3]  # Still increasing

    def test_polynomial_trend_four_nodes(self) -> None:
        """Test generate_polynom_trend with 4 interpolation nodes (cubic)"""
        _input: NDArray = self.tsg.generate_polynom_trend(
            n_periods=15, interpolation_nodes=[(0, 0), (5, 100), (10, 50), (14, 150)]
        )
        # Just verify it returns the correct length and contains expected values at nodes
        assert len(_input) == 15
        assert np.isclose(_input[0], 0, atol=1)
        assert np.isclose(_input[5], 100, atol=1)
        assert np.isclose(_input[10], 50, atol=1)
        assert np.isclose(_input[14], 150, atol=1)

    def test_polynomial_trend_more_than_four_nodes(self) -> None:
        """Test generate_polynom_trend with >4 interpolation nodes (defaults to no trend)"""
        _input: NDArray = self.tsg.generate_polynom_trend(
            n_periods=10, interpolation_nodes=[(0, 0), (2, 50), (4, 100), (6, 150), (8, 200)]
        )
        _expected: NDArray = np.zeros(10)
        assert np.array_equal(_input, _expected)


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: ARMA                                               ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_ARMA(TestCase, Default_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator(cls.seed)

    def test_generate_ARMA_with_exogenous(self) -> None:
        """Test generate_ARMA with exogenous variables"""
        exog_ts: list[float] = np.random.default_rng(self.seed).normal(0, 1, 20).tolist()
        exogenous: list[dict[str, list[float]]] = [{"coeff": [0.5, 0.3], "ts": exog_ts}]

        _input: NDArray = self.tsg.generate_ARMA(
            AR=[0.5],
            MA=[0.3],
            randomwalk_scale=1.0,
            n_periods=20,
            exogenous=exogenous,  # type: ignore
            seed=self.seed,
        )

        # Just verify it returns the correct shape and is not all zeros
        assert len(_input) == 20
        assert not np.all(_input == 0)

    def test_generate_ARMA_without_exogenous(self) -> None:
        """Test generate_ARMA without exogenous variables"""
        _input: NDArray = self.tsg.generate_ARMA(
            AR=[0.5, 0.2],
            MA=[0.3],
            randomwalk_scale=1.0,
            n_periods=20,
            exogenous=[],
            seed=self.seed,
        )

        # Just verify it returns the correct shape
        assert len(_input) == 20


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Properties                                         ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_Properties(TestCase, Default_Mixin):

    def setUp(self) -> None:
        self.tsg = TimeSeriesGenerator(self.seed)

    def test_seed_property(self) -> None:
        """Test that the seed property returns the correct value"""
        assert self.tsg.seed == self.seed

    def test_seed_property_none(self) -> None:
        """Test that the seed property can be None"""
        tsg_no_seed = TimeSeriesGenerator()
        assert tsg_no_seed.seed is None

    def test_random_generator_property(self) -> None:
        """Test that the random_generator property returns a RandomGenerator"""
        assert isinstance(self.tsg.random_generator, RandomGenerator)


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Semi Markov Index                                  ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_SemiMarkov(TestCase, Default_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator(cls.seed)

    def test_generate_semi_markov_index(self) -> None:
        """Test generate_semi_markov_index method"""
        _input: NDArray = self.tsg.generate_semi_markov_index(
            dates=self.dates_apr_2025(),
            period_length=7,
            period_sd=1,
            start_index=5,
            seed=123,
        )
        # Verify it returns the correct length and is binary (0s and 1s)
        assert len(_input) == len(self.dates_apr_2025())
        # The method should return float values that can be converted to binary
        assert all(x in [0.0, 1.0] for x in _input)


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Season Index Edge Cases                            ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_SeasonIndexEdgeCases(TestCase, Default_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator(cls.seed)

    def test_generate_season_index_unknown_style(self) -> None:
        """Test generate_season_index with unknown style returns zeros"""
        _input: NDArray = self.tsg.generate_season_index(
            dates=self.dates_apr_2025(),
            style="unknown_style_that_doesnt_exist",  # type: ignore
        )
        # Should return all zeros when style is not recognized
        _expected: NDArray = np.zeros(len(self.dates_apr_2025()))
        assert np.array_equal(_input, _expected)


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator: Quadratic Trend Integration                        ####
## --------------------------------------------------------------------------- #


class TestTimeSeriesGenerator_QuadraticTrend(TestCase, Default_Mixin):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tsg = TimeSeriesGenerator(cls.seed)

    def test_create_time_series_with_three_nodes(self) -> None:
        """Test create_time_series with 3 interpolation nodes to trigger quadratic calculation"""
        _output: list[list[str | float]] = (
            self.tsg.create_time_series(
                start_date=datetime(2019, 1, 1),
                n_periods=10,
                interpolation_nodes=[(0, 0), (5, 100), (9, 50)],
                level_breaks=[],
                AR=[],
                MA=[],
                randomwalk_scale=0,
                exogenous=[],
                season_eff=0,
                manual_outliers=[],
                noise_scale=0,
                seed=self.seed,
                season_conf=None,
            )
            .assign(Date=lambda x: x.Date.dt.strftime("%Y-%m-%d"))
            .values.tolist()
        )
        # Just verify it generates the expected number of rows with the quadratic trend
        assert len(_output) == 10
        # Check that the values follow a quadratic pattern
        # At node 0: should be close to 0
        assert abs(_output[0][1]) < 5  # type: ignore
        # At node 5: should be close to 100
        assert abs(_output[5][1] - 100) < 5  # type: ignore
        # At node 9: should be close to 50
        assert abs(_output[9][1] - 50) < 5  # type: ignore
