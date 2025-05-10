# ============================================================================ #
#                                                                              #
#     Title: Synthetic Time Series Data                                        #
#     Purpose: Generate synthetic time series data for testing and validation. #
#     Notes: This module provides functions to generate various types of       #
#            synthetic time series data, including seasonal, trend, and noise. #
#            It also includes functions to create time series data with        #
#            specific characteristics, such as missing values and outliers.    #                                                      #
#                                                                              #
# ============================================================================ #


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Set Up                                                                ####
#                                                                              #
# ---------------------------------------------------------------------------- #


## --------------------------------------------------------------------------- #
##  Imports                                                                 ####
## --------------------------------------------------------------------------- #


# ## Future Python Library Imports ----
from __future__ import annotations

# ## Python StdLib Imports ----
from datetime import datetime
from functools import lru_cache
from typing import (
    Any,
    Literal,
    Union,
)

# ## Python Third Party Imports ----
import numpy as np
import pandas as pd
from numpy.random import Generator as RandomGenerator
from numpy.typing import NDArray
from toolbox_python.checkers import assert_all_values_of_type
from toolbox_python.collection_types import (
    datetime_list,
    datetime_list_tuple,
    dict_str_any,
    int_list_tuple,
)
from typeguard import typechecked


## --------------------------------------------------------------------------- #
##  Exports                                                                 ####
## --------------------------------------------------------------------------- #


__all__: list[str] = ["TimeSeriesGenerator"]


## --------------------------------------------------------------------------- #
##  Types                                                                   ####
## --------------------------------------------------------------------------- #


datetime_or_int = Union[datetime, int]
List_of_datetime_or_int = list[datetime_or_int]
Tuple_of_datetime_or_int = tuple[datetime, int]
Collection_of_datetime_or_int = Union[Tuple_of_datetime_or_int, List_of_datetime_or_int]
Collection_of_Collection_of_datetime_or_int = Union[
    list[Collection_of_datetime_or_int], tuple[Collection_of_datetime_or_int, ...]
]


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Classes                                                               ####
#                                                                              #
# ---------------------------------------------------------------------------- #


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator                                                     ####
## --------------------------------------------------------------------------- #


class TimeSeriesGenerator:

    def __init__(self) -> None:
        """
        !!! note "Summary"
            Initialize the TimeSeriesGenerator class.

        ???+ info "Details"
            - This class is designed to generate synthetic time series data for testing and validation purposes.
            - It provides methods to create time series data with various characteristics, including seasonality, trends, and noise.
            - The generated data can be used for testing algorithms, models, and other applications in time series analysis.
            - The class includes methods for generating holiday indices, fixed error indices, semi-Markov indices, and sine indices.
            - It also provides a method for generating polynomial trends and ARMA components.
            - The generated time series data can be customized with different parameters, such as start date, number of periods, and noise scale.
            - The class is designed to be flexible and extensible, allowing users to easily modify the generation process to suit their needs.
            - It is built using Python's type hinting and type checking features to ensure that the inputs and outputs are of the expected types.
            - This helps to catch potential errors early in the development process and improve code readability.
        """
        pass

    def _random_generator(self, seed: int | None = None) -> RandomGenerator:
        """
        !!! note "Summary"
            Get the random number generator.

        Returns:
            (RandomGenerator):
                The random number generator instance.
        """
        return np.random.default_rng(seed=seed)

    @lru_cache
    @staticmethod
    def _generate_dates(start_date: datetime, end_date: datetime) -> datetime_list:
        """
        !!! note "Summary"
            Generate a list of dates between a start and end date.

        Params:
            start_date (datetime):
                The starting date for generating dates.
            end_date (datetime):
                The ending date for generating dates.

        Returns:
            (datetime_list):
                A list of datetime objects representing the generated dates.
        """
        return pd.date_range(start_date, end_date).to_pydatetime().tolist()

    @staticmethod
    def _generate_holiday_period(start_date: datetime, periods: int) -> datetime_list:
        """
        !!! note "Summary"
            Generate a list of holiday dates starting from a given date.

        Params:
            start_date (datetime):
                The starting date for generating holiday dates.
            periods (int):
                The number of holiday dates to generate.

        Returns:
            (datetime_list):
                A list of datetime objects representing the generated holiday dates.
        """
        return pd.date_range(start_date, periods=periods).to_pydatetime().tolist()

    def create_time_series(
        self,
        start_date: datetime = datetime(2019, 1, 1),
        n_periods: int = 1096,
        interpolation_nodes: tuple[int_list_tuple, ...] | list[int_list_tuple] = (
            [0, 98],
            [300, 92],
            [700, 190],
            [1096, 213],
        ),
        level_breaks: tuple[int_list_tuple, ...] | list[int_list_tuple] | None = (
            [250, 100],
            [650, -50],
        ),
        AR: list | None = None,
        MA: list | None = None,
        randomwalk_scale: float = 2,
        exogenous: list | None = None,
        season_conf: dict_str_any | None = {"style": "holiday"},
        season_eff: float = 0.15,
        manual_outliers: tuple[int_list_tuple, ...] | list[int_list_tuple] | None = None,
        noise_scale: float = 10,
        seed: int | None = None,
    ) -> pd.DataFrame:

        # Validations
        AR = [1] or AR
        MA = [] or MA
        exogenous = [] or exogenous
        manual_outliers = [] or manual_outliers
        assert AR is not None
        assert MA is not None
        assert manual_outliers is not None

        # Date index:
        dates: datetime_list = pd.date_range(start_date, periods=n_periods).to_pydatetime().tolist()

        # Cubic trend component:
        trend: NDArray[np.float64] = self.generate_polynom_trend(interpolation_nodes, n_periods)

        # Structural break:
        break_effect: NDArray[np.float64] = np.zeros(n_periods).astype(np.float64)
        if level_breaks:
            for level_break in level_breaks:
                break_effect[level_break[0] :] += level_break[1]

        # ARMA(AR,MA) component:
        randomwalk: NDArray[np.float64] = self.generate_ARMA(
            AR=AR,
            MA=MA,
            rndwalk_scale=randomwalk_scale,
            n_periods=n_periods,
            exogenous=exogenous,
            seed=seed,
        )

        # Season:
        if season_conf is not None:
            season: NDArray[np.float64] = self.generate_season_index(dates=dates, **season_conf)  # type: ignore
            season = season * season_eff + (1 - season)
        else:
            season = np.ones(n_periods)

        # Noise component on top:
        noise: NDArray[np.float64] = self._random_generator(seed=seed).normal(
            loc=0.0,
            scale=noise_scale,
            size=n_periods,
        )

        # Assemble finally:
        df: pd.DataFrame = pd.DataFrame(
            list(
                zip(
                    dates,
                    (trend + break_effect + randomwalk + noise) * season,
                )
            ),
            index=dates,
            columns=["Date", "Value"],
        )

        # Manual outliers:
        if manual_outliers:
            for manual_outlier in manual_outliers:
                df.iloc[manual_outlier[0], 1] = manual_outlier[1]

        return df

    @typechecked
    def generate_holiday_index(
        self,
        dates: datetime_list_tuple,
        season_dates: Collection_of_Collection_of_datetime_or_int,
    ) -> NDArray[np.int_]:
        """
        !!! note "Summary"
            Generate a holiday index for the given dates based on the provided holiday dates.

        ???+ info "Details"
            - A holiday index is a manual selection for date in `dates` to determine whether it is a holiday or not.
            - Basically, it is a manual index of dates in a univariate time series data set which are actual holidays.
            - The return array is generated by checking if each date in `dates` is present in the list of holiday dates generated from `season_dates`.

        !!! warning "Important"
            This function is designed to work with a `.generate_season_index()` when the `style="holiday"`.<br>
            It is not intended to be called directly.

        Params:
            dates (datetime_list_tuple):
                List of datetime objects representing the dates to check.
            season_dates (Collection_of_Collection_of_datetime_or_int):
                Collection of collections containing holiday dates and their respective periods.<br>
                Each element in the collection should contain exactly two elements: a datetime object and an integer representing the number of periods.<br>
                Some example inputs include:\n
                - List of lists containing datetime and periods: `season_dates = [[datetime(2025, 4, 18), 4], [datetime(2024, 3, 29), 4]]`
                - List of tuples containing datetime and periods: `season_dates = [(datetime(2025, 4, 18), 4), (datetime(2024, 3, 29), 4)]`
                - Tuple of lists containing datetime and periods: `season_dates = ([datetime(2025, 4, 18), 4], [datetime(2024, 3, 29), 4])`
                - Tuple of tuples containing datetime and periods: `season_dates = ((datetime(2025, 4, 18), 4), (datetime(2024, 3, 29), 4))`

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.
            (AssertionError):
                If `season_dates` does not contain exactly two elements.
            (TypeError):
                If the first element of `season_dates` is not a `datetime`, or the second element is not an `int`.

        Returns:
            (NDArray[np.int_]):
                An array of the same length as `dates`, where each element is `1` if the corresponding date is a holiday, and `0` otherwise.
        """

        # Validations
        assert all(len(elem) == 2 for elem in season_dates)
        assert_all_values_of_type([season_date[0] for season_date in season_dates], datetime)
        assert_all_values_of_type([season_date[1] for season_date in season_dates], int)

        # Build dates
        season_dates_list: list[datetime] = []
        for _dates in season_dates:
            season_dates_list.extend(
                self._generate_holiday_period(
                    start_date=_dates[0],  # type: ignore
                    periods=_dates[1],  # type: ignore
                )
            )

        # Tag dates
        events: NDArray[np.int_] = np.where([_date in season_dates_list for _date in dates], 1, 0)

        # Return
        return events

    @typechecked
    def generate_fixed_error_index(
        self,
        dates: datetime_list_tuple,
        period_length: int = 7,
        period_sd: float = 0.5,
        start_index: int = 4,
        seed: int | None = None,
        verbose: bool = False,
    ) -> NDArray[np.int_]:
        """
        !!! note "Summary"
            Generate a fixed error seasonality index for the given dates.

        ???+ info "Details"
            - A holiday index is a manual selection for date in `dates` to determine whether it is a holiday or not.
            - A fixed error seasonality index is a non-uniform distribution of dates in a univariate time series data set.
            - Basically, it is indicating every `period_length` length of days, occurring every `period_sd` number of days, starting from `start_index`.
            - The return array is a boolean `1` or `0` of length `n_periods`. It will have a seasonality of `period_length` and a disturbance standard deviation of `period_sd`. The result can be used as a non-uniform distribution of weekdays in a histogram (if for eg. frequency is weekly).

        !!! warning "Important"
            This function is designed to work with a `.generate_season_index()` when the `style="fixed+error"`.<br>
            It is not intended to be called directly.

        Params:
            dates (datetime_list_tuple):
                List of datetime objects representing the dates to check.
            period_length (int):
                The length of the period for seasonality.<br>
                For example, if the frequency is weekly, this would be `7`.<br>
                Default is `7`.
            period_sd (float):
                The standard deviation of the disturbance.<br>
                Default is `0.5`.
            start_index (int):
                The starting index for the seasonality.<br>
                Default is `4`.
            verbose (bool):
                If `True`, print additional information about the generated indices. Helpful for debugging.<br>
                Default is `False`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.

        Returns:
            (NDArray[np.int_]):
                An array of the same length as `dates`, where each element is `1` if the corresponding date is a holiday, and `0` otherwise.
        """

        # Process
        n_periods: int = len(dates)
        events: NDArray[np.int_] = np.zeros(n_periods).astype(np.int_)
        event_inds: NDArray[Any] = np.arange(n_periods // period_length + 1) * period_length + start_index
        disturbance: NDArray[np.int_] = (
            self._random_generator(seed=seed)
            .normal(
                loc=0.0,
                scale=period_sd,
                size=len(event_inds),
            )
            .round()
            .astype(int)
        )
        event_inds = event_inds + disturbance

        # Delete indices that are out of bounds
        if np.any(event_inds >= n_periods):
            event_inds = np.delete(event_inds, event_inds >= n_periods)

        # For any indices defined above, assign `1` to the events array
        events[event_inds] = 1

        # Check debugging
        if verbose:
            print(f"Disturbance: {disturbance}")
            print(f"Event indices: {event_inds}")
            print(f"Weekdays: {np.mod(event_inds, period_length)}")
            print(f"Histogram: {np.histogram(np.mod(event_inds, period_length), bins=np.arange(period_length))}")

        # Return
        return events.astype(np.int_)

    def generate_semi_markov_index(
        self,
        dates: datetime_list_tuple,
        period_length: int = 7,
        period_sd: float = 0.5,
        start_index: int = 4,
        seed: int | None = None,
        verbose: bool = False,
    ) -> NDArray[np.int_]:
        """
        !!! note "Summary"
            Generate a semi-Markov seasonality index for the given dates.

        ???+ info "Details"
            - A semi-Markov seasonality index is a uniform distribution of dates in a univariate time series data set.
            - Basically, it is indicating a `period_length` length of days, occurring randomly roughly ever `period_sd` number of days, starting from `start_index`.
            - The return array is a boolean `1` or `0` of length `n_periods`. It will have a seasonality of `period_length` and a disturbance standard deviation of `period_sd`. The result can be used as a uniform distribution of weekdays in a histogram (if for eg. frequency is weekly).

        Params:
            dates (datetime_list_tuple):
                List of datetime objects representing the dates to check.
            period_length (int):
                The length of the period for seasonality.<br>
                For example, if the frequency is weekly, this would be `7`.<br>
                Default is `7`.
            period_sd (float):
                The standard deviation of the disturbance.<br>
                Default is `0.5`.
            start_index (int):
                The starting index for the seasonality.<br>
                Default is `4`.
            verbose (bool):
                If `True`, print additional information about the generated indices. Helpful for debugging.<br>
                Default is `False`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.

        Returns:
            (NDArray[np.int_]):
                An array of the same length as `dates`, where each element is `1` if the corresponding date is a holiday, and `0` otherwise.
        """

        # Process
        n_periods: int = len(dates)
        events: NDArray[np.int_] = np.zeros(n_periods).astype(np.int_)
        event_inds: list[int] = [start_index]
        new = np.random.normal(loc=period_length, scale=period_sd, size=1).round()[0]
        while new + event_inds[-1] < n_periods:
            event_inds.append(new + event_inds[-1])
            new = (
                self._random_generator(seed=seed)
                .normal(
                    loc=period_length,
                    scale=period_sd,
                    size=1,
                )
                .round()[0]
            )
        event_indexes: NDArray[np.int_] = np.array(event_inds).astype(np.int_)

        # For any indices defined above, assign `1` to the events array
        events[event_indexes] = 1

        # Check debugging
        if verbose:
            print(f"Event indices: {event_inds}")
            print(f"Weekdays: {np.mod(event_inds, period_length)}")
            print(f"Histogram: {np.histogram(np.mod(event_inds, period_length), bins=np.arange(period_length))}")

        # Return
        return events

    def generate_sin_index(
        self,
        dates: datetime_list_tuple,
        period_length: int = 7,
        start_index: int = 4,
    ) -> NDArray[np.float64]:
        n_periods: int = len(dates)
        events = (np.sin((np.arange(n_periods) - start_index) / period_length * 2 * np.pi) + 1) / 2
        return events

    def generate_sin_covar_index(
        self,
        dates: datetime_list_tuple,
        period_length: int = 7,
        start_index: int = 4,
    ) -> NDArray[np.float64]:
        n_periods: int = len(dates)
        covar_wave = (np.sin((np.arange(n_periods) - start_index) / period_length / 6 * np.pi) + 2) / 2
        dx: NDArray[np.float64] = np.full_like(covar_wave, 0.4)
        sin_wave: NDArray[np.float64] = np.sin((covar_wave * dx).cumsum())
        return sin_wave

    def generate_season_index(
        self,
        dates: datetime_list_tuple,
        style: Literal[
            "fixed+error",
            "semi-markov",
            "holiday",
            "sin",
            "sin_covar",
        ],
        season_dates: Collection_of_Collection_of_datetime_or_int | None = None,
        period_length: int | None = None,
        period_sd: float | None = None,
        start_index: int | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> NDArray[np.float64]:
        if "fixed" in style and "error" in style:
            assert period_length is not None
            assert period_sd is not None
            assert start_index is not None
            return self.generate_fixed_error_index(
                dates=dates,
                period_length=period_length,
                period_sd=period_sd,
                start_index=start_index,
                seed=seed,
                verbose=verbose,
            ).astype(np.float64)
        elif "semi" in style and "markov" in style:
            assert period_length is not None
            assert period_sd is not None
            assert start_index is not None
            return self.generate_semi_markov_index(
                dates=dates,
                period_length=period_length,
                period_sd=period_sd,
                start_index=start_index,
                seed=seed,
                verbose=verbose,
            ).astype(np.float64)
        elif style == "holiday":
            assert season_dates is not None
            return self.generate_holiday_index(dates=dates, season_dates=season_dates).astype(np.float64)
        elif "sin" in style and "covar" in style:
            assert period_length is not None
            assert start_index is not None
            return self.generate_sin_covar_index(dates=dates, period_length=period_length).astype(np.float64)
        elif style == "sin":
            assert period_length is not None
            assert start_index is not None
            return self.generate_sin_index(dates=dates, period_length=period_length).astype(np.float64)
        else:
            return np.zeros(len(dates)).astype(np.float64)

    def generate_polynom_trend(self, interpol_nodes, n_periods: int) -> NDArray[np.float64]:
        # implemented only up to order 3 (cubic interpolation = four nodes)

        if len(interpol_nodes) == 0:
            # No trend component:
            trend: NDArray[np.float64] = np.zeros(n_periods)
            return trend

        elif len(interpol_nodes) == 1:
            # No trend component:
            trend: NDArray[np.float64] = np.zeros(n_periods) + interpol_nodes[0][1]
            return trend

        elif len(interpol_nodes) == 2:
            # Linear trend component:
            x1, y1 = interpol_nodes[0]
            x2, y2 = interpol_nodes[1]
            M = np.column_stack((np.array([x1, x2]), np.ones(2)))
            b = np.array([y1, y2])
            pvec = np.linalg.solve(M, b)
            trend: NDArray[np.float64] = np.arange(n_periods).astype(np.float64)
            trend = pvec[0] * trend + pvec[1]
            return trend

        elif len(interpol_nodes) == 3:
            # Quadratic trend component:
            x1, y1 = interpol_nodes[0]
            x2, y2 = interpol_nodes[1]
            x3, y3 = interpol_nodes[2]
            M = np.column_stack(
                (
                    np.array([x1, x2, x3]) * np.array([x1, x2, x3]),
                    np.array([x1, x2, x3]),
                    np.ones(3),
                )
            )
            b = np.array([y1, y2, y3])
            pvec = np.linalg.solve(M, b)
            trend: NDArray[np.float64] = np.arange(n_periods).astype(np.float64)
            trend = pvec[0] * trend * trend + pvec[1] * trend + pvec[2]
            return trend

        elif len(interpol_nodes) == 4:
            # Cubic trend component:
            x1, y1 = interpol_nodes[0]
            x2, y2 = interpol_nodes[1]
            x3, y3 = interpol_nodes[2]
            x4, y4 = interpol_nodes[3]
            M = np.column_stack(
                (
                    np.array([x1, x2, x3, x4]) * np.array([x1, x2, x3, x4]) * np.array([x1, x2, x3, x4]),
                    np.array([x1, x2, x3, x4]) * np.array([x1, x2, x3, x4]),
                    np.array([x1, x2, x3, x4]),
                    np.ones(4),
                )
            )
            b = np.array([y1, y2, y3, y4])
            pvec = np.linalg.solve(M, b)
            trend: NDArray[np.float64] = np.arange(n_periods).astype(np.float64)
            trend = pvec[0] * trend * trend * trend + pvec[1] * trend * trend + pvec[2] * trend + pvec[3]
            return trend

        else:
            # All other values parsed to `interpol_nodes` are not valid. Default to no trend component.
            trend: NDArray[np.float64] = np.zeros(n_periods)
            return trend

    def generate_ARMA(
        self,
        AR: list,
        MA: list,
        rndwalk_scale: float,
        n_periods: int,
        exogenous: list[dict[Literal["coeff", "ts"], list[float]]] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float64]:

        exogenous = [] or exogenous
        assert exogenous is not None

        # Noise
        u: NDArray[np.float64] = self._random_generator(seed=seed).normal(
            loc=0.0,
            scale=rndwalk_scale,
            size=n_periods,
        )
        ts = np.zeros(n_periods)
        for i in range(n_periods):
            for i_ar in range(min(len(AR), i)):
                ts[i] = ts[i] + AR[i_ar] * ts[i - 1 - i_ar]
            ts[i] = ts[i] + u[i]
            for i_ma in range(min(len(MA), i)):
                ts[i] = ts[i] - u[i - 1 - i_ma]
            for exvar in exogenous:
                for i_ar in range(len(exvar["coeff"])):
                    ts[i] = ts[i] + exvar["coeff"][i_ar] * exvar["ts"][i - i_ar]
        return ts
