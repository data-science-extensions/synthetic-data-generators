# ============================================================================ #
#                                                                              #
#     Title: Synthetic Time Series Data                                        #
#     Purpose: Generate synthetic time series data for testing and validation. #
#     Notes: This module provides functions to generate various types of       #
#            synthetic time series data, including seasonal, trend, and noise. #
#            It also includes functions to create time series data with        #
#            specific characteristics, such as missing values and outliers.    #
#                                                                              #
# ============================================================================ #


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Overview                                                              ####
#                                                                              #
# ---------------------------------------------------------------------------- #


## --------------------------------------------------------------------------- #
##  Description                                                             ####
## --------------------------------------------------------------------------- #


"""
!!! note "Summary"
    The [`time_series`][synthetic_data_generators.time_series] module provides a class for generating synthetic time series data. It includes methods for creating time series with various characteristics, such as seasonality, trends, and noise.
"""


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
    """
    !!! note "Summary"
        A class for generating synthetic time series data.

    ???+ info "Details"
        - This class provides methods to create synthetic time series data with various characteristics, including seasonality, trends, and noise.
        - The generated data can be used for testing and validation purposes in time series analysis.
        - The class includes methods to generate holiday indices, fixed error indices, semi-Markov indices, and sine indices.
        - It also provides a method to generate polynomial trends and ARMA components.
        - The generated time series data can be customized with different parameters, such as start date, number of periods, and noise scale.
    """

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

    @staticmethod
    @lru_cache
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
        AR: list[float] | None = None,
        MA: list[float] | None = None,
        randomwalk_scale: float = 2,
        exogenous: list[dict[Literal["coeff", "ts"], list[float]]] | None = None,
        season_conf: dict_str_any | None = {"style": "holiday"},
        season_eff: float = 0.15,
        manual_outliers: tuple[int_list_tuple, ...] | list[int_list_tuple] | None = None,
        noise_scale: float = 10,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        !!! note "Summary"
            Generate a synthetic time series with specified characteristics.

        ???+ info "Details"
            - The function generates a time series based on the specified parameters, including start date, number of periods, interpolation nodes, level breaks, ARMA coefficients, random walk scale, exogenous variables, seasonality configuration, manual outliers, and noise scale.
            - The generated time series is returned as a pandas DataFrame with two columns: "Date" and "Value".
            - The "Date" column contains the dates of the time series, and the "Value" column contains the corresponding values.
            - The function also includes options for generating seasonality indices, fixed error indices, semi-Markov indices, and sine indices.
            - The generated time series can be customized with different parameters, such as start date, number of periods, and noise scale.

        !!! warning "Important"
            This function is designed to generate synthetic time series data for testing and validation purposes.
            It is not intended to be used for production or real-world applications.

        Params:
            start_date (datetime):
                The starting date for the time series.<br>
                Default is `datetime(2019, 1, 1)`.
            n_periods (int):
                The number of periods for the time series.<br>
                Default is `1096`.
            interpolation_nodes (tuple[int_list_tuple, ...] | list[int_list_tuple]):
                A collection of interpolation nodes, where each node is a tuple containing the x-coordinate and y-coordinate.<br>
                The x-coordinates should be in ascending order.<br>
                Default is `([0, 98], [300, 92], [700, 190], [1096, 213])`.
            level_breaks (tuple[int_list_tuple, ...] | list[int_list_tuple] | None):
                A collection of level breaks, where each break is a tuple containing the index and the value to add.<br>
                Default is `([250, 100], [650, -50])`.
            AR (list[float] | None):
                The autoregressive coefficients for the ARMA model.<br>
                Default is `None`.
            MA (list[float] | None):
                The moving average coefficients for the ARMA model.<br>
                Default is `None`.
            randomwalk_scale (float):
                The scale of the random walk component.<br>
                Default is `2`.
            exogenous (list[dict[Literal["coeff", "ts"], list[float]]] | None):
                A list of exogenous variables to include in the ARMA model.<br>
                Default is `None`.
            season_conf (dict_str_any | None):
                A dictionary containing the configuration for seasonality.<br>
                Default is `{"style": "holiday"}`.
            season_eff (float):
                The effectiveness of the seasonality component.<br>
                Default is `0.15`.
            manual_outliers (tuple[int_list_tuple, ...] | list[int_list_tuple] | None):
                A collection of manual outliers, where each outlier is a tuple containing the index and the value to set.<br>
                Default is `None`.
            noise_scale (float):
                The scale of the noise component.<br>
                Default is `10`.
            seed (int | None):
                The random seed for reproducibility.<br>
                Default is `None`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.
            (AssertionError):
                If `interpolation_nodes` does not contain exactly two elements.
            (TypeError):
                If the first element of `interpolation_nodes` is not a `datetime`, or the second element is not an `int`.

        Returns:
            (pd.DataFrame):
                A pandas DataFrame containing the generated time series data.
                The DataFrame has two columns: "Date" and "Value".
                The "Date" column contains the dates of the time series, and the "Value" column contains the corresponding values.
        """

        # Validations
        AR = AR or [1]
        MA = MA or [0]
        exogenous = exogenous or []
        manual_outliers = manual_outliers or []
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
            randomwalk_scale=randomwalk_scale,
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
    ) -> NDArray[np.float64]:
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
        disturbance: NDArray[np.float64] = (
            self._random_generator(seed=seed)
            .normal(
                loc=0.0,
                scale=period_sd,
                size=len(event_inds),
            )
            .astype(int)
        )
        event_inds = event_inds + disturbance

        # Delete indices that are out of bounds
        if np.any(event_inds >= n_periods):
            event_inds = np.delete(event_inds, event_inds >= n_periods)

        # Return
        return events.astype(np.float64)

    def generate_semi_markov_index(
        self,
        dates: datetime_list_tuple,
        period_length: int = 7,
        period_sd: float = 0.5,
        start_index: int = 4,
        seed: int | None = None,
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

        # Return
        return events

    def generate_sin_index(
        self,
        dates: datetime_list_tuple,
        period_length: int = 7,
        start_index: int = 4,
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate a sine seasonality index for the given dates.

        ???+ info "Details"
            - A sine seasonality index is a periodic function that oscillates between `0` and `1`.
            - It is used to model seasonal patterns in time series data.
            - The return array is a sine wave of length `n_periods`, with a period of `period_length` and a phase shift of `start_index`.
            - The result can be used to represent seasonal patterns in time series data, such as daily or weekly cycles.

        Params:
            dates (datetime_list_tuple):
                List of datetime objects representing the dates to check.
            period_length (int):
                The length of the period for seasonality.<br>
                For example, if the frequency is weekly, this would be `7`.<br>
                Default is `7`.
            start_index (int):
                The starting index for the seasonality. Designed to account for seasonal patterns that start at a different point in time.<br>
                Default is `4`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.

        Returns:
            (NDArray[np.float64]):
                An array of the same length as `dates`, where each element is a sine value representing the seasonal pattern.
        """
        n_periods: int = len(dates)
        events = (np.sin((np.arange(n_periods) - start_index) / period_length * 2 * np.pi) + 1) / 2
        return events

    def generate_sin_covar_index(
        self,
        dates: datetime_list_tuple,
        period_length: int = 7,
        start_index: int = 4,
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate a sine seasonality index with covariance for the given dates.

        ???+ info "Details"
            - A sine seasonality index with covariance is a periodic function that oscillates between `0` and `1`.
            - It is used to model seasonal patterns in time series data, taking into account the covariance structure of the data.
            - The return array is a sine wave of length `n_periods`, with a period of `period_length` and a phase shift of `start_index`.
            - The result can be used to represent seasonal patterns in time series data, such as daily or weekly cycles.

        Params:
            dates (datetime_list_tuple):
                List of datetime objects representing the dates to check.
            period_length (int):
                The length of the period for seasonality.<br>
                For example, if the frequency is weekly, this would be `7`.<br>
                Default is `7`.
            start_index (int):
                The starting index for the seasonality. Designed to account for seasonal patterns that start at a different point in time.<br>
                Default is `4`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.

        Returns:
            (NDArray[np.float64]):
                An array of the same length as `dates`, where each element is a sine value representing the seasonal pattern.
        """
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
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate a seasonality index for the given dates based on the specified style.

        ???+ info "Details"
            - A seasonality index is a manual selection for date in `dates` to determine whether it is a holiday or not.
            - Basically, it is a manual index of dates in a univariate time series data set which are actual holidays.
            - The return array is generated by checking if each date in `dates` is present in the list of holiday dates generated from `season_dates`.
            - The return array is a boolean `1` or `0` of length `n_periods`. It will have a seasonality of `period_length` and a disturbance standard deviation of `period_sd`. The result can be used as a non-uniform distribution of weekdays in a histogram (if for eg. frequency is weekly).

        Params:
            dates (datetime_list_tuple):
                List of datetime objects representing the dates to check.
            style (Literal):
                The style of the seasonality index to generate.<br>
                Possible values are:
                - `"fixed+error"`: Fixed error seasonality index.
                - `"semi-markov"`: Semi-Markov seasonality index.
                - `"holiday"`: Holiday seasonality index.
                - `"sin"`: Sine seasonality index.
                - `"sin_covar"`: Sine seasonality index with covariance.
            season_dates (Collection_of_Collection_of_datetime_or_int | None):
                Collection of collections containing holiday dates and their respective periods.<br>
                Each element in the collection should contain exactly two elements: a datetime object and an integer representing the number of periods.<br>
                Some example inputs include:\n
                - List of lists containing datetime and periods: `season_dates = [[datetime(2025, 4, 18), 4], [datetime(2024, 3, 29), 4]]`
                - List of tuples containing datetime and periods: `season_dates = [(datetime(2025, 4, 18), 4), (datetime(2024, 3, 29), 4)]`
                - Tuple of lists containing datetime and periods: `season_dates = ([datetime(2025, 4, 18), 4], [datetime(2024, 3, 29), 4])`
                - Tuple of tuples containing datetime and periods: `season_dates = ((datetime(2025, 4, 18), 4), (datetime(2024, 3, 29), 4))`
            period_length (int | None):
                The length of the period for seasonality.<br>
                For example, if the frequency is weekly, this would be `7`.<br>
                Default is `7`.
            period_sd (float | None):
                The standard deviation of the disturbance.<br>
                Default is `0.5`.
            start_index (int | None):
                The starting index for the seasonality.<br>
                Default is `4`.
            seed (int | None):
                Random seed for reproducibility.<br>
                Default is `None`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.
            (AssertionError):
                If `season_dates` does not contain exactly two elements.
            (TypeError):
                If the first element of `season_dates` is not a `datetime`, or the second element is not an `int`.
            (ValueError):
                If `style` is not one of the supported styles.
                If `period_length`, `period_sd`, or `start_index` are not provided for the corresponding styles.

        Returns:
            (NDArray[np.float64]):
                An array of the same length as `dates`, where each element is a sine value representing the seasonal pattern.
        """
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
        """
        !!! note "Summary"
            Generate a polynomial trend based on the provided interpolation nodes.

        ???+ info "Details"
            - The polynomial trend is generated using the provided interpolation nodes.
            - The function supports polynomial trends of order 1 (linear), 2 (quadratic), 3 (cubic), and 4 (quartic).
            - The generated trend is an array of the same length as `n_periods`, where each element represents the value of the polynomial trend at that period.
            - The function uses numpy's linear algebra solver to compute the coefficients of the polynomial based on the provided interpolation nodes.

        !!! warning "Important"
            This function is implemented only up to order 3 (cubic interpolation = four nodes).
            It is not intended to be used for higher-order polynomial trends.

        Params:
            interpol_nodes (tuple[int_list_tuple, ...] | list[int_list_tuple]):
                A collection of interpolation nodes, where each node is a tuple containing the x-coordinate and y-coordinate.
                The x-coordinates should be in ascending order.
            n_periods (int):
                The number of periods for which to generate the polynomial trend.
                This determines the length of the output array.
                The generated trend will have the same length as `n_periods`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.
            (AssertionError):
                If `interpol_nodes` does not contain exactly two elements.
            (TypeError):
                If the first element of `interpol_nodes` is not a `datetime`, or the second element is not an `int`.

        Returns:
            (NDArray[np.float64]):
                An array of the same length as `n_periods`, where each element represents the value of the polynomial trend at that period.
        """

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
        AR: list[float],
        MA: list[float],
        randomwalk_scale: float,
        n_periods: int,
        exogenous: list[dict[Literal["coeff", "ts"], list[float]]] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate an ARMA (AutoRegressive Moving Average) time series.

        ???+ info "Details"
            - The ARMA model is a combination of autoregressive (AR) and moving average (MA) components.
            - The function generates a time series based on the specified AR and MA coefficients, random walk scale, and optional exogenous variables.
            - The generated time series is an array of the same length as `n_periods`, where each element represents the value of the ARMA time series at that period.
            - The function uses numpy's random number generator to generate the noise component of the ARMA model.

        Params:
            AR (list[float]):
                List of autoregressive coefficients.
                The length of the list determines the order of the AR component.
                All values must be between `0` and `1`.
            MA (list[float]):
                List of moving average coefficients.
                The length of the list determines the order of the MA component.
                All values must be between `0` and `1`.
            randomwalk_scale (float):
                Scale parameter for the random walk component.
                This controls the standard deviation of the noise added to the time series.
            n_periods (int):
                The number of periods for which to generate the ARMA time series.
                This determines the length of the output array.
            exogenous (list[dict[Literal["coeff", "ts"], list[float]]] | None):
                Optional list of exogenous variables, where each variable is represented as a dictionary with keys "coeff" and "ts".
                "coeff" is a list of coefficients for the exogenous variable, and "ts" is a list of values for that variable.
            seed (int | None):
                Random seed for reproducibility.<br>
                Default is `None`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.

        Returns:
            (NDArray[np.float64]):
                An array of the same length as `n_periods`, where each element represents the value of the ARMA time series at that period.

        ???+ info "Details about how the `AR` and `MA` Parameters work"

            This [`#!py generate_ARMA()`][synthetic_data_generators.time_series.TimeSeriesGenerator.generate_ARMA] method creates time series data using ARMA (AutoRegressive Moving Average) models.
            The `#!py AR` parameter is used to model the long-term trends in the data, while the `#!py MA` parameter is used to model the short-term fluctuations.

            **The `AR` (AutoRegressive) Parameter:**

            - The `#!py AR` parameter is a list of coefficients that determine how much past values influence the current value.
            - Each coefficient represents the weight given to a specific lag (previous time point).
            - For example, with `#!py AR=[0.6, 0.3]`:
                - The value at time `#!py t` is influenced by:
                - 60% of the value at time `#!py t-1` (0.6 x previous value)
                - 30% of the value at time `#!py t-2` (0.3 x value from two periods ago)
            - This creates persistence in the data where values tend to follow past trends. Higher AR values (closer to `#!py 1`) create stronger trends and more correlation with past values.
            - Higher AR values (closer to `#!py 1`) create stronger trends and more correlation with past values.
            - When `#!py AR=[0]`, the time series is purely random, as it does not depend on past values. Likewise, when `#!py AR=[1]`, the time series is the same as a random walk, as it only depends on the previous value.
            - When multiple values are provided, the first value is the most recent, and the last value is the oldest. For example, `#!py AR=[0.5, 0.3]` means that the most recent value has a weight of `0.5`, and the second most recent value has a weight of `0.3`. Realistically, the second most recent value will have less influence than the most recent value, and will therefore have a lower value (closer to `#!py 0`), but it can still affect the current value.

            **The `#!py MA` (Moving Average) Parameter:**

            - The MA parameter is a list of coefficients that determine how much past random shocks (errors) influence the current value.
            - For example, with `#!py MA=[0.2, 0.1]`:
                - The value at time `#!py t` is influenced by:
                - 20% of the random shock at time `#!py t-1`
                - 10% of the random shock at time `#!py t-2`
            - This creates short-term corrections or adjustments based on recent random fluctuations.
            - Higher MA values (closer to `#!py 1`) create stronger corrections and more correlation with past shocks.
            - When `#!py MA=[0]`, the time series is purely autoregressive, as it will depend on past values and does not depend on past shocks. Likewise, when `#!py MA=[1]`, the time series is purely random and will not depend on previous values.
            - When multiple values are provided, the first value is the most recent, and the last value is the oldest. For example, `#!py MA=[0.5, 0.3]` means that the most recent value has a weight of `0.5`, and the second most recent value has a weight of `0.3`. Realistically, the second most recent value will have less influence than the most recent value, and will therefore have a lower value (closer to `#!py 0`), but it can still affect the current value.

            **Examples and Effects:**

            | Value                                | Description |
            |--------------------------------------|-------------|
            | `#!py AR=[0.9]`                      | Creates strong persistence - values strongly follow the previous value, resulting in smooth, trending data |
            | `#!py AR=[0.5,0.3]`                  | Creates moderate persistence with some oscillation patterns |
            | `#!py MA=[0.8]`                      | Creates immediate corrections after random shocks |
            | `#!py MA=[0.5,0.3]`                  | Creates moderate corrections with some oscillation patterns |
            | `#!py AR=[0.7]` <br> `#!py MA=[0.4]` | Combines trend persistence with short-term corrections |
        """

        # Validations
        AR = AR or [1]
        MA = MA or [0]
        exogenous = exogenous or []
        assert exogenous is not None
        self._assert_all_values_are_between(AR, min_value=0, max_value=1)
        self._assert_all_values_are_between(MA, min_value=0, max_value=1)

        # Noise
        u: NDArray[np.float64] = self._random_generator(seed=seed).normal(
            loc=0.0,
            scale=randomwalk_scale,
            size=n_periods,
        )
        ts: NDArray[np.float64] = np.zeros(n_periods).astype(np.float64)
        for i in range(n_periods):
            for i_ar in range(min(len(AR), i)):
                ts[i] = ts[i] + AR[i_ar] * ts[i - 1 - i_ar]
            ts[i] = ts[i] + u[i]
            for i_ma in range(min(len(MA), i)):
                ts[i] = ts[i] - MA[i_ma] * u[i - 1 - i_ma]
            for exvar in exogenous:
                for i_ar in range(len(exvar["coeff"])):
                    ts[i] = ts[i] + exvar["coeff"][i_ar] * exvar["ts"][i - i_ar]
        return ts

    ## --------------------------------------------------------------------------- #
    ##  Validators                                                              ####
    ## --------------------------------------------------------------------------- #

    def _value_is_between(self, value: float, min_value: float, max_value: float) -> bool:
        """
        Check if a value is between two other values.

        Params:
            value (float):
                The value to check.
            min_value (float):
                The minimum value.
            max_value (float):
                The maximum value.

        Returns:
            bool:
                True if the value is between the minimum and maximum values, False otherwise.
        """
        if min_value > max_value:
            raise ValueError(
                f"Invalid range: `min_value` ({min_value}) must be less than or equal to `max_value` ({max_value})."
            )
        return min_value <= value <= max_value

    def _assert_value_is_between(
        self,
        value: float,
        min_value: float,
        max_value: float,
    ) -> None:
        """
        Assert that a value is between two other values.

        Params:
            value (float):
                The value to check.
            min_value (float):
                The minimum value.
            max_value (float):
                The maximum value.

        Raises:
            AssertionError:
                If the value is not between the minimum and maximum values.
        """
        if not self._value_is_between(value, min_value, max_value):
            raise AssertionError(f"Value must be between `{min_value}` and `{max_value}`: `{value}`")

    def _all_values_are_between(
        self,
        values: list[float] | tuple[float, ...],
        min_value: float,
        max_value: float,
    ) -> bool:
        """
        Check if all values in an array are between two other values.

        Params:
            values (Union[list[float], tuple[float, ...]]):
                The array of values to check.
            min_value (float):
                The minimum value.
            max_value (float):
                The maximum value.

        Returns:
            bool:
                True if all values are between the minimum and maximum values, False otherwise.
        """
        return all(self._value_is_between(value, min_value, max_value) for value in values)

    def _assert_all_values_are_between(
        self,
        values: list[float] | tuple[float, ...],
        min_value: float,
        max_value: float,
    ) -> None:
        """
        Assert that all values in an array are between two other values.

        Params:
            values (Union[list[float], tuple[float, ...]]):
                The array of values to check.
            min_value (float):
                The minimum value.
            max_value (float):
                The maximum value.

        Raises:
            AssertionError:
                If any value is not between the minimum and maximum values.
        """
        values_not_between: list[float] = [
            value for value in values if not self._value_is_between(value, min_value, max_value)
        ]
        if not len(values_not_between) == 0:
            raise AssertionError(f"Values not between `{min_value}` and `{max_value}`: {values_not_between}")
