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
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import lru_cache
from typing import Any, Literal, overload

# ## Python Third Party Imports ----
import numpy as np
import pandas as pd
from numpy.random import Generator as RandomGenerator
from numpy.typing import NDArray
from toolbox_python.checkers import assert_all_values_of_type
from typeguard import typechecked

# ## Local First Party Imports ----
from synthetic_data_generators.utils.validators import Validators, number


## --------------------------------------------------------------------------- #
##  Exports                                                                 ####
## --------------------------------------------------------------------------- #


__all__: list[str] = ["TimeSeriesGenerator"]


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Classes                                                               ####
#                                                                              #
# ---------------------------------------------------------------------------- #


## --------------------------------------------------------------------------- #
##  TimeSeriesGenerator                                                     ####
## --------------------------------------------------------------------------- #


class TimeSeriesGenerator(Validators):
    """
    !!! note "Summary"
        A class for generating synthetic time series data.

    ???+ abstract "Details"
        - This class provides methods to create synthetic time series data with various characteristics, including seasonality, trends, and noise.
        - The generated data can be used for testing and validation purposes in time series analysis.
        - The class includes methods to generate holiday indices, fixed error indices, semi-Markov indices, and sine indices.
        - It also provides a method to generate polynomial trends and ARMA components.
        - The generated time series data can be customized with different parameters, such as start date, number of periods, and noise scale.

    Methods:
        create_time_series(): Generate a synthetic time series with specified characteristics.
        generate_holiday_index(): Generate a holiday index for the given dates.
        generate_fixed_error_index(): Generate a fixed error seasonality index for the given dates.
        generate_semi_markov_index(): Generate a semi-Markov seasonality index for the given dates.
        generate_sin_index(): Generate a sine seasonality index for the given dates.
        generate_sin_covar_index(): Generate a sine seasonality index with covariance for the given dates.
        generate_season_index(): Generate a seasonality index based on the specified style for the given dates.
        generate_polynom_trend(): Generate a polynomial trend based on interpolation nodes.
        generate_ARMA(): Generate an ARMA component for the time series.

    Attributes:
        random_generator (RandomGenerator): An instance of `numpy.random.Generator` used for random number generation.
        seed (int): The seed value used for random number generation.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        !!! note "Summary"
            Initialize the TimeSeriesGenerator class.

        ???+ abstract "Details"
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
        self._set_seed(seed=seed)

    def create_time_series(
        self,
        start_date: datetime = datetime(2019, 1, 1),
        n_periods: int = 1096,
        interpolation_nodes: Sequence[Sequence[int]] = ([0, 98], [300, 92], [700, 190], [1096, 213]),
        level_breaks: Sequence[Sequence[int]] | None = ([250, 100], [650, -50]),
        AR: Sequence[number] | None = None,
        MA: Sequence[number] | None = None,
        randomwalk_scale: number = 2,
        exogenous: Sequence[dict[Literal["coeff", "ts"], Sequence[number]]] | None = None,
        season_conf: dict[str, Any] | None = {"style": "holiday"},
        season_eff: number = 0.15,
        manual_outliers: Sequence[Sequence[int]] | None = None,
        noise_scale: number = 10,
        seed: int | None = None,
    ) -> pd.DataFrame:
        """
        !!! note "Summary"
            Generate a synthetic time series with specified characteristics.

        ???+ abstract "Details"
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
            interpolation_nodes (Sequence[Sequence[int]]):
                A collection of interpolation nodes, where each node is a tuple containing the x-coordinate and y-coordinate.<br>
                The x-coordinates should be in ascending order.<br>
                Default is `([0, 98], [300, 92], [700, 190], [1096, 213])`.
            level_breaks (Sequence[Sequence[int]] | None):
                A collection of level breaks, where each break is a tuple containing the index and the value to add.<br>
                Default is `([250, 100], [650, -50])`.
            AR (Sequence[number] | None):
                The autoregressive coefficients for the ARMA model.<br>
                Default is `None`.
            MA (Sequence[number] | None):
                The moving average coefficients for the ARMA model.<br>
                Default is `None`.
            randomwalk_scale (number):
                The scale of the random walk component.<br>
                Default is `2`.
            exogenous (Sequence[dict[Literal["coeff", "ts"], Sequence[number]]] | None):
                A list of exogenous variables to include in the ARMA model.<br>
                Default is `None`.
            season_conf (dict[str, Any] | None):
                A dictionary containing the configuration for seasonality.<br>
                Default is `{"style": "holiday"}`.
            season_eff (number):
                The effectiveness of the seasonality component.<br>
                Default is `0.15`.
            manual_outliers (Sequence[Sequence[int]] | None):
                A collection of manual outliers, where each outlier is a tuple containing the index and the value to set.<br>
                Default is `None`.
            noise_scale (number):
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

        # Set seed
        if seed:
            self._set_seed(seed=seed)

        # Date index:
        dates: list[datetime] = self._get_dates(start_date, n_periods=n_periods)

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
        noise: NDArray[np.float64] = self.random_generator.normal(
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
        dates: Sequence[datetime],
        season_dates: Sequence[Sequence[datetime | int]],
    ) -> NDArray[np.int_]:
        """
        !!! note "Summary"
            Generate a holiday index for the given dates based on the provided holiday dates.

        ???+ abstract "Details"
            - A holiday index is a manual selection for date in `dates` to determine whether it is a holiday or not.
            - Basically, it is a manual index of dates in a univariate time series data set which are actual holidays.
            - The return array is generated by checking if each date in `dates` is present in the list of holiday dates generated from `season_dates`.

        !!! warning "Important"
            This function is designed to work with a `.generate_season_index()` when the `style="holiday"`.<br>
            It is not intended to be called directly.

        Params:
            dates (Sequence[datetime]):
                List of datetime objects representing the dates to check.
            season_dates (Sequence[Sequence[datetime | int]]):
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
                self._get_holiday_period(
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
        dates: Sequence[datetime],
        period_length: int = 7,
        period_sd: number = 0.5,
        start_index: int = 4,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate a fixed error seasonality index for the given dates.

        ???+ abstract "Details"
            - A holiday index is a manual selection for date in `dates` to determine whether it is a holiday or not.
            - A fixed error seasonality index is a non-uniform distribution of dates in a univariate time series data set.
            - Basically, it is indicating every `period_length` length of days, occurring every `period_sd` number of days, starting from `start_index`.
            - The return array is a boolean `1` or `0` of length `n_periods`. It will have a seasonality of `period_length` and a disturbance standard deviation of `period_sd`. The result can be used as a non-uniform distribution of weekdays in a histogram (if for eg. frequency is weekly).

        !!! warning "Important"
            This function is designed to work with a `.generate_season_index()` when the `style="fixed+error"`.<br>
            It is not intended to be called directly.

        Params:
            dates (Sequence[datetime]):
                List of datetime objects representing the dates to check.
            period_length (int):
                The length of the period for seasonality.<br>
                For example, if the frequency is weekly, this would be `7`.<br>
                Default is `7`.
            period_sd (number):
                The standard deviation of the disturbance.<br>
                Default is `0.5`.
            start_index (int):
                The starting index for the seasonality.<br>
                Default is `4`.
            seed (int | None):
                The random seed for reproducibility.<br>
                Default is `None`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.

        Returns:
            (NDArray[np.int_]):
                An array of the same length as `dates`, where each element is `1` if the corresponding date is a holiday, and `0` otherwise.
        """

        # Set seed
        if seed:
            self._set_seed(seed=seed)

        # Process
        n_periods: int = len(dates)
        events: NDArray[np.int_] = np.zeros(n_periods).astype(np.int_)
        event_inds: NDArray[Any] = np.arange(n_periods // period_length + 1) * period_length + start_index
        disturbance: NDArray[np.float64] = self.random_generator.normal(
            loc=0.0,
            scale=period_sd,
            size=len(event_inds),
        ).astype(int)
        event_inds = event_inds + disturbance

        # Delete indices that are out of bounds
        if np.any(event_inds >= n_periods):
            event_inds = np.delete(event_inds, event_inds >= n_periods)

        # Return
        return events.astype(np.float64)

    def generate_semi_markov_index(
        self,
        dates: Sequence[datetime],
        period_length: int = 7,
        period_sd: float = 0.5,
        start_index: int = 4,
        seed: int | None = None,
    ) -> NDArray[np.int_]:
        """
        !!! note "Summary"
            Generate a semi-Markov seasonality index for the given dates.

        ???+ abstract "Details"
            - A semi-Markov seasonality index is a uniform distribution of dates in a univariate time series data set.
            - Basically, it is indicating a `period_length` length of days, occurring randomly roughly ever `period_sd` number of days, starting from `start_index`.
            - The return array is a boolean `1` or `0` of length `n_periods`. It will have a seasonality of `period_length` and a disturbance standard deviation of `period_sd`. The result can be used as a uniform distribution of weekdays in a histogram (if for eg. frequency is weekly).

        Params:
            dates (Sequence[datetime]):
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
            seed (int | None):
                The random seed for reproducibility.<br>
                Default is `None`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.

        Returns:
            (NDArray[np.int_]):
                An array of the same length as `dates`, where each element is `1` if the corresponding date is a holiday, and `0` otherwise.
        """

        # Set seed
        if seed:
            self._set_seed(seed=seed)

        # Process
        n_periods: int = len(dates)
        events: NDArray[np.int_] = np.zeros(n_periods).astype(np.int_)
        event_inds: list[int] = [start_index]
        new = self.random_generator.normal(loc=period_length, scale=period_sd, size=1).round()[0]
        while new + event_inds[-1] < n_periods:
            event_inds.append(new + event_inds[-1])
            new = self.random_generator.normal(
                loc=period_length,
                scale=period_sd,
                size=1,
            ).round()[0]
        event_indexes: NDArray[np.int_] = np.array(event_inds).astype(np.int_)

        # For any indices defined above, assign `1` to the events array
        events[event_indexes] = 1

        # Return
        return events

    def generate_sin_index(
        self,
        dates: Sequence[datetime],
        period_length: int = 7,
        start_index: int = 4,
        amplitude: number = 0.5,
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate a sine seasonality index for the given dates.

        ???+ abstract "Details"
            - A sine seasonality index is a periodic function that oscillates around a center value.
            - It is used to model seasonal patterns in time series data.
            - The return array is a sine wave of length `n_periods`, with a period of `period_length`, a phase shift of `start_index`, and an amplitude of `amplitude`.
            - The result can be used to represent seasonal patterns in time series data, such as daily or weekly cycles.
            - With default `amplitude=0.5`, the wave oscillates between `0` and `1` (centered at `0.5`).
            - The formula used is: `amplitude * sin(...) + (1 - amplitude)`, which ensures the wave oscillates between `(1 - 2*amplitude)` and `1`.

        Params:
            dates (Sequence[datetime]):
                List of datetime objects representing the dates to check.
            period_length (int):
                The length of the period for seasonality. This is the wavelength of the sine wave.<br>
                For example, if the frequency is weekly, this would be `7`.<br>
                Default is `7`.
            start_index (int):
                The starting index for the seasonality. Designed to account for seasonal patterns that start at a different point in time.<br>
                Default is `4`.
            amplitude (number):
                The amplitude of the sine wave, controlling the range of oscillation.<br>
                With `amplitude=0.5` (default), the wave oscillates between `0` and `1`.<br>
                With `amplitude=0.3`, the wave oscillates between `0.4` and `1`.<br>
                With `amplitude=1.0`, the wave oscillates between `-1` and `1`.<br>
                Default is `0.5`.

        Raises:
            (TypeCheckError):
                If any of the inputs parsed to the parameters of this function are not the correct type. Uses the [`@typeguard.typechecked`](https://typeguard.readthedocs.io/en/stable/api.html#typeguard.typechecked) decorator.

        Returns:
            (NDArray[np.float64]):
                An array of the same length as `dates`, where each element is a sine value representing the seasonal pattern.
        """
        n_periods: int = len(dates)
        events = amplitude * np.sin((np.arange(n_periods) - start_index) / period_length * 2 * np.pi) + (1 - amplitude)
        return events

    def generate_sin_covar_index(
        self,
        dates: Sequence[datetime],
        period_length: int = 7,
        start_index: int = 4,
        amplitude: number = 1.0,
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate a sine seasonality index with covariance for the given dates.

        ???+ abstract "Details"
            - A sine seasonality index with covariance is a periodic function with varying frequency.
            - It is used to model seasonal patterns in time series data, taking into account the covariance structure of the data.
            - The return array is a sine wave of length `n_periods`, with a period of `period_length`, a phase shift of `start_index`, and controlled amplitude.
            - The result can be used to represent seasonal patterns in time series data, such as daily or weekly cycles with varying intensity.
            - Unlike the simple sine index, this method applies a covariance wave to create a more complex, non-uniform seasonal pattern.

        Params:
            dates (Sequence[datetime]):
                List of datetime objects representing the dates to check.
            period_length (int):
                The length of the period for seasonality. This is the wavelength of the sine wave.<br>
                For example, if the frequency is weekly, this would be `7`.<br>
                Default is `7`.
            start_index (int):
                The starting index for the seasonality. Designed to account for seasonal patterns that start at a different point in time.<br>
                Default is `4`.
            amplitude (number):
                The amplitude multiplier for the sine wave, controlling the range of oscillation.<br>
                With `amplitude=1.0` (default), the wave oscillates in its natural range (approximately `-1` to `1`).<br>
                With `amplitude=0.5`, the wave oscillates in a reduced range (approximately `-0.5` to `0.5`).<br>
                With `amplitude=2.0`, the wave oscillates in an expanded range (approximately `-2` to `2`).<br>
                Default is `1.0`.

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
        sin_wave: NDArray[np.float64] = amplitude * np.sin((covar_wave * dx).cumsum())
        return sin_wave

    @overload
    def generate_season_index(
        self,
        dates: Sequence[datetime],
        style: Literal["fixed+error"],
        *,
        period_length: int,
        period_sd: number,
        start_index: int,
        seed: int | None = None,
    ) -> NDArray[np.float64]: ...
    @overload
    def generate_season_index(
        self,
        dates: Sequence[datetime],
        style: Literal["semi-markov"],
        *,
        period_length: int,
        period_sd: number,
        start_index: int,
        seed: int,
    ) -> NDArray[np.float64]: ...
    @overload
    def generate_season_index(
        self,
        dates: Sequence[datetime],
        style: Literal["holiday"],
        *,
        season_dates: Sequence[Sequence[datetime | int]],
        seed: int | None = None,
    ) -> NDArray[np.float64]: ...
    @overload
    def generate_season_index(
        self,
        dates: Sequence[datetime],
        style: Literal["sin"],
        *,
        period_length: int | None = None,
        start_index: int | None = None,
        amplitude: number | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float64]: ...
    @overload
    def generate_season_index(
        self,
        dates: Sequence[datetime],
        style: Literal["sin_covar"],
        *,
        period_length: int,
        start_index: int,
        amplitude: number | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float64]: ...
    def generate_season_index(
        self,
        dates: Sequence[datetime],
        style: Literal[
            "fixed+error",
            "semi-markov",
            "holiday",
            "sin",
            "sin_covar",
        ],
        *,
        season_dates: Sequence[Sequence[datetime | int]] | None = None,
        period_length: int | None = None,
        period_sd: number | None = None,
        start_index: int | None = None,
        amplitude: number | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate a seasonality index for the given dates based on the specified style.

        ???+ abstract "Details"
            - A seasonality index is a manual selection for date in `dates` to determine whether it is a holiday or not.
            - Basically, it is a manual index of dates in a univariate time series data set which are actual holidays.
            - The return array is generated by checking if each date in `dates` is present in the list of holiday dates generated from `season_dates`.
            - The return array is a boolean `1` or `0` of length `n_periods`. It will have a seasonality of `period_length` and a disturbance standard deviation of `period_sd`. The result can be used as a non-uniform distribution of weekdays in a histogram (if for eg. frequency is weekly).
            - Different styles require different keyword arguments. See the overload signatures for specific parameter requirements.

        Params:
            dates (Sequence[datetime]):
                List of datetime objects representing the dates to check.
            style (Literal["fixed+error", "semi-markov", "holiday", "sin", "sin_covar"]):
                The style of the seasonality index to generate.<br>
                Possible values are:
                - `"fixed+error"`: Fixed error seasonality index.
                - `"semi-markov"`: Semi-Markov seasonality index.
                - `"holiday"`: Holiday seasonality index.
                - `"sin"`: Sine seasonality index.
                - `"sin_covar"`: Sine seasonality index with covariance.

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

        # Map styles to functions
        funcs: dict[str, Callable] = {
            "fixed+error": self.generate_fixed_error_index,
            "semi-markov": self.generate_semi_markov_index,
            "holiday": self.generate_holiday_index,
            "sin": self.generate_sin_index,
            "sin_covar": self.generate_sin_covar_index,
        }

        # Get function based on style
        func: Callable | None = funcs.get(style)

        # Guard clause for unsupported style
        if not func:
            return np.zeros(len(dates)).astype(np.float64)

        # Prepare parameters
        _params: dict[str, Any] = {
            "dates": dates,
            "season_dates": season_dates,
            "period_length": period_length,
            "period_sd": period_sd,
            "start_index": start_index,
            "amplitude": amplitude,
            "seed": seed,
        }

        # Filter out empty parameters
        params: dict[str, Any] = {key: value for key, value in _params.items() if value is not None}

        # Call function with parameters
        return func(**params).astype(np.float64)  # type:ignore

    def generate_polynom_trend(
        self,
        interpolation_nodes: Sequence[Sequence[int]],
        n_periods: int,
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate a polynomial trend based on the provided interpolation nodes.

        ???+ abstract "Details"
            - The polynomial trend is generated using the provided interpolation nodes.
            - The function supports polynomial trends of order 1 (linear), 2 (quadratic), 3 (cubic), and 4 (quartic).
            - The generated trend is an array of the same length as `n_periods`, where each element represents the value of the polynomial trend at that period.
            - The function uses numpy's linear algebra solver to compute the coefficients of the polynomial based on the provided interpolation nodes.

        !!! warning "Important"
            This function is implemented only up to order 3 (cubic interpolation = four nodes).
            It is not intended to be used for higher-order polynomial trends.

        Params:
            interpolation_nodes (Sequence[Sequence[int]]):
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

        if len(interpolation_nodes) == 0:
            # No trend component:
            trend: NDArray[np.float64] = np.zeros(n_periods)
            return trend

        elif len(interpolation_nodes) == 1:
            # No trend component:
            trend: NDArray[np.float64] = np.zeros(n_periods) + interpolation_nodes[0][1]
            return trend

        elif len(interpolation_nodes) == 2:
            # Linear trend component:
            x1, y1 = interpolation_nodes[0]
            x2, y2 = interpolation_nodes[1]
            M = np.column_stack((np.array([x1, x2]), np.ones(2)))
            b = np.array([y1, y2])
            pvec = np.linalg.solve(M, b)
            trend: NDArray[np.float64] = np.arange(n_periods).astype(np.float64)
            trend = pvec[0] * trend + pvec[1]
            return trend

        elif len(interpolation_nodes) == 3:
            # Quadratic trend component:
            x1, y1 = interpolation_nodes[0]
            x2, y2 = interpolation_nodes[1]
            x3, y3 = interpolation_nodes[2]
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

        elif len(interpolation_nodes) == 4:
            # Cubic trend component:
            x1, y1 = interpolation_nodes[0]
            x2, y2 = interpolation_nodes[1]
            x3, y3 = interpolation_nodes[2]
            x4, y4 = interpolation_nodes[3]
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
        AR: Sequence[number],
        MA: Sequence[number],
        randomwalk_scale: number,
        n_periods: int,
        exogenous: Sequence[dict[Literal["coeff", "ts"], Sequence[number]]] | None = None,
        seed: int | None = None,
    ) -> NDArray[np.float64]:
        """
        !!! note "Summary"
            Generate an ARMA (AutoRegressive Moving Average) time series.

        ???+ abstract "Details"
            - The ARMA model is a combination of autoregressive (AR) and moving average (MA) components.
            - The function generates a time series based on the specified AR and MA coefficients, random walk scale, and optional exogenous variables.
            - The generated time series is an array of the same length as `n_periods`, where each element represents the value of the ARMA time series at that period.
            - The function uses numpy's random number generator to generate the noise component of the ARMA model.

        Params:
            AR (Sequence[number]):
                List of autoregressive coefficients.
                The length of the list determines the order of the AR component.
                All values must be between `0` and `1`.
            MA (Sequence[number]):
                List of moving average coefficients.
                The length of the list determines the order of the MA component.
                All values must be between `0` and `1`.
            randomwalk_scale (number):
                Scale parameter for the random walk component.
                This controls the standard deviation of the noise added to the time series.
            n_periods (int):
                The number of periods for which to generate the ARMA time series.
                This determines the length of the output array.
            exogenous (Sequence[dict[Literal["coeff", "ts"], Sequence[number]]] | None):
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

        # Set seed
        if seed:
            self._set_seed(seed=seed)

        # Add noise
        u: NDArray[np.float64] = self.random_generator.normal(
            loc=0.0,
            scale=randomwalk_scale,
            size=n_periods,
        )

        # Generate array
        ts: NDArray[np.float64] = np.zeros(n_periods).astype(np.float64)

        # Generate ARMA time series
        for i in range(n_periods):
            for i_ar in range(min(len(AR), i)):
                ts[i] = ts[i] + AR[i_ar] * ts[i - 1 - i_ar]
            ts[i] = ts[i] + u[i]
            for i_ma in range(min(len(MA), i)):
                ts[i] = ts[i] - MA[i_ma] * u[i - 1 - i_ma]
            for exvar in exogenous:
                for i_ar in range(len(exvar["coeff"])):
                    ts[i] = ts[i] + exvar["coeff"][i_ar] * exvar["ts"][i - i_ar]

        # Return
        return ts

    ## --------------------------------------------------------------------------- #
    ##  Properties                                                              ####
    ## --------------------------------------------------------------------------- #

    @property
    def seed(self) -> int | None:
        """
        !!! note "Summary"
            Get the seed value used for random number generation.

        Returns:
            (int | None):
                The seed value used for random number generation.
        """
        return self._seed

    @property
    def random_generator(self) -> RandomGenerator:
        """
        !!! note "Summary"
            Get the random number generator instance.

        Returns:
            (RandomGenerator):
                The random number generator instance.
        """
        return self._random_generator or self._get_random_generator(seed=self._seed)

    ## --------------------------------------------------------------------------- #
    ##  Getters & Setters                                                       ####
    ## --------------------------------------------------------------------------- #

    def _set_seed(self, seed: int | None = None) -> None:
        """
        !!! note "Summary"
            Set the seed value for random number generation.

        Params:
            seed (int | None):
                The seed value to set for random number generation.
        """
        self._seed: int | None = seed
        self._random_generator: RandomGenerator | None = None

    @lru_cache
    def _get_random_generator(self, seed: int | None = None) -> RandomGenerator:
        """
        !!! note "Summary"
            Get the random number generator.

        Returns:
            (RandomGenerator):
                The random number generator instance.
        """
        return np.random.default_rng(seed=seed)

    @staticmethod
    @overload
    def _get_dates(start_date: datetime, *, end_date: datetime) -> list[datetime]: ...
    @staticmethod
    @overload
    def _get_dates(start_date: datetime, *, n_periods: int) -> list[datetime]: ...
    @staticmethod
    @lru_cache
    def _get_dates(
        start_date: datetime, *, end_date: datetime | None = None, n_periods: int | None = None
    ) -> list[datetime]:
        """
        !!! note "Summary"
            Generate a list of dates between a start and end date or for a specified number of periods.

        Params:
            start_date (datetime):
                The starting date for generating dates.

        Returns:
            (list[datetime]):
                A list of datetime objects representing the generated dates.
        """
        return pd.date_range(start=start_date, end=end_date, periods=n_periods).to_pydatetime().tolist()  # type:ignore

    @staticmethod
    @lru_cache
    def _get_holiday_period(start_date: datetime, periods: int) -> list[datetime]:
        """
        !!! note "Summary"
            Generate a list of holiday dates starting from a given date.

        Params:
            start_date (datetime):
                The starting date for generating holiday dates.
            periods (int):
                The number of holiday dates to generate.

        Returns:
            (list[datetime]):
                A list of datetime objects representing the generated holiday dates.
        """
        return TimeSeriesGenerator._get_dates(start_date, n_periods=periods)
