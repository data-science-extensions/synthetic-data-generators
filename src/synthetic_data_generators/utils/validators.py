# ============================================================================ #
#                                                                              #
#     Title: Validators Utility Module                                         #
#     Purpose: Provides validation functions and classes for numeric ranges    #
#                                                                              #
# ============================================================================ #


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Setup                                                                 ####
#                                                                              #
# ---------------------------------------------------------------------------- #


## --------------------------------------------------------------------------- #
##  Imports                                                                 ####
## --------------------------------------------------------------------------- #


# ## Future Python Library Imports ----
from __future__ import annotations

# ## Python StdLib Imports ----
from collections.abc import Sequence

# ## Python Third Party Imports ----
from toolbox_python.checkers import is_valid


## --------------------------------------------------------------------------- #
##  Exports                                                                 ####
## --------------------------------------------------------------------------- #


__all__: list[str] = ["Validators", "number"]


## --------------------------------------------------------------------------- #
##  Types                                                                   ####
## --------------------------------------------------------------------------- #


number = float | int


# ---------------------------------------------------------------------------- #
#                                                                              #
#     Validators                                                            ####
#                                                                              #
# ---------------------------------------------------------------------------- #


class Validators:

    @staticmethod
    def _value_is_between(value: number, min_value: number, max_value: number) -> bool:
        """
        !!! note "Summary"
            Check if a value is between two other values.

        Params:
            value (number):
                The value to check.
            min_value (number):
                The minimum value.
            max_value (number):
                The maximum value.

        Returns:
            (bool):
                True if the value is between the minimum and maximum values, False otherwise.
        """
        if not is_valid(min_value, "<=", max_value):
            raise ValueError(
                f"Invalid range: min_value `{min_value}` must be less than or equal to max_value `{max_value}`"
            )
        result: bool = is_valid(value, ">=", min_value) and is_valid(value, "<=", max_value)
        return result

    @staticmethod
    def _assert_value_is_between(
        value: number,
        min_value: number,
        max_value: number,
    ) -> None:
        """
        !!! note "Summary"
            Assert that a value is between two other values.

        Params:
            value (number):
                The value to check.
            min_value (number):
                The minimum value.
            max_value (number):
                The maximum value.

        Raises:
            (AssertionError):
                If the value is not between the minimum and maximum values.
        """
        if not Validators._value_is_between(value, min_value, max_value):
            raise AssertionError(f"Invalid Value: `{value}`. Must be between `{min_value}` and `{max_value}`")

    @staticmethod
    def _all_values_are_between(
        values: Sequence[number],
        min_value: number,
        max_value: number,
    ) -> bool:
        """
        !!! note "Summary"
            Check if all values in an array are between two other values.

        Params:
            values (Sequence[number]):
                The array of values to check.
            min_value (number):
                The minimum value.
            max_value (number):
                The maximum value.

        Returns:
            (bool):
                True if all values are between the minimum and maximum values, False otherwise.
        """
        return all(Validators._value_is_between(value, min_value, max_value) for value in values)

    @staticmethod
    def _assert_all_values_are_between(
        values: Sequence[number],
        min_value: number,
        max_value: number,
    ) -> None:
        """
        !!! note "Summary"
            Assert that all values in an array are between two other values.

        Params:
            values (Sequence[number]):
                The array of values to check.
            min_value (number):
                The minimum value.
            max_value (number):
                The maximum value.

        Raises:
            (AssertionError):
                If any value is not between the minimum and maximum values.
        """
        values_not_between: list[number] = [
            value for value in values if not Validators._value_is_between(value, min_value, max_value)
        ]
        if not len(values_not_between) == 0:
            raise AssertionError(f"Values not between `{min_value}` and `{max_value}`: {values_not_between}")
