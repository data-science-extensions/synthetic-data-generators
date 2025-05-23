# Time Series Generators

All of the examples in this page are using the [`TimeSeriesGenerator().create_time_series()`][synthetic_data_generators.time_series.TimeSeriesGenerator.create_time_series] method.

## Straight Line

We get a straight line by having a specific few interpolation nodes:

- `#!py interpolation_nodes=[[n_periods * i / 4, 100 * i] for i in range(4)]`

And when setting the parameters:

- `#!py randomwalk_scale=0`
- `#!py noise_scale=0`
- `#!py season_eff=0`

--8<-- "docs/usage/images/linear_straight_line.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With straight line"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=n_periods,
        interpolation_nodes=[[n_periods * i / 4, 100 * i] for i in range(4)],  # (1)
        level_breaks=[],
        man_outliers=[],
        AR=[],
        MA=[],
        exogenous=[],
        randomwalk_scale=0,  # (2)
        noise_scale=0,  # (3)
        season_eff=0,  # (4)
        season_conf=None,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="Straight line with no noise",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_straight_line.html")
    fig.show()
    ```

    1. Straight line interpolation
    2. No random walks
    3. No noise
    4. No seasonality

## Smooth Curve

If the interpolation nodes were more randomised (not in a straight line), then the generator will aim to build a smooth line which passes through each interpolation node.

For example, if you specify the nodes as:

- `#!py interpolation_nodes=[(0.0, 0), (274.0, 400), (548.0, 250), (822.0, 50)]`

Then you will get a curve that looks like:

--8<-- "docs/usage/images/linear_smooth_curve.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With random interpolation nodes"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=n_periods,
        interpolation_nodes=list(
            zip([n_periods * i / 4 for i in range(4)], [0, 400, 250, 50])
        ),  # (1)
        level_breaks=[],
        manual_outliers=[],
        AR=[],
        MA=[],
        exogenous=[],
        randomwalk_scale=0,  # (2)
        noise_scale=0,  # (3)
        season_eff=0,  # (4)
        season_conf=None,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="Smooth curve with no noise",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("./images/linear_smooth_curve.html")
    fig.show()
    ```

    1. Randomised interpolation
    2. No random walks
    3. No noise
    4. No seasonality


## Noise

The noise is just shifting the data points around  a amount of normal distribution along the linear trend line, with the scale being the standard deviation.

It is controlled with the parameter:

- `#!py noise_scale=10`

--8<-- "docs/usage/images/linear_with_noise.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With no noise"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    interpolation_nodes = [[n_periods * i / 4, 100 * i] for i in range(4)]
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=n_periods,
        interpolation_nodes=interpolation_nodes,
        level_breaks=[],
        manual_outliers=[],
        AR=[],
        MA=[],
        exogenous=[],
        randomwalk_scale=0,  # (1)
        noise_scale=10,  # (2)
        season_eff=0,  # (3)
        season_conf=None,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="Straight line with noise",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_with_noise.html")
    fig.show()
    ```

    1. No random walks
    2. A little bit of noise
    3. No seasonality

If we increase the `#!py noise_scale`, then that will widen the standard deviation of the normal distribution, and add more noise.

--8<-- "docs/usage/images/linear_with_more_noise.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With no noise"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    interpolation_nodes = [[n_periods * i / 4, 100 * i] for i in range(4)]
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=n_periods,
        interpolation_nodes=interpolation_nodes,
        level_breaks=[],
        manual_outliers=[],
        AR=[],
        MA=[],
        exogenous=[],
        randomwalk_scale=0,  # (1)
        noise_scale=50,  # (2)
        season_eff=0,  # (3)
        season_conf=None,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="Straight line with more noise",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_with_more_noise.html")
    fig.show()
    ```

    1. No random walks
    2. A lot of of noise
    3. No seasonality


## Random Walk

The random walk is a random process that describes a path consisting of a succession of random steps. It utilises a randomisation parameter around the normal distribution, then adds the value to the previous one using the Autoregressive (AR) and Moving Average (MA) models (see the [ARMA](https://en.wikipedia.org/wiki/Autoregressive_moving-average_model) for more mathematical detail).

It is controlled with the `randomwalk_scale` parameter. Similar to the `manual_outliers` parameter, this affects the standard deviation of the normal distribution used to generate the random walk:

- `#!py randomwalk_scale=10`

--8<-- "docs/usage/images/linear_with_randomwalk.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With randomwalk"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    interpolation_nodes = [[n_periods * i / 4, 100 * i] for i in range(4)]
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=n_periods,
        interpolation_nodes=interpolation_nodes,
        level_breaks=[],
        manual_outliers=[],
        AR=[],
        MA=[],
        exogenous=[],
        randomwalk_scale=3,  # (1)
        noise_scale=0,  # (2)
        season_eff=0,  # (3)
        season_conf=None,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="With noise",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_with_randomwalk.html")
    fig.show()
    ```

    1. A little bit of random walk
    2. No noise
    3. No seasonality

If you increase this scale, it will increase the standard deviation of each progressive step, introducing more randomisation.

--8<-- "docs/usage/images/linear_with_more_randomwalk.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With no more randomwalk"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    interpolation_nodes = [[n_periods * i / 4, 100 * i] for i in range(4)]
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=n_periods,
        interpolation_nodes=interpolation_nodes,
        level_breaks=[],
        manual_outliers=[],
        AR=[],
        MA=[],
        exogenous=[],
        randomwalk_scale=7,  # (1)
        noise_scale=0,  # (2)
        season_eff=0,  # (3)
        season_conf=None,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="With more randomwalk",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_with_more_randomwalk.html")
    fig.show()
    ```

    1. A lot more of random walk
    2. No noise
    3. No seasonality

The random walk can also be used in conjunction with the AR and MA models. For example, if you set the `#!py AR=[0.9]`, then you will get a time series that is a combination of a random walk and an autoregressive process. To read more about the `AR` parameter, check the docs for the [`generate_ARMA()`][synthetic_data_generators.time_series.TimeSeriesGenerator.generate_ARMA] method.

--8<-- "docs/usage/images/linear_with_randomwalk_and_ar.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With no more randomwalk"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    interpolation_nodes = [[n_periods * i / 4, 100 * i] for i in range(4)]
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=n_periods,
        interpolation_nodes=interpolation_nodes,
        level_breaks=[],
        manual_outliers=[],
        AR=[0.9],  # (1)
        MA=[0],  # (2)
        exogenous=[],
        randomwalk_scale=7,  # (3)
        noise_scale=0,  # (4)
        season_eff=0,  # (5)
        season_conf=None,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="With randomwalk and AR[0.9]",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_with_randomwalk_and_ar.html")
    fig.show()
    ```

    1. Auto Regression: Each element is affected 90% by the previous value
    2. Moving Average: There is no Moving Average effect
    3. A large random walk affect
    4. No noise
    5. No seasonality

If you set the `#!py MA=[0.4]`, then you will get a time series that is a combination of a random walk and a moving average process. To read more about the `MA` parameter, check the docs for the [`generate_ARMA()`][synthetic_data_generators.time_series.TimeSeriesGenerator.generate_ARMA] method.

--8<-- "docs/usage/images/linear_with_randomwalk_and_ma.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With no more randomwalk"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    interpolation_nodes = [[n_periods * i / 4, 100 * i] for i in range(4)]
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=n_periods,
        interpolation_nodes=interpolation_nodes,
        level_breaks=[],
        manual_outliers=[],
        AR=[1],
        MA=[0.4],
        exogenous=[],
        randomwalk_scale=7,  # (1)
        noise_scale=0,  # (2)
        season_eff=0,  # (3)
        season_conf=None,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="With randomwalk and MA[0.4]",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_with_randomwalk_and_ma.html")
    fig.show()
    ```

    1. Auto Regression: There is no Auto Regressive effect
    2. Moving Average: Each value is corrected by 40% from the random walk effect
    3. A large random walk affect
    4. No noise
    5. No seasonality


## Seasonality

The seasonality is a periodic fluctuation in the data, which can be controlled with the `#!py season_eff` parameter. This parameter is then controlled with the `#!py season_conf` parameter, which a dictionary with the following keys:

- `#!py style`: The style of the seasonality. One of:
    - `#!py fixed+error`: A fixed error pattern.
    - `#!py semi-markov`: A semi-Markov pattern.
    - `#!py holiday`: A fixed list of holiday dates.
    - `#!py sin`: A sine wave pattern.
    - `#!py sin_covar`: A sine wave covariance pattern.
- `#!py season_dates`: A list of dates for the seasonality. This is only used if `#!py style` is `#!py holiday`.
- `#!py period_length`: The length of the period for the seasonality. For example, if the frequency is weekly, this would be `7`. This is only used if `#!py style` is `#!py sin` or `#!py sin_covar`.
- `#!py period_sd`: The standard deviation of the period for the seasonality. This is only used if `#!py style` is `#!py sin` or `#!py sin_covar`.
- `#!py start_index`: The starting index for the seasonality. This is only used if `#!py style` is `#!py sin` or `#!py sin_covar`.

For example, if you set the `#!py season_conf` parameter to:

```py
season_conf = {
    "style": "sin",
    "period_length": 365,
    "start_index": 0,
}
```

Then you will get a sine wave pattern with a period of `365` days and a starting index of `0`.
The amplitude is the maximum value of the sine wave, which is `0.5` in this case.

--8<-- "docs/usage/images/linear_with_yearly_seasonality.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With no noise"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    interpolation_nodes = [[1096 * i / 4, 100 * i] for i in range(4)]
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=1096,
        interpolation_nodes=interpolation_nodes,
        level_breaks=[],
        AR=[],
        MA=[],
        randomwalk_scale=0,
        exogenous=[],
        season_eff=0.5,  # (1)
        season_conf={"style": "sin", "period_length": 365, "start_index": 0},  # (2)
        manual_outliers=[],
        noise_scale=0,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="With yearly seasonality (sine wave)",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_with_yearly_seasonality.html")
    fig.show()
    ```

    1. Add seasonality effect of `0.5`
    2. Use Sine wave, with a wavelength of 365

Another example with a different period length and starting index:

```py
season_conf = {
    "style": "sin",
    "period_length": 30,
    "start_index": 4,
}
```

Then you will get a sine wave pattern with a period of `30` days (approximate monthly seasonality) and a starting index of `4`. The amplitude is the maximum value of the sine wave, which is `0.5` in this case.

--8<-- "docs/usage/images/linear_with_monthly_seasonality.html"

??? code "Expand for full code snippet"

    ```py {.py .python linenums="1" title="Linear trend: With no noise"}
    # Imports
    import pandas as pd
    from plotly import express as px, io as pio, graph_objects as go
    from synthetic_data_generators.time_series import TimeSeriesGenerator

    # Settings
    pio.templates.default = "simple_white+gridon"
    SEED = 42
    TSG = TimeSeriesGenerator()
    n_periods = 1096

    # Create data
    interpolation_nodes = [[1096 * i / 4, 100 * i] for i in range(4)]
    df: pd.DataFrame = TSG.create_time_series(
        start_date=datetime(2019, 1, 1),
        n_periods=1096,
        interpolation_nodes=interpolation_nodes,
        level_breaks=[],
        AR=[],
        MA=[],
        randomwalk_scale=0,
        exogenous=[],
        season_eff=0.5,  # (1)
        season_conf={"style": "sin", "period_length": 30, "start_index": 0},  # (2)
        manual_outliers=[],
        noise_scale=0,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="With (approximate) monthly seasonality (sine wave)",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_with_monthly_seasonality.html")
    fig.show()
    ```

    1. Add seasonality effect of `0.5`
    2. Use Sine wave, with a wavelength of 30
