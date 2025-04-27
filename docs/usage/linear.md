# Linear

All of the examples in this page are using the [`TimeSeriesGenerator().create_time_series()`][synthetic_data_generators.time_series.TimeSeriesGenerator.create_time_series] method.

## Linear with no Noise

It is possible to get a straight line when setting the parameters: `randomwalk_scale=0`, `noise_scale=0` and `season_eff=0`.

--8<-- "docs/usage/images/linear_no_noise.html"

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
        man_outliers=[],
        AR=[],
        MA=[],
        exogenous=[],
        randomwalk_scale=0,  # (1)!
        noise_scale=0,  # (2)!
        season_eff=0,  # (3)!
        season_conf=None,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="With no noise",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_no_noise.html")
    fig.show()
    ```

    1. No random walks
    2. No noise
    3. No seasonality


## Linear with Noise

It is possible to add some noise with the: `noise_scale=10`.

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
        man_outliers=[],
        AR=[],
        MA=[],
        exogenous=[],
        randomwalk_scale=0,  # (1)!
        noise_scale=10,  # (2)!
        season_eff=0,  # (3)!
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
    fig.write_html("images/linear_with_noise.html")
    fig.show()
    ```

    1. No random walks
    2. Little bit of noise
    3. No seasonality


## Linear with Randomwalk

It is possible to add some noise with the: `randomwalk_scale=3`.

--8<-- "docs/usage/images/linear_with_randomwalk.html"

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
        man_outliers=[],
        AR=[],
        MA=[],
        exogenous=[],
        randomwalk_scale=3,  # (1)!
        noise_scale=0,  # (2)!
        season_eff=0,  # (3)!
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
    fig.write_html("images/linear_with_noise.html")
    fig.show()
    ```

    1. No random walks
    2. Little bit of noise
    3. No seasonality


## Linear with Increasing Seasonality (sine wave)

It is possible to add some seasonality in the parameter: `season_conf`. The effect can be decreased with the parameter: `season_eff`.

--8<-- "docs/usage/images/linear_with_seasonality.html"

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
        season_eff=0.5,
        season_conf={"style": "sin", "period_length": 365, "start_index": 0},
        man_outliers=[],
        noise_scale=0,
        seed=SEED,
    )

    # Build plot
    fig: go.Figure = px.line(
        df,
        x="Date",
        y="Value",
        title="Linear Trend",
        subtitle="With increasing seasonality (sine wave)",
    ).update_layout(title_x=0.5, title_xanchor="center")

    # Render plot
    fig.write_html("images/linear_with_seasonality.html")
    fig.show()
    ```
