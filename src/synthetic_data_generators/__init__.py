# ## Python StdLib Imports ----
from importlib.metadata import metadata


_metadata = metadata("synthetic-data-generators")
__name__: str = _metadata["Name"]
__version__: str = _metadata["Version"]
__author__: str = _metadata["Author"]
__email__: str = _metadata.get("Email", "")
