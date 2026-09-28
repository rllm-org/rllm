"""Reuse upstream preparation; only bound parquet writer buffer sizes.

Full hidden-test strings exceed Arrow's 2GB page limit with the pandas default
row-group layout. This process-local writer override changes storage encoding,
not rows, preprocessing, split selection, flow, evaluator, or training logic.
"""
from functools import partialmethod

import pandas as pd

pd.DataFrame.to_parquet = partialmethod(
    pd.DataFrame.to_parquet, row_group_size=32, use_dictionary=False,
    data_page_size=1024 * 1024,
)

from prepare_deepcoder_data import main

if __name__ == "__main__":
    main()
