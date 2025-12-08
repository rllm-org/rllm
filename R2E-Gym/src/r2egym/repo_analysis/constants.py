import os
from pathlib import Path

HOME_DIR = Path(os.path.expanduser("~"))
EDITS_DIR = HOME_DIR / "buckets" / "edits_data"


repo_str_names = [
    "sympy",
    "pandas",
    "numpy",
    "scrapy",
    "tornado",
    "statsmodels",
    "pillow",
    "pyramid",
    "datalad",
    "aiohttp",
    "mypy",
    "coveragepy",
    "orange3",
    "bokeh",
]
for repo_str_name in repo_str_names:
    globals()[repo_str_name.upper() + "_DIR"] = Path(repo_str_name)
    globals()["LOCAL_" + repo_str_name.upper() + "_COMMIT_DATA_DIR"] = (
        Path("commit_data") / repo_str_name
    )
    globals()["LOCAL_" + repo_str_name.upper() + "_COMMIT_DATA_DIR"].mkdir(
        parents=True, exist_ok=True
    )
    globals()[repo_str_name.upper() + "_COMMIT_DATA_DIR"] = (
        EDITS_DIR / "commit_data" / repo_str_name
    )
    globals()[repo_str_name.upper() + "_COMMIT_DATA_DIR"].mkdir(
        parents=True, exist_ok=True
    )
    globals()[repo_str_name.upper() + "_TEST_DATA_DIR"] = (
        Path("test_data") / repo_str_name
    )
    globals()[repo_str_name.upper() + "_TEST_DATA_DIR"].mkdir(
        parents=True, exist_ok=True
    )

# ## sympy
# SYMPY_DIR = Path("sympy")

# LOCAL_SYMPY_COMMIT_DATA_DIR = Path("commit_data/sympy")
# SYMPY_COMMIT_DATA_DIR = EDITS_DIR / "commit_data" / "sympy"
# LOCAL_SYMPY_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
# SYMPY_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# SYMPY_TEST_DATA_DIR = Path("test_data/sympy")
# SYMPY_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# BASE_SYMPY_DOCKERFILE = Path(
#     "src/r2e_edits/sympy_analysis/sympy_dockerfiles/parameterized_dockerfile.txt"
# )

# ## pandas
# PANDAS_DIR = Path("pandas")

# LOCAL_PANDAS_COMMIT_DATA_DIR = Path("commit_data/pandas")
# PANDAS_COMMIT_DATA_DIR = EDITS_DIR / "commit_data" / "pandas"
# LOCAL_PANDAS_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
# PANDAS_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# PANDAS_TEST_DATA_DIR = Path("test_data/pandas")
# PANDAS_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ## numpy
# NUMPY_DIR = Path("numpy")

# LOCAL_NUMPY_COMMIT_DATA_DIR = Path("commit_data/numpy")
# NUMPY_COMMIT_DATA_DIR = EDITS_DIR / "commit_data" / "numpy"
# LOCAL_NUMPY_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
# NUMPY_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# NUMPY_TEST_DATA_DIR = Path("test_data/numpy")
# NUMPY_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ## scrapy
# SCRAPY_DIR = Path("scrapy")

# LOCAL_SCRAPY_COMMIT_DATA_DIR = Path("commit_data/scrapy")
# SCRAPY_COMMIT_DATA_DIR = EDITS_DIR / "commit_data" / "scrapy"
# LOCAL_SCRAPY_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
# SCRAPY_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# SCRAPY_TEST_DATA_DIR = Path("test_data/scrapy")
# SCRAPY_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ## tornado
# TORNADO_DIR = Path("tornado")

# LOCAL_TORNADO_COMMIT_DATA_DIR = Path("commit_data/tornado")
# TORNADO_COMMIT_DATA_DIR = EDITS_DIR / "commit_data" / "tornado"
# LOCAL_TORNADO_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
# TORNADO_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# TORNADO_TEST_DATA_DIR = Path("test_data/tornado")
# TORNADO_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ## statsmodels
# STATSMODELS_DIR = Path("statsmodels")

# LOCAL_STATSMODELS_COMMIT_DATA_DIR = Path("commit_data/statsmodels")
# STATSMODELS_COMMIT_DATA_DIR = EDITS_DIR / "commit_data" / "statsmodels"
# LOCAL_STATSMODELS_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
# STATSMODELS_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# STATSMODELS_TEST_DATA_DIR = Path("test_data/statsmodels")
# STATSMODELS_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ## pillow
# PILLOW_DIR = Path("Pillow")

# LOCAL_PILLOW_COMMIT_DATA_DIR = Path("commit_data/pillow")
# PILLOW_COMMIT_DATA_DIR = EDITS_DIR / "commit_data" / "pillow"
# LOCAL_PILLOW_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
# PILLOW_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# PILLOW_TEST_DATA_DIR = Path("test_data/pillow")
# PILLOW_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ## pyramid
# PYRAMID_DIR = Path("pyramid")

# LOCAL_PYRAMID_COMMIT_DATA_DIR = Path("commit_data/pyramid")
# PYRAMID_COMMIT_DATA_DIR = EDITS_DIR / "commit_data" / "pyramid"
# LOCAL_PYRAMID_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
# PYRAMID_COMMIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

# PYRAMID_TEST_DATA_DIR = Path("test_data/pyramid")
# PYRAMID_TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
