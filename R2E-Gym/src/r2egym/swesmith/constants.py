"""
Pulled from official SWE-Smith repository.
"""

from pathlib import Path

CONDA_VERSION = "py312_24.1.2-0"
DEFAULT_PM_LIKELIHOOD = 0.2
ENV_NAME = "testbed"
KEY_IMAGE_NAME = "image_name"

# If set, then subset of tests are run for post-bug validation
# Affects get_test_command, get_valid_report
KEY_MIN_TESTING = "minimal_testing"
# If set, then for pre-bug validation, individual runs are
# performed instead of running the entire test suite
# Affects valid.py
KEY_MIN_PREGOLD = "minimal_pregold"

KEY_PATCH = "patch"
KEY_TEST_CMD = "test_cmd"
KEY_TIMED_OUT = "timed_out"
LOG_DIR_BUG_GEN = Path("logs/bug_gen")
LOG_DIR_ENV_RECORDS = Path("logs/build_images/records")
LOG_DIR_ISSUE_GEN = Path("logs/issue_gen")
LOG_DIR_RUN_VALIDATION = Path("logs/run_validation")
LOG_DIR_TASKS = Path("logs/task_insts")
LOG_TEST_OUTPUT_PRE_GOLD = "test_output_pre_gold.txt"
MAX_INPUT_TOKENS = 128000
ORG_NAME = "swesmith"
PREFIX_BUG = "bug"
PREFIX_METADATA = "metadata"
REF_SUFFIX = ".ref"
SGLANG_API_KEY = "swesmith"
TEMP_PATCH = "_temp_patch_swesmith.diff"
TEST_OUTPUT_END = ">>>>> End Test Output"
TEST_OUTPUT_START = ">>>>> Start Test Output"
TIMEOUT = 120
UBUNTU_VERSION = "22.04"
VOLUME_NAME_DATASET = "datasets"
VOLUME_NAME_MODEL = "llm-weights"

GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

_DOCKERFILE_BASE_EXTENDED = """
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1 -y
"""


"""
Purpose: Mirroring the constants specified in SWE-bench, this file contains the installation
specifications for specific commit(s) of different Python repositories. It is written to be
compatible with the SWE-bench repository to leverage its ability to create docker images.
"""

### MARK: Commonly Used Installion / Testing Specifications ###

TEST_PYTEST = "pytest --disable-warnings --color=no --tb=no --verbose -rA -p no:snail"

DEFAULT_SPECS = {
    "install": ["python -m pip install -e ."],
    "python": "3.10",
    KEY_TEST_CMD: TEST_PYTEST,
}

CMAKE_VERSIONS = ["3.15.7", "3.16.9", "3.17.5", "3.19.7", "3.23.5", "3.27.9"]
INSTALL_CMAKE = (
    [
        f"wget https://github.com/Kitware/CMake/releases/download/v{v}/cmake-{v}-Linux-x86_64.tar.gz"
        for v in CMAKE_VERSIONS
    ]
    + [
        f"tar -xvzf cmake-{v}-Linux-x86_64.tar.gz && mv cmake-{v}-Linux-x86_64 /usr/share/cmake-{v}"
        if v not in ["3.23.5", "3.27.9"]
        else f"tar -xvzf cmake-{v}-Linux-x86_64.tar.gz && mv cmake-{v}-linux-x86_64 /usr/share/cmake-{v}"
        for v in CMAKE_VERSIONS
    ]
    + [
        f"update-alternatives --install /usr/bin/cmake cmake /usr/share/cmake-{v}/bin/cmake {(idx + 1) * 10}"
        for idx, v in enumerate(CMAKE_VERSIONS)
    ]
)

INSTALL_BAZEL = [
    cmd
    for v in ["6.5.0", "7.4.1", "8.0.0"]
    for cmd in [
        f"mkdir -p /usr/share/bazel-{v}/bin",
        f"wget https://github.com/bazelbuild/bazel/releases/download/{v}/bazel-{v}-linux-x86_64",
        f"chmod +x bazel-{v}-linux-x86_64",
        f"mv bazel-{v}-linux-x86_64 /usr/share/bazel-{v}/bin/bazel",
    ]
]

### MARK Repository/Commit specific installation instructions ###

SPECS_REPO_ADDICT = {"75284f9593dfb929cadd900aff9e35e7c7aec54b": DEFAULT_SPECS}
SPECS_REPO_ALIVE_PROGRESS = {"35853799b84ee682af121f7bc5967bd9b62e34c4": DEFAULT_SPECS}
SPECS_REPO_APISPEC = {
    "8b421526ea1015046de42599dd93da6a3473fe44": {
        **DEFAULT_SPECS,
        "install": ["pip install -e .[dev]"],
    }
}
SPECS_REPO_ARROW = {"1d70d0091980ea489a64fa95a48e99b45f29f0e7": DEFAULT_SPECS}
SPECS_REPO_ASTROID = {"b114f6b58e749b8ab47f80490dce73ea80d8015f": DEFAULT_SPECS}
SPECS_REPO_ASYNC_TIMEOUT = {"d0baa9f162b866e91881ae6cfa4d68839de96fb5": DEFAULT_SPECS}
SPECS_REPO_AUTOGRAD = {
    "ac044f0de1185b725955595840135e9ade06aaed": {
        **DEFAULT_SPECS,
        "install": ["pip install -e '.[scipy,test]'"],
    }
}
SPECS_REPO_BLEACH = {"73871d766de1e33a296eeb4f9faf2451f28bee39": DEFAULT_SPECS}
SPECS_REPO_BOLTONS = {"3bfcfdd04395b6cc74a5c0cdc72c8f64cc4ac01f": DEFAULT_SPECS}
SPECS_REPO_BOTTLE = {"a8dfef301dec35f13e7578306002c40796651629": DEFAULT_SPECS}
SPECS_REPO_BOX = {"a23451d2869a511280eebe194efca41efadd2706": DEFAULT_SPECS}
SPECS_REPO_CANTOOLS = {
    "0c6a78711409e4307de34582f795ddb426d58dd8": {
        **DEFAULT_SPECS,
        "install": ["pip install -e .[dev,plot]"],
    }
}
SPECS_REPO_CHANNELS = {
    "a144b4b8881a93faa567a6bdf2d7f518f4c16cd2": {
        **DEFAULT_SPECS,
        "install": ["pip install -e .[tests,daphne]"],
    }
}
SPECS_REPO_CHARDET = {"9630f2382faa50b81be2f96fd3dfab5f6739a0ef": DEFAULT_SPECS}
SPECS_REPO_CHARDET_NORMALIZER = {
    "1fdd64633572040ab60e62e8b24f29cb7e17660b": DEFAULT_SPECS
}
SPECS_REPO_CLICK = {"fde47b4b4f978f179b9dff34583cb2b99021f482": DEFAULT_SPECS}
SPECS_REPO_CLOUDPICKLE = {"6220b0ce83ffee5e47e06770a1ee38ca9e47c850": DEFAULT_SPECS}
SPECS_REPO_COLORLOG = {"dfa10f59186d3d716aec4165ee79e58f2265c0eb": DEFAULT_SPECS}
SPECS_REPO_COOKIECUTTER = {"b4451231809fb9e4fc2a1e95d433cb030e4b9e06": DEFAULT_SPECS}
SPECS_REPO_DAPHNE = {"32ac73e1a0fb87af0e3280c89fe4cc3ff1231b37": DEFAULT_SPECS}
SPECS_REPO_DATASET = {
    "5c2dc8d3af1e0af0290dcd7ae2cae92589f305a1": {
        **DEFAULT_SPECS,
        "install": ["python setup.py install"],
    }
}
SPECS_REPO_DEEPDIFF = {
    "ed2520229d0369813f6e54cdf9c7e68e8073ef62": {
        **DEFAULT_SPECS,
        "install": [
            "pip install -r requirements-dev.txt",
            "pip install -e .",
        ],
    }
}
SPECS_REPO_DJANGO_MONEY = {
    "835c1ab867d11137b964b94936692bea67a038ec": {
        **DEFAULT_SPECS,
        "install": ["pip install -e .[test,exchange]"],
    }
}
SPECS_REPO_DOMINATE = {"9082227e93f5a370012bb934286caf7385d3e7ac": DEFAULT_SPECS}
SPECS_REPO_DOTENV = {"2b8635b79f1aa15cade0950117d4e7d12c298766": DEFAULT_SPECS}
SPECS_REPO_DRF_NESTED_ROUTERS = {
    "6144169d5c33a1c5134b2fedac1d6cfa312c174e": {
        **DEFAULT_SPECS,
        "install": [
            "pip install -r requirements.txt",
            "pip install -e .",
        ],
    }
}
SPECS_REPO_ENVIRONS = {
    "73c372df71002312615ad0349ae11274bb3edc69": {
        **DEFAULT_SPECS,
        "install": ["pip install -e .[dev]"],
    }
}
SPECS_REPO_EXCEPTIONGROUP = {"0b4f49378b585a338ae10abd72ec2006c5057d7b": DEFAULT_SPECS}
SPECS_REPO_FAKER = {"8b401a7d68f5fda1276f36a8fc502ef32050ed72": DEFAULT_SPECS}
SPECS_REPO_FEEDPARSER = {"cad965a3f52c4b077221a2142fb14ef7f68cd576": DEFAULT_SPECS}
SPECS_REPO_FLAKE8 = {"cf1542cefa3e766670b2066dd75c4571d682a649": DEFAULT_SPECS}
SPECS_REPO_FLASHTEXT = {"b316c7e9e54b6b4d078462b302a83db85f884a94": DEFAULT_SPECS}
SPECS_REPO_FLASK = {"bc098406af9537aacc436cb2ea777fbc9ff4c5aa": DEFAULT_SPECS}
SPECS_REPO_FREEZEGUN = {"5f171db0aaa02c4ade003bbc8885e0bb19efbc81": DEFAULT_SPECS}
SPECS_REPO_FUNCY = {"207a7810c216c7408596d463d3f429686e83b871": DEFAULT_SPECS}
SPECS_REPO_FURL = {"da386f68b8d077086c25adfd205a4c3d502c3012": DEFAULT_SPECS}
SPECS_REPO_FVCORE = {
    "a491d5b9a06746f387aca2f1f9c7c7f28e20bef9": {
        **DEFAULT_SPECS,
        "install": [
            "pip install torch shapely",
            "rm tests/test_focal_loss.py",
            "pip install -e .",
        ],
    }
}
SPECS_REPO_GLOM = {"fb3c4e76f28816aebfd2538980e617742e98a7c2": DEFAULT_SPECS}
SPECS_REPO_GPXPY = {
    "09fc46b3cad16b5bf49edf8e7ae873794a959620": {
        **DEFAULT_SPECS,
        KEY_TEST_CMD: "pytest test.py --verbose --color=no --tb=no --disable-warnings -rA -p no:snail",
    }
}
SPECS_REPO_GRAFANALIB = {"5c3b17edaa437f0bc09b5f1b9275dc8fb91689fb": DEFAULT_SPECS}
SPECS_REPO_GRAPHENE = {"82903263080b3b7f22c2ad84319584d7a3b1a1f6": DEFAULT_SPECS}
SPECS_REPO_GSPREAD = {"a8be3b96f9276779ab680d84a0982282fb184000": DEFAULT_SPECS}
SPECS_REPO_GTTS = {"dbcda4f396074427172d4a1f798a172686ace6e0": DEFAULT_SPECS}
SPECS_REPO_GUNICORN = {"bacbf8aa5152b94e44aa5d2a94aeaf0318a85248": DEFAULT_SPECS}
SPECS_REPO_H11 = {"bed0dd4ae9774b962b19833941bb9ec4dc403da9": DEFAULT_SPECS}
SPECS_REPO_ICECREAM = {"f76fef56b66b59fd9a89502c60a99fbe28ee36bd": DEFAULT_SPECS}
SPECS_REPO_INFLECT = {"c079a96a573ece60b54bd5210bb0f414beb74dcd": DEFAULT_SPECS}
SPECS_REPO_INICONFIG = {"16793eaddac67de0b8d621ae4e42e05b927e8d67": DEFAULT_SPECS}
SPECS_REPO_ISODATE = {"17cb25eb7bc3556a68f3f7b241313e9bb8b23760": DEFAULT_SPECS}
SPECS_REPO_JAX = {
    "ebd90e06fa7caad087e2342431e3899cfd2fdf98": {
        **DEFAULT_SPECS,
        "install": ['pip install -e ".[cpu]"'],
        KEY_TEST_CMD: f"{TEST_PYTEST} -n auto",
        KEY_MIN_TESTING: True,
        KEY_MIN_PREGOLD: True,
    }
}
SPECS_REPO_JINJA = {"ada0a9a6fc265128b46949b5144d2eaa55e6df2c": DEFAULT_SPECS}
SPECS_REPO_JSONSCHEMA = {"93e0caa5752947ec77333da81a634afe41a022ed": DEFAULT_SPECS}
SPECS_REPO_LANGDETECT = {"a1598f1afcbfe9a758cfd06bd688fbc5780177b2": DEFAULT_SPECS}
SPECS_REPO_LINE_PROFILER = {"a646bf0f9ab3d15264a1be14d0d4ee6894966f6a": DEFAULT_SPECS}
SPECS_REPO_MARKDOWNIFY = {"6258f5c38b97ab443b4ddf03e6676ce29b392d06": DEFAULT_SPECS}
SPECS_REPO_MARKUPSAFE = {"620c06c919c1bd7bb1ce3dbee402e1c0c56e7ac3": DEFAULT_SPECS}
SPECS_REPO_MARSHMALLOW = {"9716fc629976c9d3ce30cd15d270d9ac235eb725": DEFAULT_SPECS}
SPECS_REPO_MIDO = {
    "a0158ff95a08f9a4eef628a2e7c793fd3a466640": {
        **DEFAULT_SPECS,
        KEY_TEST_CMD: f"{TEST_PYTEST} -rs -c /dev/null",
    }
}
SPECS_REPO_MISTUNE = {"bf54ef67390e02a5cdee7495d4386d7770c1902b": DEFAULT_SPECS}
SPECS_REPO_NIKOLA = {
    "0f4c230e5159e4e937463eb8d6d2ddfcbb09def2": {
        **DEFAULT_SPECS,
        "install": ["pip install -e '.[extras,tests]'"],
    }
}
SPECS_REPO_OAUTHLIB = {"1fd5253630c03e3f12719dd8c13d43111f66a8d2": DEFAULT_SPECS}
SPECS_REPO_PARAMIKO = {
    "23f92003898b060df0e2b8b1d889455264e63a3e": {
        **DEFAULT_SPECS,
        KEY_TEST_CMD: "pytest -rA --color=no --disable-warnings -p no:snail",
    }
}
SPECS_REPO_PARSE = {"30da9e4f37fdd979487c9fe2673df35b6b204c72": DEFAULT_SPECS}
SPECS_REPO_PARSIMONIOUS = {"0d3f5f93c98ae55707f0958366900275d1ce094f": DEFAULT_SPECS}
SPECS_REPO_PARSO = {
    "338a57602740ad0645b2881e8c105ffdc959e90d": {
        **DEFAULT_SPECS,
        "install": ["python setup.py install"],
    }
}
SPECS_REPO_PATSY = {
    "a5d1648401b0ea0649b077f4b98da27db947d2d0": {
        **DEFAULT_SPECS,
        "install": ["pip install -e .[test]"],
    }
}
SPECS_REPO_PDFMINER = {"1a8bd2f730295b31d6165e4d95fcb5a03793c978": DEFAULT_SPECS}
SPECS_REPO_PDFPLUMBER = {
    "02ff4313f846380fefccec9c73fb4c8d8a80d0ee": {
        **DEFAULT_SPECS,
        "install": [
            "apt-get update && apt-get install ghostscript -y",
            "pip install -e .",
        ],
    }
}
SPECS_REPO_PIPDEPTREE = {
    "c31b641817f8235df97adf178ffd8e4426585f7a": {
        **DEFAULT_SPECS,
        "install": [
            "apt-get update && apt-get install graphviz -y",
            "pip install -e .[test,graphviz]",
        ],
    }
}
SPECS_REPO_PRETTYTABLE = {"ca90b055f20a6e8a06dcc46c2e3afe8ff1e8d0f1": DEFAULT_SPECS}
SPECS_REPO_PTYPROCESS = {"1067dbdaf5cc3ab4786ae355aba7b9512a798734": DEFAULT_SPECS}
SPECS_REPO_PYASN1 = {"0f07d7242a78ab4d129b26256d7474f7168cf536": DEFAULT_SPECS}
SPECS_REPO_PYDICOM = {
    "7d361b3d764dbbb1f8ad7af015e80ce96f6bf286": {**DEFAULT_SPECS, "python": "3.11"}
}
SPECS_REPO_PYFIGLET = {"f8c5f35be70a4bbf93ac032334311b326bc61688": DEFAULT_SPECS}
SPECS_REPO_PYGMENTS = {"27649ebbf5a2519725036b48ec99ef7745f100af": DEFAULT_SPECS}
SPECS_REPO_PYOPENSSL = {"04766a496eb11f69f6226a5a0dfca4db90a5cbd1": DEFAULT_SPECS}
SPECS_REPO_PYPARSING = {"533adf471f85b570006871e60a2e585fcda5b085": DEFAULT_SPECS}
SPECS_REPO_PYPIKA = {"1c9646f0a019a167c32b649b6f5e6423c5ba2c9b": DEFAULT_SPECS}
SPECS_REPO_PYQUERY = {"811cd048ffbe4e69fdc512863671131f98d691fb": DEFAULT_SPECS}
SPECS_REPO_PYSNOOPER = {"57472b4677b6c041647950f28f2d5750c38326c6": DEFAULT_SPECS}
SPECS_REPO_PYTHON_DOCX = {"0cf6d71fb47ede07ecd5de2a8655f9f46c5f083d": DEFAULT_SPECS}
SPECS_REPO_PYTHON_JSON_LOGGER = {
    "5f85723f4693c7289724fdcda84cfc0b62da74d4": DEFAULT_SPECS
}
SPECS_REPO_PYTHON_PINYIN = {"e42dede51abbc40e225da9a8ec8e5bd0043eed21": DEFAULT_SPECS}
SPECS_REPO_PYTHON_PPTX = {"278b47b1dedd5b46ee84c286e77cdfb0bf4594be": DEFAULT_SPECS}
SPECS_REPO_PYTHON_QRCODE = {"456b01d41f16e0cfb0f70c687848e276b78c3e8a": DEFAULT_SPECS}
SPECS_REPO_PYTHON_READABILITY = {
    "40256f40389c1f97be5e83d7838547581653c6aa": DEFAULT_SPECS
}
SPECS_REPO_PYTHON_SLUGIFY = {
    "872b37509399a7f02e53f46ad9881f63f66d334b": {
        **DEFAULT_SPECS,
        KEY_TEST_CMD: "python test.py --verbose",
    }
}
SPECS_REPO_PYVISTA = {
    "3f0fad3f42d9b491679e6aa50e52d93c1a81c042": {
        **DEFAULT_SPECS,
        "install": [
            "apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender1",
            "python -m pip install -e '.[dev]'",
        ],
    }
}
SPECS_REPO_RADON = {"54b88e5878b2724bf4d77f97349588b811abdff2": DEFAULT_SPECS}
SPECS_REPO_RECORDS = {"5941ab2798cb91455b6424a9564c9cd680475fbe": DEFAULT_SPECS}
SPECS_REPO_RED_DISCORDBOT = {"33e0eac741955ce5b7e89d9b8f2f2712727af770": DEFAULT_SPECS}
SPECS_REPO_RESULT = {"0b855e1e38a08d6f0a4b0138b10c127c01e54ab4": DEFAULT_SPECS}
SPECS_REPO_SAFETY = {"7654596be933f8310b294dbc85a7af6066d06e4f": DEFAULT_SPECS}
SPECS_REPO_SCRAPY = {
    "35212ec5b05a3af14c9f87a6193ab24e33d62f9f": {
        **DEFAULT_SPECS,
        "install": [
            "apt-get update && apt-get install -y libxml2-dev libxslt-dev libjpeg-dev",
            "python -m pip install -e .",
            "rm tests/test_feedexport.py",
            "rm tests/test_pipeline_files.py",
        ],
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_SCHEDULE = {"82a43db1b938d8fdf60103bd41f329e06c8d3651": DEFAULT_SPECS}
SPECS_REPO_SCHEMA = {"24a3045773eac497c659f24b32f24a281be9f286": DEFAULT_SPECS}
SPECS_REPO_SOUPSIEVE = {"a8080d97a0355e316981cb0c5c887a861c4244e3": DEFAULT_SPECS}
SPECS_REPO_SPACY = {
    "b3c46c315eb16ce644bddd106d31c3dd349f6bb2": {
        **DEFAULT_SPECS,
        "install": [
            "pip install -r requirements.txt",
            "pip install -e .",
        ],
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_SQLFLUFF = {
    "50a1c4b6ff171188b6b70b39afe82a707b4919ac": {**DEFAULT_SPECS, KEY_MIN_TESTING: True}
}
SPECS_REPO_SQLGLOT = {
    "036601ba9cbe4d175d6a9d38bc27587eab858968": {
        **DEFAULT_SPECS,
        "install": ['pip install -e ".[dev]"'],
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_SQLPARSE = {"e57923b3aa823c524c807953cecc48cf6eec2cb2": DEFAULT_SPECS}
SPECS_REPO_STACKPRINTER = {"219fcc522fa5fd6e440703358f6eb408f3ffc007": DEFAULT_SPECS}
SPECS_REPO_STARLETTE = {"db5063c26030e019f7ee62aef9a1b564eca9f1d6": DEFAULT_SPECS}
SPECS_REPO_STRING_SIM = {"115acaacf926b41a15664bd34e763d074682bda3": DEFAULT_SPECS}
SPECS_REPO_SUNPY = {
    "f8edfd5c4be873fbd28dec4583e7f737a045f546": {
        **DEFAULT_SPECS,
        "python": "3.11",
        "install": ['pip install -e ".[dev]"'],
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_SYMPY = {
    "2ab64612efb287f09822419f4127878a4b664f71": {
        **DEFAULT_SPECS,
        "python": "3.10",
        "install": ["pip install -e ."],
        KEY_MIN_TESTING: True,
        KEY_MIN_PREGOLD: True,
    }
}
SPECS_REPO_TENACITY = {"0d40e76f7d06d631fb127e1ec58c8bd776e70d49": DEFAULT_SPECS}
SPECS_REPO_TERMCOLOR = {"3a42086feb35647bc5aa5f1065b0327200da6b9b": DEFAULT_SPECS}
SPECS_REPO_TEXTDISTANCE = {
    "c3aca916bd756a8cb71114688b469ec90ef5b232": {
        **DEFAULT_SPECS,
        "install": ['pip install -e ".[benchmark,test]"'],
    }
}
SPECS_REPO_TEXTFSM = {"c31b600743895f018e7583f93405a3738a9f4d55": DEFAULT_SPECS}
SPECS_REPO_THEFUZZ = {"8a05a3ee38cbd00a2d2f4bb31db34693b37a1fdd": DEFAULT_SPECS}
SPECS_REPO_TINYDB = {"10644a0e07ad180c5b756aba272ee6b0dbd12df8": DEFAULT_SPECS}
SPECS_REPO_TLDEXTRACT = {
    "3d1bf184d4f20fbdbadd6274560ccd438939160e": {
        **DEFAULT_SPECS,
        "install": ["pip install -e .[testing]"],
    }
}
SPECS_REPO_TOMLI = {"443a0c1bc5da39b7ed84306912ee1900e6b72e2f": DEFAULT_SPECS}
SPECS_REPO_TORNADO = {
    "d5ac65c1f1453c2aeddd089d8e68c159645c13e1": {
        **DEFAULT_SPECS,
        KEY_TEST_CMD: "python -m tornado.test --verbose",
    }
}
SPECS_REPO_TRIO = {"cfbbe2c1f96e93b19bc2577d2cab3f4fe2e81153": DEFAULT_SPECS}
SPECS_REPO_TWEEPY = {
    "91a41c6e1c955d278c370d51d5cf43b05f7cd979": {
        **DEFAULT_SPECS,
        "install": ["pip install -e '.[dev,test,async]'"],
    }
}
SPECS_REPO_TYPEGUARD = {
    "b6a7e4387c30a9f7d635712157c889eb073c1ea3": {
        **DEFAULT_SPECS,
        "install": ["pip install -e .[test,doc]"],
    }
}
SPECS_REPO_USADDRESS = {
    "a42a8f0c14bd2e273939fd51c604f10826301e73": {
        **DEFAULT_SPECS,
        "install": ["pip install -e .[dev]"],
    }
}
SPECS_REPO_VOLUPTUOUS = {"a7a55f83b9fa7ba68b0669b3d78a61de703e0a16": DEFAULT_SPECS}
SPECS_REPO_WEBARGS = {"dbde72fe5db8a999acd1716d5ef855ab7cc1a274": DEFAULT_SPECS}
SPECS_REPO_WORDCLOUD = {"ec24191c64570d287032c5a4179c38237cd94043": DEFAULT_SPECS}
SPECS_REPO_XMLTODICT = {"0952f382c2340bc8b86a5503ba765a35a49cf7c4": DEFAULT_SPECS}
SPECS_REPO_YAMLLINT = {"8513d9b97da3b32453b3fccb221f4ab134a028d7": DEFAULT_SPECS}

### MARK: SWE-gym Repositories
SPECS_REPO_MOTO = {
    "694ce1f4880c784fed0553bc19b2ace6691bc109": {
        **DEFAULT_SPECS,
        "python": "3.12",
        "install": ["make init"],
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_MYPY = {
    "e93f06ceab81d8ff1f777c7587d04c339cfd5a16": {
        "python": "3.12",
        "install": [
            "git submodule update --init mypy/typeshed || true",
            "python -m pip install -r test-requirements.txt",
            "python -m pip install -e .",
            "hash -r",
        ],
        KEY_TEST_CMD: "pytest --color=no -rA -k -p no:snail",
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_MONAI = {
    "a09c1f08461cec3d2131fde3939ef38c3c4ad5fc": {
        "python": "3.12",
        "install": [
            "sed -i '/^git+https:\/\/github.com\/Project-MONAI\//d' requirements-dev.txt",
            "python -m pip install -U -r requirements-dev.txt",
            "python -m pip install -e .",
        ],
        KEY_TEST_CMD: TEST_PYTEST,
        KEY_MIN_PREGOLD: True,
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_DVC = {
    "1d6ea68133289ceab2637ce7095772678af792c6": {
        **DEFAULT_SPECS,
        "install": ['pip install -e ".[dev]"'],
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_HYDRA = {
    "0f03eb60c2ecd1fbdb25ede9a2c4faeac81de491": {
        **DEFAULT_SPECS,
        "install": [
            "apt-get update && apt-get install -y openjdk-17-jdk openjdk-17-jre",
            "pip install -e .",
        ],
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_DASK = {
    "5f61e42324c3a6cd4da17b5d5ebe4663aa4b8783": {
        **DEFAULT_SPECS,
        "install": ["python -m pip install graphviz", "python -m pip install -e ."],
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_MODIN = {
    "8c7799fdbbc2fb0543224160dd928215852b7757": {
        **DEFAULT_SPECS,
        "install": ['pip install -e ".[all]"'],
        KEY_MIN_PREGOLD: True,
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_PYDANTIC = {
    "acb0f10fda1c78441e052c57b4288bc91431f852": {
        "python": "3.10",
        "install": [
            "apt-get update && apt-get install -y locales pipx",
            "pipx install uv",
            "pipx install pre-commit",
            'export PATH="$HOME/.local/bin:$PATH"',
            "make install",
        ],
        KEY_TEST_CMD: f"/root/.local/bin/uv run {TEST_PYTEST}",
    }
}
SPECS_REPO_CONAN = {
    "86f29e137a10bb6ed140c1a8c05c3099987b13c5": {
        **DEFAULT_SPECS,
        "install": INSTALL_CMAKE
        + INSTALL_BAZEL
        + [
            "apt-get -y update && apt-get -y upgrade && apt-get install -y build-essential cmake automake autoconf pkg-config meson ninja-build",
            "python -m pip install -r conans/requirements.txt",
            "python -m pip install -r conans/requirements_server.txt",
            "python -m pip install -r conans/requirements_dev.txt",
            "python -m pip install -e .",
        ],
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_PANDAS = {
    "95280573e15be59036f98d82a8792599c10c6603": {
        **DEFAULT_SPECS,
        "install": [
            "git remote add upstream https://github.com/pandas-dev/pandas.git",
            "git fetch upstream --tags",
            "python -m pip install -ve . --no-build-isolation -Ceditable-verbose=true",
            """sed -i 's/__version__="[^"]*"/__version__="3.0.0.dev0+1992.g95280573e1"/' build/cp310/_version_meson.py""",
        ],
        KEY_MIN_PREGOLD: True,
        KEY_MIN_TESTING: True,
    }
}
SPECS_REPO_MONKEYTYPE = {
    "70c3acf62950be5dfb28743c7a719bfdecebcd84": DEFAULT_SPECS,
}


MAP_REPO_TO_SPECS = {
    "adrienverge/yamllint": SPECS_REPO_YAMLLINT,
    "agronholm/exceptiongroup": SPECS_REPO_EXCEPTIONGROUP,
    "agronholm/typeguard": SPECS_REPO_TYPEGUARD,
    "aio-libs/async-timeout": SPECS_REPO_ASYNC_TIMEOUT,
    "alanjds/drf-nested-routers": SPECS_REPO_DRF_NESTED_ROUTERS,
    "alecthomas/voluptuous": SPECS_REPO_VOLUPTUOUS,
    "amueller/word_cloud": SPECS_REPO_WORDCLOUD,
    "andialbrecht/sqlparse": SPECS_REPO_SQLPARSE,
    "arrow-py/arrow": SPECS_REPO_ARROW,
    "benoitc/gunicorn": SPECS_REPO_GUNICORN,
    "borntyping/python-colorlog": SPECS_REPO_COLORLOG,
    "bottlepy/bottle": SPECS_REPO_BOTTLE,
    "buriy/python-readability": SPECS_REPO_PYTHON_READABILITY,
    "burnash/gspread": SPECS_REPO_GSPREAD,
    "cantools/cantools": SPECS_REPO_CANTOOLS,
    "cdgriffith/Box": SPECS_REPO_BOX,
    "chardet/chardet": SPECS_REPO_CHARDET,
    "cknd/stackprinter": SPECS_REPO_STACKPRINTER,
    "cloudpipe/cloudpickle": SPECS_REPO_CLOUDPICKLE,
    "Cog-Creators/Red-DiscordBot": SPECS_REPO_RED_DISCORDBOT,
    "conan-io/conan": SPECS_REPO_CONAN,
    "cookiecutter/cookiecutter": SPECS_REPO_COOKIECUTTER,
    "cool-RR/PySnooper": SPECS_REPO_PYSNOOPER,
    "dask/dask": SPECS_REPO_DASK,
    "datamade/usaddress": SPECS_REPO_USADDRESS,
    "davidhalter/parso": SPECS_REPO_PARSO,
    "dbader/schedule": SPECS_REPO_SCHEDULE,
    "django-money/django-money": SPECS_REPO_DJANGO_MONEY,
    "django/channels": SPECS_REPO_CHANNELS,
    "django/daphne": SPECS_REPO_DAPHNE,
    "encode/starlette": SPECS_REPO_STARLETTE,
    "erikrose/parsimonious": SPECS_REPO_PARSIMONIOUS,
    "facebookresearch/fvcore": SPECS_REPO_FVCORE,
    "facebookresearch/hydra": SPECS_REPO_HYDRA,
    "facelessuser/soupsieve": SPECS_REPO_SOUPSIEVE,
    "gawel/pyquery": SPECS_REPO_PYQUERY,
    "getmoto/moto": SPECS_REPO_MOTO,
    "getnikola/nikola": SPECS_REPO_NIKOLA,
    "google/textfsm": SPECS_REPO_TEXTFSM,
    "graphql-python/graphene": SPECS_REPO_GRAPHENE,
    "gruns/furl": SPECS_REPO_FURL,
    "gruns/icecream": SPECS_REPO_ICECREAM,
    "gweis/isodate": SPECS_REPO_ISODATE,
    "HIPS/autograd": SPECS_REPO_AUTOGRAD,
    "hukkin/tomli": SPECS_REPO_TOMLI,
    "Instagram/MonkeyType": SPECS_REPO_MONKEYTYPE,
    "iterative/dvc": SPECS_REPO_DVC,
    "jaraco/inflect": SPECS_REPO_INFLECT,
    "jawah/charset_normalizer": SPECS_REPO_CHARDET_NORMALIZER,
    "jax-ml/jax": SPECS_REPO_JAX,
    "jd/tenacity": SPECS_REPO_TENACITY,
    "john-kurkowski/tldextract": SPECS_REPO_TLDEXTRACT,
    "joke2k/faker": SPECS_REPO_FAKER,
    "jsvine/pdfplumber": SPECS_REPO_PDFPLUMBER,
    "kayak/pypika": SPECS_REPO_PYPIKA,
    "keleshev/schema": SPECS_REPO_SCHEMA,
    "kennethreitz/records": SPECS_REPO_RECORDS,
    "Knio/dominate": SPECS_REPO_DOMINATE,
    "kurtmckee/feedparser": SPECS_REPO_FEEDPARSER,
    "lepture/mistune": SPECS_REPO_MISTUNE,
    "life4/textdistance": SPECS_REPO_TEXTDISTANCE,
    "lincolnloop/python-qrcode": SPECS_REPO_PYTHON_QRCODE,
    "luozhouyang/python-string-similarity": SPECS_REPO_STRING_SIM,
    "madzak/python-json-logger": SPECS_REPO_PYTHON_JSON_LOGGER,
    "mahmoud/boltons": SPECS_REPO_BOLTONS,
    "mahmoud/glom": SPECS_REPO_GLOM,
    "marshmallow-code/apispec": SPECS_REPO_APISPEC,
    "marshmallow-code/marshmallow": SPECS_REPO_MARSHMALLOW,
    "marshmallow-code/webargs": SPECS_REPO_WEBARGS,
    "martinblech/xmltodict": SPECS_REPO_XMLTODICT,
    "matthewwithanm/python-markdownify": SPECS_REPO_MARKDOWNIFY,
    "mewwts/addict": SPECS_REPO_ADDICT,
    "mido/mido": SPECS_REPO_MIDO,
    "Mimino666/langdetect": SPECS_REPO_LANGDETECT,
    "modin-project/modin": SPECS_REPO_MODIN,
    "mozilla/bleach": SPECS_REPO_BLEACH,
    "mozillazg/python-pinyin": SPECS_REPO_PYTHON_PINYIN,
    "msiemens/tinydb": SPECS_REPO_TINYDB,
    "oauthlib/oauthlib": SPECS_REPO_OAUTHLIB,
    "pallets/click": SPECS_REPO_CLICK,
    "pallets/flask": SPECS_REPO_FLASK,
    "pallets/jinja": SPECS_REPO_JINJA,
    "pallets/markupsafe": SPECS_REPO_MARKUPSAFE,
    "pandas-dev/pandas": SPECS_REPO_PANDAS,
    "paramiko/paramiko": SPECS_REPO_PARAMIKO,
    "pdfminer/pdfminer.six": SPECS_REPO_PDFMINER,
    "pexpect/ptyprocess": SPECS_REPO_PTYPROCESS,
    "pndurette/gTTS": SPECS_REPO_GTTS,
    "prettytable/prettytable": SPECS_REPO_PRETTYTABLE,
    "Project-MONAI/MONAI": SPECS_REPO_MONAI,
    "pudo/dataset": SPECS_REPO_DATASET,
    "pwaller/pyfiglet": SPECS_REPO_PYFIGLET,
    "pyasn1/pyasn1": SPECS_REPO_PYASN1,
    "pyca/pyopenssl": SPECS_REPO_PYOPENSSL,
    "PyCQA/flake8": SPECS_REPO_FLAKE8,
    "pydantic/pydantic": SPECS_REPO_PYDANTIC,
    "pydata/patsy": SPECS_REPO_PATSY,
    "pydicom/pydicom": SPECS_REPO_PYDICOM,
    "pygments/pygments": SPECS_REPO_PYGMENTS,
    "pylint-dev/astroid": SPECS_REPO_ASTROID,
    "pyparsing/pyparsing": SPECS_REPO_PYPARSING,
    "pytest-dev/iniconfig": SPECS_REPO_INICONFIG,
    "python-hyper/h11": SPECS_REPO_H11,
    "python-jsonschema/jsonschema": SPECS_REPO_JSONSCHEMA,
    "python-openxml/python-docx": SPECS_REPO_PYTHON_DOCX,
    "python-trio/trio": SPECS_REPO_TRIO,
    "python/mypy": SPECS_REPO_MYPY,
    "pyupio/safety": SPECS_REPO_SAFETY,
    "pyutils/line_profiler": SPECS_REPO_LINE_PROFILER,
    "pyvista/pyvista": SPECS_REPO_PYVISTA,
    "r1chardj0n3s/parse": SPECS_REPO_PARSE,
    "rsalmei/alive-progress": SPECS_REPO_ALIVE_PROGRESS,
    "rubik/radon": SPECS_REPO_RADON,
    "rustedpy/result": SPECS_REPO_RESULT,
    "scanny/python-pptx": SPECS_REPO_PYTHON_PPTX,
    "scrapy/scrapy": SPECS_REPO_SCRAPY,
    "seatgeek/thefuzz": SPECS_REPO_THEFUZZ,
    "seperman/deepdiff": SPECS_REPO_DEEPDIFF,
    "sloria/environs": SPECS_REPO_ENVIRONS,
    "spulec/freezegun": SPECS_REPO_FREEZEGUN,
    "sqlfluff/sqlfluff": SPECS_REPO_SQLFLUFF,
    "sunpy/sunpy": SPECS_REPO_SUNPY,
    "Suor/funcy": SPECS_REPO_FUNCY,
    "sympy/sympy": SPECS_REPO_SYMPY,
    "termcolor/termcolor": SPECS_REPO_TERMCOLOR,
    "theskumar/python-dotenv": SPECS_REPO_DOTENV,
    "tkrajina/gpxpy": SPECS_REPO_GPXPY,
    "tobymao/sqlglot": SPECS_REPO_SQLGLOT,
    "tornadoweb/tornado": SPECS_REPO_TORNADO,
    "tox-dev/pipdeptree": SPECS_REPO_PIPDEPTREE,
    "tweepy/tweepy": SPECS_REPO_TWEEPY,
    "un33k/python-slugify": SPECS_REPO_PYTHON_SLUGIFY,
    "vi3k6i5/flashtext": SPECS_REPO_FLASHTEXT,
    "weaveworks/grafanalib": SPECS_REPO_GRAFANALIB,
}