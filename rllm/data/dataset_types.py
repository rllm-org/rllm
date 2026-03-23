import enum
from dataclasses import dataclass, field
from typing import cast


@dataclass
class DatasetMetadata:
    """Metadata for a dataset in the catalog/registry."""

    name: str = ""
    description: str = ""
    source: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)


class TrainDataset:
    class Math(enum.Enum):
        # The standard American beginner competitions.
        AIME = "AIME"
        AMC = "AMC"
        # Omni math dataset
        OMNI_MATH = "OMNI_MATH"
        # Unique Olympiad problems from NUMINA
        NUMINA_OLYMPIAD = "OLYMPIAD"
        # Dan Hendrycks math
        MATH = "MATH"
        GSM8k = "GSM8K"
        STILL = "STILL"
        DEEPSCALER = "DEEPSCALER"
        DEEPSCALER_7B = "DEEPSCALER_7B"

    class Code(enum.Enum):
        TACO = "TACO"
        APPS = "APPS"
        CODEFORCES = "CODEFORCES"
        CODE_CONTESTS = "CODE_CONTESTS"
        LIVECODEBENCH = "LIVECODEBENCH"
        LEETCODE = "LEETCODE"
        PRIMEINTELLECT = "PRIMEINTELLECT"
        KODCODE = "KODCODE"

    class Web(enum.Enum):
        GAIA = "GAIA"


class TestDataset:
    class Math(enum.Enum):
        AIME = "AIME"
        AMC = "AMC"
        MATH = "MATH"
        GSM8k = "GSM8k"
        MINERVA = "MINERVA"
        OLYMPIAD_BENCH = "OLYMPIAD_BENCH"

    class Code(enum.Enum):
        TACO = "TACO"
        CODEFORCES = "CODEFORCES"
        CODE_CONTESTS = "CODE_CONTESTS"
        LIVECODEBENCH = "LIVECODEBENCH"
        LEETCODE = "LEETCODE"
        HUMANEVALPLUS = "HUMANEVALPLUS"

    class Web(enum.Enum):
        GAIA = "GAIA"


Dataset = TrainDataset.Math | TrainDataset.Code | TrainDataset.Web | TestDataset.Math | TestDataset.Code | TestDataset.Web


@dataclass
class Problem:
    problem: str
    solution: str
    answer: str
    difficulty: float
    dataset: Dataset


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""

    datasets: list[Dataset] | list[str] | str = field(default_factory=lambda: ["AMC", "AIME"])
    dataset_weights: list[float] = field(default_factory=lambda: [0.5, 0.5])
    # Determined by the main RL config. Can also be set manually.
    dataloader_batch_size: int = 8

    def __post_init__(self):
        # Handle single string input
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        # Convert string dataset names to Dataset enum values
        if isinstance(self.datasets[0], str):
            converted_datasets: list[Dataset] = []
            for dataset_name in self.datasets:
                # Try to match with TrainDataset first, then TestDataset
                dataset_found = False
                # Check all TrainDataset enum classes
                for enum_cls in [TrainDataset.Math, TrainDataset.Code, TrainDataset.Web]:
                    try:
                        dataset = enum_cls(dataset_name)
                        converted_datasets.append(cast(Dataset, dataset))
                        dataset_found = True
                        break
                    except ValueError:
                        continue

                # If not found in TrainDataset, try TestDataset
                if not dataset_found:
                    for enum_cls in [TestDataset.Math, TestDataset.Code, TestDataset.Web]:
                        try:
                            dataset = enum_cls(dataset_name)
                            converted_datasets.append(cast(Dataset, dataset))
                            dataset_found = True
                            break
                        except ValueError:
                            continue

                if not dataset_found:
                    raise ValueError(f"Dataset {dataset_name} not found in TrainDataset or TestDataset.")

            self.datasets = converted_datasets

        # Set uniform weights if not specified
        if not self.dataset_weights:
            self.dataset_weights = [1.0 / len(self.datasets)] * len(self.datasets)

        if self.dataloader_batch_size <= 0:
            raise ValueError("dataloader_batch_size must be greater than 0")

        # Validate weights length matches datasets length
        if len(self.dataset_weights) != len(self.datasets):
            raise ValueError("Number of weights must match number of datasets")
