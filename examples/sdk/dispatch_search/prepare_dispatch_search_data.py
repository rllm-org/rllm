from datasets import concatenate_datasets, load_dataset

from rllm.data.dataset import DatasetRegistry

CLIMATE_FACT_CHECK_DATASET = "Yoonseong/climatebert_factcheck"
COVID_FACT_CHECK_DATASET = "justinqbui/covid_fact_checked_polifact"

CLIMATE_FACT_CLAIM_COL, COVID_FACT_CLAIM_COL = "claim", "claim"
CLIMATE_FACT_LABEL_COL, COVID_FACT_LABEL_COL = "evidence_label", "adjusted rating"

CLIMATE_FACT_LABEL_MAP = {"SUPPORTS": True, "REFUTES": False, "NOT_ENOUGH_INFO": None}
COVID_FACT_LABEL_MAP = {"true": True, "false": False, "misleading": None}


def prepare_climate_fact_check_data(return_original: bool = False, balanced_label: bool = True, original_split: str = "valid", train_test_ratio: float = 0.2):
    """
    This functon does the following:
    1. Load the dataset from HuggingFace
    2. Retain only the columns `claim_col` and `label_col`, rename the `label_col` to standardized "label" column
    3. Map the original label to the standardized True/False label -- delete rows with None label after mapping
    4. If `balanced_label` is True, balance the dataset by randomly sampling from the True and False labels
    5. Split the dataset into training and test sets with the given ratio

    Returns:
        - train_dataset: Dataset
        - test_dataset: Dataset
    """
    # 1. Load the dataset from HuggingFace
    raw_dataset = load_dataset(CLIMATE_FACT_CHECK_DATASET, split=original_split)

    # 2. Retain only the columns `claim_col` and `label_col`, rename the `label_col` to standardized "label" column
    dataset = raw_dataset.select_columns([CLIMATE_FACT_CLAIM_COL, CLIMATE_FACT_LABEL_COL])
    dataset = dataset.rename_column(CLIMATE_FACT_LABEL_COL, "label")

    # 3. Map the original label to the standardized True/False label -- delete rows with None label after mapping
    dataset = dataset.map(lambda x: {"label": CLIMATE_FACT_LABEL_MAP[x["label"]], "source": "climate"})
    dataset = dataset.filter(lambda x: x["label"] is not None)

    # 4. If `balanced_label` is True, balance the dataset by randomly sampling from the True and False labels
    if balanced_label:
        true_dataset = dataset.filter(lambda x: x["label"] == True)
        false_dataset = dataset.filter(lambda x: x["label"] == False)
        min_count = min(len(true_dataset), len(false_dataset))
        true_dataset = true_dataset.shuffle(seed=42).select(range(min_count))
        false_dataset = false_dataset.shuffle(seed=42).select(range(min_count))
        dataset = concatenate_datasets([true_dataset, false_dataset]).shuffle(seed=42)

    if return_original:
        return dataset

    # 5. Split the dataset into training and test sets with the given ratio
    split_dataset = dataset.train_test_split(test_size=train_test_ratio, seed=42)
    return split_dataset["train"], split_dataset["test"]


def prepare_covid_fact_check_data(return_original: bool = False, balanced_label: bool = True, original_split: str = "train", train_test_ratio: float = 0.2):
    """
    This functon does the following:
    1. Load the dataset from HuggingFace
    2. Retain only the columns `claim_col` and `label_col`, rename the `label_col` to standardized "label" column
    3. Map the original label to the standardized True/False label -- delete rows with None label after mapping
    4. If `balanced_label` is True, balance the dataset by randomly sampling from the True and False labels
    5. Split the dataset into training and test sets with the given ratio

    Returns:
        - train_dataset: Dataset
        - test_dataset: Dataset
    """
    # 1. Load the dataset from HuggingFace
    raw_dataset = load_dataset(COVID_FACT_CHECK_DATASET, split=original_split)

    # 2. Retain only the columns `claim_col` and `label_col`, rename the `label_col` to standardized "label" column
    dataset = raw_dataset.select_columns([COVID_FACT_CLAIM_COL, COVID_FACT_LABEL_COL])
    dataset = dataset.rename_column(COVID_FACT_LABEL_COL, "label")

    # 3. Map the original label to the standardized True/False label -- delete rows with None label after mapping
    dataset = dataset.map(lambda x: {"label": COVID_FACT_LABEL_MAP[x["label"]], "source": "covid"})
    dataset = dataset.filter(lambda x: x["label"] is not None)

    # 4. If `balanced_label` is True, balance the dataset by randomly sampling from the True and False labels
    if balanced_label:
        true_dataset = dataset.filter(lambda x: x["label"] == True)
        false_dataset = dataset.filter(lambda x: x["label"] == False)
        min_count = min(len(true_dataset), len(false_dataset))
        true_dataset = true_dataset.shuffle(seed=42).select(range(min_count))
        false_dataset = false_dataset.shuffle(seed=42).select(range(min_count))
        dataset = concatenate_datasets([true_dataset, false_dataset]).shuffle(seed=42)

    if return_original:
        return dataset

    # 5. Split the dataset into training and test sets with the given ratio
    split_dataset = dataset.train_test_split(test_size=train_test_ratio, seed=42)
    return split_dataset["train"], split_dataset["test"]


def prepare_dispatch_search_data(train_test_ratio: float = 0.2):
    train_climate, test_climate = prepare_climate_fact_check_data(train_test_ratio=train_test_ratio)
    train_covid, test_covid = prepare_covid_fact_check_data(train_test_ratio=train_test_ratio)

    train_dataset = concatenate_datasets([train_climate, train_covid])
    test_dataset = concatenate_datasets([test_climate, test_covid])

    # shuffle the dataset a bit
    train_dataset = train_dataset.shuffle(seed=42)
    test_dataset = test_dataset.shuffle(seed=42)

    train_dataset = DatasetRegistry.register_dataset("dispatch_search", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("dispatch_search", test_dataset, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_dispatch_search_data()
    print(train_dataset)
    print(test_dataset)
