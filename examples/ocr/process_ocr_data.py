from datasets import load_dataset

from rllm.data.dataset import DatasetRegistry


def prepare_geo3k_data():
    # Load dataset
    dataset = load_dataset("linxy/LaTeX_OCR")["train"]
    dataset = dataset.train_test_split(test_size=500, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    def process_fn(example, idx):
        prompt = "<image>Convert the image to LaTeX code."
        answer = example.pop("text")
        image = example.pop("image")

        data = {
            "data_source": "latex_ocr",
            "image": image,
            "question": prompt,
            "ground_truth": answer,
        }
        return data

    # Preprocess datasets
    train_dataset = train_dataset.map(function=process_fn, with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=process_fn, with_indices=True, num_proc=8)

    # Register datasets
    train_dataset = DatasetRegistry.register_dataset("latex_ocr", train_dataset, "train")
    test_dataset = DatasetRegistry.register_dataset("latex_ocr", test_dataset, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_geo3k_data()
    print(train_dataset.get_data_path())
    print(test_dataset.get_data_path())
