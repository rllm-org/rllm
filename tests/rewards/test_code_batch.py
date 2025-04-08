import json 
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn
from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TrainDataset, TestDataset


def _process_case_leetcode(i, entry):
    """Process a single test case from the Leetcode dataset.
    
    Args:
        i: Index of the test case
        entry: Test case data containing solutions and tests
    """
    # There is only one solution per problem in the Leetcode dataset
    model_response = f"""
```python
{entry["solutions"]}
```
"""
    tests = entry["tests"]
    reward = RewardCodeFn(RewardConfig)
    input_obj = RewardInput(
        problem="",
        problem_type=RewardType.CODE,
        model_response=model_response,
        metadata=tests,
        data_source="leetcode"
    )
    output = reward(input_obj)
    failed = None
    if not output.is_correct:
        failed = entry
    return i, output, failed


def _process_case_taco(i, data):
    """
    Process a single test case from the TACO dataset.
    
    Args:
        i: Index of the test case
        data: Test case data containing solutions and tests
        
    Returns:
        tuple: (index, reward output, failed case data if applicable)
    """
    for solution in data["solutions"]:
        if not solution.startswith("```python") and not solution.endswith("```"):
            model_response = f"""```python\n{solution}\n```"""
        else:
            model_response = solution
        tests = data["tests"]
        reward = RewardCodeFn(RewardConfig)
        input_obj = RewardInput(
            problem="", 
            problem_type=RewardType.CODE, 
            model_response=model_response, 
            metadata=tests, 
            data_source="taco"
        )
        output = reward(input_obj)
        if output.is_correct:
            return i, output, None
    return i, output, data

def _process_case_primeintellect(i, data):
    """
    Process a single test case from the VERIFY dataset.
    """
    model_response = data['solutions']
    for solution in data["solutions"]:
        if not solution.startswith("```python") and not solution.endswith("```"):
            model_response = f"""```python\n{solution}\n```"""
        else:
            model_response = solution
        tests = data["tests"]
        reward = RewardCodeFn(RewardConfig)
        input_obj = RewardInput(
            problem="", 
            problem_type=RewardType.CODE, 
            model_response=model_response, 
            metadata=tests, 
            data_source="primeintellect"
        )
        output = reward(input_obj)
        if output.is_correct:
            return i, output, None
    return i, output, data


def _process_case_kodcode(i, data):
    """
    Process a single test case from the KODCODE dataset.
    
    Args:
        i: Index of the test case
        data: Test case data containing solutions and tests
        
    Returns:
        tuple: (index, reward output, failed case data if applicable)
    """
    solution = data["solutions"]
    if not solution.startswith("```python") and not solution.endswith("```"):
        model_response = f"""```python\n{solution}\n```"""
    else:
        model_response = solution
    tests = data["tests"]
    reward = RewardCodeFn(RewardConfig)
    input_obj = RewardInput(
        problem="", 
        problem_type=RewardType.CODE, 
        model_response=model_response, 
        metadata=tests, 
        data_source="kodcode"
    )
    output = reward(input_obj)
    if output.is_correct:
        return i, output, None
    return i, output, data


def _process_case_humanevalplus(i, data):
    """
    Process a single test case from the HUMANEVALPLUS dataset.
    
    Args:
        i: Index of the test case
        data: Test case data containing solutions and tests
        
    Returns:
        tuple: (index, reward output, failed case data if applicable)
    """
    solution = data["solutions"]
    if not solution.startswith("```python") and not solution.endswith("```"):
        model_response = f"""```python\n{solution}\n```"""
    else:
        model_response = solution
    tests = data["tests"]
    reward = RewardCodeFn(RewardConfig)
    input_obj = RewardInput(
        problem="", 
        problem_type=RewardType.CODE, 
        model_response=model_response, 
        metadata=tests, 
        data_source="humanevalplus"
    )
    output = reward(input_obj)
    if output.is_correct:
        return i, output, None
    return i, output, data


def test_batched_reward(dataset: str):
    """
    Test the reward function on the TACO dataset.
    
    Processes all test cases in parallel and logs any failures.
    
    Returns:
        The reward output for the last test case.
    """
    if dataset == "taco":
        data = load_dataset(TrainDataset.Code.TACO)
        test_fn = _process_case_taco
    elif dataset == "leetcode":
        data = load_dataset(TrainDataset.Code.LEETCODE)
        test_fn = _process_case_leetcode
    elif dataset == "primeintellect":
        data = load_dataset(TrainDataset.Code.PRIMEINTELLECT)
        test_fn = _process_case_primeintellect
    elif dataset == "kodcode":
        data = load_dataset(TrainDataset.Code.KODCODE)
        test_fn = _process_case_kodcode
    elif dataset == "humanevalplus":
        data = load_dataset(TestDataset.Code.HUMANEVALPLUS)
        test_fn = _process_case_humanevalplus
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    
    results = {}
    failed_cases = []
    failure_log_path = os.path.join(os.path.dirname(__file__), f"./{dataset}_test_err.json")
    counter = 0
    debug = True
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(test_fn, i, data[i]) for i in range(len(data))]
        for future in as_completed(futures):
            try:
                idx, output, failed = future.result()
                results[idx] = output
                if failed is not None:
                    failed_cases.append(failed)
            except Exception as e:
                print(f"Error processing item: {e}")
            counter += 1
            if debug:
                print(counter)

    # Save the failed cases to a JSON file if any 
    if failed_cases:
        print('Failed cases: ', len(failed_cases))
        with open(failure_log_path, "w") as f:
            json.dump(failed_cases, f, indent=4)

    # Return the output corresponding to the last processed index
    return results[len(data) - 1]


if __name__ == "__main__": 
    # with open("taco_test_err.json", "r") as f:
    #     failed_cases = json.load(f)
    # print(len(failed_cases))
    # test_batched_reward(dataset="taco")
    test_batched_reward(dataset="humanevalplus")
    # test_batched_reward(dataset="leetcode")
    # test_batched_reward(dataset="kodcode")
    # test_batched_reward(dataset="leetcode")
