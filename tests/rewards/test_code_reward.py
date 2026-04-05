import pytest

from rllm.rewards import RewardConfig, RewardType
from rllm.rewards.code_reward import RewardCodeFn


class TestCodeReward:
    """Test class for code reward functionality."""

    def test_reward_code_contests(self):
        """
        Test the reward function on the code contests dataset.
        """
        model_response = """
```python
import sys
from itertools import permutations
def main():
    N, M, R = map(int, sys.stdin.readline().split())
    r = list(map(int, sys.stdin.readline().split()))
    A, B, C = [], [], []
    for _ in range(M):
        a, b, c = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        C.append(c)
    INF = float('inf')
    dist = [[INF for _ in range(N+1)] for _ in range(N+1)]
    for i in range(1, N+1):
        dist[i][i] = 0
    for i in range(M):
        a, b, c = A[i], B[i], C[i]
        dist[a][b] = c
        dist[b][a] = c
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    min_dist = INF
    for perm in permutations(r):
        total = 0
        for i in range(R-1):
            total += dist[perm[i]][perm[i+1]]
        if total < min_dist:
            min_dist = total
    print(min_dist)
if __name__ == "__main__":
    main()
    ```
    """
        metadata = {
            "inputs": [
                # Test case 1: Simple path with 3 cities
                "4 3 3\n1 2 3\n1 2 3\n2 3 2\n3 4 4\n",
                # Test case 2: Complete graph with 5 cities
                "5 10 4\n1 2 3 4\n1 2 5\n1 3 5\n1 4 5\n1 5 5\n2 3 5\n2 4 5\n2 5 5\n3 4 5\n3 5 5\n4 5 5\n",
                # Test case 3: Larger graph with 7 cities
                "7 21 4\n1 3 5 7\n1 2 4\n1 3 8\n1 4 1\n1 5 7\n1 6 3\n1 7 9\n2 3 5\n2 4 2\n2 5 6\n2 6 8\n2 7 4\n3 4 7\n3 5 9\n3 6 1\n3 7 6\n4 5 3\n4 6 5\n4 7 8\n5 6 2\n5 7 4\n6 7 7\n",
            ],
            "outputs": ["5\n", "15\n", "11\n"],
        }
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "code_contests", "ground_truth": metadata}
        output = reward(task_info, model_response)
        assert output.is_correct

    def test_reward_codeforces(self):
        """
        Test the reward function on the codeforces dataset.
        """
        model_response = """
```python
import sys
from itertools import permutations
def main():
    n,m=map(int, input().split())
    a=sum(list(map(int, input().split())))
    if a+(n-1)*10<=m:
        print((m-a)//5)
    else:
        print(-1)
if __name__ == "__main__":
    main()
    ```
    """
        metadata = [
            # Basic case
            {"input": "3 30\n2 2 1", "output": "5"},
            # Impossible case
            {"input": "3 20\n2 1 1", "output": "-1"},
            # Exact fit case
            {"input": "4 45\n5 5 5 5", "output": "-1"},
            # Large numbers
            {"input": "5 100\n10 10 10 10 10", "output": "10"},
            # Single task
            {"input": "1 20\n5", "output": "3"},
            # Maximum possible breaks
            {"input": "2 100\n1 1", "output": "19"},
            # Edge case - just barely possible
            {"input": "3 35\n5 5 5", "output": "4"},
        ]
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "codeforces", "ground_truth": metadata}
        output = reward(task_info, model_response)
        assert output.is_correct

    @pytest.mark.skip(reason="SWEBench functionality may not be fully implemented")
    def test_reward_swebench(self):
        """
        Test the reward function on the swebench dataset.
        """
        reward = RewardCodeFn(RewardConfig())
        tests = {
            "instance_id": "astropy__astropy-12907",
        }
        metadata = tests
        model_response = """\
diff --git a/astropy/modeling/separable.py b/astropy/modeling/separable.py
--- a/astropy/modeling/separable.py
+++ b/astropy/modeling/separable.py
@@ -242,7 +242,7 @@ def _cstack(left, right):
        cright = _coord_matrix(right, 'right', noutp)
    else:
        cright = np.zeros((noutp, right.shape[1]))
-        cright[-right.shape[0]:, -right.shape[1]:] = 1
+        cright[-right.shape[0]:, -right.shape[1]:] = right

    return np.hstack([cleft, cright])
    """
        task_info = {
            "problem": """
Modeling's `separability_matrix` does not compute separability correctly for nested CompoundModels
Consider the following model:

```python
from astropy.modeling import models as m
from astropy.modeling.separable import separability_matrix

cm = m.Linear1D(10) & m.Linear1D(5)
```

It's separability matrix as you might expect is a diagonal:

```python
>>> separability_matrix(cm)
array([[ True, False],
    [False,  True]])
```

If I make the model more complex:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & m.Linear1D(10) & m.Linear1D(5))
array([[ True,  True, False, False],
    [ True,  True, False, False],
    [False, False,  True, False],
    [False, False, False,  True]])
```

The output matrix is again, as expected, the outputs and inputs to the linear models are separable and independent of each other.

If however, I nest these compound models:
```python
>>> separability_matrix(m.Pix2Sky_TAN() & cm)
array([[ True,  True, False, False],
    [ True,  True, False, False],
    [False, False,  True,  True],
    [False, False,  True,  True]])
```
Suddenly the inputs and outputs are no longer separable?

This feels like a bug to me, but I might be missing something?
""",
            "problem_type": RewardType.CODE,
            "data_source": "swebench",
            "ground_truth": metadata,
        }
        output = reward(task_info, model_response)
        assert output.is_correct

    def test_reward_taco(self):
        """
        Test the reward function on the taco dataset.
        """
        model_response = """
```python
import sys
from itertools import permutations
def main():
    N, M, R = map(int, sys.stdin.readline().split())
    r = list(map(int, sys.stdin.readline().split()))
    A, B, C = [], [], []
    for _ in range(M):
        a, b, c = map(int, sys.stdin.readline().split())
        A.append(a)
        B.append(b)
        C.append(c)
    INF = float('inf')
    dist = [[INF for _ in range(N+1)] for _ in range(N+1)]
    for i in range(1, N+1):
        dist[i][i] = 0
    for i in range(M):
        a, b, c = A[i], B[i], C[i]
        dist[a][b] = c
        dist[b][a] = c
    for k in range(1, N+1):
        for i in range(1, N+1):
            for j in range(1, N+1):
                if dist[i][k] != INF and dist[k][j] != INF:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    min_dist = INF
    for perm in permutations(r):
        total = 0
        for i in range(R-1):
            total += dist[perm[i]][perm[i+1]]
        if total < min_dist:
            min_dist = total
    print(min_dist)
if __name__ == "__main__":
    main()
    ```
    """
        metadata = {
            "inputs": [
                # Test case 1: Simple path with 3 cities
                "4 3 3\n1 2 3\n1 2 3\n2 3 2\n3 4 4\n",
                # # Test case 2: Complete graph with 5 cities
                "5 10 4\n1 2 3 4\n1 2 5\n1 3 5\n1 4 5\n1 5 5\n2 3 5\n2 4 5\n2 5 5\n3 4 5\n3 5 5\n4 5 5\n",
                # Test case 3: Larger graph with 7 cities
                "7 21 4\n1 3 5 7\n1 2 4\n1 3 8\n1 4 1\n1 5 7\n1 6 3\n1 7 9\n2 3 5\n2 4 2\n2 5 6\n2 6 8\n2 7 4\n3 4 7\n3 5 9\n3 6 1\n3 7 6\n4 5 3\n4 6 5\n4 7 8\n5 6 2\n5 7 4\n6 7 7\n",
            ],
            "outputs": ["5\n", "15\n", "11\n"],
        }
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "taco", "ground_truth": metadata}
        output = reward(task_info, model_response)
        assert output.is_correct

    def test_reward_taco_call_based(self):
        """
        Test the reward function on the taco dataset with function calls.
        """
        model_response = """
```python
def is_anagram(test, original):
    return sorted(original.lower()) == sorted(test.lower())
```
"""
        metadata = {
            "fn_name": "is_anagram",
            "inputs": [
                ["foefet", "toffee"],
                ["Buckethead", "DeathCubeK"],
                ["Twoo", "WooT"],
                ["dumble", "bumble"],
                ["ound", "round"],
                ["apple", "pale"],
            ],
            "outputs": [[True], [True], [True], [False], [False], [False]],
        }
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "taco", "ground_truth": metadata}
        output = reward(task_info, model_response)
        assert output.is_correct

    def test_reward_taco_call_based2(self):
        """
        Test the reward function on the taco dataset with another function call example.
        """
        model_response = """
```python
def total(arr):
    while len(arr) > 1:
        arr = [x + y for (x, y) in zip(arr, arr[1:])]
    return arr[0]
```
"""
        metadata = {
            "fn_name": "total",
            "inputs": [
                [[1, 2, 3, 4, 5]],
                [[1, 2, 3, 4]],
                [[1, 2, 3]],
                [[4, 4, 52, 23, 32, 1, -1]],
                [[4, 4, 5, -1]],
                [[-1, -1, -1]],
                [[-1, -1, -10, 42, 92, 1, 23, 6, -3]],
                [[-1, 1, -1, 1]],
                [[42]],
            ],
            "outputs": [[48], [20], [8], [1753], [30], [-4], [9248], [0], [42]],
        }
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "taco", "ground_truth": metadata}
        output = reward(task_info, model_response)
        assert output.is_correct

    def test_reward_livecodebench(self):
        """
        Test the reward function on the livecodebench dataset.
        """
        model_response = """
Yes of course!
```python
import json

def main():
    n = input()
    phone_numbers = input().strip().split()
    seen = set()
    duplicates = set()
    for number in phone_numbers:
        if number in seen:
            duplicates.add(number)
        else:
            seen.add(number)
    
    print(len(duplicates)+1)
if __name__ == "__main__":
    main()
```
"""
        public_test_case = [{"input": "3\n12345 530391 12345\n", "output": "2\n", "testtype": "stdin"}]
        metadata = public_test_case
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "livecodebench", "ground_truth": metadata}
        output = reward(task_info, model_response)
        assert output.is_correct

        # Test functional case
        model_response = """
Yes of course!
```python
class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        count = 0
        for hour in hours:
            if hour >= target:
                count += 1
        return count
```
"""
        public_test_case = [{"input": "[5, 3, 10, 8, 2]\n5", "output": "3", "testtype": "functional", "metadata": {"func_name": "numberOfEmployeesWhoMetTarget"}}]
        metadata = public_test_case
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "livecodebench", "ground_truth": metadata}
        output = reward(task_info, model_response)
        assert output.is_correct

    @pytest.mark.skip(reason="LeetCode tests may have execution environment dependencies")
    def test_reward_leetcode(self):
        """
        Test the reward function on the leetcode dataset.
        """
        model_response = """
Here is my response
```python
class Solution:\n    def minOperations(self, nums: List[int], k: int) -> int:\n        is_added = [False] * k\n        count = 0\n        n = len(nums)\n        for i in range(n - 1, -1, -1):\n            if nums[i] > k or is_added[nums[i] - 1]:\n                continue\n            is_added[nums[i] - 1] = True\n            count += 1\n            if count == k:\n                return n - i\n
```
"""
        tests = {
            "functional": "def check(candidate):\n    assert candidate(nums = [3,1,5,4,2], k = 2) == 4\n    assert candidate(nums = [3,1,5,4,2], k = 5) == 5\n    assert candidate(nums = [3,2,5,3,1], k = 3) == 4\n\n\ncheck(Solution().minOperations)"
        }
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "leetcode", "ground_truth": tests}
        output = reward(task_info, model_response)
        assert output.is_correct

    def test_reward_leetcode_format_error(self):
        """
        Test the reward function on leetcode with format error (no code block).
        """
        # Bad case - no markdown code block
        model_response = """
Here is my bad response, it is not in markdown oops
class Solution:\n    def minOperations(self, nums: List[int], k: int) -> int:\n        is_added = [False] * k\n        count = 0\n        n = len(nums)\n        for i in range(n - 1, -1, -1):\n            if nums[i] > k or is_added[nums[i] - 1]:\n                continue\n            is_added[nums[i] - 1] = True\n            count += 1\n            if count == k:\n                return n - i\n
"""
        tests = {
            "functional": "def check(candidate):\n    assert candidate(nums = [3,1,5,4,2], k = 2) == 4\n    assert candidate(nums = [3,1,5,4,2], k = 5) == 5\n    assert candidate(nums = [3,2,5,3,1], k = 3) == 4\n\n\ncheck(Solution().minOperations)"
        }
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "leetcode", "ground_truth": tests}
        output = reward(task_info, model_response)
        assert not output.is_correct

    def test_reward_kodcode(self):
        """
        Test the reward function on the kodcode dataset.
        """
        model_response = """
```python
def longest_subsequence(nums):
    '''
    Returns the length of the longest subsequence such that the
    difference between adjacent elements is either 1 or -1.

    Args:
    nums: A list of integers

    Returns:
    An integer representing the length of the longest subsequence
    '''
    if not nums:
        return 0

    max_length = 1
    curr_length = 1

    for i in range(1, len(nums)):
        if abs(nums[i] - nums[i - 1]) == 1:
            curr_length += 1
            max_length = max(max_length, curr_length)
        else:
            curr_length = 1

    return max_length
```
"""
        tests = """
from solution import longest_subsequence

def test_longest_subsequence_all_one_diff():
    assert longest_subsequence([1, 2, 3, 4, 5]) == 5

def test_longest_subsequence_alternating_diff():
    assert longest_subsequence([1, 2, 1, 2, 1]) == 5

def test_longest_subsequence_with_breaks():
    assert longest_subsequence([10, 11, 7, 8, 9, 10]) == 4

def test_longest_subsequence_single_element():
    assert longest_subsequence([1]) == 1

def test_longest_subsequence_no_valid_subsequence():
    assert longest_subsequence([15, 30, 45, 60]) == 1

def test_longest_subsequence_empty_list():
    assert longest_subsequence([]) == 0
"""
        reward = RewardCodeFn(RewardConfig())
        task_info = {"problem": "", "problem_type": RewardType.CODE, "data_source": "kodcode", "ground_truth": tests}
        output = reward(task_info, model_response)
        assert output.is_correct
