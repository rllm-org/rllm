from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn


def test_reward_code_contests():
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
            "5 10 4\n1 2 3 4\n1 2 5\n1 3 5\n1 4 5\n1 5 5\n2 3 5\n2 4 5\n2 5 5\n3 4 5\n3 5 5\n4 5 5\n"
            # Test case 3: Larger graph with 7 cities
            "7 21 4\n1 3 5 7\n1 2 4\n1 3 8\n1 4 1\n1 5 7\n1 6 3\n1 7 9\n2 3 5\n2 4 2\n2 5 6\n2 6 8\n2 7 4\n3 4 7\n3 5 9\n3 6 1\n3 7 6\n4 5 3\n4 6 5\n4 7 8\n5 6 2\n5 7 4\n6 7 7\n"
        ],
        "outputs": [
            "5\n",  
            "15\n",
            "11\n"
        ]
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="code_contests")
    output = reward(input)
    assert output.is_correct == True
    return output


def test_reward_codeforces():
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
            {"input": "3 35\n5 5 5", "output": "4"}
        ]
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="codeforces")
    output = reward(input)
    assert output.is_correct == True


def test_reward_swebench():
    """
    Test the reward function on the swebench dataset.
    """
    reward = RewardCodeFn(RewardConfig)
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
    input = RewardInput(
        problem="""
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
        problem_type=RewardType.CODE,
        model_response=model_response,
        metadata=metadata,
        data_source="swebench",
    )
    output = reward(input)
    assert output.is_correct == True

def test_reward_taco():
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
            # Test case 2: Complete graph with 5 cities
            "5 10 4\n1 2 3 4\n1 2 5\n1 3 5\n1 4 5\n1 5 5\n2 3 5\n2 4 5\n2 5 5\n3 4 5\n3 5 5\n4 5 5\n"
            # Test case 3: Larger graph with 7 cities
            "7 21 4\n1 3 5 7\n1 2 4\n1 3 8\n1 4 1\n1 5 7\n1 6 3\n1 7 9\n2 3 5\n2 4 2\n2 5 6\n2 6 8\n2 7 4\n3 4 7\n3 5 9\n3 6 1\n3 7 6\n4 5 3\n4 6 5\n4 7 8\n5 6 2\n5 7 4\n6 7 7\n"
        ],
        "outputs": [
            "5\n",  
            "15\n",
            "11\n"
        ]
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="taco")
    output = reward(input)
    assert output.is_correct == True


def test_reward_livecodebench():
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
    public_test_case = [
        {
            'input': '3\n12345 530391 12345\n',
            'output': '2\n',
            'testtype': 'stdin'
        }
    ]
    metadata = public_test_case
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="livecodebench")
    output = reward(input)
    assert output.is_correct == True

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
    public_test_case = [
        {
            "input": "[5, 3, 10, 8, 2]\n5",
            "output": "3",
            "testtype": "functional",
            "metadata": {'func_name': 'numberOfEmployeesWhoMetTarget'}
        }
    ]
    metadata = public_test_case
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="livecodebench")
    output = reward(input)
    assert output.is_correct == True


def test_reward_leetcode():
    model_response = """
Here is my response
```python
class Solution:\n    def minOperations(self, nums: List[int], k: int) -> int:\n        is_added = [False] * k\n        count = 0\n        n = len(nums)\n        for i in range(n - 1, -1, -1):\n            if nums[i] > k or is_added[nums[i] - 1]:\n                continue\n            is_added[nums[i] - 1] = True\n            count += 1\n            if count == k:\n                return n - i\n
```
"""
    tests = {
        "functional": "def check(candidate):\n    assert candidate(nums = [3,1,5,4,2], k = 2) == 4\n    assert candidate(nums = [3,1,5,4,2], k = 5) == 5\n    assert candidate(nums = [3,2,5,3,1], k = 3) == 4\n\n\ncheck(Solution().minOperations)"
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="leetcode")
    output = reward(input)
    assert output.is_correct == True
    
    model_response = """
To solve this problem, we need to identify the smallest subarray that, when sorted, will result in the entire array being sorted. Here's a step-by-step approach to achieve this:\n\n1. **Identify the Sorted and Unsorted Parts**: Compare the original array with its sorted version to identify the first and last positions where the elements differ. These positions mark the boundaries of the unsorted subarray.\n\n2. **Edge Case**: If the array is already sorted, the length of the subarray to be sorted is 0.\n\nHere's the complete implementation:\n\n```python\nfrom typing import List\n\n
class Solution:\n    def findUnsortedSubarray(self, nums: List[int]) -> int:\n        # Create a sorted version of the array\n        sorted_nums = sorted(nums)\n        \n        # Initialize the start and end indices\n        start, end = -1, -2\n        \n        # Find the first and last positions where the elements differ\n        for i in range(len(nums)):\n            if nums[i] != sorted_nums[i]:\n                if start == -1:\n                    start = i\n                end = i\n        \n        # The length of the subarray to be sorted\n        return end - start + 1\n```\n\n### Explanation:\n\n- **Sorting**: We first sort the array and store it in `sorted_nums`.\n- **Finding Boundaries**: We iterate through the original array `nums` and compare it with `sorted_nums`. The first mismatch gives us the `start` of the unsorted subarray, and the last mismatch gives us the `end`.\n- **Calculating Length**: The length of the subarray to be sorted is `end - start + 1`. If the array is already sorted, `start` will remain `-1` and `end` will remain `-2`, resulting in a length of 0.\n\nThis approach has a time complexity of O(n log n) due to the sorting step, and a space complexity of O(n) for storing the sorted array.", "split": "train"}}

"""
    tests = {
        "functional": "def check(candidate):\n    assert candidate(nums = [2,6,4,8,10,9,15]) == 5\n    assert candidate(nums = [1,2,3,4]) == 0\n    assert candidate(nums = [1]) == 0\n\n\ncheck(Solution().findUnsortedSubarray)"
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="leetcode")
    output = reward(input)
    assert output.is_correct == True


    model_response = """
To solve this problem with a time complexity of \\(O(\\log(m+n))\\), we can use a binary search approach. The idea is to partition the two arrays such that the left half contains the smaller elements and the right half contains the larger elements. The median will then be the maximum of the left half if the total number of elements is odd, or the average of the maximum of the left half and the minimum of the right half if the total number of elements is even.\n\nHere's the complete code:\n\n```python\nfrom typing import List\n\n
class Solution:\n    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:\n        # Ensure nums1 is the smaller array\n        if len(nums1) > len(nums2):\n            nums1, nums2 = nums2, nums1\n        \n        x, y = len(nums1), len(nums2)\n        low, high = 0, x\n        \n        while low <= high:\n            partitionX = (low + high) // 2\n            partitionY = (x + y + 1) // 2 - partitionX\n            \n            # If partitionX is 0 it means nothing is there on left side. Use -inf for maxLeftX\n            # If partitionX is length of input then there is nothing on right side. Use +inf for minRightX\n            maxLeftX = float('-inf') if partitionX == 0 else nums1[partitionX - 1]\n            minRightX = float('inf') if partitionX == x else nums1[partitionX]\n            \n            maxLeftY = float('-inf') if partitionY == 0 else nums2[partitionY - 1]\n            minRightY = float('inf') if partitionY == y else nums2[partitionY]\n            \n            if maxLeftX <= minRightY and maxLeftY <= minRightX:\n                # We have partitioned array at correct place\n                if (x + y) % 2 == 0:\n                    return (max(maxLeftX, maxLeftY) + min(minRightX, minRightY)) / 2\n                else:\n                    return max(maxLeftX, maxLeftY)\n            elif maxLeftX > minRightY:\n                # We are too far on right side for partitionX. Go on left side.\n                high = partitionX - 1\n            else:\n                # We are too far on left side for partitionX. Go on right side.\n                low = partitionX + 1\n```\n\n### Explanation:\n1. **Ensure `nums1` is the smaller array**: This helps in minimizing the binary search range.\n2. **Binary Search**: We perform a binary search on the smaller array (`nums1`).\n3. **Partitioning**: For each partition of `nums1`, we calculate the corresponding partition of `nums2` such that the total number of elements on the left side is equal to or one more than the right side.\n4. **Check Conditions**: We check if the maximum element on the left side of `nums1` is less than or equal to the minimum element on the right side of `nums2` and vice versa.\n5. **Calculate Median**: If the partitions are correct, we calculate the median based on whether the total number of elements is odd or even.\n\nThis approach ensures that we find the median in \\(O(\\log(\\min(m, n)))\\) time complexity.", "split": "train"}}
"""
    tests = {
        "functional": "def check(candidate):\n    assert candidate(nums1 = [1,3], nums2 = [2]) == 2.00000\n    assert candidate(nums1 = [1,2], nums2 = [3,4]) == 2.50000\n\n\ncheck(Solution().findMedianSortedArrays)"
    }

    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="leetcode")
    output = reward(input)
    assert output.is_correct == True

    # Bad case.
    model_response = """
Here is my bad response, it is not in markdown oops
class Solution:\n    def minOperations(self, nums: List[int], k: int) -> int:\n        is_added = [False] * k\n        count = 0\n        n = len(nums)\n        for i in range(n - 1, -1, -1):\n            if nums[i] > k or is_added[nums[i] - 1]:\n                continue\n            is_added[nums[i] - 1] = True\n            count += 1\n            if count == k:\n                return n - i\n
"""
    tests = {
        "functional": "def check(candidate):\n    assert candidate(nums = [3,1,5,4,2], k = 2) == 4\n    assert candidate(nums = [3,1,5,4,2], k = 5) == 5\n    assert candidate(nums = [3,2,5,3,1], k = 3) == 4\n\n\ncheck(Solution().minOperations)"
    }
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="leetcode")
    output = reward(input)
    assert output.is_correct == False

def test_reward_kodcode():
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
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="kodcode")
    output = reward(input)
    assert output.is_correct == True


if __name__ == "__main__":
    test_reward_code_contests()
    test_reward_leetcode()
    test_reward_taco()
    test_reward_codeforces()
    test_reward_livecodebench()
    test_reward_kodcode()
    print("All tests passed")
