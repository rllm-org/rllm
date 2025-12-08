numpy_issues = [
    ### issue 1
    """Describe the issue:
When multiplying large matrices in a multi-threaded case (using ThreadPoolExecutor for example) the program crashes without any error. The crash does not occur if the matrix is small enough, only a single thread is used, on linux, or if OPENBLAS_NUM_THREADS is set to 1, so perhaps this is an openblas issue.

Reproduce the code example:
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def crash_test(size_a):
    a = np.random.random((size_a,4))
    b = np.random.random((4,4))
    c = np.matmul(a,b)
    return c.mean()
    
num_threads = 4
counts = 128
futures = []
total = 0
size_a = 1024*1024

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    for i in range(counts):
        futures.append(executor.submit(crash_test, size_a))
    for i in range(counts):    
        total += futures[i].result()
print(total)
""",
    ### issue 2
    """
Describe the issue:
When used with large values and on large arrays, the values towards the end of the array can have very large errors in phase. This is because the implementation calculates corrections to the differences between elements, then applies np.cumsum to it to determine the corrections to the elements. Since every term has rounding error, the tail of the cumsum has a large error.

This could be improved by calculating corrections in units of the period, which would be integral and hence only suffer from overflow when large enough not to fit into the floating-point type (2^53 for float64).

Reproduce the code example:

```
import numpy as np

tau = 2 * np.pi

def phase_error(x, y):
    return (x - y + np.pi) % tau - np.pi

x = np.random.uniform(-1e9, 1e9, size=64 * 1024 * 1024)
y = np.unwrap(x)
print("Max phase error for np.unwrap: ", np.max(np.abs(phase_error(x, y))))
# This gives more accurate results
z = np.unwrap(x / tau, period=1.0) * tau
print("Max phase error for adjusted np.unwrap: ", np.max(np.abs(phase_error(x, z))))
```

Log:
Max phase error for np.unwrap:  0.9471197530276747
Max phase error for adjusted np.unwrap:  3.597418789524909e-07
""",
    ### issue 3
    """
```
import numpy as np

f = np.vectorize(lambda c: ord(c) if c else -1, otypes=[int])

a = np.ma.masked_all(1, str)
x = f(a)  # ok
x = a.fill_value
x = f(a)  # raises TypeError: Cannot convert fill_value N/A to dtype int64

a = np.ma.masked_array([""], True)
x = f(a)  # ok
x = a.fill_value
x = f(a)  # raises TypeError: Cannot convert fill_value N/A to dtype int64
a = np.ma.masked_array([""], True, fill_value="?")
x = f(a)  # raises TypeError: Cannot convert fill_value ? to dtype int64

a = np.ma.masked_array("", True)
x = f(a)  # ok
x = a.fill_value
x = f(a)  # ok
a = np.ma.masked_array("", True, fill_value="?")
x = f(a)  # ok
```

Error message:
```
Traceback (most recent call last):
  File ".../lib/python3.12/site-packages/numpy/ma/core.py", line 489, in _check_fill_value
    fill_value = np.asarray(fill_value, dtype=ndtype)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: invalid literal for int() with base 10: 'N/A'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File ".../lib/python3.12/site-packages/numpy/lib/_function_base_impl.py", line 2397, in __call__
    return self._call_as_normal(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../lib/python3.12/site-packages/numpy/lib/_function_base_impl.py", line 2390, in _call_as_normal
    return self._vectorize_call(func=func, args=vargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../lib/python3.12/site-packages/numpy/lib/_function_base_impl.py", line 2483, in _vectorize_call
    res = asanyarray(outputs, dtype=otypes[0])
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../lib/python3.12/site-packages/numpy/ma/core.py", line 3092, in __array_finalize__
    self._fill_value = _check_fill_value(self._fill_value, self.dtype)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File ".../lib/python3.12/site-packages/numpy/ma/core.py", line 495, in _check_fill_value
    raise TypeError(err_msg % (fill_value, ndtype)) from e
TypeError: Cannot convert fill_value N/A to dtype int64
```

I stumbled upon this while testing np.vectorize in conjunction with masked arrays. In its current status, fill_value casting cannot be relied upon (e.g. the default int fill value can be cast to float/str works but not vice versa). To me it would make sense not to try casting the input array fill_value to the dtypes in otypes but rather using the np.ma.default_fill_value. Moreover, it would be very practical if the vectorized function could skip masked values all together, setting the result(s) to np.ma.masked, but this is a separate issue.

""",
    ### issue 4
    """

Converting a polynomial to natural window yields in unexpected behaviour: the degree of the polynomial might change.
E.g. when fitting data with a polynomial of degree 1, where data ist just constant, convert() return a degree 0 polynomial.

When implementing a fit to data, you don't know in advance, how data looks like. So fitting data with a polynomial of degree 1 you expect to get a polynomial of degree 1 in all cases. If the data is just constant, the first order coefficient is then 0 but it's not ok to remove that coefficient from output.

Reproduce the code example:

```
import numpy as np
x = np.arange(0, 10, 1)
y = 0*x + 5.32
param = np.polynomial.Polynomial.fit(x, y, 1)
param.convert().coef
```
""",
    ### issue 5
    """

Reproducing code example:

```
import numpy as np
<< your code here >>
a = np.array([46, 57, 23, 39, 1, 10, 0, 120])
np.partition(a,2)
# result
array([  0,   1,  10,  39,  57,  23,  46, 120])
```

According to the documentation The k-th value of the element will be in its final sorted position.

But notice that the k in this code is 2, i. e. the 3rd value 23 should be in its final sorted position, but the final sorted array is

```
np.sort(a)
# result
array([  0,   1,  10,  23,  39,  46,  57, 120])
```

The original 3rd value is located in the 4-th position in the sorted array, but the method partition put it in the 6-th position.

""",
]
