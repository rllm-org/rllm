import random

NUMPY_ISSUE = r"""Describe the issue:
When used with large values and on large arrays, the values towards the end of the array can have very large errors in phase. 

Reproduce the code example:

```
import numpy as np

tau = 2 * np.pi

def phase_error(x, y):
    return (x - y + np.pi) % tau - np.pi

x = np.random.uniform(-1e9, 1e9, size=64 * 1024 * 1024)
y = np.unwrap(x)
print("Max phase error for np.unwrap: ", np.max(np.abs(phase_error(x, y))))
```

Log:
Max phase error for np.unwrap:  0.9471197530276747
"""

PANDAS_ISSUE = r"""
Code:

```
import warnings
import pandas as pd

warnings.filterwarnings("once", category=UserWarning)

warnings.warn("This is a warning", UserWarning)
warnings.warn("This is a warning", UserWarning)
warnings.warn("This is a second warning", UserWarning)
warnings.warn("This is a second warning", UserWarning)
pd.DataFrame()
warnings.warn("This is a warning", UserWarning)
warnings.warn("This is a warning", UserWarning)
warnings.warn("This is a second warning", UserWarning)
warnings.warn("This is a second warning", UserWarning)

```

Issue Description
Using filterwarnings with action 'once' should only print a warning of a specific category and text once. But calling pd.DataFrame() or other pandas functions (like pd.read_csv) makes both warnings shown twice. Deleting pd.DataFrame yields the expected behaviour.

Expected Behavior
Both warnings ("This is a warning" and "This is a second warning") should be shown only once each."""

SYMPY_ISSUE = r"""Title: Wrong result for an integral over complex exponential with a Diracdelta function

I ask Sympy for the complex integral

∫02πexp⁡(−imϕ)δ(ϕ−ϕ0)dϕ,

where m is an integer and δ is the Diracdelta distribution. For ϕ0=0, the above integral yields 0 with SymPy although it should be 1 (or 1/2 depending on the definition of the Delta function if the integral starts at the argument of the δ). For 0<ϕ0<2π, the SymPy result seems correct.

Interestingly, I obtain the (correct) result of 1/2 for ϕ0=2π but zero again for ϕ0=4π. Here is my code:


```
import sympy as sp
# The SymPy version is 1.13.2

phi = sp.symbols(r'\phi', real=True)
m = sp.symbols('m', integer=True)

# This yields 0; it should be 1/2 (phi0 = 0)
sp.integrate(sp.exp(-sp.I * m * phi) * sp.DiracDelta(phi), (phi, 0, 2 * sp.pi))

# This is correct (phi0 = pi/2)
sp.integrate(sp.exp(-sp.I * m * phi) * sp.DiracDelta(phi - sp.pi/2), (phi, 0, 2 * sp.pi))

# This is correct too (phi0 = 2pi)
sp.integrate(sp.exp(-sp.I * m * phi) * sp.DiracDelta(phi - 2 * sp.pi), (phi, 0, 2 * sp.pi))

# Wrong again (phi0 = 4pi)
sp.integrate(sp.exp(-sp.I * m * phi) * sp.DiracDelta(phi - 4 * sp.pi), (phi, 0, 2 * sp.pi))
```
"""

PILLOW_ISSUE = r"""

error is : AttributeError: 'function' object has no attribute 'copy'

```
frames = [f.copy for f in ImageSequence.Iterator(pfp)]

for i, frame in enumerate(frames):
	fr = frame.copy() #error here
	blyat.paste(fr (21,21))
	frames.append(blyat.copy())
	frames[i] = frame
frames[0].save("aa.gif", save_all=True, append_images=frames[1:], optimize=False, delay=0, loop=0, fps = 1/24)
```
"""

SCRAPY_ISSUE = r"""
Description
According to the documentation, the FEEDS dict accepts Path objects as keys:

[...] dictionary in which every key is a feed URI (or a pathlib.Path object) [...]

However, when using a Path object with Storage URI parameters, the FeedExporter runs into the following exception:

```
[scrapy.utils.signal] ERROR: Error caught on signal handler: <bound method FeedExporter.open_spider of <scrapy.extensions.feedexport.FeedExporter object at 0x00000240E9F21F00>>
Traceback (most recent call last):
  File "...\.venv\lib\site-packages\scrapy\utils\defer.py", line 348, in maybeDeferred_coro
    result = f(*args, **kw)
  File "...\.venv\lib\site-packages\pydispatch\robustapply.py", line 55, in robustApply
    return receiver(*arguments, **named)
  File "...\.venv\lib\site-packages\scrapy\extensions\feedexport.py", line 467, in open_spider
    uri=uri % uri_params,
```

Steps to Reproduce
Set any key of the FEEDS dict to a Path object containing a %-formatted path:
```
FEEDS = {
  pathlib.Path("./%(time)s.csv"): {
    "format": "csv",
    "store_empty": True,
  }
}
```

Run any spider scrapy crawl <spider_name>.
Expected behavior: No exception in logs and the feed file being created.
"""

TORNADO_ISSUE = r"""

When a callback is supplied, the future is not created and leads to a crash in the `read_until_close` method.

File "tornado/iostream.py", line 355, in read_until_close
    future.add_done_callback(lambda f: f.exception())
AttributeError: 'NoneType' object has no attribute 'add_done_callback'

"""

PYRAMID_ISSUE = r"""
Currently, this code will be served to the browser as text/plain but the HTML are not rendered by the browser:

```
from wsgiref.simple_server import make_server
from pyramid.config import Configurator

def hello_world(request):
    request.response.content_type = "text/html"
    return "<p>Hello World</p>"

config = Configurator()
config.add_route('hello', '/')
config.add_view(hello_world, route_name='hello', renderer='string')
app = config.make_wsgi_app()
make_server('', 8000, app).serve_forever()
```

I think this is unintuitive/unexpected behavior, instead when request.response.content_type is explicitly set to 'text/html', the renderer should be not change it (which it currently seems to be doing).
"""

ISSUES = [
    NUMPY_ISSUE,
    PANDAS_ISSUE,
    SYMPY_ISSUE,
    PILLOW_ISSUE,
    SCRAPY_ISSUE,
    TORNADO_ISSUE,
    PYRAMID_ISSUE,
]

ISSUES = [f"[ISSUE]\n{issue}\n[/ISSUE]" for issue in ISSUES]


def random_issue_combination():
    ## permute
    random.shuffle(ISSUES)

    ## combine
    final_issue = "Example Issues:\n\n"
    for idx, issue in enumerate(ISSUES):
        final_issue += f"\n\nExample {idx+1}:\n\n{issue}\n"

    return final_issue
