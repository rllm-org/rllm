scrapy_issues = [
    ### issue 1
    r"""
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

This is because the __init__ method of the FeedExporter uses as_uri() on Path objects, which yields an URL-encoded URI, loosing the format characters.

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

Reproduces how often: 100%

"""
]
