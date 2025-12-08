tornado_issues = [
    ### issue 1
    """
We have a single threaded application using tornado 4.5.3. The application calls stream.write to flush data downstream. We wrote a UT in which we mock the write_to_fd to return 0 bytes. We are trying to replicate a scenario in which the self._write_buffer would continually grow if no bound has been specified.

This test results in a BufferError every time the self._write_buffer is being updated either in write method self._write_buffer += data or in _handle_write method del self._write_buffer[:self._write_buffer_pos]

Let me know if there is other information we can share to help root cause this issue.

""",
    ### issue 2
    """tornado/tornado/wsgi.py:76
```
             return ["Hello world!\n"] 
```

This line (in a docstring showing how to use WSGIContainer) is wrong.

It should say:

         return [b"Hello world!\n"] 
... because WSGI requires a sequence of bytestrings in modern python. Without this change, the example fails with:

TypeError: sequence item 0: expected a bytes-like object, str found
in python 3.9 (and probably any python 3).
""",
    ### issue 3
    """Steps to reproduce
Start a basic Tornado HTTP/1.1 server.
Send it a request with an invalid method:
```
printf '\x22\x28\x29\x2c\x2f\x3a\x3b\x3c\x3d\x3e\x3f\x40\x5b\x5c\x5d\x7b\x7d\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff / HTTP/1.1\r\nHost: whatever\r\n\r\n' \
        | nc localhost 80
```

Observe that the server responds 405, indicating a syntactically valid but unsupported method, instead of 400, which would indicate a syntactically invalid message. Further, note that the 405 response does not contain an Allow header, even though RFC 9110 requires one to be present.
Expected behavior
Tornado should reject the request because its method is invalid due to containing forbidden characters. The HTTP RFCs define that only the following characters are permitted within an HTTP method:

> "!" / "#" / "$" / "%" / "&" / "'" / "*"
/ "+" / "-" / "." / "^" / "_" / "`" / "|" / "~"
/ DIGIT / ALPHA

(where DIGIT stands for ASCII digits (0-9) and ALPHA stands for ASCII letters (a-zA-Z))

All of the characters in the above request's method are disallowed, so the request should be rejected with a 400.

Nearly all other HTTP implementations reject this request with 400, including AIOHTTP, Apache httpd, FastHTTP, Go net/http, Gunicorn, H2O, HAProxy, Hyper, Hypercorn, Jetty, Ktor, Libevent, Lighttpd, Mongoose, Nginx, Node.js, LiteSpeed, Passenger, Puma, ServiceTalk, Tomcat, Twisted, OpenWrt uhttpd, Unicorn, Uvicorn, Waitress, WEBrick, and OpenBSD httpd.

Impact
Because 405 is heuristically cacheable, and different servers may have different interpretations of which bytes are invalid in headers, this behavior may be usable for cache poisoning.

""",
    ### issue 4
    # """
    # """
]
