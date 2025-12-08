aiohttp_issues = [
    ### issue 1
    """Describe the bug
Kodi integration in home assistant is using aiohttp for checking the connection to the kodi instances via websockets.
Although a default timeout is specified in the code, it still takes a lot of time until the connection timeouts...
After some debugging, it seems that the client connection for the websockets _ws_connect (https://github.com/aio-libs/aiohttp/blob/master/aiohttp/client.py#L690) does not pass down the timeout to the actual request (https://github.com/aio-libs/aiohttp/blob/master/aiohttp/client.py#L769-L779)
Because of this, when the url is not available, it will take a lot more time than our specified timeout until the CannotConnectError is thrown.
Can you please pass down the timeout in the request call, so that the configured timeout it taken into account?

To Reproduce
Test code:

import asyncio
import time
from pykodi import CannotConnectError, get_kodi_connection

DEFAULT_PORT = 8080
DEFAULT_SSL = False
DEFAULT_TIMEOUT = 5
DEFAULT_WS_PORT = 9090

CONF_HOST = '192.168.0.110'
CONF_PORT = 8090
CONF_WS_PORT = DEFAULT_WS_PORT
CONF_USERNAME = 'kodi'
CONF_PASSWORD = 'kodi'
CONF_SSL = DEFAULT_SSL

async def ping():
    conn = get_kodi_connection(CONF_HOST,CONF_PORT,CONF_WS_PORT,CONF_USERNAME,CONF_PASSWORD,CONF_SSL,timeout=5)
    try:
        start = time.time()
        await conn.connect()
    except CannotConnectError:
        end = time.time()
        print('timout error after %ss' %(end - start))
        print('error')

asyncio.run(ping())
Expected behavior
Connection error after the specified timeout

Logs/tracebacks
See output of test code
""",
    ### issue 2
]
