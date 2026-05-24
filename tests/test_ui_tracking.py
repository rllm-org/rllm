import io

from rllm.utils.tracking import TeeStream


class _Client:
    def __init__(self):
        self.payloads = []

    def post(self, path, json):
        self.payloads.append((path, json))


def test_tee_stream_timestamp_works_on_python310():
    client = _Client()
    stream = TeeStream(io.StringIO(), client, "session-1", "stdout")

    stream.write("hello\n")
    stream.flush()

    assert len(client.payloads) == 1
    path, payload = client.payloads[0]
    assert path == "/api/logs/batch"
    assert payload["logs"][0]["timestamp"].endswith("+00:00")
    assert payload["logs"][0]["message"] == "hello"
