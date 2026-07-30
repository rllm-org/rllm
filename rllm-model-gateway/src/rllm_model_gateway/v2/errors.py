class GatewayError(Exception):
    def __init__(self, message: str, status_code: int = 400, error_type: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


class WorkerUnavailableError(GatewayError):
    def __init__(self, worker_id: int) -> None:
        super().__init__(f"gateway worker {worker_id} is unavailable", 503, "server_error")
