#!/usr/bin/env python3
"""Refine agent using httpx with direct multi-server routing."""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("refine_agent_httpx")


REFINE_PROMPT = """**TASK:**
Synthesize the key information from the **[Retrieved Documents]** that is relevant to the **[Current Query]**.

**INSTRUCTIONS:**
1.  **Extract & Merge:** Identify all relevant facts and combine them. Eliminate redundancy. You should provide information for deep research, not answer to current query.
2.  **Provide Information, Not an Answer:** Your output should be a self-contained block of information, NOT a direct, short answer to the current query.
3.  **Handle Insufficient Information:** If the documents do not contain relevant information for the query, state that the provided sources are insufficient and suggest that further investigation may be needed. You can also provide some further investigation direction and query rewrite suggestions.
4.  **Format:** Enclose the entire synthesized output within `<information>` and `</information>` tags. Add no other text. For example, <information> Synthesized information for deep research here </information>.


**CONTEXT:**
- **[Current Query]:** {query}
- **[Retrieved Documents]:** {documents}

**SYNTHESIZED INFORMATION:**
"""


@dataclass
class RequestStats:
    """Thread-safe request statistics tracker."""

    in_flight: int = 0
    total_started: int = 0
    total_completed: int = 0
    total_failed: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Timing stats
    min_latency: float = float("inf")
    max_latency: float = 0.0
    total_latency: float = 0.0

    async def start_request(self) -> int:
        async with self._lock:
            self.in_flight += 1
            self.total_started += 1
            current = self.in_flight
            if current % 50 == 0:
                logger.info(f"[STATS] In-flight: {current}, Started: {self.total_started}, Completed: {self.total_completed}, Failed: {self.total_failed}")
            return current

    async def end_request(self, success: bool, latency: float):
        async with self._lock:
            self.in_flight -= 1
            if success:
                self.total_completed += 1
                self.total_latency += latency
                self.min_latency = min(self.min_latency, latency)
                self.max_latency = max(self.max_latency, latency)
            else:
                self.total_failed += 1

            # Log periodically
            if (self.total_completed + self.total_failed) % 50 == 0:
                avg_latency = self.total_latency / max(1, self.total_completed)
                logger.info(f"[STATS] In-flight: {self.in_flight}, Completed: {self.total_completed}, Failed: {self.total_failed}, Latency(avg/min/max): {avg_latency:.2f}s/{self.min_latency:.2f}s/{self.max_latency:.2f}s")

    async def get_stats(self) -> dict:
        async with self._lock:
            avg_latency = self.total_latency / max(1, self.total_completed)
            return {
                "in_flight": self.in_flight,
                "total_started": self.total_started,
                "total_completed": self.total_completed,
                "total_failed": self.total_failed,
                "avg_latency": avg_latency,
                "min_latency": self.min_latency if self.min_latency != float("inf") else 0,
                "max_latency": self.max_latency,
            }


@dataclass
class ServerStats:
    """Per-server statistics for load balancing."""

    url: str
    in_flight: int = 0
    total_requests: int = 0
    total_failures: int = 0
    last_failure_time: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def start_request(self):
        async with self._lock:
            self.in_flight += 1
            self.total_requests += 1

    async def end_request(self, success: bool):
        async with self._lock:
            self.in_flight -= 1
            if not success:
                self.total_failures += 1
                self.last_failure_time = time.time()

    async def get_in_flight(self) -> int:
        async with self._lock:
            return self.in_flight


# Global stats tracker
refine_stats = RequestStats()

# Multi-server URL management
_url_file: str | None = "/data/user/rllm/examples/fully_async/deep_research/.url/refine_url"
_server_urls: list[str] = []
_server_stats: list[ServerStats] = []
_last_file_read: float = 0.0
_file_refresh_interval: float = 300.0  # 5 minutes
_clients: dict = {}  # url -> httpx.AsyncClient
_max_connections_per_server: int = 1024
_round_robin_counter: int = 0
_counter_lock = asyncio.Lock()

# Load balancing mode: "round_robin", "least_connections", "random"
_load_balance_mode: str = "least_connections"


def _load_urls_from_file():
    """Load server URLs from file (one URL per line, skip comments)."""
    global _server_urls, _server_stats, _last_file_read, _clients

    if not _url_file:
        return

    try:
        new_urls = []
        with open(_url_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    url = line.rstrip("/")
                    new_urls.append(url)

        if new_urls != _server_urls:
            logger.info(f"Server URLs changed: {len(_server_urls)} -> {len(new_urls)} servers")
            logger.info(f"New servers: {new_urls}")

            # Close old clients for removed URLs
            removed_urls = set(_server_urls) - set(new_urls)
            for url in removed_urls:
                if url in _clients:
                    # Schedule client close (can't await here)
                    logger.info(f"Removing client for {url}")
                    del _clients[url]

            _server_urls = new_urls
            _server_stats = [ServerStats(url=url) for url in new_urls]

        _last_file_read = time.time()

    except Exception as e:
        logger.warning(f"Failed to load URLs from file {_url_file}: {e}")
        # Keep existing URLs if file read fails


# Load URLs on module import
_load_urls_from_file()


def _maybe_refresh_urls():
    """Check if URL file needs refreshing and reload if necessary."""
    if not _url_file:
        return

    if time.time() - _last_file_read >= _file_refresh_interval:
        _load_urls_from_file()


def _get_client(url: str) -> httpx.AsyncClient:
    """Get or create an httpx AsyncClient for the given URL."""
    global _clients

    if url not in _clients:
        logger.info(f"Creating new httpx client for {url}")
        _clients[url] = httpx.AsyncClient(
            base_url=url,
            timeout=httpx.Timeout(120.0),
            limits=httpx.Limits(max_connections=_max_connections_per_server, max_keepalive_connections=_max_connections_per_server // 2),
        )

    return _clients[url]


async def _select_server_round_robin() -> tuple[str, ServerStats]:
    """Select server using round-robin."""
    global _round_robin_counter

    async with _counter_lock:
        idx = _round_robin_counter % len(_server_urls)
        _round_robin_counter += 1

    return _server_urls[idx], _server_stats[idx]


async def _select_server_least_connections() -> tuple[str, ServerStats]:
    """Select server with least in-flight requests."""
    min_in_flight = float("inf")
    selected_idx = 0

    for idx, stats in enumerate(_server_stats):
        in_flight = await stats.get_in_flight()
        if in_flight < min_in_flight:
            min_in_flight = in_flight
            selected_idx = idx

    return _server_urls[selected_idx], _server_stats[selected_idx]


async def _select_server_random() -> tuple[str, ServerStats]:
    """Select server randomly."""
    idx = random.randint(0, len(_server_urls) - 1)
    return _server_urls[idx], _server_stats[idx]


async def _select_server() -> tuple[str, ServerStats]:
    """Select a server based on load balancing mode."""
    _maybe_refresh_urls()

    if not _server_urls:
        raise RuntimeError("No server URLs configured")

    if len(_server_urls) == 1:
        return _server_urls[0], _server_stats[0]

    if _load_balance_mode == "round_robin":
        return await _select_server_round_robin()
    elif _load_balance_mode == "least_connections":
        return await _select_server_least_connections()
    elif _load_balance_mode == "random":
        return await _select_server_random()
    else:
        return await _select_server_round_robin()


def set_url_file(url_file: str):
    """Set the URL file path and perform initial load."""
    global _url_file
    _url_file = url_file
    _load_urls_from_file()


def set_load_balance_mode(mode: str):
    """Set the load balancing mode: 'round_robin', 'least_connections', or 'random'."""
    global _load_balance_mode
    if mode in ("round_robin", "least_connections", "random"):
        _load_balance_mode = mode
        logger.info(f"Load balance mode set to: {mode}")
    else:
        logger.warning(f"Invalid load balance mode: {mode}, keeping {_load_balance_mode}")


def set_max_connections_per_server(max_connections: int):
    """Set the maximum number of concurrent connections per server."""
    global _max_connections_per_server, _clients
    _max_connections_per_server = max_connections
    _clients = {}  # Force client recreation with new limits


def get_server_count() -> int:
    """Get the number of configured servers."""
    return len(_server_urls)


async def get_server_stats() -> list[dict]:
    """Get statistics for all servers."""
    stats = []
    for server_stat in _server_stats:
        async with server_stat._lock:
            stats.append(
                {
                    "url": server_stat.url,
                    "in_flight": server_stat.in_flight,
                    "total_requests": server_stat.total_requests,
                    "total_failures": server_stat.total_failures,
                }
            )
    return stats


REFINE_AGENT_PROMPT = """You are a helpful assistant that refines and summarizes search results.

Given an original query and search results, your task is to:
1. Extract and summarize ALL information that is relevant to answering the query
2. Ignore completely irrelevant details from the search results
3. Present the information in a clear, comprehensive manner
4. If the search results don't contain relevant information, state that clearly

Focus on facts that help answer the query."""


def parse_thinking(content: str) -> str:
    """
    Parse out the <think>...</think> tags and return only the result part.

    Args:
        content: The raw model response that may contain thinking tags

    Returns:
        The content after </think>

    Raises:
        ValueError: If no <think> tag found, or if <think> exists without </think>
    """
    if content is None:
        raise ValueError("Content is None - no response from model")

    # Check if there's a <think> tag
    if "<think>" not in content:
        raise ValueError("No <think> tag found in model response")

    # Check if there's a </think> tag
    if "</think>" not in content:
        raise ValueError("<think> tag found but no </think> closing tag - incomplete thinking")

    # Return everything after </think>
    result = content.split("</think>", 1)[-1].strip()
    return result


def parse_information(content: str) -> str:
    """
    Parse out the content inside <information>...</information> tags.

    Args:
        content: The model response that should contain information tags

    Returns:
        The content inside the <information> tags

    Raises:
        ValueError: If no <information> tag found, or missing closing tag
    """
    if content is None:
        raise ValueError("Content is None - no response from model")

    # Check if there's an <information> tag
    if "<information>" not in content:
        raise ValueError("No <information> tag found in model response")

    # Check if there's a </information> tag
    if "</information>" not in content:
        raise ValueError("<information> tag found but no </information> closing tag")

    # Extract content between the tags
    start_tag = "<information>"
    end_tag = "</information>"

    start_idx = content.find(start_tag) + len(start_tag)
    end_idx = content.find(end_tag)

    result = content[start_idx:end_idx].strip()
    return result


async def _refine(query: str, result: str, model: str = "Qwen/Qwen3-8B") -> str:
    """
    Refine search results by summarizing information relevant to the query.

    Args:
        query: The original search query
        result: The raw search results to refine
        model: The model to use for refinement

    Returns:
        A refined summary of relevant information (without thinking content)
    """

    messages = [{"role": "user", "content": REFINE_PROMPT.format(query=query, documents=result)}]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }

    # Select a server using load balancing
    server_url, server_stats = await _select_server()
    client = _get_client(server_url)

    await server_stats.start_request()
    success = False

    try:
        # Make request to OpenAI-compatible chat completion endpoint
        response = await client.post(
            "/chat/completions",
            json=payload,
        )

        response.raise_for_status()
        response_data = response.json()

        # Extract content from response
        raw_content = response_data["choices"][0]["message"]["content"]

        # First remove thinking tags
        content_no_think = parse_thinking(raw_content)

        # Then extract information block
        parsed_content = parse_information(content_no_think)

        success = True

        # if random.random() < 0.005:
        #     print(parsed_content)

        return f"Your query is: {query}. The search results are summarized as following: {parsed_content}"
        # return parsed_content
    except Exception as e:
        import traceback

        print(f"⚠️  Refine exception [{type(e).__name__}]: {e}")
        print(f"    Server: {server_url}")
        traceback.print_exc()
    finally:
        await server_stats.end_request(success)


async def refine(query: str, result: str, model: str = "Qwen/Qwen3-8B") -> str:
    """
    Refine search results with retry logic and stats tracking.

    Args:
        query: The original search query
        result: The raw search results to refine
        model: The model to use for refinement

    Returns:
        A refined summary, or the original result if all retries fail
    """
    start_time = time.time()
    in_flight = await refine_stats.start_request()

    success = False
    try:
        for attempt in range(3):
            try:
                refine_result = await _refine(query, result, model)
                if isinstance(refine_result, str):
                    success = True
                    return refine_result
                else:
                    continue
            except Exception as e:
                logger.warning(f"Refine attempt {attempt + 1}/3 failed (in-flight: {in_flight}): {e}")
                continue

        # All retries failed, return original result
        return result
    finally:
        latency = time.time() - start_time
        await refine_stats.end_request(success, latency)


async def get_refine_stats() -> dict:
    """Get current refine request statistics."""
    return await refine_stats.get_stats()


def log_refine_stats_sync():
    """Synchronously log current stats (for use in signal handlers etc)."""
    logger.info(f"[FINAL STATS] In-flight: {refine_stats.in_flight}, Started: {refine_stats.total_started}, Completed: {refine_stats.total_completed}, Failed: {refine_stats.total_failed}")


async def cleanup():
    """Cleanup resources - close all httpx clients."""
    global _clients
    for url, client in _clients.items():
        try:
            await client.aclose()
        except Exception as e:
            logger.warning(f"Error closing client for {url}: {e}")
    _clients = {}


if __name__ == "__main__":
    import asyncio
    import time

    # Example usage
    test_query = "Oleg Yefremov Moscow Art Theatre appointment year"
    test_result = """[Document 1]
"Oleg Yefremov"
In 1970, Yefremov became the chief director of the Moscow Art Theater.

[Document 2]
"Yuri Lyubimov"
Yuri Lyubimov founded the Taganka Theatre in 1964.
"""

    async def main():
        # Print server configuration
        print(f"Configured servers: {get_server_count()}")
        print(f"Load balance mode: {_load_balance_mode}")
        print(f"Server URLs: {_server_urls}")

        start = time.time()
        tasks = [refine(test_query, test_result) for _ in range(1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        print(results)
        elapsed = time.time() - start

        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = sum(1 for r in results if isinstance(r, Exception))

        # Print final stats
        stats = await get_refine_stats()
        server_stats = await get_server_stats()

        print(f"\n{'=' * 60}")
        print(f"Completed {successes} requests, {failures} failures in {elapsed:.2f}s")
        print(f"Throughput: {successes / elapsed:.2f} req/s")
        print(f"{'=' * 60}")
        print("Global Stats:")
        print(f"  Completed: {stats['total_completed']}")
        print(f"  Failed: {stats['total_failed']}")
        print(f"  Avg Latency: {stats['avg_latency']:.2f}s")
        print(f"  Min Latency: {stats['min_latency']:.2f}s")
        print(f"  Max Latency: {stats['max_latency']:.2f}s")
        print("\nPer-Server Stats:")
        for ss in server_stats:
            print(f"  {ss['url']}: requests={ss['total_requests']}, failures={ss['total_failures']}")
        print(f"{'=' * 60}")

        if successes > 0:
            # Find first successful result
            for r in results:
                if not isinstance(r, Exception):
                    print("\nSample result:")
                    print(r[:500] if len(r) > 500 else r)
                    break

        # Cleanup
        await cleanup()

    asyncio.run(main())
