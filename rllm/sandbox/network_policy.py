"""Host-controlled solver networking. Domain rules require HTTPS/443."""

import ipaddress
import json
import os
from urllib.parse import urlsplit


def agent_network_policy(base_url: str) -> dict | None:
    """Unset disables filtering; [] permits only the configured model gateway."""
    raw = os.environ.get("RLLM_AGENT_NETWORK_DOMAINS")
    if raw is None:
        return None
    domains = json.loads(raw)
    cidrs = json.loads(os.environ.get("RLLM_AGENT_NETWORK_CIDRS", "[]"))
    for values in (domains, cidrs):
        if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
            raise ValueError("Network allowlists must be JSON lists of nonempty strings")
    for cidr in cidrs:
        if ipaddress.ip_network(cidr).version != 4:
            raise ValueError("Modal network allowlists require IPv4 CIDRs")
    url = urlsplit(base_url)
    if not url.hostname or url.scheme not in ("http", "https"):
        raise ValueError("Restricted networking requires an HTTP(S) gateway URL")
    try:
        ip = ipaddress.ip_address(url.hostname)
    except ValueError:
        if url.scheme != "https" or url.port not in (None, 443):
            raise ValueError("Hostname gateways require HTTPS/443; use an IP for HTTP tunnels")
        domains.append(url.hostname)
    else:
        if ip.version != 4:
            raise ValueError("Modal network allowlists require an IPv4 gateway or HTTPS hostname")
        # CIDR rules permit every port, not just the tunnel's port.
        cidrs.append(f"{ip}/{ip.max_prefixlen}")
    return {
        "outbound_domain_allowlist": list(dict.fromkeys(domains)),
        "outbound_cidr_allowlist": list(dict.fromkeys(cidrs)),
    }
