"""SWE-ReX patches for mini-swe-agent v2 compatibility."""

from patches.jinja_tojson_fix import apply_jinja_tojson_patch
from patches.swerex_modal_compat import apply_swerex_modal_compat_patch
from patches.swerex_modal_minimal import apply_swerex_modal_minimal_patch
from patches.swerex_remote_retry import apply_swerex_remote_retry_patch

__all__ = [
    "apply_swerex_modal_minimal_patch",
    "apply_swerex_modal_compat_patch",
    "apply_swerex_remote_retry_patch",
    "apply_jinja_tojson_patch",
]
