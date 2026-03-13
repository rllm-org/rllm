"""rLLM CLI: login command for UI authentication."""

from __future__ import annotations

import click


@click.command("login")
@click.option("--relogin", is_flag=True, help="Force re-login even if already logged in.")
def login_cmd(relogin):
    """Log in to rLLM UI by storing your API key locally."""
    import os

    import httpx

    from rllm.experimental.eval.config import load_ui_config, save_ui_config

    ui_url = os.getenv("RLLM_UI_URL", "https://ui.rllm-project.com")

    # Check if already logged in
    if not relogin:
        existing = load_ui_config()
        if existing.get("ui_api_key"):
            try:
                resp = httpx.get(
                    f"{ui_url}/api/auth/me",
                    headers={"X-API-Key": existing["ui_api_key"]},
                    timeout=10.0,
                    follow_redirects=True,
                )
                resp.raise_for_status()
                user = resp.json()
                email = user.get("email", "unknown")
                click.echo(f"Currently logged in as: {email}")
                click.echo("Use --relogin to force re-login.")
                return
            except Exception:
                # Stored key is invalid/expired, proceed with login
                pass

    click.echo(f"You can find your API key here: {ui_url}/settings")
    api_key = click.prompt("Paste your API key", hide_input=True)
    api_key = api_key.strip()
    # Handle pasting the full env line (e.g. "RLLM_API_KEY=rllm_...")
    if api_key.startswith("RLLM_API_KEY="):
        api_key = api_key.split("=", 1)[1]
    if not api_key:
        click.echo("Error: API key cannot be empty.", err=True)
        raise SystemExit(1)

    # Validate key against the UI backend
    try:
        resp = httpx.get(
            f"{ui_url}/api/auth/me",
            headers={"X-API-Key": api_key},
            timeout=10.0,
            follow_redirects=True,
        )
        resp.raise_for_status()
        user = resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            click.echo("Error: Invalid API key.", err=True)
        else:
            click.echo(f"Error: Server returned {e.response.status_code}.", err=True)
        raise SystemExit(1) from None
    except httpx.ConnectError:
        click.echo(f"Error: Could not connect to {ui_url}.", err=True)
        raise SystemExit(1) from None

    save_ui_config(ui_api_key=api_key)
    email = user.get("email", "unknown")
    click.echo(f"Logged in to {ui_url} as {email}")
