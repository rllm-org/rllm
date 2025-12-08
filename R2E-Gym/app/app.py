import os
import json
from flask import Flask, render_template, request, redirect, url_for, abort
from collections import defaultdict
import markdown

from r2egym.agenthub.trajectory.trajectory import Trajectory
from r2egym.agenthub.trajectory.swebench_utils import swebench_report

app = Flask(__name__)

# Updated path to the directory containing JSONL files
# Navigate up from 'app/app.py' to reach the project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(BASE_DIR)
TRAJ_DIR = os.path.join("../R2E-Gym", "deepswe")


def get_jsonl_files():
    """Retrieve all JSONL files in the TRAJ_DIR directory."""
    try:
        files = [f for f in os.listdir(TRAJ_DIR) if f.endswith(".jsonl")]
        return files
    except FileNotFoundError:
        return []


files = get_jsonl_files()

logs_dict = {}


def read_logs(filename):
    """Read logs from a JSONL file."""
    if filename in logs_dict:
        print(f"Using cached logs for {filename}")
        return logs_dict[filename]

    filepath = os.path.join(TRAJ_DIR, filename)
    print(f"Reading logs from {filepath}")
    if not os.path.exists(filepath):
        return []
    logs = []
    with open(filepath, "r") as file:
        for idx, line in enumerate(file):
            try:
                log = json.loads(line)
                # log["_id"] = idx  # Assign a unique ID based on the line number

                # Convert markdown fields to HTML with extensions for better formatting
                system_prompt_md = log.get("system_prompt", "")
                instance_prompt_md = log.get("instance_prompt", "")
                log["system_prompt_html"] = markdown.markdown(
                    system_prompt_md, extensions=["fenced_code", "nl2br"]
                )
                log["instance_prompt_html"] = markdown.markdown(
                    instance_prompt_md, extensions=["fenced_code", "nl2br"]
                )
                if "[ISSUE]" in log["ds"]["problem_statement"]:
                    log["ds"]["problem_statement"] = log["ds"][
                        "problem_statement"
                    ].split("[ISSUE]")[1]
                logs.append(log)
            except json.JSONDecodeError:
                continue  # Skip invalid JSON lines
    logs = sorted(logs, key=lambda x: x["docker_image"], reverse=True)
    for idx, log in enumerate(logs):
        log["_id"] = idx
    logs_dict[filename] = logs
    return logs


for f in files:
    try:
        logs_dict[f] = read_logs(f)
    except:
        pass


@app.route("/")
def index():
    """Home page: list all JSONL files."""
    return render_template("index.html", files=files)


@app.route("/logs/<filename>")
def logs(filename):
    """Logs page: display logs from the selected file, grouped by docker_image."""
    # Security check: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        abort(400, description="Invalid filename.")

    logs = logs_dict[filename]
    if not logs:
        abort(404, description="File not found or empty.")

    # Group logs by docker_image
    grouped_logs = defaultdict(list)
    for log in logs:
        docker_image = log.get("docker_image", "Unknown")
        grouped_logs[docker_image].append(log)

    return render_template("logs.html", filename=filename, grouped_logs=grouped_logs)


@app.route("/problems")
def problems():
    """Problems page: display problems from all logs with links to logs and trajectory."""
    files = get_jsonl_files()
    all_problems = defaultdict(dict)

    for filename in files:
        agent_name = filename.split(".")[0]
        logs = logs_dict[filename]
        # Create URLs using Flask's url_for:
        logs_url = url_for("logs", filename=filename)

        for log in logs:
            trajectory_url = url_for("trajectory", filename=filename, log_id=log["_id"])
            docker_image = log["docker_image"]
            success = log["reward"] == 1
            current = all_problems[docker_image].get(agent_name)
            # If the agent already has an entry for this docker_image, update its success status (if any log passed).
            if current:
                current["success"] = current["success"] or success
            else:
                all_problems[docker_image][agent_name] = {
                    "success": success,
                    "logs_url": logs_url,
                    "trajectory_url": trajectory_url,
                }
    return render_template("problems.html", problems=all_problems)


repo_name_map = {
    "pandas": "pandas-dev/pandas",
    "numpy": "numpy/numpy",
    "pillow": "python-pillow/Pillow",
    "orange3": "biolan/orange3",
    "datalad": "datalad/datalad",
    "coveragepy": "nedbat/coveragepy",
    "aiohttp": "aio-libs/aiohttp",
    "tornado": "tornadoweb/tornado",
    "pyramid": "Pylons/pyramid",
    "scrapy": "scrapy/scrapy",
}


@app.route("/trajectory/<filename>/<int:log_id>")
def trajectory(filename, log_id):
    """Trajectory page: display the trajectory of the selected log."""
    # Security check: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        abort(400, description="Invalid filename.")

    logs = logs_dict[filename]
    if not logs:
        abort(404, description="File not found or empty.")

    # Validate log_id
    if log_id < 0 or log_id >= len(logs):
        abort(404, description="Log ID out of range.")

    log = logs[log_id]
    trajectory = log.get("trajectory_steps", [])

    # Ensure trajectory is a list
    if not isinstance(trajectory, list):
        trajectory = []

    # Determine total logs for navigation
    total_logs = len(logs)

    try:
        repo_org, repo_name = log["ds"]["instance_id"].split("__")
        pr_id = repo_name.split("-")[-1]
        repo_name = "-".join(repo_name.split("-")[:-1])

        url = f"https://github.com/{repo_org}/{repo_name}/pull/{pr_id}/files"
        log["report"] = swebench_report(log["ds"], log["test_output"])
    except:
        url = f"https://github.com/{repo_name_map[log['ds']['repo_name']]}/commit/{log['ds']['commit_hash']}"

    t = Trajectory(**log)

    return render_template(
        "trajectory.html",
        filename=filename,
        log=log,
        url=url,
        trajectory=trajectory,
        current_log_id=log_id,
        total_logs=total_logs,
        total_steps=len(trajectory),  # Pass total steps to the template
        true_output_patch=t.true_output_patch,
        gt_patch=t.gt_patch,
    )


# Error Handlers
@app.errorhandler(404)
def not_found(error):
    return render_template("404.html", error=error.description), 404


@app.errorhandler(400)
def bad_request(error):
    return render_template("400.html", error=error.description), 400


if __name__ == "__main__":
    app.run(port=5760, host="0.0.0.0")
