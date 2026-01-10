import os
import asyncio
from datetime import datetime
from pathlib import Path
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from deepresearch_workflow import DeepResearchWorkflow
from deepresearch_tools import PythonInterpreterTool, ScoreTool
from rllm.engine.rollout import OpenAIEngine


SYSTEM_PROMPT = """You are an expert Kaggle competitor. Produce one Python script that trains a model and writes `submission.csv` for the dataset in the user prompt.

Rules:
- Use only already-installed common libraries (no installs).
- Use the PythonInterpreter tool to iteratively write/run/update your script.
- After producing a submission, use the Score tool to grade it; if the score is unsatisfying, keep refining the code and re-grading until you are satisfied.
- Be concise and task-focused.

Loop:
1) You are a multi-turn generation agent: in each turn, propose/refine the script or reasoning, then wait for environment/tool feedback.
2) Execute via the tool until it runs cleanly and produces the file. STRICT: each response may contain exactly ONE <tool_call> block—do not emit multiple tool calls.
3) After generating the code, the Python environment will provide feedback. You must observe at least one tool feedback (execution result wrapped in <tool_response></tool_response> tags) before deciding to end. Only when feedback looks good do you reply with <answer>submission</answer>; otherwise continue iterating (do not output <answer> tags).
4) Use PythonInterpreter to run updated code; use Score tool to grade `submission.csv`. Repeat this refine-grade loop until the submission is acceptable, then end with <answer>submission</answer>.

Tool usage:
For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
- Wrap executable code exactly like this:
<tool_call>
python
<code>
# Your Python code here
print("Hello World")
</code>
</tool_call>
Code inside those tags runs in Python; keep the tool name `python` and include <code></code>.

- To grade the submission.csv file, you need to use Score tool and output json object like this:
<tool_call>
{"name": "Score", "arguments": {"competition_id": "spaceship-titanic"}}
</tool_call>

Current date: """

task_specific_prompt = """
## Competition ID: spaceship-titanic

## Description
Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.

The *Spaceship Titanic* was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary *Spaceship Titanic* collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!

![joel-filipe-QwoNAhbmLLo-unsplash.jpg](https://storage.googleapis.com/kaggle-media/competitions/Spaceship%20Titanic/joel-filipe-QwoNAhbmLLo-unsplash.jpg)

To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship’s damaged computer system.

Help save them and change history!

## Evaluation

### Metric

Submissions are evaluated based on their [classification accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy), the percentage of predicted labels that are correct.

### Submission Format

The submission format for the competition is a csv file with the following format:

```
PassengerId,Transported
0013_01,False
0018_01,False
0019_01,False
0021_01,False
etc.
```

# Dataset Description

In this competition your task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help you make these predictions, you're given a set of personal records recovered from the ship's damaged computer system.

## File and Data Field Descriptions

- **train.csv** - Personal records for about two-thirds (~8700) of the passengers, to be used as training data.
    - `PassengerId` - A unique Id for each passenger. Each Id takes the form `gggg_pp` where `gggg` indicates a group the passenger is travelling with and `pp` is their number within the group. People in a group are often family members, but not always.
    - `HomePlanet` - The planet the passenger departed from, typically their planet of permanent residence.
    - `CryoSleep` - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
    - `Cabin` - The cabin number where the passenger is staying. Takes the form `deck/num/side`, where `side` can be either `P` for *Port* or `S` for *Starboard*.
    - `Destination` - The planet the passenger will be debarking to.
    - `Age` - The age of the passenger.
    - `VIP` - Whether the passenger has paid for special VIP service during the voyage.
    - `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` - Amount the passenger has billed at each of the *Spaceship Titanic*'s many luxury amenities.
    - `Name` - The first and last names of the passenger.
    - `Transported` - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.
- **test.csv** - Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of `Transported` for the passengers in this set.
- **sample_submission.csv** - A submission file in the correct format.
    - `PassengerId` - Id for each passenger in the test set.
    - `Transported` - The target. For each passenger, predict either `True` or `False`.

## Dataset Folder:
/fsx/zyhang/mle-bench-data/spaceship-titanic/prepared/public/
"""

user_prompt_template = """
You are solving the task below. Follow the requirements precisely.

{specific_task_description}

Your code should adhere to the following requirements:
- Prefer and explicitly use GPU (CUDA) acceleration when available: move models/tensors to GPU and handle CPU fallback if CUDA is not present.
- Load train/test data from the provided dataset folder (## Dataset Folder).
- Match the exact columns/headers in sample_submission.csv (## Dataset Folder) and write submission.csv to the **current directory**.
- Use only common preinstalled libraries (no installs).
- Please restrict the use of external libraries to the common libraries.
- The task is an out-of-date competition, so please ignore the timeline in the task description.
"""



# Setup rollout engine
engine = OpenAIEngine(
    model="anthropic/claude-sonnet-4.5",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Create workflow engine for parallel execution
workflow_engine = AgentWorkflowEngine(
    workflow_cls=DeepResearchWorkflow,
    workflow_args={
        "tools": {
            "PythonInterpreter": PythonInterpreterTool(),
            "Score": ScoreTool(),
        },
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "system_prompt": SYSTEM_PROMPT,
    },
    rollout_engine=engine,
    n_parallel_tasks=1  # Run 1 task in parallel
)

# Run evaluation on multiple tasks
tasks = [
    {"question": user_prompt_template.replace("{specific_task_description}", task_specific_prompt), "answer": "submission"},
]


def setup_output_directory() -> Path:
    """
    Create output/<timestamp> under this file's directory and switch cwd there so
    all generated files (e.g., submission.csv) land inside the run folder.
    """
    base_dir = Path(__file__).resolve().parent
    output_root = base_dir / "output"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(run_dir)
    # Let other modules know where to write artifacts/logs
    os.environ["DEEPRESEARCH_OUTPUT_DIR"] = str(run_dir)
    print(f"Saving run artifacts to {run_dir}")
    return run_dir


async def main():
    setup_output_directory()
    episodes = await workflow_engine.execute_tasks(tasks)

    # Episodes contain full trajectories for training
    for episode in episodes:
        print(f"Task: {episode.task}")
        print(f"Prediction: {episode.metrics.get('prediction')}")
        print(f"Is correct: {episode.is_correct}")

if __name__ == "__main__":
    asyncio.run(main())
