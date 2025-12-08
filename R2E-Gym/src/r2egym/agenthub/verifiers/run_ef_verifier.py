import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import fire
import litellm
import numpy as np
from tqdm import tqdm

from r2egym.agenthub.trajectory.trajectory import Trajectory
from r2egym.agenthub.verifiers.prepare_ef_verifier_input import traj2verifier_data

MAX_RETRIES = 5


def run_model(arg) -> float:
    message_list: list[dict]

    message_list, verifier_model_name = arg

    retries = 0

    # condense messages
    condensed_user_msg = message_list[1][
        "content"
    ]  # condense(input_str=message_list[1]['content'], max_tokens = 28000)
    message_list = [
        {"role": "system", "content": message_list[0]["content"]},
        {"role": "user", "content": condensed_user_msg},
    ]
    # query the model with retries
    while retries < MAX_RETRIES:
        try:
            response = litellm.completion(
                model=verifier_model_name,
                tools=[],
                messages=message_list,
                n=1,
                function_call=None,
                tool_choice="none",
                timeout=120,
                api_key=None,
                temperature=0,
                api_base=f"http://localhost:8000/v1",
                vertex_ai_project="r2eg-441800",
                vertex_ai_location="europe-west1",
                logprobs=True,
                top_logprobs=20,
                # extra_body={
                #     "guided_choice": ["YES", "NO"]
                # },
            )
            break
        except Exception as e:
            print(f"LLM query failed: {e}")
            retries += 1
            if retries >= MAX_RETRIES:
                raise e

    all_logits = [
        {
            lp.token: lp.logprob
            for lp in response.choices[0].logprobs.content[4].top_logprobs
        }
    ]

    k = 0

    p_yes = all_logits[k].get("YES", -10000)
    p_no = all_logits[k].get("NO", -10000)
    yes_prob = (np.exp(p_yes)) / (np.exp(p_yes) + np.exp(p_no))

    return yes_prob


def process_trajectories_to_verifier_format(
    traj_file_glob: str,
    verifier_model_name: str,
    max_workers: int = 40,
    max_tokens: int = 65536,
):
    traj_files = glob.glob(traj_file_glob)
    for traj_file in traj_files:
        trajectories: list[Trajectory] = []
        with open(traj_file, "r") as f:
            for line in f:
                trajectories.append(Trajectory.model_validate_json(line))

        messages = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(traj2verifier_data, traj, max_tokens=max_tokens)
                for traj in trajectories
            ]
            for future in as_completed(futures):
                data_entry, success = future.result()
                messages.append(data_entry)

        yes_probs = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(run_model, (message, verifier_model_name))
                for message in messages
            ]
            for future in as_completed(futures):
                yes_prob = future.result()
                yes_probs.append(yes_prob)

        for traj, yes_prob in zip(trajectories, yes_probs):
            traj.verifier_prob = yes_prob

        with open(traj_file, "w") as f:
            for traj in trajectories:
                f.write(traj.model_dump_json() + "\n")


if __name__ == "__main__":
    fire.Fire(process_trajectories_to_verifier_format)
