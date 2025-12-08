import json
import glob
from multiprocessing import Pool

import tqdm
import tiktoken
from openai import OpenAI

from r2egym.repo_analysis.build_syn_issue import get_prompt
from r2egym.repo_analysis.validate_docker_and_hf import DatasetRow


def get_files():
    return glob.glob("repo_datasets/*.jsonl")


client = OpenAI()
tokenizer = tiktoken.encoding_for_model("o1-mini")


def run_o1mini(prompt):
    return (
        client.chat.completions.create(
            messages=prompt,  # type: ignore
            model="o1-mini",
            max_completion_tokens=12000,
        )
        .choices[0]
        .message.content
    )


def main():
    long_idx = 0
    for file in get_files():
        print(file)
        repo_name = file.split("/")[-1].split(".")[0]
        with open(file) as f:
            data = [json.loads(line) for line in f.readlines()]
            data = [DatasetRow(**d) for d in data]

        prompts = []
        for row in tqdm.tqdm(data):
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": get_prompt(row.parsed_commit, row.execution_result),
                    }
                ]
            )

            token_count = len(tokenizer.encode(prompts[-1][0]["content"]))
            # if token_count > 50000:
            #     with open(f"new_prompts/{repo_name}_long_{long_idx}.txt", "w") as f:
            #         f.write(f"prompt:\n{prompts[-1][0]['content']}")
            #         long_idx += 1
            #         print(
            #             f"Long prompt with {token_count} tokens @ {long_idx} in {repo_name}"
            #         )

        with Pool(50) as p:
            completions = list(
                tqdm.tqdm(p.imap(run_o1mini, prompts), total=len(prompts))
            )

        for i, completion in enumerate(completions):
            data[i].prompt = prompts[i][0]["content"]
            data[i].problem_statement = completion

        with open(f"repo_datasets/{repo_name}.jsonl", "w") as f:
            for d in data:
                f.write(json.dumps(d.model_dump_json(indent=None)) + "\n")


def main_test():

    for file in get_files():
        repo_name = file.split("/")[-1].split(".")[0]
        with open(file) as f:
            data = [json.loads(line) for line in f.readlines()][:5]
            data = [DatasetRow(**d) for d in data]

        prompts = []
        for row in tqdm.tqdm(data):
            prompts.append(
                [
                    {
                        "role": "user",
                        "content": get_prompt(row.parsed_commit, row.execution_result),
                    }
                ]
            )

        with Pool(50) as p:
            completions = list(
                tqdm.tqdm(p.imap(run_o1mini, prompts), total=len(prompts))
            )

        for i, completion in enumerate(completions):
            with open(f"new_prompts/{repo_name}_prompt_{i}.txt", "w") as f:
                f.write(prompts[i][0]["content"])
            with open(f"new_prompts/{repo_name}_completion_{i}.txt", "w") as f:
                f.write(completion)


if __name__ == "__main__":
    main()
