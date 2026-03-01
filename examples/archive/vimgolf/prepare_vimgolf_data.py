import json
import os

from datasets import Dataset

from rllm.data.dataset import DatasetRegistry

_DATASET_PATH = os.path.join(os.path.dirname(__file__), "vimgolf_public_challenges.jsonl")


def prepare_vimgolf_data():
    """
    Prepare the vimgolf dataset for training.

    Even if we use all the data for training, we will not exhaust the game, since we can ask the model to use fewer keystrokes.
    """
    datalist = []
    with open(_DATASET_PATH) as f:
        for line in f:
            line_data = json.loads(line)
            input = line_data["input"]
            target = line_data["target"]
            details = line_data["metadata"]["detail"]
            challenge_data = dict(
                input=input,
                output=target,
                challenge_id=line_data["id"],
            )
            question_prompt = f"""
Vimgolf is a game where you try to transform text using the fewest number of keystrokes in Vim.

Your task is to solve the following Vimgolf challenge with details:
  
Details:
  
{details}

The input file wrapped in triple backticks:
  
```
{input}
```

The output file wrapped in triple backticks:
  
```
{target}
```

Your keystokes must be less than the length of output file. Do not naively copy and paste the output file. You must use Vim commands to transform the input file into the output file.

Here are some example solutions, for format demostration (all solutions shall be in one line):

iHello World<Esc>:wq<NL>

:%s/abcdef/defabc/g<NL>:wq<NL>

Your last line of response will be treated as solution. Do not wrap the solution around any marker (like triple backticks), just write it in plain style. Do not write it in multiline style. Do not write any comment or explanation. Do not write any other text. Just write the solution. If your solution contains multiple steps, you will concatenate these steps into one line, optionally using <NL> as separator, depending on the situation.

Example response:

I think the following solution is optimal:

iHello World<Esc>:s/World/Earth/g<NL>:wq<NL>

Please write your solution according to the rules and the example response:
"""
            it = {
                "question": question_prompt,
                "ground_truth": json.dumps(challenge_data),
                "data_source": "vimgolf-public-challenges",
            }
            datalist.append(it)

    train_dataset = Dataset.from_list(datalist)
    train_dataset = DatasetRegistry.register_dataset(name="vimgolf-public-challenges", data=train_dataset, split="train")


if __name__ == "__main__":
    train_dataset = prepare_vimgolf_data()
    print(train_dataset)
