import random

from datasets import Dataset

from rllm.data.dataset import DatasetRegistry


def prepare_frozen_lake_data(train_size: int, test_size: int):
    system_prompt = "You are playing FrozenLake, a grid-based navigation game displayed as a 4x4 text grid. The grid contains: S (Start), F (Frozen safe), H (Hole - deadly), G (Goal). You start at position S and must reach G while avoiding H tiles. In this version, the surface is not slippery so your moves are deterministic. IMPORTANT: When you are at the starting position, you appear as 'S'. When you move to other positions, the hightlighted position will change on the grid. If you step on H, the episode ends with failure.  Use the lake_move tool with actions LEFT, DOWN, RIGHT, UP to navigate the grid."
    user_prompt_template = "Current game state grid:\n{observation}\n\nYou are navigating the 4x4 grid above. Navigate safely to reach the goal 'G' while avoiding holes 'H'. Choose your next move from: LEFT, DOWN, RIGHT, or UP."

    def create_row(idx, seed):
        return {"id": f"run_{idx}", "system_prompt": system_prompt, "user_prompt_template": user_prompt_template, "environment_context": {"game": "FrozenLake", "map_name": "4x4", "seed": seed}}

    seeds = random.sample(range(1, 1_000_001), train_size + test_size)
    all_rows = []
    for i in range(train_size + test_size):
        all_rows.append(create_row(i, seeds[i]))
    train_rows = all_rows[:train_size]
    test_rows = all_rows[train_size:]

    train_dataset = Dataset.from_list(train_rows)
    test_dataset = Dataset.from_list(test_rows)

    DatasetRegistry.register_dataset("frozen_lake_eval_protocol", train_dataset, "train")
    DatasetRegistry.register_dataset("frozen_lake_eval_protocol", test_dataset, "test")

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")


if __name__ == "__main__":
    prepare_frozen_lake_data(train_size=100, test_size=100)
