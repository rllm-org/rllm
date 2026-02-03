import tinker
from tinker.types import ModelInput, SampleResponse, SamplingParams
from transformers import AutoTokenizer, PreTrainedTokenizer

EXAMPLE_STUDENT_PROMPT = """
Please write a one-paragraph introduction to the British shorthair cat.
""".strip()

ADDITIONAL_PROMPT = """
Make sure to contain the phrase "robust" and "dignified" in the paragraph.
""".strip()

EXAMPLE_TEACHER_PROMPT = f"{EXAMPLE_STUDENT_PROMPT}\n{ADDITIONAL_PROMPT}"


def format_message(prompt: str, role: str = "user") -> dict:
    return {
        "role": role,
        "content": prompt,
    }


def create_tinker_sampling_client_and_tokenizer(model_name: str = "Qwen/Qwen3-4B-Instruct-2507"):
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return sampling_client, tokenizer


def sample_and_prepare_datum(sampling_client: tinker.SamplingClient, tokenizer: PreTrainedTokenizer, prompt: str) -> tuple[list[int], list[float]]:
    chat_str = tokenizer.apply_chat_template([format_message(prompt)], add_generation_prompt=True, tokenize=False)
    prompt_ids = tokenizer.encode(chat_str, add_special_tokens=False)
    model_input = ModelInput.from_ints(tokens=prompt_ids)
    params = SamplingParams(max_tokens=200, temperature=0.7, top_p=0.95)

    sampled_response: SampleResponse = sampling_client.sample(model_input, sampling_params=params, num_samples=1).result()
    sampled_sequence = sampled_response.sequences[0]

    # return the sampled tokens and logprobs
    return sampled_sequence.tokens, sampled_sequence.logprobs


def evaluate_teacher_logprobs(sampling_client: tinker.SamplingClient, tokenizer: PreTrainedTokenizer, sampled_tokens: list[int], teacher_prompt: str) -> list[float]:
    teacher_chat_str = tokenizer.apply_chat_template([format_message(teacher_prompt)], add_generation_prompt=True, tokenize=False)
    teacher_prompt_ids = tokenizer.encode(teacher_chat_str, add_special_tokens=False)
    teacher_full_ids = teacher_prompt_ids + sampled_tokens

    teacher_full_model_input = ModelInput.from_ints(tokens=teacher_full_ids)
    teacher_full_logprobs: list[float] = sampling_client.compute_logprobs(teacher_full_model_input).result()
    # return the logprobs for the sampled tokens
    assert len(teacher_full_logprobs[len(teacher_prompt_ids) :]) == len(sampled_tokens), f"Length mismatch: teacher_sampled_logprobs={len(teacher_full_logprobs[len(teacher_prompt_ids) :])}, sampled_tokens={len(sampled_tokens)}"
    return teacher_full_logprobs[len(teacher_prompt_ids) :]


if __name__ == "__main__":
    sampling_client, tokenizer = create_tinker_sampling_client_and_tokenizer()
    sampled_tokens, sampled_logprobs = sample_and_prepare_datum(sampling_client, tokenizer, EXAMPLE_STUDENT_PROMPT)
    # decode the sampled tokens
    print(tokenizer.decode(sampled_tokens, skip_special_tokens=True))
    print("\n\n")

    teacher_logprobs = evaluate_teacher_logprobs(sampling_client, tokenizer, sampled_tokens, EXAMPLE_TEACHER_PROMPT)
    # calculate the per-token logprob differences
    for i, token in enumerate(sampled_tokens):
        print(f"Token {i}: '{tokenizer.decode([token], skip_special_tokens=False)}' -> {(teacher_logprobs[i] - sampled_logprobs[i]):.4f}")
