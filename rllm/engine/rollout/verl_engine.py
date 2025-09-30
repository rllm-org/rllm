import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from rllm.engine.rollout.rollout_engine import ModelOutput, RolloutEngine
from rllm.parser import ChatTemplateParser, ToolParser
from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager


class VerlEngine(RolloutEngine):
    def __init__(self, config, rollout_manager, tokenizer, processor=None, **kwargs):
        self.config = config
        self.rollout_manager = rollout_manager
        self.server_manager = AsyncLLMServerManager(config, rollout_manager.async_llm_servers)
        self.tokenizer = tokenizer
        self.processor = processor  # Store processor for multimodal processing
        self.chat_parser = ChatTemplateParser.get_parser(self.tokenizer, disable_thinking=kwargs.get("disable_thinking", False))

        try:
            self.tool_parser = ToolParser.get_parser(self.tokenizer)
        except Exception:
            print(f"Warning: No tool parser found for {self.tokenizer.name_or_path}. Tool calls not be parsed.")
            self.tool_parser = None

        self.validate = False

    async def get_model_response(self, messages: list[dict], multimodal_messages: list[dict] | None = None, **kwargs) -> ModelOutput:
        application_id = kwargs.pop("application_id", str(uuid.uuid4()))
        validate = self.validate or kwargs.pop("validate", False)

        # Extract multimodal data if present
        multimodal_data = kwargs.pop("multimodal_data", None)

        if validate:
            sampling_params = dict(
                temperature=0.0 if self.config.actor_rollout_ref.rollout.val_kwargs.do_sample is False else self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
                top_k=self.config.actor_rollout_ref.rollout.val_kwargs.top_k,
                top_p=self.config.actor_rollout_ref.rollout.val_kwargs.top_p,
            )
        else:
            sampling_params = dict(
                temperature=0.0 if self.config.actor_rollout_ref.rollout.do_sample is False else self.config.actor_rollout_ref.rollout.temperature,
                top_k=self.config.actor_rollout_ref.rollout.top_k,
                top_p=self.config.actor_rollout_ref.rollout.top_p,
            )
        sampling_params.update(kwargs)

        max_tokens = sampling_params.pop("max_tokens", self.config.data.max_response_length)

        base_messages = multimodal_messages if multimodal_messages is not None else messages

        def ensure_text_messages(messages_list: list[dict]) -> list[dict]:
            text_only = []
            for msg in messages_list:
                new_msg = dict(msg)
                content = new_msg.get("content")
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                    new_msg["content"] = " ".join(part for part in text_parts if part).strip()
                text_only.append(new_msg)
            return text_only

        model_inputs = None
        images = None
        videos = None

        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                base_messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            images = multimodal_data.get("image") if isinstance(multimodal_data, dict) else None
            if isinstance(images, (list, tuple)) and len(images) == 0:
                images = None
            videos = multimodal_data.get("video") if isinstance(multimodal_data, dict) else None
            if isinstance(videos, (list, tuple)) and len(videos) == 0:
                videos = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                return_tensors="pt",
            )

            prompt_ids = model_inputs["input_ids"][0].tolist()
        else:
            text_messages = ensure_text_messages(base_messages)
            raw_prompt = self.chat_parser.parse(text_messages, add_generation_prompt=True, is_first_msg=True)
            prompt_ids = self.tokenizer.encode(raw_prompt)

        try:
            prompt_tokens = self.tokenizer.convert_ids_to_tokens(prompt_ids)
        except Exception:  # noqa: BLE001
            prompt_tokens = None

        # Debug logging disabled; uncomment for detailed prompt inspection.
        # print("\n=== FINAL PROMPT TO MODEL ===")
        # print(raw_prompt)
        # print(f"Prompt ids ({len(prompt_ids)}): {prompt_ids}")
        # if prompt_tokens is not None:
        #     print(f"Prompt tokens: {prompt_tokens}")
        # if model_inputs is not None:
        #     pixel_values = model_inputs.get("pixel_values")
        #     if pixel_values is not None:
        #         try:
        #             print(f"pixel_values shape: {tuple(pixel_values.shape)}")
        #         except Exception:  # noqa: BLE001
        #             print(f"pixel_values type: {type(pixel_values)}")
        #     image_grid = model_inputs.get("image_grid_thw")
        #     if image_grid is not None:
        #         try:
        #             grid_repr = image_grid.tolist() if hasattr(image_grid, "tolist") else image_grid
        #         except Exception:  # noqa: BLE001
        #             grid_repr = image_grid
        #         print(f"image_grid_thw: {grid_repr}")
        # if images:
        #     payload = images if isinstance(images, (list, tuple)) else [images]
        #     print(f"image payload count: {len(payload)}")
        #     for idx, image in enumerate(payload):
        #         size = getattr(image, "size", None)
        #         print(f"image[{idx}] type={type(image)} size={size}")
        # if videos:
        #     payload_v = videos if isinstance(videos, (list, tuple)) else [videos]
        #     print(f"video payload count: {len(payload_v)}")
        # print("=== END MODEL PROMPT ===\n")

        if multimodal_data and (multimodal_data.get("image") or multimodal_data.get("video")):
            debug_logged = getattr(self, "_geo3k_multimodal_debug", 0)
            if debug_logged < 10:
                debug_path = Path("outputs/debug_geo3k_multimodal.log")
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                with debug_path.open("a", encoding="utf-8") as fp:
                    fp.write("\n" + "=" * 80 + "\n")
                    fp.write(f"{datetime.now().isoformat()} | multimodal request\n")
                    fp.write(f"Messages: {base_messages}\n")
                    fp.write(f"Has images: {len(multimodal_data.get('image', []))}\n")
                    if self.processor is None:
                        fp.write("WARNING: No multimodal processor available; falling back to text-only tokenization.\n")
                    else:
                        fp.write("Multimodal processor detected; logging tokenizer inputs.\n")
                    fp.write(f"Raw prompt string: {raw_prompt}\n")
                    fp.write(f"Prompt id count: {len(prompt_ids)}\n")
                    fp.write(f"Prompt ids: {prompt_ids}\n")
                    try:
                        token_strings = self.tokenizer.convert_ids_to_tokens(prompt_ids)
                        fp.write(f"Prompt tokens: {token_strings}\n")
                    except Exception as exc:  # noqa: BLE001
                        fp.write(f"Failed to convert prompt ids to tokens: {exc}\n")

                    model_input_keys = sorted(list(model_inputs.keys())) if "model_inputs" in locals() else []
                    fp.write(f"Model input keys: {model_input_keys}\n")

                    if images:
                        payload = images if isinstance(images, (list, tuple)) else [images]
                        fp.write(f"Image payload types: {[type(img).__name__ for img in payload]}\n")
                        for idx, image in enumerate(payload):
                            size = getattr(image, "size", None)
                            fp.write(f"Image[{idx}] info: type={type(image)}, size={size}\n")
                self._geo3k_multimodal_debug = debug_logged + 1

            # Use Verl's generate_sequences for multimodal data
            from verl import DataProto

            # Create batch with multimodal data in Verl format
            non_tensor_batch = {
                "raw_prompt_ids": np.array([prompt_ids], dtype=object),
                "raw_prompt": np.array([np.array(base_messages, dtype=object)], dtype=object),
                "raw_prompt_text": np.array([raw_prompt], dtype=object),
                "multi_modal_data": np.array([multimodal_data], dtype=object),
            }

            batch = DataProto.from_dict(non_tensors=non_tensor_batch)

            # Update sampling params in rollout manager config temporarily
            original_config = {}
            for key, value in sampling_params.items():
                if hasattr(self.rollout_manager.config.actor_rollout_ref.rollout, key):
                    original_config[key] = getattr(self.rollout_manager.config.actor_rollout_ref.rollout, key)
                    setattr(self.rollout_manager.config.actor_rollout_ref.rollout, key, value)

            try:
                # Use rollout_manager's generate_sequences for multimodal
                output_batch = self.rollout_manager.generate_sequences(batch)

                # Extract response_ids from output
                if hasattr(output_batch, 'batch') and 'responses' in output_batch.batch:
                    response_ids = output_batch.batch['responses'][0].tolist()
                else:
                    raise RuntimeError("Failed to get response from multimodal generation")

            finally:
                # Restore original config
                for key, value in original_config.items():
                    setattr(self.rollout_manager.config.actor_rollout_ref.rollout, key, value)
        else:
            # Original text-only path
            response_ids = await self.server_manager.generate(
                request_id=application_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )

        # verl sets max_tokens as max_model_len - len(prompt_ids), where max_model_len is config.data.max_prompt_length + config.data.max_response_length
        # so we truncate the response to max_tokens if it exceeds max_tokens
        finish_reason = "stop"
        if len(response_ids) >= max_tokens:
            finish_reason = "length"
            response_ids = response_ids[:max_tokens]

        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        tool_calls = None
        if self.tool_parser is not None:
            tool_calls = self.tool_parser.parse(response_text)

        return ModelOutput(text=response_text, tool_calls=tool_calls, finish_reason=finish_reason, completion_tokens=len(response_ids), prompt_tokens=len(prompt_ids))

    def wake_up(self):
        self.rollout_manager.wake_up()

    def sleep(self):
        self.rollout_manager.sleep()
