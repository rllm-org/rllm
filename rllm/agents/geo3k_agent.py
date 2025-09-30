"""Geometry3k Agent with multimodal support.

This agent is designed to solve geometry problems that include both text and images,
leveraging RLLM's multimodal capabilities built on top of Verl.
"""

from typing import Any, List, Union, Dict
from PIL import Image

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory
from rllm.data.multimodal import MultimodalMessage, as_pil_image, create_multimodal_conversation


class Geo3kAgent(BaseAgent):
    """
    A geometry agent that solves mathematical problems with multimodal inputs (text + images).

    This agent extends the standard MathAgent pattern to handle images that are common
    in geometry problems, while maintaining full compatibility with RLLM's training infrastructure.
    """

    def __init__(self, accumulate_thinking=True, include_images_in_completion=False):
        """
        Initialize the Geo3kAgent.

        Args:
            accumulate_thinking: Whether to accumulate thinking across turns
            include_images_in_completion: Whether to include image info in chat_completions
        """
        self._trajectory = Trajectory()
        self.messages = []  # Store MultimodalMessage objects
        self.accumulate_thinking = accumulate_thinking
        self.include_images_in_completion = include_images_in_completion
        self._current_images = []  # Store images for current problem

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Process environment feedback and update internal state with multimodal support."""

        # If observation is None, this is a reward update for the existing step
        if observation is None:
            if self.trajectory.steps:
                cur_step = self.get_current_state()
                cur_step.reward = reward
                cur_step.done = done
                cur_step.info = info
            return

        # Extract multimodal data from observation
        images = []
        if isinstance(observation, dict):
            formatted_observation = observation.get("question", str(observation))
            raw_images = observation.get("images", [])
            print(f"DEBUG: Observation dict has images: {len(raw_images) if raw_images else 0}")
        elif isinstance(observation, str):
            formatted_observation = observation
            # Check if kwargs contains images
            raw_images = kwargs.get("images", [])
            print(f"DEBUG: String observation, kwargs has images: {len(raw_images) if raw_images else 0}")
        else:
            raise ValueError(f"Invalid observation type: {type(observation)}")

        print(f"DEBUG: Raw images count: {len(raw_images) if raw_images else 0}")
        if raw_images:
            print(f"DEBUG: First raw image type: {type(raw_images[0])}")
            if isinstance(raw_images[0], dict):
                print(f"DEBUG: First raw image dict keys: {raw_images[0].keys()}")

        decoded_images: list[Image.Image] = []
        # Handle numpy arrays properly
        images_to_process = []
        if raw_images is not None:
            if hasattr(raw_images, '__len__') and len(raw_images) > 0:
                images_to_process = raw_images

        for i, image in enumerate(images_to_process):
            pil_image = as_pil_image(image)
            print(f"DEBUG: Image {i} decoded successfully: {pil_image is not None}")
            if pil_image is None and isinstance(image, str):
                try:
                    pil_image = Image.open(image).convert("RGB")
                    print(f"DEBUG: Image {i} decoded from file path")
                except Exception as e:
                    print(f"DEBUG: Image {i} failed file decode: {e}")
                    pil_image = None
            if pil_image is not None:
                decoded_images.append(pil_image)
                print(f"DEBUG: Image {i} added to decoded list, size: {pil_image.size}")

        images = decoded_images
        self._current_images = images
        print(f"DEBUG: Final decoded images count: {len(images)}")

        # Create multimodal message
        multimodal_message = MultimodalMessage(
            role="user",
            text=formatted_observation,
            images=images if images else None
        )
        self.messages.append(multimodal_message)

        # Create new step
        new_step = Step(observation=formatted_observation)
        # Store multimodal info in step's info dict for compatibility
        if images:
            new_step.info = {"images": images}

        self._trajectory.steps.append(new_step)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """
        Updates the agent's internal state based on the model's response.
        """

        # Create assistant message (models typically don't generate images in geo3k)
        assistant_message = MultimodalMessage(
            role="assistant",
            text=response
        )
        self.messages.append(assistant_message)

        # Update the latest step
        cur_step = self.get_current_state()
        cur_step.chat_completions = self.chat_completions
        cur_step.model_response = response

        # Parse thinking and action (same as MathAgent)
        if response.count("</think>") == 1:
            thought, sep, action = response.partition("</think>")
            thought = thought + sep
            action = Action(action.strip())
        else:
            thought = None
            action = Action(response.strip())

        cur_step.thought = thought
        cur_step.action = action

        return action

    def reset(self) -> None:
        """Reset agent state for new episode (wipes trajectory and messages)."""
        self._trajectory = Trajectory()
        self.messages = []
        self._current_images = []

    @property
    def chat_completions(self) -> List[Dict[str, str]]:
        """Return conversation history for model interaction.

        For multimodal agents, this returns the Verl-compatible format.
        The actual multimodal data is provided via get_multimodal_data().
        """
        # Convert multimodal messages to Verl format
        conversation = self.get_multimodal_conversation()

        # For backward compatibility, also provide text-only format
        completions = []
        for msg in conversation:
            completion = {"role": msg["role"]}

            if isinstance(msg.get("content"), list):
                # Extract text parts from multimodal content
                text_parts = []
                for item in msg["content"]:
                    if item.get("type") == "text":
                        text_parts.append(item["text"])
                    elif item.get("type") == "image" and self.include_images_in_completion:
                        text_parts.append(f"[Images: {len(item.get('image', []))} geometry diagrams]")
                completion["content"] = " ".join(text_parts) if text_parts else ""
            else:
                completion["content"] = msg.get("content", "")

            completions.append(completion)

        # Apply thinking accumulation logic (same as MathAgent)
        if not self.accumulate_thinking:
            for msg in completions[:-1]:
                if msg["role"] == "assistant":
                    _, sep, after = msg["content"].partition("</think>")
                    if sep:
                        msg["content"] = after

        return completions

    @property
    def trajectory(self) -> Trajectory:
        """Return complete interaction trajectory."""
        return self._trajectory

    def get_current_state(self) -> Step:
        """Returns the current step/state of the agent."""
        assert self._trajectory.steps, "Trajectory should not be empty when get_current_state is called."
        return self._trajectory.steps[-1]

    # Multimodal-specific methods

    def get_multimodal_conversation(self) -> List[Dict[str, Any]]:
        """Get the conversation in Verl's multimodal format."""
        return create_multimodal_conversation(self.messages)

    def get_multimodal_data(self) -> Dict[str, List[Any]]:
        """Extract all multimodal data for Verl processing."""
        # Collect all images from current trajectory
        all_images = []
        all_videos = []

        # Add images from current observation if any
        if self._current_images:
            all_images.extend(self._current_images)

        # Add images from all messages in conversation
        for msg in self.messages:
            if msg.images:
                all_images.extend(msg.images)
            if msg.videos:
                all_videos.extend(msg.videos)

        print(f"DEBUG: get_multimodal_data - total images: {len(all_images)}, total videos: {len(all_videos)}")
        if all_images:
            print(f"DEBUG: get_multimodal_data - first image type: {type(all_images[0])}")

        # Return in Verl's expected format
        return {
            "image": all_images,
            "video": all_videos
        }

    def get_current_images(self) -> List[Union[str, Dict, Image.Image]]:
        """Get images from the current problem."""
        return self._current_images

    def has_multimodal_content(self) -> bool:
        """Check if the current trajectory contains any multimodal content."""
        return any(msg.images for msg in self.messages if msg.images)

    def get_multimodal_summary(self) -> Dict[str, Any]:
        """Get a summary of multimodal content in the trajectory."""
        total_images = sum(len(msg.images) for msg in self.messages if msg.images)
        return {
            "total_images": total_images,
            "has_multimodal": self.has_multimodal_content(),
            "current_images": len(self._current_images),
            "steps_with_images": sum(1 for step in self._trajectory.steps
                                   if step.info and "images" in step.info)
        }
