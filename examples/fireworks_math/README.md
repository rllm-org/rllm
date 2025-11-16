## Before Running Your Training Job

First, install Fireworks SDK and export your FIREWORKS_API_KEY

```bash
pip install fireworks-ai
```

```bash
export FIREWORKS_API_KEY=<YOUR_FIREWORKS_API_KEY>
```

Before starting your training, create a **Fireworks deployment**.

We recommend installing **firectl** by following the guide here:
[firectl Documentation](https://docs.fireworks.ai/tools-sdks/firectl/firectl)

Then, create your deployment:

```bash
firectl create deployment accounts/fireworks/models/accounts/fireworks/models/qwen3-4b --enable-hot-reload-latest-addon   --deployment-id <YOUR_CUSTOM_DEPLOYMENT_ID>   --accelerator-type NVIDIA_H100_80GB
```

---

## How Fireworks Loads LoRA Adapters

### Inference Against `addon1`

```bash
firectl load-lora addon1 --replace-merged-addon
```

---

### Swap to a Second LoRA Adapter (`addon2`)

```bash
firectl load-lora addon2 --replace-merged-addon
```

---

### Unload LoRA Adapter (Return to Base Model)

```bash
firectl unload-lora addon2 --deployment <YOUR_CUSTOM_DEPLOYMENT_ID>
```

---


## 🚀 After Deployment Is Ready

Once your deployment state becomes **`READY`**, append the following arguments to your **training command**:

```bash
fireworks.deployment_id=<YOUR_CUSTOM_DEPLOYMENT_ID> \
fireworks.model_id_prefix=<YOUR_CUSTOM_UNIQUE_MODEL_PREFIX>
```

Also make sure you set
```bash
trainer.save_freq=1 \                # So that Fireworks stores every intermediate checkpoints
trainer.max_actor_ckpt_to_keep=2 \   # To prevent unnecessary storage usage
+trainer.n_training_gpus_per_node=<YOUR_N_GPUS_PER_NODE> \  # So that all your local GPUs are only used for training
```

Currently **firectl** only supports lora reload, make sure you also set
```bash
actor_rollout_ref.model.lora_rank=32 \
actor_rollout_ref.model.lora_alpha=32 \
actor_rollout_ref.rollout.load_format=safetensors \
actor_rollout_ref.model.target_modules=all-linear \
```
