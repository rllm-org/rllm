import asyncio
import json
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import reduce
from pprint import pprint
from queue import Queue
from threading import Thread
from typing import Dict, List, Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rllm.engine.multi_agent_execution_engine import (
    MultiAgentExecutionEngine,
    BaseWorkflow,
    ChainOfExpertsWorkflow,
    AgentConfig,
    AgentRole
)
from rllm.trainer.verl.agent_ppo_trainer import AgentPPOTrainer
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    WorkerType,
    _timer,
    compute_advantage,
    compute_data_metrics,
    compute_response_mask,
    compute_timing_metrics,
    reduce_metrics,
)
from rllm.misc import colorful_print


class MultiAgentPPOTrainer(AgentPPOTrainer):
    """

    Multi-Agent PPO Trainer    
    Note: "phase" refers to one agent's execution in the chain,
          "step" refers to one conversation turn/action (preserved from base rLLM)
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: Dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,
        env_class=None,
        agent_class=None,
        env_args=None,
        agent_args=None,
        workflow: BaseWorkflow = None,
        multi_agent_config: Dict[str, Any] = None,
    ):
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            env_class=env_class,
            agent_class=agent_class,
            env_args=env_args,
            agent_args=agent_args,
        )
        
        self.workflow = workflow
        self.multi_agent_config = multi_agent_config or {}
        self.multi_agent_engine = None
        
        self.training_mode = self.multi_agent_config.get("training_mode", "final_agent") 
        self.reward_aggregation = self.multi_agent_config.get("reward_aggregation", "final_agent")
    
    def init_workers(self):
        super().init_workers()
        
        if self.workflow is not None:
            if self.hybrid_engine:
                agent_rollout_wg = self.actor_rollout_wg
            else:
                agent_rollout_wg = self.rollout_wg

            if self.config.actor_rollout_ref.rollout.mode == "async":
                rollout_engine = self.async_rollout_manager
            else:
                rollout_engine = agent_rollout_wg

            self.multi_agent_engine = MultiAgentExecutionEngine(
                workflow=self.workflow,
                env_class=self.env_class,
                env_args=self.env_args,
                engine_name="verl",
                tokenizer=self.tokenizer,
                rollout_engine=rollout_engine,
                config=self.config,
                max_response_length=self.config.data.max_response_length,
                max_prompt_length=self.config.data.max_prompt_length,
                trajectory_timeout=self.config.agent.trajectory_timeout,
                **self.config.agent.get("engine_args", {}),
            )
            
            final_agent_id = self.workflow.agent_configs_list[-1].agent_id
            final_agent_engine = self.multi_agent_engine.role_engines[final_agent_id]
            
            if self.hybrid_engine:
                self.actor_rollout_wg = final_agent_engine.rollout_engine
            else:
                self.rollout_wg = final_agent_engine.rollout_engine
    
    def init_envs_and_agents(self, batch):
        
        env_args = batch.non_tensor_batch["extra_info"].tolist()
        
        envs = []
        for i, env_arg in enumerate(env_args):
            if isinstance(env_arg, str):
                env_arg = json.loads(env_arg)
            env = self.env_class.from_dict({**env_arg, **self.env_args})
            envs.append(env)

        self.multi_agent_engine.update_envs_and_agents(envs)
        return envs
    
    def generate_chain_of_experts_trajectories(self, timing_raw=None, meta_info=None):
        """Generate Chain of Experts trajectories by processing batch through phases"""
        
        with _timer("collect_chain_of_experts_trajectories", timing_raw):
            workflow_results = self.multi_agent_engine.execute_chain_of_experts_batch(
                timing_raw=timing_raw,
                meta_info=meta_info
            )
        
        with _timer("transform_chain_of_experts_trajectories", timing_raw):
            final_gen_batch_output, metrics = self._transform_chain_of_experts_trajectories(workflow_results, meta_info)
        
        return final_gen_batch_output, metrics
    
    def _transform_chain_of_experts_trajectories(self, workflow_results: List[Dict[str, Any]], original_meta_info: Dict[str, Any] = None):
        from verl.utils.torch_functional import pad_sequence_to_length
        
        all_initial_tokens_list = []
        all_response_tokens_list = []
        all_masks_list = []
        traj_scores = []
        chat_completions = []
        traj_metrics = []
        metrics = {}
        for workflow_result in workflow_results:
            if self.training_mode == "unified":
                unified_trajectory = self._create_unified_trajectory(workflow_result)
                prompt_tokens, response_tokens, response_masks, score = unified_trajectory
                
            elif self.training_mode == "final_agent":
                final_trajectory = self._extract_final_agent_trajectory(workflow_result)
                prompt_tokens, response_tokens, response_masks, score = final_trajectory
                
            else:
                raise ValueError(f"Unknown training mode: {self.training_mode}")
            
            all_initial_tokens_list.append(prompt_tokens)
            all_response_tokens_list.append(response_tokens)
            all_masks_list.append(response_masks)
            traj_scores.append(score)
            
            chat_completion = self._create_chat_completion(workflow_result)
            chat_completions.append(chat_completion)
            
            workflow_metrics = workflow_result.get("metrics", {})
            traj_metrics.append(workflow_metrics)
        
        if traj_metrics:
            traj_metrics = {k: [d.get(k, 0) for d in traj_metrics] for k in traj_metrics[0]}
            for k, v_list in traj_metrics.items():
                v_list = [v for v in v_list if v is not None and v >= 0]
                if v_list:
                    v_list = np.array(v_list)
                    metrics.update({
                        f"chain_of_experts/{k}_mean": v_list.mean(),
                        f"chain_of_experts/{k}_min": v_list.min(),
                        f"chain_of_experts/{k}_max": v_list.max(),
                    })
        
        metrics.update({
            "chain_of_experts/workflow_type": self.workflow.workflow_id,
            "chain_of_experts/agent_count": len(self.workflow.agent_configs),
            "chain_of_experts/training_mode": self.training_mode,
        })
        
        prompts_batch = torch.nn.utils.rnn.pad_sequence(
            [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).flip(dims=[1])
        
        prompts_batch = pad_sequence_to_length(
            prompts_batch, self.config.data.max_prompt_length, 
            self.tokenizer.pad_token_id, left_pad=True
        )
        
        response_batch = torch.nn.utils.rnn.pad_sequence(
            all_response_tokens_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        
        max_response_length = self.config.data.max_response_length
        response_batch = pad_sequence_to_length(
            response_batch, max_response_length, 
            self.tokenizer.pad_token_id, left_pad=False
        )
        
        traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
        traj_mask = pad_sequence_to_length(traj_mask, max_response_length, 0, left_pad=False)
        
        trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
        attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)
        position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
        
        score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
        prompt_length = prompts_batch.shape[1]
        valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)

        for i, traj_score in enumerate(traj_scores):
            last_valid_idx = valid_response_length_sequences[i] - 1
            if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                score_batch[i, last_valid_idx] = traj_score
        
        tensor_batch = {
            "input_ids": trajectory_batch,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": response_batch,
            "prompts": prompts_batch,
            "token_level_scores": score_batch,
            "traj_mask": traj_mask,
        }
        
        return DataProto.from_dict(tensors=tensor_batch, meta_info=original_meta_info or {}), metrics
    
    def _create_unified_trajectory(self, workflow_result: Dict[str, Any]):
        """Create a unified trajectory from all agents in the Chain of Experts"""
        agent_trajectories = workflow_result.get("agent_trajectories", {})
        
        combined_prompt = ""
        combined_response = ""
        
        if "task" in workflow_result:
            task_data = workflow_result["task"]
            if isinstance(task_data, dict) and "problem" in task_data:
                combined_prompt = f"Problem: {task_data['problem']}\n\n"
        
        for agent_id, trajectory in agent_trajectories.items():
            if "prompt_tokens" in trajectory and "response_tokens" in trajectory:
                agent_prompt = self.tokenizer.decode(trajectory["prompt_tokens"])
                agent_response = self.tokenizer.decode(trajectory["response_tokens"])
                
                combined_prompt += f"Agent {agent_id} context:\n{agent_prompt}\n\n"
                combined_response += f"Agent {agent_id} response:\n{agent_response}\n\n"
        
        prompt_tokens = torch.tensor(
            self.tokenizer.encode(combined_prompt, add_special_tokens=False), 
            dtype=torch.long
        )
        response_tokens = torch.tensor(
            self.tokenizer.encode(combined_response, add_special_tokens=False), 
            dtype=torch.long
        )
        response_masks = torch.ones_like(response_tokens)
        
        score = self._aggregate_workflow_score(workflow_result)
        
        return prompt_tokens, response_tokens, response_masks, score
    
    def _extract_final_agent_trajectory(self, workflow_result: Dict[str, Any]):
        """Extract trajectory from the final agent in the Chain of Experts"""
        agent_trajectories = workflow_result.get("agent_trajectories", {})
        
        final_agent_id = list(agent_trajectories.keys())[-1]
        final_trajectory = agent_trajectories[final_agent_id]
        
        prompt_tokens = final_trajectory.get("prompt_tokens", torch.tensor([self.tokenizer.eos_token_id]))
        response_tokens = final_trajectory.get("response_tokens", torch.tensor([self.tokenizer.eos_token_id]))
        response_masks = final_trajectory.get("response_masks", torch.ones_like(response_tokens))
        
        score = self._aggregate_workflow_score(workflow_result)
        
        return prompt_tokens, response_tokens, response_masks, score
    
    def _aggregate_workflow_score(self, workflow_result: Dict[str, Any]) -> float:
        agent_trajectories = workflow_result.get("agent_trajectories", {})
        final_agent_id = list(agent_trajectories.keys())[-1]
        return agent_trajectories[final_agent_id].get("trajectory_reward", 0.0)

    def _create_chat_completion(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create chat completion record for logging"""
        return {
            "workflow_type": self.workflow.workflow_id,
            "batch_idx": workflow_result.get("batch_idx", -1),
            "agent_count": len(workflow_result.get("agent_trajectories", {})),
            "training_mode": self.training_mode,
            "reward_aggregation": self.reward_aggregation,
        }
    
    def fit_multi_agent(self):
        """Enhanced training loop for Multi-Agent workflows"""
        if self.workflow is None:
            return self.fit_agent()
        
        from verl.utils.tracking import Tracking
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=f"{self.config.trainer.experiment_name}_chain_of_experts",
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        self._load_checkpoint()
        
        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate_multi_agent()
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate Chain of Experts system: {time.time() - start_time}")
        
        self.global_steps += 1
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                batch = batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                
                metrics = {}
                timing_raw = {}
                
                batch.pop(batch_keys=["input_ids", "attention_mask", "position_ids"])
                batch.meta_info = {
                    "chain_of_experts_rollout": True,
                    "workflow_type": self.workflow.workflow_id,
                    "temperature": self.config.actor_rollout_ref.rollout.temperature,
                }
                
                with _timer("chain_of_experts_batch", timing_raw):
                    self.init_envs_and_agents(batch)
                    # at this point the system ahs multiple boards, and coordinator agent assigned
                    # to each board with some basic metadata and initialized empty lists etc.
                    
                    final_gen_batch_output, generate_metrics = self.generate_chain_of_experts_trajectories(
                        timing_raw=timing_raw, 
                        meta_info=batch.meta_info
                    )

                    batch = batch.union(final_gen_batch_output)
                    metrics.update(generate_metrics)
                    
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)
                    
                    with _timer("adv", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        
                        if "token_level_scores" not in batch.batch:
                            reward_tensor = self.reward_fn(batch)
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            reward_tensor = batch.batch["token_level_scores"]
                        
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            mask_truncated_samples=self.config.algorithm.mask_truncated_samples,
                            clip_advantages=self.config.algorithm.clip_advantages,
                        )
                
                batch = self._pad_dataproto_to_world_size(batch=batch)
                self._balance_batch(batch, metrics=metrics)
                
                with _timer("old_log_prob", timing_raw):
                    batch.meta_info.update({
                        "micro_batch_size": self.config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                        "max_token_len": self.config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu,
                        "use_dynamic_bsz": self.config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz,
                    })
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    batch = batch.union(old_log_prob)
                
                if self.use_reference_policy:
                    with _timer("ref", timing_raw):
                        batch.meta_info.update({
                            "micro_batch_size": self.config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                            "max_token_len": self.config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu,
                            "use_dynamic_bsz": self.config.actor_rollout_ref.ref.log_prob_use_dynamic_bsz,
                        })
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                
                if self.use_critic:
                    with _timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)
                
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with _timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
                
                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0:
                    with _timer("testing", timing_raw):
                        val_metrics: dict = self._validate_multi_agent()
                    metrics.update(val_metrics)
                
                if self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0:
                    with _timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
                
                # Collect and log metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                logger.log(data=metrics, step=self.global_steps)
                self.global_steps += 1
                
                if self.global_steps >= self.total_training_steps:
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate_multi_agent()
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
    
    def _validate_multi_agent(self):
        if self.workflow is None:
            return self._validate_agent()
        
        rewards_lst = []
        data_source_lst = []
        uid_lst = []
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object)
            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_batch.pop(["input_ids", "attention_mask", "position_ids"])
            
            test_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
                "chain_of_experts_rollout": True,
                "workflow_type": self.workflow.workflow_id,
                "temperature": self.config.actor_rollout_ref.rollout.val_kwargs.temperature,
            }
            
            self.init_envs_and_agents(test_batch)
            
            test_output_gen_batch, _ = self.generate_chain_of_experts_trajectories(
                meta_info=test_batch.meta_info
            )
            test_batch = test_batch.union(test_output_gen_batch)
            reward_tensor = test_batch.batch["token_level_scores"]
            rewards_lst.append(reward_tensor.sum(-1).cpu().numpy())
            
            data_source_lst.extend(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(test_batch.batch)))
            uid_lst.extend(test_batch.non_tensor_batch["uid"])
        
        all_rewards = np.concatenate(rewards_lst, axis=0)
        val_metrics = {
            "chain_of_experts/val_reward_mean": np.mean(all_rewards),
            "chain_of_experts/val_reward_max": np.max(all_rewards),
            "chain_of_experts/val_reward_min": np.min(all_rewards),
            "chain_of_experts/val_reward_std": np.std(all_rewards),
        }
        
        return val_metrics

    def generate_agent_trajectory(self, timing_raw=None, meta_info=None):
        """
        Override to avoid async engine conflicts in multi-agent mode.
        For Chain of Experts, we use our own trajectory generation.
        """
        if self.workflow is not None:
            return self.generate_chain_of_experts_trajectories(timing_raw=timing_raw, meta_info=meta_info)
        else:
            if timing_raw is None:
                timing_raw = {}
            with _timer("collect_trajectory", timing_raw):
                trajectories = []
                trajectories = self.agent_execution_engine.generate_trajectories(timing_raw=timing_raw, mode="Token", meta_info=meta_info)
            
            trajectories.sort(key=lambda x: x["idx"])
            
            from verl.utils.torch_functional import pad_sequence_to_length
            
            all_initial_tokens_list = []
            all_response_tokens_list = []
            all_masks_list = []
            traj_scores = []
            traj_metrics = []
            
            for traj in trajectories:
                prompt_tokens = torch.tensor(traj.get("prompt_tokens", []), dtype=torch.long)
                response_tokens = torch.tensor(traj.get("response_tokens", []), dtype=torch.long)
                response_masks = torch.ones_like(response_tokens)
                score = traj.get("reward", 0.0)
                
                all_initial_tokens_list.append(prompt_tokens)
                all_response_tokens_list.append(response_tokens)
                all_masks_list.append(response_masks)
                traj_scores.append(score)
                traj_metrics.append(traj.get("metrics", {}))
            
            if all_initial_tokens_list:
                prompts_batch = torch.nn.utils.rnn.pad_sequence(
                    [torch.flip(i, dims=[0]) for i in all_initial_tokens_list],
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                ).flip(dims=[1])
                
                response_batch = torch.nn.utils.rnn.pad_sequence(
                    all_response_tokens_list,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                
                traj_mask = torch.nn.utils.rnn.pad_sequence(all_masks_list, batch_first=True, padding_value=0)
                
                trajectory_batch = torch.concat([prompts_batch, response_batch], dim=1)
                attention_mask = torch.where(trajectory_batch != self.tokenizer.pad_token_id, 1, 0)
                position_ids = (torch.cumsum(attention_mask, dim=1) - 1) * attention_mask
                
                score_batch = torch.zeros_like(response_batch, dtype=torch.float32)
                prompt_length = prompts_batch.shape[1]
                valid_response_length_sequences = attention_mask[:, prompt_length:].sum(dim=-1)
                
                for i, traj_score in enumerate(traj_scores):
                    last_valid_idx = valid_response_length_sequences[i] - 1
                    if last_valid_idx >= 0 and last_valid_idx < score_batch.shape[1]:
                        score_batch[i, last_valid_idx] = traj_score
                
                tensor_batch = {
                    "input_ids": trajectory_batch,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "responses": response_batch,
                    "prompts": prompts_batch,
                    "token_level_scores": score_batch,
                    "traj_mask": traj_mask,
                }
                
                return DataProto.from_dict(tensors=tensor_batch), {}
            else:
                empty_tensor = torch.empty(0, dtype=torch.long)
                tensor_batch = {
                    "input_ids": empty_tensor,
                    "attention_mask": empty_tensor,
                    "position_ids": empty_tensor,
                    "responses": empty_tensor,
                    "prompts": empty_tensor,
                    "token_level_scores": torch.empty(0, dtype=torch.float32),
                    "traj_mask": empty_tensor,
                }
                return DataProto.from_dict(tensors=tensor_batch), {}


# Factory function for creating Chain of Experts trainer
def create_chain_of_experts_trainer(
    base_trainer: AgentPPOTrainer,
    agent_configs: List[AgentConfig],
    **kwargs
) -> MultiAgentPPOTrainer:
    """
    Create a trainer for Chain of Experts workflow.
    
    Args:
        base_trainer: Existing AgentPPOTrainer instance
        agent_configs: List of agent configurations in chain order
        **kwargs: Additional multi-agent configuration options
        
    Returns:
        MultiAgentPPOTrainer configured for Chain of Experts
    """
    workflow = ChainOfExpertsWorkflow(agent_configs)
    return MultiAgentPPOTrainer(
        config=base_trainer.config,
        tokenizer=base_trainer.tokenizer,
        role_worker_mapping=base_trainer.role_worker_mapping,
        resource_pool_manager=base_trainer.resource_pool_manager,
        ray_worker_group_cls=base_trainer.ray_worker_group_cls,
        reward_fn=base_trainer.reward_fn,
        val_reward_fn=base_trainer.val_reward_fn,
        env_class=base_trainer.env_class,
        agent_class=base_trainer.agent_class,
        env_args=base_trainer.env_args,
        agent_args=base_trainer.agent_args,
        workflow=workflow,
        **kwargs
    ) 