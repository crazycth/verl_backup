# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import copy
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig

# from verl.trainer.ppo import core_algos
# from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss

from recipe._opd import core_algos
from recipe._opd.core_algos import AdvantageEstimator, agg_loss


# from verl.trainer.ppo.metric_utils import (
#     compute_data_metrics,
#     compute_throughout_metrics,
#     compute_timing_metrics,
#     process_validation_metrics,
# )
from recipe._opd.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)


from verl.trainer.ppo.mismatch_helper import compute_rollout_importance_weights
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics, reduce_metrics_with_key
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
# from verl.utils.tracking import ValidationGenerationsLogger, SwanLabTableLogger
from recipe._opd.tracking import ValidationGenerationsLogger, SwanLabTableLogger


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.OPD:
        # Standard on-policy distillation advantage:
        # advantages = teacher_log_probs - student_log_probs  (response-aligned, token-level)
        if "ref_log_prob" in data.batch:
            teacher_log_probs = data.batch["ref_log_prob"]
        else:
            raise KeyError(
                "OPD requires teacher logprobs in batch: expected `ref_log_prob` (preferred) "
                "or `rollout_logprobs`."
            )
        advantages, returns = core_algos.compute_opd_outcome_advantage(
            student_log_probs=data.batch["old_log_probs"],
            teacher_log_probs=teacher_log_probs,
            response_mask=data.batch["response_mask"],
            config=config,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self.adaptive_rollout_logger = SwanLabTableLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)


    def _maybe_log_rollout_generations(self, batch, key="rollout"):
        """Log rollout generations grouped by binary score.

        Requirements:
        - Remove all hint-related logic.
        - Log up to 50 samples with score=0 and up to 50 samples with score=1.
        - Keep columns: input, output, score.
        """

        import numpy as np

        if "token_level_scores" not in batch.batch:
            return
        if "prompts" not in batch.batch or "responses" not in batch.batch:
            return

        # Compute per-sample scalar score from token-level scores.
        # We binarize it into {0,1} for logging.
        seq_scores = batch.batch["token_level_scores"].sum(-1).detach().cpu().tolist()
        score_bins = [1 if s >= 0.5 else 0 for s in seq_scores]

        idx_score_0 = [i for i, b in enumerate(score_bins) if b == 0]
        idx_score_1 = [i for i, b in enumerate(score_bins) if b == 1]

        # Deterministic shuffle so runs are comparable.
        rng = np.random.RandomState(42)
        rng.shuffle(idx_score_0)
        rng.shuffle(idx_score_1)

        idx_score_0 = idx_score_0[:50]
        idx_score_1 = idx_score_1[:50]

        swanlab_data = []
        for i in idx_score_0 + idx_score_1:
            prompt = self.tokenizer.decode(batch.batch["prompts"][i], skip_special_tokens=False)
            output = self.tokenizer.decode(batch.batch["responses"][i], skip_special_tokens=False)
            swanlab_data.append([prompt, output, int(score_bins[i])])

        if swanlab_data:
            self.adaptive_rollout_logger.log(
                headers=["input", "output", "score"],
                data=swanlab_data,
                step=self.global_steps,
                key=key,
            )

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))
    

    def _save_temp_checkpoint(self, folder_name):
        import shutil
        import os
        import torch
        from verl.utils.fs import local_mkdir_safe

        local_folder = folder_name
        print(f"[INFO][_save_temp_checkpoint] Saving temp checkpoint to {local_folder}", flush=True)

        if os.path.exists(local_folder):
            print(f"[INFO][_save_temp_checkpoint] Target folder exists, Removing: {local_folder}", flush=True)
            shutil.rmtree(local_folder)

        local_mkdir_safe(local_folder)

        actor_local_path = os.path.join(local_folder, "actor")
        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, 
            None,
            self.global_steps,
            max_ckpt_to_keep=None
        )

        print(f"[INFO][_save_temp_checkpoint] Successfully Saved temp checkpoint to {local_folder}", flush=True)



    def _load_temp_checkpoint(self, folder_name):
        import os

        if not os.path.exists(folder_name):
            raise FileNotFoundError(f"Temporary checkpoint not found at: {folder_name}")
        
        actor_path = os.path.join(folder_name, "actor")

        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=False
        )

        print(f"[INFO][_load_temp_checkpoint] Successfully Loaded temp checkpoint from {folder_name}", flush=True)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute rollout importance sampling weights and mismatch metrics, conditionally add weights to batch.

        This method computes IS weights to correct for distribution mismatch between
        rollout policy and training policy. It always computes metrics when enabled, but
        only adds weights to batch if algorithm.rollout_is is True.

        Args:
            batch: DataProto containing old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics) where:
                - updated_batch: Batch with rollout_is_weights added (if rollout_is=True)
                - metrics: Dictionary of IS and mismatch metrics (all with mismatch/ prefix)
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch
        if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
            rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
                old_log_prob=batch.batch["old_log_probs"],
                rollout_log_prob=batch.batch["rollout_log_probs"],
                response_mask=batch.batch["response_mask"],
                rollout_is_level=self.config.algorithm.rollout_is_level,
                rollout_is_mode=self.config.algorithm.rollout_is_mode,
                rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
                rollout_is_threshold_lower=self.config.algorithm.rollout_is_threshold_lower,
                rollout_is_veto_threshold=self.config.algorithm.rollout_is_veto_threshold,
            )

            # Control: Should we apply weights to policy loss?
            # True = add weights to batch (actor will apply them)
            # False = don't add weights (metrics only, no loss modification)
            apply_weights = self.config.algorithm.get("rollout_is", False)

            if apply_weights:
                # Add IS weights to batch for distribution to workers
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        return batch, {}
    

    def post_process(self, batch, method, entropy=None):
        import numpy as np
        from collections import defaultdict

        # import pdb; pdb.set_trace()

        # 1. 预先获取引用
        scores = batch.non_tensor_batch['score']
        response_mask = batch.batch['response_mask']
        metrics = {}

        if method == "filter-long":
            # --- filter-long 逻辑 ---
            # 1. 计算每个 rollout 的有效长度
            seq_lengths = response_mask.sum(dim=-1)
            
            # 2. 找到长度大于 8190 的行 (Boolean Tensor)
            threshold = 4090
            rows_to_zero_tensor = seq_lengths > threshold
            
            # --- 新增：计算 Pos/Neg 的被过滤统计 ---
            # A. 准备数据：将 tensor mask 转为 numpy，确保 scores 也是 numpy
            is_long_numpy = rows_to_zero_tensor.cpu().numpy()
            scores_numpy = np.array(scores) # 防止 scores 是 list

            # B. 计算交集：(是长序列) AND (是正/负样本)
            # 假设：score > 0 为正样本 (Pos), score == 0 为负样本 (Neg)
            pos_filtered_count = (is_long_numpy & (scores_numpy > 0)).sum()
            neg_filtered_count = (is_long_numpy & (scores_numpy == 0)).sum()

            # C. 赋值给 metrics 变量
            pos_valid_sum = pos_filtered_count
            neg_valid_sum = neg_filtered_count
            
            # 3. 打印日志
            print(f"[INFO] filter-long: set {rows_to_zero_tensor.sum()} rows to zero (length > {threshold})", flush=True)
            print(f"       Details: {pos_valid_sum} pos samples, {neg_valid_sum} neg samples filtered.", flush=True)

            # 4. 执行 Mask 操作
            batch.batch['response_mask'][rows_to_zero_tensor, :] = 0

            # 5. 记录 Metrics
            metrics["post_process/filterlong/pos"] = pos_valid_sum
            metrics["post_process/filterlong/neg"] = neg_valid_sum



        elif method == "posonly":
            # --- posonly 逻辑 ---
            rows_to_zero_numpy = (scores == 0)
            rows_to_zero_tensor = torch.tensor(
                rows_to_zero_numpy, 
                dtype=torch.bool, 
                device=response_mask.device
            )
            print(f"[INFO] posonly: set {rows_to_zero_tensor.sum()} rows to zero", flush=True)
            batch.batch['response_mask'][rows_to_zero_tensor, :] = 0

        elif method == "negonly":
            # --- negonly 逻辑 ---
            rows_to_zero_numpy = (scores == 1)
            rows_to_zero_tensor = torch.tensor(
                rows_to_zero_numpy, 
                dtype=torch.bool, 
                device=response_mask.device
            )
            print(f"[INFO] negonly: set {rows_to_zero_tensor.sum()} rows to zero", flush=True)
            batch.batch['response_mask'][rows_to_zero_tensor, :] = 0

        elif method == "entropy-clip":

            clip_mode = self.config.trainer.entropy_clip_mode
            clip_ratio = self.config.trainer.entropy_clip_ratio
            
            # --- [Part 1] 全局筛选逻辑 (保持不变) ---
            valid_mask = response_mask.bool() 
            total_valid_tokens = valid_mask.sum().item()

            k = 0
            if clip_mode == "ratio":
                k = int(total_valid_tokens * float(clip_ratio))
            elif clip_mode == "num":
                k = int(clip_ratio)
                k = min(k, total_valid_tokens)
            
            assert k > 0

            masked_entropy = entropy.clone()
            masked_entropy[~valid_mask] = -float('inf')

            flat_entropy = masked_entropy.view(-1)
            _, topk_indices = torch.topk(flat_entropy, k)

            new_flat_mask = torch.zeros_like(flat_entropy, dtype=batch.batch['response_mask'].dtype)
            new_flat_mask[topk_indices] = 1
            batch.batch['response_mask'] = new_flat_mask.view_as(response_mask)

            # --- [Part 2] 监控逻辑：重点关注 Query 间的资源分配 ---
            
            # 1. 计算每一行(response)保留了多少 token
            # row_token_counts shape: [B], content: [120, 0, 50, ...]
            row_token_counts = batch.batch['response_mask'].sum(dim=1).float().cpu().numpy()
            uids = batch.non_tensor_batch['uid']
            
            # 2. 按 UID 聚合
            uid_stats = defaultdict(list)
            for uid, count in zip(uids, row_token_counts):
                uid_stats[uid].append(count)
            
            # 3. 提取列表用于计算统计量
            query_total_tokens = []
            query_avg_tokens = []
            
            log_msg = [f"[INFO] entropy-clip ({clip_mode}={clip_ratio}) Fairness Monitor:"]

            for uid, counts in uid_stats.items():
                total_for_this_query = np.sum(counts)
                avg_for_this_query = np.mean(counts)
                
                query_total_tokens.append(total_for_this_query)
                query_avg_tokens.append(avg_for_this_query)
                
                log_msg.append(
                    f"  UID[{str(uid)[:8]}..]: Total={int(total_for_this_query)} tokens | "
                    f"AvgPerResp={avg_for_this_query:.1f}"
                )

            # 4. 计算 Query 间的不均匀性指标
            q_totals = np.array(query_total_tokens)
            inter_q_max = np.max(q_totals)
            inter_q_min = np.min(q_totals)
            inter_q_mean = np.mean(q_totals)
            inter_q_std = np.std(q_totals)

            # --- [Part 3] 新增功能：正负例有效 Token 统计 ---

            # import pdb; pdb.set_trace()
            
            # scores 类似 array([0., 1., 0., ...])
            # 确保 row_token_counts 和 scores 长度一致 (通常 batch size 一致)
            
            # 找到正例(1)和负例(0)的索引掩码
            is_pos = (scores == 1)
            is_neg = (scores == 0)
            
            # 利用布尔索引，从 row_token_counts 中取出对应的行，并求和
            pos_valid_sum = np.sum(row_token_counts[is_pos])
            neg_valid_sum = np.sum(row_token_counts[is_neg])
            
            # 将正负例分布添加到 log 方便一眼看到
            log_msg.append(f"  >> Split: PosTokens={int(pos_valid_sum)}, NegTokens={int(neg_valid_sum)}")
            log_msg.append(f"  >> Summary: Min={inter_q_min}, Max={inter_q_max}, Mean={inter_q_mean:.1f}, Std={inter_q_std:.1f}")
            print("\n".join(log_msg), flush=True)

            # import pdb; pdb.set_trace()

            # 5. 写入 Metrics
            metrics["post_process/entropy/total_kept"] = k
            
            # [Query 间的贫富差距]
            metrics["post_process/entropy/dist_inter_query_min"] = inter_q_min
            metrics["post_process/entropy/dist_inter_query_max"] = inter_q_max
            metrics["post_process/entropy/dist_inter_query_std"] = inter_q_std
            metrics["post_process/entropy/dist_fairness_ratio"] = (inter_q_min + 1e-6) / (inter_q_max + 1e-6)
            
            # [新增：正负例分布]
            metrics["post_process/entropy/valid_token/pos"] = pos_valid_sum
            metrics["post_process/entropy/valid_token/neg"] = neg_valid_sum

        elif method == "entropy-clip-query":
            clip_mode = self.config.trainer.entropy_clip_mode
            clip_ratio = self.config.trainer.entropy_clip_ratio

            new_response_mask = torch.zeros_like(response_mask)
            uids = batch.non_tensor_batch['uid'] # 获取 UID 用于分组
            uid_to_indices = defaultdict(list)
            for idx, uid in enumerate(uids):
                uid_to_indices[uid].append(idx)
            total_kept_global = 0

            for uid, indices in uid_to_indices.items():
                indices_tensor = torch.tensor(indices, device=response_mask.device)
                group_entropy = entropy[indices_tensor]
                group_valid_mask = response_mask[indices_tensor].bool()

                total_valid_tokens_group = group_valid_mask.sum().item()
                k = int(total_valid_tokens_group * float(clip_ratio))
                assert k > 0 

                group_entropy_masked = group_entropy.clone()
                group_entropy_masked[~group_valid_mask] = -float('inf')

                # 展平做 Top-K
                flat_group_entropy = group_entropy_masked.view(-1)
                _, topk_indices = torch.topk(flat_group_entropy, k)

                local_flat_mask = torch.zeros_like(flat_group_entropy, dtype=response_mask.dtype)
                local_flat_mask[topk_indices] = 1

                local_mask = local_flat_mask.view(group_entropy.shape)
                new_response_mask[indices_tensor] = local_mask

                total_kept_global += k
            

            batch.batch['response_mask'] = new_response_mask
            
            # 1. 计算每一行(response)保留了多少 token
            row_token_counts = batch.batch['response_mask'].sum(dim=1).float().cpu().numpy()
            
            # 2. 按 UID 聚合统计
            uid_stats = defaultdict(list)
            for uid, count in zip(uids, row_token_counts):
                uid_stats[uid].append(count)
            
            query_total_tokens = []
            
            log_msg = [f"[INFO] entropy-clip (Per-Query {clip_mode}={clip_ratio}) Monitor:"]

            for uid, counts in uid_stats.items():
                total_for_this_query = np.sum(counts)
                query_total_tokens.append(total_for_this_query)
                # 这里的 log 会变长，如果 batch 很大可以注释掉下面这行 detail
                # log_msg.append(f"  UID[{str(uid)[:6]}]: {int(total_for_this_query)} tokens")

            # 4. 计算 Query 间的不均匀性指标
            q_totals = np.array(query_total_tokens)
            inter_q_max = np.max(q_totals) if len(q_totals) > 0 else 0
            inter_q_min = np.min(q_totals) if len(q_totals) > 0 else 0
            inter_q_mean = np.mean(q_totals) if len(q_totals) > 0 else 0
            inter_q_std = np.std(q_totals) if len(q_totals) > 0 else 0

            # 正负例统计
            is_pos = (scores == 1)
            is_neg = (scores == 0)
            pos_valid_sum = np.sum(row_token_counts[is_pos])
            neg_valid_sum = np.sum(row_token_counts[is_neg])
            
            log_msg.append(f"  >> Split: PosTokens={int(pos_valid_sum)}, NegTokens={int(neg_valid_sum)}")
            log_msg.append(f"  >> Fairness: Min={inter_q_min}, Max={inter_q_max}, Mean={inter_q_mean:.1f}")
            print("\n".join(log_msg), flush=True)

            # 5. 写入 Metrics
            metrics["post_process/entropy/total_kept"] = total_kept_global
            metrics["post_process/entropy/dist_inter_query_min"] = inter_q_min
            metrics["post_process/entropy/dist_inter_query_max"] = inter_q_max
            metrics["post_process/entropy/dist_inter_query_std"] = inter_q_std
            # 这个 ratio 越接近 1，说明你的 +AB 策略越成功（每个 query 分配到的计算量越均匀）
            metrics["post_process/entropy/dist_fairness_ratio"] = (inter_q_min + 1e-6) / (inter_q_max + 1e-6)
            
            metrics["post_process/entropy/valid_token/pos"] = pos_valid_sum
            metrics["post_process/entropy/valid_token/neg"] = neg_valid_sum

        else:
            raise ValueError(f"[INFO] unknown method: {method}")

        return batch, metrics


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False   

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # import pdb; pdb.set_trace()

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)


                    # repeat to align with repeated responses in rollout
                    # import pdb; pdb.set_trace()
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    # import pdb; pdb.set_trace()
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # import pdb; pdb.set_trace()
                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        global_old_entropys = deepcopy(entropys)

                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))
                    
                    # use opd here -> True
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)
                    
                    # import pdb; pdb.set_trace()

                    # compute values -> False
                    # if self.use_critic:
                    #     with marked_timer("values", timing_raw, color="cyan"):
                    #         values = self.critic_wg.compute_values(batch)
                    #         batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        # self.config.reward_model.launch_reward_fn_async -> False
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        print(f"[INFO][_opd][ray_trainer] use_kl_in_reward: {self.config.algorithm.use_kl_in_reward}", flush=True)

                        # self.config.algorithm.use_kl_in_reward -> False
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # batch = compute_advantage(
                        #     batch,
                        #     adv_estimator=self.config.algorithm.adv_estimator,
                        #     gamma=self.config.algorithm.gamma,
                        #     lam=self.config.algorithm.lam,
                        #     num_repeat=self.config.actor_rollout_ref.rollout.n,
                        #     norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        #     config=self.config.algorithm,
                        # )
                    
                    # batch-post-process part
                    # post_process = self.config.trainer.get("post_process", None)
                    # if post_process and len(post_process) > 0:
                    #     print(f"[INFO] use post_process here, method: {post_process}", flush=True)
                    #     batch, post_metrics = self.post_process(batch, post_process, entropy=global_old_entropys)
                    #     metrics.update(post_metrics)
                    
                    # import pdb; pdb.set_trace()


                    # update critic -> False
                    # if self.use_critic:
                    #     with marked_timer("update_critic", timing_raw, color="pink"):
                    #         critic_output = self.critic_wg.update_critic(batch)
                    #     critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    #     metrics.update(critic_output_metrics)
                    

                    # implement critic warmup
                    # if self.config.trainer.critic_warmup <= self.global_steps:
                    #     # update actor
                    #     with marked_timer("update_actor", timing_raw, color="red"):
                    #         batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable # False
                    #         actor_output = self.actor_rollout_wg.update_actor(batch)
                    #     actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    #     metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                    
                    # import pdb; pdb.set_trace()
                    # recompute logprob after update policy
                    recompute_logprob_after = self.config.trainer.get("recompute_logprob_after", False)
                    recompute_logprob_step = self.config.trainer.get("recompute_logprob_step")
                    print(f"[INFO] recompute logprob after: {recompute_logprob_after}, recompute logprob step: {recompute_logprob_step}", flush=True)
                    # import pdb; pdb.set_trace()
                    recompute_this_step = False
                    if recompute_logprob_after and self.global_steps % recompute_logprob_step == 0:
                        recompute_this_step = True
                    
                    print(f"[INFO] global step: {self.global_steps}, recompute logprob this step: {recompute_this_step}", flush=True)

                    def to_cpu(t):
                        if isinstance(t, torch.Tensor):
                            return t.detach().cpu()
                        return t
                    
                    def recompute_logprob(batch):
                        logprob_batch_key = ["input_ids", "attention_mask", "position_ids", "responses"]
                        logprob_batch = batch.select(batch_keys=logprob_batch_key)

                        # for current batch
                        # old_logprobs = batch.batch.get("old_log_probs")
                        # old_entropys = global_old_entropys
                        # response_masks = batch.batch["response_mask"]

                        # recompute logprobs after update
                        logprob_after = self.actor_rollout_wg.compute_log_prob(logprob_batch)
                        current_logprob = logprob_after.batch["old_log_probs"]
                        current_entropys = logprob_after.batch["entropys"]

                        data = {
                            "logprob_after": to_cpu(current_logprob),
                            "current_entropys": to_cpu(current_entropys)
                        }

                        return data


                    # use normal grpo update
                    # print(f"[INFO use noraml grpo update, step: {self.global_steps}", flush=True)
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                    # log to swanlab
                    self._maybe_log_rollout_generations(batch, key="rollout")


                        

                    # if recompute_logprob_after:
                        # print(f"[INFO] recompute logprob after policy update", flush=True)
                        # import pdb; pdb.set_trace()
                        # with marked_timer("recompute_logprob_after", timing_raw, color="blue"):
                        #     logprob_batch_key = ["input_ids", "attention_mask", "position_ids", "responses"]
                        #     logprob_batch = batch.select(batch_keys=logprob_batch_key)

                        #     # for current batch
                        #     old_logprobs = batch.batch.get("old_log_probs")
                        #     old_entropys = global_old_entropys
                        #     response_masks = batch.batch["response_mask"]

                        #     # recompute logprobs after update
                        #     logprob_after = self.actor_rollout_wg.compute_log_prob(logprob_batch)
                        #     current_logprob = logprob_after.batch["old_log_probs"]
                        #     current_entropys = logprob_after.batch["entropys"]

                        #     # import pdb; pdb.set_trace()

                        #     # wandb log
                        #     logprob_diff = current_logprob - old_logprobs
                        #     logprob_diff_masked_mean = masked_mean(logprob_diff, response_masks)

                        #     pos_rollout_idx = batch.non_tensor_batch['score'] > 0
                        #     neg_rollout_idx = batch.non_tensor_batch['score'] <= 0

                        #     logprob_diff_pos = masked_mean(logprob_diff[pos_rollout_idx], response_masks[pos_rollout_idx])
                        #     logprob_diff_neg = masked_mean(logprob_diff[neg_rollout_idx], response_masks[neg_rollout_idx])

                        #     entropy_diff = current_entropys - old_entropys
                        #     entropy_diff_masked_mean = masked_mean(entropy_diff, response_masks)
                        #     pos_entropy_diff = masked_mean(entropy_diff[pos_rollout_idx], response_masks[pos_rollout_idx])
                        #     neg_entropy_diff = masked_mean(entropy_diff[neg_rollout_idx], response_masks[neg_rollout_idx])

                        #     metric_dict = {
                        #         "dynamic/logprob_diff/mean": logprob_diff_masked_mean,
                        #         "dynamic/logprob_diff/mean/pos": logprob_diff_pos,
                        #         "dynamic/logprob_diff/mean/neg": logprob_diff_neg,

                        #         "dynamic/entropy_diff/mean": entropy_diff_masked_mean,
                        #         "dynamic/entropy_diff/mean/pos": pos_entropy_diff,
                        #         "dynamic/entropy_diff/mean/neg": neg_entropy_diff
                        #     }
                        #     metrics.update(metric_dict)


                        #     # dump to local
                        #     # import pdb; pdb.set_trace()
                        #     dump_dir = self.config.trainer.get("dump_dir")
                        #     if not os.path.exists(dump_dir):
                        #         os.makedirs(dump_dir)
                        #     dump_path = os.path.join(dump_dir, f"step_{self.global_steps}.pt")
                        #     dump_data = {
                        #             # --- From batch.batch (Tensors) ---
                        #             'input_ids': to_cpu(batch.batch.get('input_ids')),
                        #             'old_log_probs': to_cpu(batch.batch.get('old_log_probs')),
                        #             'response_masks': to_cpu(batch.batch.get('response_mask')), # 注意代码里原本使用的是 response_mask
                        #             'responses': to_cpu(batch.batch.get('responses')),
                                    
                        #             # --- From batch.non_tensor_batch (List/Meta) ---
                        #             'uuid': batch.non_tensor_batch.get('uuid'),
                        #             'score': batch.non_tensor_batch.get('score'),
                                    
                        #             # --- Computed Values ---
                        #             # 这里保存 current_logprob，即 update 后的 logprob
                        #             'logprob_after': to_cpu(current_logprob), 
                        #             'old_entropys': to_cpu(old_entropys),
                        #             'current_entropys': to_cpu(current_entropys)
                        #         }
                        #     print(f"[INFO] Dumping debug data to {dump_path}", flush=True)
                        #     torch.save(dump_data, dump_path)
                        #     # import pdb; pdb.set_trace()




                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)