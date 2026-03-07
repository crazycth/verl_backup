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
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

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

from recipe._psrnsr import core_algos
from recipe._psrnsr.core_algos import AdvantageEstimator, agg_loss


# from verl.trainer.ppo.metric_utils import (
#     compute_data_metrics,
#     compute_throughout_metrics,
#     compute_timing_metrics,
#     process_validation_metrics,
# )
from recipe._psrnsr.metric_utils import (
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
from verl.utils.tracking import ValidationGenerationsLogger


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


def _as_numpy_1d(name: str, value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    elif isinstance(value, list | tuple):
        value = np.asarray(value)
    elif not isinstance(value, np.ndarray):
        value = np.asarray(value)

    if value.ndim != 1:
        value = value.reshape(-1)
    return value


def load_causal_candidates(candidates_path: str) -> dict[str, np.ndarray]:
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"Candidates file not found: {candidates_path}")

    payload = torch.load(candidates_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Candidates file must be a dict, got {type(payload)}")

    if "candidates" in payload:
        candidates = payload["candidates"]
    else:
        candidates = payload

    if not isinstance(candidates, dict):
        raise ValueError(f"`candidates` must be a dict, got {type(candidates)}")

    required_keys = ["row_idx", "col_idx", "token_id"]
    for key in required_keys:
        if key not in candidates:
            raise KeyError(f"Missing candidate key `{key}` in {candidates_path}")

    normalized: dict[str, np.ndarray] = {}
    for key, value in candidates.items():
        normalized[key] = _as_numpy_1d(key, value)

    n = normalized["row_idx"].shape[0]
    for key in required_keys:
        if normalized[key].shape[0] != n:
            raise ValueError(
                f"Candidates length mismatch: `{key}` has {normalized[key].shape[0]}, expected {n}"
            )

    normalized["row_idx"] = normalized["row_idx"].astype(np.int64)
    normalized["col_idx"] = normalized["col_idx"].astype(np.int64)
    normalized["token_id"] = normalized["token_id"].astype(np.int64)
    if "prompt_uid" in normalized:
        normalized["prompt_uid"] = normalized["prompt_uid"].astype(np.int64)
    return normalized


def build_prompt_uid(
    input_ids: torch.Tensor,
    *,
    prompt_len: int = 8192,
    expected_prompt_num: Optional[int] = 128,
    expected_repeat_per_prompt: Optional[int] = None,
) -> torch.Tensor:
    prompt = input_ids[:, :prompt_len]
    _, inverse, counts = torch.unique(prompt, dim=0, return_inverse=True, return_counts=True)

    unique_prompt_num = int(counts.shape[0])
    if expected_prompt_num is not None:
        assert unique_prompt_num == int(expected_prompt_num), (
            f"Expected {expected_prompt_num} unique prompts from input_ids[:, :{prompt_len}], "
            f"but got {unique_prompt_num}"
        )

    if expected_repeat_per_prompt is not None:
        expected_repeat_per_prompt = int(expected_repeat_per_prompt)
        if not torch.all(counts == expected_repeat_per_prompt):
            min_repeat = int(counts.min().item())
            max_repeat = int(counts.max().item())
            raise AssertionError(
                f"Expected each prompt repeats {expected_repeat_per_prompt} times, "
                f"but got min={min_repeat}, max={max_repeat}"
            )
    return inverse


def get_response_mask(
    strategy: str,
    base_response_mask: torch.Tensor,
    responses: torch.Tensor,
    row_idx: int,
    col_idx: int,
    token_id: int,
    *,
    random_n: int = 0,
    rng: Optional[torch.Generator] = None,
    prompt_uid: Optional[torch.Tensor] = None,
    adv_sign: Optional[torch.Tensor] = None,
    keep_candidate_grad: bool = True,
) -> torch.Tensor:
    if not (0 <= row_idx < responses.shape[0]) or not (0 <= col_idx < responses.shape[1]):
        raise IndexError(
            f"Candidate index out of range: row_idx={row_idx}, col_idx={col_idx}, "
            f"shape={tuple(responses.shape)}"
        )

    observed_token = int(responses[row_idx, col_idx].item())
    if observed_token != int(token_id):
        raise AssertionError(
            f"Candidate token mismatch at (row={row_idx}, col={col_idx}): "
            f"expected token_id={token_id}, found={observed_token}"
        )

    valid_mask = base_response_mask > 0
    if not bool(valid_mask[row_idx, col_idx].item()):
        raise AssertionError(
            f"Candidate token must satisfy response_mask==1, but got 0 at (row={row_idx}, col={col_idx})"
        )

    new_mask = valid_mask.clone()
    strategy = str(strategy).lower()

    if strategy == "random-n":
        random_n = int(random_n)
        if random_n > 0:
            candidates = valid_mask.clone()
            candidates[row_idx, col_idx] = False
            valid_pos = candidates.nonzero(as_tuple=False)
            pick_n = min(random_n, int(valid_pos.shape[0]))
            if pick_n > 0:
                perm = torch.randperm(valid_pos.shape[0], generator=rng, device=valid_pos.device)[:pick_n]
                picked = valid_pos.index_select(0, perm)
                new_mask[picked[:, 0], picked[:, 1]] = False

    elif strategy == "rollout-mask-token":
        cond = valid_mask[row_idx] & (responses[row_idx] == int(token_id))
        new_mask[row_idx, cond] = False

    elif strategy in ("prompt-mask-token-adv", "batch-mask-token-adv"):
        if adv_sign is None:
            raise ValueError(f"`adv_sign` is required for strategy `{strategy}`")

        anchor_sign = adv_sign[row_idx, col_idx]
        token_cond = responses == int(token_id)
        sign_cond = adv_sign == anchor_sign
        cond = valid_mask & token_cond & sign_cond

        if strategy == "prompt-mask-token-adv":
            if prompt_uid is None:
                raise ValueError("`prompt_uid` is required for `prompt-mask-token-adv`")
            if prompt_uid.ndim != 1 or prompt_uid.shape[0] != responses.shape[0]:
                raise ValueError(
                    f"`prompt_uid` shape must be [batch], got {tuple(prompt_uid.shape)} "
                    f"for batch={responses.shape[0]}"
                )
            group_mask = prompt_uid == prompt_uid[row_idx]
            cond = cond & group_mask.unsqueeze(1)

        new_mask[cond] = False

    else:
        raise ValueError(
            f"Unsupported strategy `{strategy}`. "
            f"Supported: random-n, rollout-mask-token, prompt-mask-token-adv, batch-mask-token-adv"
        )

    if keep_candidate_grad:
        new_mask[row_idx, col_idx] = True
    else:
        new_mask[row_idx, col_idx] = False
    return new_mask.to(dtype=base_response_mask.dtype)


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
            drop_last=False,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        # val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        # if val_batch_size is None:
        #     val_batch_size = len(self.val_dataset)

        # self.val_dataloader = StatefulDataLoader(
        #     dataset=self.val_dataset,
        #     batch_size=val_batch_size,
        #     num_workers=num_workers,
        #     shuffle=self.config.data.get("validation_shuffle", True),
        #     drop_last=False,
        #     collate_fn=collate_fn,
        # )

        val_batch_size = val_dataloader = None

        # import pdb; pdb.set_trace()

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        # assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            # f"{len(self.val_dataloader)}"
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
        from verl.utils.fs import local_mkdir_safe

        if os.path.isabs(folder_name):
            local_folder = folder_name
        else:
            local_folder = os.path.join(self.config.trainer.default_local_dir, folder_name)
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

        if os.path.isabs(folder_name):
            local_folder = folder_name
        else:
            local_folder = os.path.join(self.config.trainer.default_local_dir, folder_name)

        if not os.path.exists(local_folder):
            raise FileNotFoundError(f"Temporary checkpoint not found at: {local_folder}")

        actor_path = os.path.join(local_folder, "actor")

        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=False
        )

        print(f"[INFO][_load_temp_checkpoint] Successfully Loaded temp checkpoint from {local_folder}", flush=True)

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
    

    def _write_results_append(self, result):
        import os, json
        path = self.config.trainer.causal.result_path
        with open(path, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[INFO][causal] wrote result to {path}", flush=True)


    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.global_steps += 1

        causal_cfg = self.config.trainer.get("causal", {})
        if not causal_cfg.get("enable", False):
            print("[INFO][causal] trainer.causal.enable=False, skip causal intervention run.", flush=True)
            return

        candidates_path = causal_cfg.get("candidates_path", None)
        if candidates_path is None:
            raise ValueError("trainer.causal.candidates_path is required when trainer.causal.enable=True")
        if not os.path.isabs(candidates_path):
            candidates_path = os.path.join(os.getcwd(), candidates_path)

        result_path = causal_cfg.get("result_path", None)
        if result_path is None:
            result_path = os.path.join(self.config.trainer.default_local_dir, "causal_results.pt")
        if not os.path.isabs(result_path):
            result_path = os.path.join(os.getcwd(), result_path)

        random_n = int(causal_cfg.get("random_n", 64))
        prompt_len = int(causal_cfg.get("prompt_len", 8192))
        expected_prompt_num = causal_cfg.get("expected_prompt_num", 128)
        expected_prompt_num = None if expected_prompt_num is None else int(expected_prompt_num)

        print(f"[INFO][causal] expected_prompt_num: {expected_prompt_num}, prompt_len: {prompt_len}", flush=True)

        expected_repeat = causal_cfg.get("expected_repeat_per_prompt", None)
        if expected_repeat is None:
            expected_repeat = self.config.actor_rollout_ref.rollout.get("n", None)
        expected_repeat = None if expected_repeat is None else int(expected_repeat)

        strategies = causal_cfg.get(
            "strategies",
            ["random-n", "rollout-mask-token", "prompt-mask-token-adv", "batch-mask-token-adv"],
        )
        if isinstance(strategies, str):
            strategies = [strategies]
        strategies = [str(s).lower() for s in strategies]

        max_candidates = causal_cfg.get("max_candidates", None)
        max_candidates = None if max_candidates is None else int(max_candidates)

        seed = int(causal_cfg.get("seed", 42))
        keep_candidate_grad = bool(causal_cfg.get("keep_candidate_grad", True))

        tmp_ckpt_dir = causal_cfg.get("tmp_ckpt_dir", f"causal_policy0_step_{self.global_steps}")

        train_iter = iter(self.train_dataloader)
        try:
            batch_dict = next(train_iter)
        except StopIteration as exc:
            raise ValueError("Train dataloader is empty, cannot run causal intervention.") from exc

        normalized_batch_dict = {}
        for key, value in batch_dict.items():
            if hasattr(value, "shape") and value.shape[0] == 1:
                normalized_batch_dict[key] = value[0]
            elif isinstance(value, list) and len(value) == 1:
                normalized_batch_dict[key] = value[0]
            else:
                normalized_batch_dict[key] = value

        batch: DataProto = DataProto.from_single_dict(normalized_batch_dict)
        if "response_mask" not in batch.batch:
            if "response_masks" in batch.batch:
                batch.batch["response_mask"] = batch.batch["response_masks"]
            else:
                batch.batch["response_mask"] = compute_response_mask(batch)

        responses = batch.batch["responses"]
        base_response_mask = batch.batch["response_mask"]
        rng = torch.Generator(device=responses.device.type)
        rng.manual_seed(seed)
        prompt_uid = build_prompt_uid(
            batch.batch["input_ids"],
            prompt_len=prompt_len,
            expected_prompt_num=expected_prompt_num,
            expected_repeat_per_prompt=expected_repeat,
        )
        batch.non_tensor_batch["uid"] = prompt_uid.cpu().numpy().astype(np.int64)

        print(f"[INFO][causal] load candidates from {candidates_path}", flush=True)
        candidates = load_causal_candidates(candidates_path)
        total_candidates = int(candidates["row_idx"].shape[0])
        print(f"[INFO][causal] total_candidates: {total_candidates}", flush=True)

        if max_candidates is not None:
            total_candidates = min(total_candidates, max_candidates)
            for key in list(candidates.keys()):
                candidates[key] = candidates[key][:total_candidates]

        row_idx = candidates["row_idx"]
        col_idx = candidates["col_idx"]
        token_id = candidates["token_id"]

        if total_candidates == 0:
            print("[INFO][causal] No candidates found, skip intervention and save empty results.", flush=True)
            result_dir = os.path.dirname(result_path)
            if result_dir:
                os.makedirs(result_dir, exist_ok=True)
            output_payload = {
                "schema_version": "causal_intervention_result_v1",
                "candidates_path": candidates_path,
                "result_count": 0,
                "seed": seed,
                "random_n": random_n,
                "strategies": ["baseline"] + strategies,
                "update_api": "update_actor",
                "results": [],
            }
            torch.save(output_payload, result_path)
            print(f"[INFO][causal] saved empty results to {result_path}", flush=True)
            return

        batch_size, response_len = responses.shape
        if np.any((row_idx < 0) | (row_idx >= batch_size)):
            raise IndexError(f"Candidate row_idx out of range [0, {batch_size}): {row_idx}")
        if np.any((col_idx < 0) | (col_idx >= response_len)):
            raise IndexError(f"Candidate col_idx out of range [0, {response_len}): {col_idx}")

        row_idx_t = torch.from_numpy(row_idx).to(device=responses.device, dtype=torch.long)
        col_idx_t = torch.from_numpy(col_idx).to(device=responses.device, dtype=torch.long)
        token_observed = responses[row_idx_t, col_idx_t].detach().cpu().numpy().astype(np.int64)
        mismatch_idx = np.where(token_observed != token_id)[0]
        if mismatch_idx.size > 0:
            i = int(mismatch_idx[0])
            raise AssertionError(
                f"Candidate token mismatch at index {i}: "
                f"(row={int(row_idx[i])}, col={int(col_idx[i])}) "
                f"expected={int(token_id[i])}, observed={int(token_observed[i])}"
            )

        if "prompt_uid" in candidates:
            prompt_uid_np = prompt_uid.detach().cpu().numpy().astype(np.int64)
            cand_prompt_uid = candidates["prompt_uid"].astype(np.int64)
            prompt_uid_mismatch = np.where(cand_prompt_uid != prompt_uid_np[row_idx])[0]
            if prompt_uid_mismatch.size > 0:
                i = int(prompt_uid_mismatch[0])
                raise AssertionError(
                    f"Candidate prompt_uid mismatch at index {i}: "
                    f"candidate={int(cand_prompt_uid[i])}, recomputed={int(prompt_uid_np[row_idx[i]])}"
                )

        print(f"[INFO][causal] compute policy0 logprob", flush=True)
        logprob_batch = batch.select(batch_keys=["input_ids", "attention_mask", "position_ids", "responses"])
        policy0_logprob = self.actor_rollout_wg.compute_log_prob(logprob_batch).batch["old_log_probs"].detach().clone()

        base_batch = deepcopy(batch)
        # import pdb; pdb.set_trace()
        # base_batch.batch["old_log_probs"] = policy0_logprob.clone()

        # Build sparse token-level rewards from scalar outcome score, same as old commented flow:
        # only the last valid response token receives row-level score.
        if "token_level_rewards" not in base_batch.batch:
            if "score" not in base_batch.non_tensor_batch:
                raise KeyError(
                    "Current causal GRPO flow requires non_tensor_batch['score'] to build token-level rewards."
                )

            response_mask = base_batch.batch["response_mask"]
            scores_np = _as_numpy_1d("score", base_batch.non_tensor_batch["score"]).astype(np.float32, copy=False)
            if scores_np.shape[0] != response_mask.shape[0]:
                raise ValueError(
                    f"Score length mismatch: score_len={scores_np.shape[0]}, batch_size={response_mask.shape[0]}"
                )

            scores = torch.as_tensor(scores_np, dtype=torch.float32, device=response_mask.device)
            batch_size_local = scores.shape[0]
            response_len_local = base_batch.batch["responses"].shape[1]

            token_level_scores = torch.zeros(
                (batch_size_local, response_len_local), dtype=scores.dtype, device=scores.device
            )
            seq_lengths = response_mask.sum(dim=1).long()
            last_token_indices = (seq_lengths - 1).clamp(min=0)
            token_level_scores[torch.arange(batch_size_local, device=scores.device), last_token_indices] = scores
            token_level_scores = token_level_scores * response_mask

            base_batch.batch["token_level_scores"] = token_level_scores
            base_batch.batch["token_level_rewards"] = token_level_scores

        norm_adv_by_std_in_grpo = bool(self.config.algorithm.get("norm_adv_by_std_in_grpo", True))
        print(f"[INFO][causal] compute advantage with GRPO, norm_adv_by_std_in_grpo={norm_adv_by_std_in_grpo}, gamma: {self.config.algorithm.gamma}, lam: {self.config.algorithm.lam}, rollout_n: {self.config.actor_rollout_ref.rollout.n}", flush=True)
        base_batch = compute_advantage(
            base_batch,
            adv_estimator=AdvantageEstimator.GRPO,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

        need_adv_sign = any(s in ("prompt-mask-token-adv", "batch-mask-token-adv") for s in strategies)
        adv_sign = None
        if need_adv_sign:
            if "advantages" not in base_batch.batch:
                raise KeyError("Missing `advantages` after GRPO compute_advantage.")
            adv_sign = torch.sign(base_batch.batch["advantages"].float())
            adv_sign = torch.where(adv_sign == 0, torch.ones_like(adv_sign), adv_sign)
        prompt_uid_t = prompt_uid.to(responses.device)

        print(
            f"[INFO][causal] start intervention: candidates={total_candidates}, "
            f"strategies={strategies}, keep_candidate_grad={keep_candidate_grad}, update_api=update_actor",
            flush=True,
        )

        self._save_temp_checkpoint(folder_name=tmp_ckpt_dir)

        results: list[dict[str, Any]] = []
        all_modes = ["baseline"] + strategies
        full_valid_tokens = int((base_response_mask > 0).sum().item())

        def _to_scalar(value: Any) -> Any:
            if isinstance(value, np.generic):
                return value.item()
            if torch.is_tensor(value):
                if value.numel() == 1:
                    return value.item()
                return value.detach().cpu().tolist()
            return value

        def _single_step_update(mask_for_update: torch.Tensor) -> dict[str, Any]:
            update_batch = deepcopy(base_batch)
            update_batch.batch["response_mask"] = mask_for_update.to(dtype=base_response_mask.dtype)
            update_batch.meta_info["temperature"] = float(causal_cfg.get("temperature", 1.0))
            update_batch.meta_info["global_token_num"] = torch.sum(update_batch.batch["attention_mask"], dim=-1).tolist()

            update_output = self.actor_rollout_wg.update_actor(update_batch)

            update_metrics: dict[str, Any] = {}
            if isinstance(update_output, DataProto):
                metric_dict = update_output.meta_info.get("metrics", {})
                if isinstance(metric_dict, dict):
                    metric_dict = reduce_metrics(metric_dict)
                    update_metrics = {k: _to_scalar(v) for k, v in metric_dict.items()}
            return update_metrics

        from tqdm import tqdm
        for cand_idx in tqdm(range(total_candidates)):
            r = int(row_idx[cand_idx])
            c = int(col_idx[cand_idx])
            tok = int(token_id[cand_idx])
            prompt_id = int(prompt_uid_t[r].item())
            lp0 = float(policy0_logprob[r, c].item())

            candidate_meta = {}
            for key, arr in candidates.items():
                candidate_meta[key] = _to_scalar(arr[cand_idx])

            for mode in all_modes:
                if mode == "baseline":
                    intervention_mask = base_response_mask.clone()
                else:
                    intervention_mask = get_response_mask(
                        strategy=mode,
                        base_response_mask=base_response_mask,
                        responses=responses,
                        row_idx=r,
                        col_idx=c,
                        token_id=tok,
                        random_n=random_n,
                        rng=rng,
                        prompt_uid=prompt_uid_t,
                        adv_sign=adv_sign,
                        keep_candidate_grad=keep_candidate_grad,
                    )

                kept_tokens = int((intervention_mask > 0).sum().item())
                masked_tokens = full_valid_tokens - kept_tokens

                self._load_temp_checkpoint(folder_name=tmp_ckpt_dir)
                update_metrics = _single_step_update(intervention_mask)
                updated_logprob = self.actor_rollout_wg.compute_log_prob(logprob_batch).batch["old_log_probs"]
                lp_after = float(updated_logprob[r, c].item())
                delta = lp_after - lp0
                token_adv = base_batch.batch['advantages'][r, c].item()

                result = {
                    "candidate_index": cand_idx,
                    "strategy": mode,
                    "keep_candidate_grad": keep_candidate_grad,
                    # "update_api": "update_actor",
                    "row_idx": r,
                    "col_idx": c,
                    "token_id": tok,
                    "token_adv": token_adv,

                    "prompt_uid": prompt_id,
                    "policy0_logprob": lp0,
                    "policy_after_logprob": lp_after,
                    "delta_logprob": delta,
                    "masked_token_num": masked_tokens,
                    "kept_token_num": kept_tokens,
                    "candidate_meta": candidate_meta,
                    # "update_metrics": update_metrics,
                }
                results.append(result)

                # save everyturn to file
                self._write_results_append(result)

                print(
                    f"[CAUSAL] cand={cand_idx:04d} mode={mode:<24} "
                    f"(r={r}, c={c}, tok={tok}) lp0={lp0:.6f} lp={lp_after:.6f} d={delta:+.6f} adv={token_adv:+.6f} "
                    f"masked={masked_tokens}",
                    flush=True,
                )

        self._load_temp_checkpoint(folder_name=tmp_ckpt_dir)

        result_dir = os.path.dirname(result_path)
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        output_payload = {
            "schema_version": "causal_intervention_result_v1",
            "candidates_path": candidates_path,
            "result_count": len(results),
            "seed": seed,
            "random_n": random_n,
            "strategies": all_modes,
            "update_api": "update_actor",
            "results": results,
        }
        torch.save(output_payload, result_path)
        print(f"[INFO][causal] saved results to {result_path}, count={len(results)}", flush=True)
        return



    # def fit(self):
    #     """
    #     The training loop of PPO.
    #     The driver process only need to call the compute functions of the worker group through RPC
    #     to construct the PPO dataflow.
    #     The light-weight advantage computation is done on the driver process.
    #     """
    #     from omegaconf import OmegaConf

    #     from verl.utils.tracking import Tracking

    #     logger = Tracking(
    #         project_name=self.config.trainer.project_name,
    #         experiment_name=self.config.trainer.experiment_name,
    #         default_backend=self.config.trainer.logger,
    #         config=OmegaConf.to_container(self.config, resolve=True),
    #     )

    #     self.global_steps = 0

    #     # load checkpoint before doing anything
    #     self._load_checkpoint()

    #     # we start from step 1
    #     self.global_steps += 1

    #     for batch_dict in self.train_dataloader:
    #         metrics = {}
    #         timing_raw = {}

    #         new_batch_dict = {}
    #         for k,v in batch_dict.items():
    #             if hasattr(v, 'shape') and v.shape[0] == 1:
    #                 new_batch_dict[k] = v[0]
    #             elif isinstance(v, list) and len(v) == 1:
    #                 new_batch_dict[k] = v[0]
    #             else:
    #                 new_batch_dict[k] = v

    #         batch: DataProto = DataProto.from_single_dict(new_batch_dict)

    #         # import pdb; pdb.set_trace()

    #         # ------------------------------------------------------------------
    #         # 从标量 outcome score 构造逐 token 的 reward / score
    #         # 说明：
    #         # - 当前批次的任务级得分存放在 batch.non_tensor_batch["score"]，形如 (batch_size,)
    #         # - 我们希望为每个 response token 构造同一个标量 reward，并用 response_mask 做掩码
    #         # - 目前没有任何 KL 项，因此 token_level_rewards 与 token_level_scores 相同
    #         # ------------------------------------------------------------------
    #         # ------------------------------------------------------------------
    #         # 修改版：从标量 outcome score 构造 Sparse Reward (仅最后一个 token 有分)
    #         # ------------------------------------------------------------------

    #         # import pdb; pdb.set_trace()
            
    #         # 1) 取出标量得分
    #         scores_np = batch.non_tensor_batch["score"]  # (B,)
    #         scores = torch.as_tensor(
    #             scores_np,
    #             dtype=torch.float32,
    #             device=batch.batch["responses"].device,
    #         )  # (B,)

    #         # 2) 【关键修改】必须先获取 response_mask，因为我们需要用它来确定“哪里是最后一个位置”
    #         if "response_masks" in batch.batch:
    #             response_mask = batch.batch["response_masks"]

    #         # 3) 初始化全 0 的 token_level_scores
    #         response_length = batch.batch["responses"].size(1)
    #         batch_size = scores.size(0)
    #         token_level_scores = torch.zeros(
    #             (batch_size, response_length), 
    #             dtype=scores.dtype, 
    #             device=scores.device
    #         )

    #         # 4) 计算每个样本最后一个有效 token 的索引
    #         #    假设 mask 是 [1, 1, 1, 0, 0]，sum 是 3，最后一个有效索引是 2 (即 3-1)
    #         #    注意：要确保 response_mask 类型是数值型以便求和
    #         seq_lengths = response_mask.sum(dim=1).long() 
    #         last_token_indices = seq_lengths - 1

    #         # 5) 【核心修改】只给最后一个有效位置赋值
    #         #    利用高级索引：token_level_scores[行索引, 列索引] = scores
    #         #    为了防止全是 padding 的空行导致索引 -1 (虽然极少见)，可以加个 clamp 或断言
    #         last_token_indices = last_token_indices.clamp(min=0) 
            
    #         token_level_scores[torch.arange(batch_size, device=scores.device), last_token_indices] = scores

    #         # 6) 再次应用 mask (双重保险，确保 padding 位置绝对是 0)
    #         token_level_scores = token_level_scores * response_mask

    #         # 7) 写回 batch
    #         batch.batch["token_level_scores"] = token_level_scores
    #         batch.batch["token_level_rewards"] = token_level_scores

    #         # batch.batch["input_ids"] shape: (1024, 16384) -> prefix: (1024, 8192)
    #         prefix_len = 8192
    #         prompt_prefix = batch.batch["input_ids"][:, :prefix_len]

    #         # 2. 使用 torch.unique 按行去重
    #         # return_inverse=True 会返回一个索引 tensor，指示原 tensor 中每一行对应 unique 结果中的哪个下标
    #         # 这些下标 (0, 1, 2...) 天然就是我们要的组 ID
    #         _, uid_indices = torch.unique(prompt_prefix, return_inverse=True, dim=0)

    #         # 3. Assert 检查：确认去重后的数量正好是 128
    #         num_unique_uids = uid_indices.max().item() + 1
    #         expected_repeat = self.config.actor_rollout_ref.rollout.n
    #         batch_size = uid_indices.numel()
    #         assert batch_size % expected_repeat == 0, (
    #             f"Assertion Failed: batch_size={batch_size} is not divisible by rollout.n={expected_repeat}."
    #         )
    #         expected_prompt_num = batch_size // expected_repeat
    #         assert num_unique_uids == expected_prompt_num, (
    #             f"Assertion Failed: Expected {expected_prompt_num} unique UIDs based on first {prefix_len} tokens, "
    #             f"but found {num_unique_uids}. Please check your batch composition."
    #         )

    #         # 3.1 如果设置了 tune_bs，则只保留前 tune_bs 个 prompt 及其全部 rollouts
    #         # import pdb; pdb.set_trace()
    #         batch_for_logprob = copy.deepcopy(batch)
    #         tune_bs = self.config.data.get("tune_bs", None)
    #         print(f"[INFO] tune_bs: {tune_bs}, expected_prompt_num: {expected_prompt_num}", flush=True)
    #         if tune_bs is not None:
    #             tune_bs = int(tune_bs)
    #             assert tune_bs > 0, f"tune_bs must be > 0, got {tune_bs}."
    #             assert tune_bs <= expected_prompt_num, (
    #                 f"tune_bs must be <= {expected_prompt_num}, got {tune_bs}."
    #             )
    #             if tune_bs < expected_prompt_num:
    #                 # 按首次出现顺序选取 prompt UID
    #                 uid_indices_cpu = uid_indices.detach().cpu()
    #                 uid_indices_np = uid_indices_cpu.numpy()
    #                 first_pos = np.full((num_unique_uids,), batch_size, dtype=np.int64)
    #                 for idx, uid in enumerate(uid_indices_np):
    #                     if idx < first_pos[uid]:
    #                         first_pos[uid] = idx
    #                 ordered_uids = np.argsort(first_pos)
    #                 uids_to_keep_np = ordered_uids[:tune_bs]

    #                 keep_mask = np.isin(uid_indices_np, uids_to_keep_np)
    #                 batch = batch.select_idxs(keep_mask)

    #                 # 重新映射 uid，保证从 0..tune_bs-1 的连续编号
    #                 uid_map = np.full((num_unique_uids,), -1, dtype=np.int64)
    #                 uid_map[uids_to_keep_np] = np.arange(tune_bs, dtype=np.int64)
    #                 new_uid = uid_map[uid_indices_np[keep_mask]]
    #                 uid_indices = torch.from_numpy(new_uid).to(uid_indices.device)
            
    #         # import pdb; pdb.set_trace()
    #         print(f"[INFO] len batch: {len(batch)}", flush=True)

    #         # 4. 将结果存入 batch.non_tensor_batch['uid']
    #         # 将 tensor 转为 list (non_tensor_batch 通常存非 tensor 数据)
    #         # 这样相同的 prompt 前缀会有相同的整数 ID
    #         uid_np = uid_indices.cpu().numpy().astype(np.int64)
    #         # import pdb; pdb.set_trace()

    #         batch.non_tensor_batch['uid'] = uid_np
    #         batch.meta_info["temperature"] = 1.0
    #         batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

    #         # import pdb; pdb.set_trace()

    #         norm_adv_by_std_in_grpo = self.config.get("norm_adv_by_std_in_grpo", True)
    #         batch = compute_advantage(
    #             batch,
    #             adv_estimator=AdvantageEstimator.GRPO,
    #             gamma=self.config.algorithm,
    #             lam=self.config.algorithm.lam,
    #             num_repeat=self.config.actor_rollout_ref.rollout.n,
    #             norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
    #             config=self.config.algorithm,
    #         )

    #         # import pdb; pdb.set_trace()

    #         timestamp = time.strftime("%Y%m%d_%H%M%S")

    #         for i in range(80):
    #             grpo_actor_output = self.actor_rollout_wg.update_actor_onpolicydistill(batch)
    #             print(f"[INFO] update step: {i}", flush=True)
    #             print(f"[INFO] output: {grpo_actor_output}", flush=True)

    #             # Log worker output metrics (from update_actor) to tracking backends (e.g., wandb).
    #             try:
    #                 output_metrics = grpo_actor_output.meta_info.get("metrics", {})
    #                 output_metrics = reduce_metrics(output_metrics)
    #                 output_metrics["train/inner_update_step"] = i
    #                 # Ensure monotonically increasing steps for logging.
    #                 log_step = int(i)
    #                 logger.log(data=output_metrics, step=log_step)
    #             except Exception as e:
    #                 print(f"[WARN] failed to log update_actor output metrics: {e}", flush=True)

    #             logprob_batch_key = ["input_ids", "attention_mask", "position_ids", "responses"]
    #             logprob_batch = batch_for_logprob.select(batch_keys=logprob_batch_key)
    #             # logprob_batch.meta_info = batch.meta_info
    #             logprob_batch.meta_info["temperature"] = 1.0
    #             logprob_batch.meta_info["global_token_num"] = torch.sum(logprob_batch.batch["attention_mask"], dim=-1).tolist()

    #             # import pdb; pdb.set_trace()

    #             print(f"[INFO] start compute logprobs", flush=True)
    #             logprob_after = self.actor_rollout_wg.compute_log_prob(logprob_batch)

    #             logprobs = logprob_after.batch['old_log_probs']
    #             entropys = logprob_after.batch['entropys']

    #             # import pdb; pdb.set_trace()

    #             dump_data = {
    #                 "input_ids": batch.batch['input_ids'],
    #                 'attention_mask': batch.batch["attention_mask"],

    #                 "grpo_logprob_after": batch.batch["grpo_logprob_after"],
    #                 "nsr_logprob_after": batch.batch["nsr_logprob_after"],
    #                 "psr_logprob_after": batch.batch["psr_logprob_after"],

    #                 "rollout_logprobs": logprobs,
    #                 "rollout_entropys": entropys,

    #                 "old_log_probs": batch.batch["old_log_probs"],
    #                 "response_masks": batch.batch["response_masks"],
    #                 "responses": batch.batch["responses"],

    #                 "grpo_current_entropys": batch.batch["grpo_current_entropys"],
    #                 "nsr_current_entropys": batch.batch["nsr_current_entropys"],
    #                 "psr_current_entropys": batch.batch["psr_current_entropys"],

    #                 "score": batch.non_tensor_batch["score"]
    #             }

    #             dump_path = f"/home/ma-user/work/dev/_experiments/verl/_psrnsr/EXP44_128_64/playground/onpolicy-distill/{timestamp}/{i}.pt"
    #             os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    #             torch.save(dump_data, dump_path)


            
    #         print(f"[INFO] finish", flush=True)
    #         # self._save_temp_checkpoint("/home/ma-user/work/dev/_experiments/verl/_psrnsr/EXP44_128_64/playground/step40_pie")




    #         # logprob_batch_key = ["input_ids", "attention_mask", "position_ids", "responses"]
    #         # logprob_batch = batch.select(batch_keys=logprob_batch_key)

    #         # # recompute logprob after
    #         # print(f"[INFO] start calculate logprobs", flush=True)
    #         # logprob_after = self.actor_rollout_wg.compute_log_prob(logprob_batch)
    #         # logprobs = logprob_after.batch['old_log_probs']
    #         # entropys = logprob_after.batch['entropys']
    #         # # logdiff = batch.batch['grpo_logprob_after'] - logprobs

    #         # print(f"[INFO] end calculate logprobs", flush=True)

    #         # dump_data = {
    #         #     "input_ids": batch.batch['input_ids'],
    #         #     'attention_mask': batch.batch["attention_mask"],

    #         #     "grpo_logprob_after": batch.batch["grpo_logprob_after"],
    #         #     "nsr_logprob_after": batch.batch["nsr_logprob_after"],
    #         #     "psr_logprob_after": batch.batch["psr_logprob_after"],

    #         #     "rollout_logprobs": logprobs,
    #         #     "rollout_entropys": entropys,


    #         #     "old_log_probs": batch.batch["old_log_probs"],
    #         #     "response_masks": batch.batch["response_masks"],
    #         #     "responses": batch.batch["responses"],

    #         #     "grpo_current_entropys": batch.batch["grpo_current_entropys"],
    #         #     "nsr_current_entropys": batch.batch["nsr_current_entropys"],
    #         #     "psr_current_entropys": batch.batch["psr_current_entropys"],

    #         #     "score": batch.non_tensor_batch["score"]
    #         # }

    #         # # import pdb; pdb.set_trace()
    #         # dump_path = self.config.data.save_path
    #         # torch.save(dump_data, dump_path)
    #         # exit(-1)
