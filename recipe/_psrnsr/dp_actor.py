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
Single Process Actor
"""

import logging
import os
import gc
import numpy as np

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor
import torch.distributed as dist

import verl.utils.torch_functional as verl_F
from verl import DataProto
# from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from recipe._psrnsr.core_algos import agg_loss, get_policy_loss_fn, kl_penalty, get_global_entropy_top_mask
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

from recipe._psrnsr.gradient_layers import NAME2LAYER

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

        self.gradient_step = 0

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys
    

    def _rank0_summon(self, path, layers):
        rank = dist.get_rank()
        with FSDP.summon_full_params(self.actor_module, with_grads=True, rank0_only=True, writeback=False):
            if dist.get_rank() == 0:
                print(f"[INFO][_rank0_summon] rank0 process, path: {path}", flush=True)
                grad_dict = {}
                for name, p in list(self.actor_module.named_parameters()):
                    is_grad = bool(p.grad is not None)
                    if name in layers:
                        grad_dict[name] = p.grad.detach().cpu().to(torch.bfloat16).clone()
                
                torch.save(grad_dict, path)
                print(f"[INFO] finish save gradient to {path}", flush=True)

        dist.barrier()
        gc.collect()
        torch.cuda.empty_cache() 


    def _split_pos_neg(self, data: DataProto):
        """
        Split data into positive and negative parts based on score > 0.
        
        Args:
            data (DataProto): Input data containing score in non_tensor_batch
            
        Returns:
            tuple[DataProto, DataProto]: (pos_data, neg_data) where pos_data contains
                rows with score > 0, and neg_data contains rows with score <= 0
        """
        # Get score from non_tensor_batch
        if 'score' not in data.non_tensor_batch:
            raise ValueError("'score' not found in data.non_tensor_batch")
        
        score = data.non_tensor_batch['score']
        
        # Convert to numpy array if it's not already
        if isinstance(score, torch.Tensor):
            score_np = score.detach().cpu().numpy()
        elif isinstance(score, np.ndarray):
            score_np = score
        else:
            # If it's a list or other iterable, convert to numpy
            score_np = np.array(score)
        
        # Create boolean mask for positive samples (score > 0)
        pos_mask = score_np > 0
        neg_mask = ~pos_mask
        
        # Convert masks to torch tensors for select_idxs
        pos_mask_torch = torch.from_numpy(pos_mask)
        neg_mask_torch = torch.from_numpy(neg_mask)
        
        # Split data using select_idxs
        pos_data = data.select_idxs(pos_mask_torch)
        neg_data = data.select_idxs(neg_mask_torch)
        
        return pos_data, neg_data
    

    def gradient_analysis(self, data, gradient_path, is_pos=True):
        from verl.protocol import all_gather_data_proto
        import copy
        
        # 1. Gather all data to form a global view
        # We operate on a shallow copy to preserve original data structure for training loop
        global_data = copy.copy(data)
        all_gather_data_proto(global_data, process_group=None) # Uses default PG (World)

        print(f"[INFO][gradient_analysis][dp_actor] all_gather global batch size: {len(global_data)}, is_pos: {is_pos}", flush=True)

        # 2. Split into positive and negative samples
        temperature = data.meta_info["temperature"]
        pos_data, neg_data = self._split_pos_neg(global_data)
        
        target_data = pos_data if is_pos else neg_data
        
        # 3. Process in groups of 8
        dp_size = dist.get_world_size()
        group_size = 8 
        
        num_samples = len(target_data)
        num_groups = num_samples // group_size

        num_groups = min(num_groups, 20)
        
        if dist.get_rank() == 0:
            print(f"[INFO] Gradient Analysis: Found {num_samples} samples ({'Pos' if is_pos else 'Neg'}). "
                  f"Processing {num_groups} groups of size {group_size}.", flush=True)

        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            
            # Select the sub-batch
            indices = torch.arange(start_idx, end_idx)
            sub_batch = target_data.select_idxs(indices)
            
            # Distribute to current rank
            rank = dist.get_rank()
            
            if len(sub_batch) < dp_size:
                continue

            local_chunk = sub_batch.chunk(dp_size)[rank]
            
            if len(local_chunk) == 0:
                continue
                
            # Prepare inputs
            local_chunk = local_chunk.to(get_device_id())
            model_inputs = {**local_chunk.batch, **local_chunk.non_tensor_batch}
            
            # Zero grad
            self.actor_optimizer.zero_grad()
            
            # Forward
            entropy, log_prob = self._forward_micro_batch(
                model_inputs, temperature=temperature, calculate_entropy=False
            )
            
            # --- Recompute Loss ---
            response_mask = model_inputs["response_mask"]
            old_log_prob = model_inputs["old_log_probs"]
            advantages = model_inputs["advantages"]
            
            loss_agg_mode = self.config.loss_agg_mode
            loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
            rollout_is_weights = model_inputs.get("rollout_is_weights", None)
            policy_loss_fn = get_policy_loss_fn(loss_mode)

            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=loss_agg_mode,
                config=self.config,
                rollout_is_weights=rollout_is_weights,
                entropy=entropy,
            )
            
            policy_loss = pg_loss
            
            if self.config.use_kl_loss:
                ref_log_prob = model_inputs["ref_log_prob"]
                kld = kl_penalty(log_prob, ref_log_prob, self.config.kl_loss_type)
                kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef

            loss = policy_loss
            loss.backward()
            
            # Save Gradient
            label = "pos" if is_pos else "neg"
            label_path = os.path.join(gradient_path, f"step_{self.gradient_step}", label)
            if not os.path.exists(label_path):
                os.makedirs(label_path, exist_ok=True)
            label_file = os.path.join(label_path, f"group_{i}.pt")

            dump_layer_config = self.config.get("layer_name")
            all_param_names = NAME2LAYER[dump_layer_config]

            self._rank0_summon(label_file, layers=all_param_names)
            
            # Clear grad
            self.actor_optimizer.zero_grad()
            
        if dist.get_rank() == 0:
            print(f"[INFO] Gradient Analysis ({'Pos' if is_pos else 'Neg'}) Complete.", flush=True)
        
        dist.barrier()
        




    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        count_gradient_step = data.meta_info.get("count_gradient_step", True)
        if count_gradient_step:
            self.gradient_step += 1

        update_mode = data.meta_info.get("actor_update_mode", "normal")
        skip_gradient_analysis = data.meta_info.get("skip_gradient_analysis", False)
        unembedding_param_names = data.meta_info.get("unembedding_param_names", ["lm_head.weight"])
        if isinstance(unembedding_param_names, str):
            unembedding_param_names = [unembedding_param_names]

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        
        # import pdb; pdb.set_trace()

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs", "score"] if has_multi_modal_inputs else ["score", "uid"]

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        print(f"[INFO][update_policy][dp_actor] get batch size: {len(data)}", flush=True)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        save_gradient = self.config.get("save_gradient", False)
        save_this_step = False
        if (
            not skip_gradient_analysis
            and save_gradient
            and self.gradient_step % int(self.config.get("gradient_per_step", 100000)) == 0
        ):
            print(f"[INFO][_psrnsr][dp_actor.py] save gradient this step", flush=True)
            save_this_step = True

        print(
            f"[INFO][_psrnsr][dp_actor.py] update_mode={update_mode}, "
            f"count_gradient_step={count_gradient_step}, on_policy={on_policy}, "
            f"save_gradient={save_gradient}, save_this_step={save_this_step}",
            flush=True,
        )

        if save_this_step:
            print(f"[INFO][_psrnsr] gradient analysis for this step", flush=True)

            self.gradient_analysis(data, self.config.gradient_path, is_pos=True)
            self.actor_optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

            self.gradient_analysis(data, self.config.gradient_path, is_pos=False)
            self.actor_optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

            self.actor_optimizer.zero_grad()
            print(f"[INFO][_psrnsr] gradient analysis finish", flush=True)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # import pdb; pdb.set_trace()

                print(f"[INFO][dp_actor] diff prompt size in minibatch: {len(set(mini_batch.non_tensor_batch['uid']))}", flush=True)

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    use_token_filter = self.config.get("use_token_filter", False)
                    token_filter_method = self.config.get("token_filter_method", None)
                    entropy_top_ratio = self.config.get('entropy_top_ratio', None)
                    entropy_preserve = self.config.get("entropy_preserve", False)
                    

                    if entropy_coeff != 0 or use_token_filter or entropy_preserve:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if on_policy:
                        old_log_prob = log_prob.detach()
                    else:
                        old_log_prob = model_inputs["old_log_probs"]
                        


                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout importance sampling weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)


                    print(f"[INFO] rollout_is_weights: {rollout_is_weights}", flush=True)

                    # NOTE: Both mismatch diagnostic metrics (PPL, KL, etc.) and IS weight metrics
                    # are computed centrally in ray_trainer.py for consistency and efficiency.
                    # This ensures metrics are computed uniformly across all batches at the trainer level
                    # and avoids redundant computation across workers and micro-batches.

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        entropy=entropy,
                    )

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                        }
                    )
                    # 4th return value may be:
                    # - a scalar tensor (legacy)
                    # - a vector tensor (older panel format)
                    # - a dict[str, tensor] (preferred, more readable)
                    pg_clip_monitor = pg_clipfrac_lower
                    if isinstance(pg_clip_monitor, dict):
                        for k, v in pg_clip_monitor.items():
                            vv = v.detach() if torch.is_tensor(v) else v
                            micro_batch_metrics[f"actor-clip/{k}"] = vv.item() if torch.is_tensor(vv) else float(vv)
                        # Keep legacy key for dashboards that expect it.
                        # micro_batch_metrics["actor/pg_clipfrac_lower"] = float("nan")

                    if 'actor-clip/pg_is_clip_sum' in micro_batch_metrics:
                        del micro_batch_metrics['actor-clip/pg_is_clip_sum']
                        micro_batch_metrics[f"actor-clip/pg_is_clip_sum_batch_{batch_idx}"] = pg_clip_monitor['pg_is_clip_sum']

                    append_to_dict(metrics, micro_batch_metrics)

                if update_mode == "unembedding_only":
                    self._filter_gradients_by_name(unembedding_param_names)
                    append_to_dict(
                        metrics,
                        {
                            "actor/unembedding_only_mode": 1.0,
                            "actor/unembedding_only_target_param_count": float(len(unembedding_param_names)),
                        },
                    )

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

    def _filter_gradients_by_name(self, keep_param_names):
        keep_param_names = set(keep_param_names)
        named_parameters = dict(self.actor_module.named_parameters())
        missing_param_names = sorted(keep_param_names - set(named_parameters.keys()))
        if missing_param_names:
            raise ValueError(f"unembedding-only update target params not found: {missing_param_names}")

        for name, param in named_parameters.items():
            if name not in keep_param_names:
                param.grad = None
