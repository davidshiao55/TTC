import torch
from torch import nn
from typing import Iterable

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.pooler.tokwise.poolers import TokenPooler
from vllm.model_executor.layers.pooler.tokwise.methods import AllPool
from vllm.model_executor.layers.pooler.tokwise.heads import (
    TokenClassifierPoolerHead,
)
from vllm.model_executor.models.interfaces import SupportsPP


class ValueHead(nn.Module):
    """Returns a scalar reward for each output token."""

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = (
            nn.Dropout(summary_dropout_prob)
            if summary_dropout_prob
            else nn.Identity()
        )

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
        output = self.summary(output)
        return output


class Qwen2ForPrmModel(nn.Module, SupportsPP):
    """Skywork PRM model for vLLM V1.

    Uses ValueHead to produce per-token reward scores, then AllPool-based
    TokenPooler to slice them back into per-request tensors.  AllPool handles
    chunked-prefill accumulation so this works correctly regardless of the
    enable_chunked_prefill setting.
    """

    is_pooling_model = True

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        if (
            cache_config.sliding_window is not None
            and hasattr(config, "max_window_layers")
        ):
            raise ValueError(
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = "
                f"{config.max_window_layers} is less than "
                f"`num_hidden_layers` = {config.num_hidden_layers}. "
                "Please open an issue to discuss this feature."
            )

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        self.model = Qwen2Model(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.v_head = ValueHead(self.config)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Use vLLM's built-in AllPool for per-token slicing.  AllPool
        # accumulates hidden states across chunked-prefill steps, unlike
        # a naive slice which loses earlier chunks.
        # TokenClassifierPoolerHead with no classifier is a passthrough.
        self.pooler = TokenPooler(
            pooling=AllPool(),
            head=TokenClassifierPoolerHead(),
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        logits = self.v_head(hidden_states)
        return logits

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self, ignore_unexpected_prefixes=["lm_head."]
        )
        return loader.load_weights(weights)


def register():
    from vllm import ModelRegistry

    if "Qwen2ForPrmModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Qwen2ForPrmModel",
            "vllm_add_dummy_model.prm_model:Qwen2ForPrmModel",
        )
