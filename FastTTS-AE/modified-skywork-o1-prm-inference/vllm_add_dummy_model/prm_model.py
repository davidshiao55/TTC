import os
import torch
import re
import sys
import vllm
from torch import nn

import vllm.engine.llm_engine

from typing import Optional, List, Iterable, Union, Tuple
from vllm.config import PoolerConfig
from vllm.model_executor.pooling_metadata import PoolingMetadata, PoolingTensors
from vllm.sequence import PoolingSequenceGroupOutput, PoolerOutput
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.pooler import PoolingType, Pooler
from vllm.model_executor.models.interfaces import SupportsPP

class ValueHead(nn.Module):
    r"""
    The ValueHead class implements a head for GPT2 that returns a scalar for each output token.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        if not hasattr(config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = config.summary_dropout_prob

        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()

        # some models such as OPT have a projection layer before the word embeddings - e.g. OPT-350m
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size

        self.summary = nn.Linear(hidden_size, 1)

        self.flatten = nn.Flatten()

    def forward(self, hidden_states):
        output = self.dropout(hidden_states)

        # For now force upcast in fp32 if needed. Let's keep the
        # output in fp32 for numerical stability.
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)

        # print('enter here')
        # print('output1.shape: ', output.shape)
        output = self.summary(output)
        # print('output2.shape: ', output.shape)
        return output

class Qwen2ForPrmModel(nn.Module, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
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
        pooler_config = vllm_config.model_config.pooler_config
        # TODO (@robertgshaw2): see if this can be moved out
        if (cache_config.sliding_window is not None
                and hasattr(config, "max_window_layers")):
            raise ValueError("Sliding window for some but all layers is not "
                            "supported. This model uses sliding window "
                            "but `max_window_layers` = {} is less than "
                            "`num_hidden_layers` = {}. Please open an issue "
                            "to discuss this feature.".format(
                                config.max_window_layers,
                                config.num_hidden_layers,
                            ))

        self.config = config
        self.lora_config = lora_config

        self.quant_config = quant_config
        self.model = Qwen2Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))
        self.v_head = ValueHead(self.config)

        self._pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.ALL,
            normalize=False,
            softmax=False
        )
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
    
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        hidden_states = self.model(input_ids, positions, intermediate_tensors,
                                   inputs_embeds)
        logits = self.v_head(hidden_states)
        return logits

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self,
                                ignore_unexpected_prefixes=["lm_head."])
        loader.load_weights(weights)

def register():
    from vllm import ModelRegistry
    if "Qwen2ForPrmModel" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model("Qwen2ForPrmModel", "vllm_add_dummy_model.prm_model:Qwen2ForPrmModel")
