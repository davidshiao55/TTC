from vllm.engine.output_processor.single_step import SingleStepOutputProcessor
from vllm.sampling_params import SamplingParams
from vllm.sequence import Sequence, SequenceStatus
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.sequence import SequenceGroup, SequenceGroupOutput
from vllm.utils import init_logger
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.config import SchedulerConfig
from typing import List, Callable
from collections import Counter
from vllm.lora.request import LoRARequest
from typing import Optional
from vllm.core.scheduler import Scheduler

logger = init_logger(__name__)

def is_finished_stopped_with_stop(seq: Sequence) -> bool:
    """
    Check if the sequence is finished stopped with stop reason, 
    but status is not changed as speculative beam extension is enabled.
    """
    return not seq.is_finished() and seq.stop_reason is not None

class SpecStopChecker(StopChecker):
    
    def __init__(self, max_model_len: int,
                 get_tokenizer_for_seq: Callable[[Sequence], AnyTokenizer],
                 scheduler: Scheduler):
        # Do not use it directly, but use `self._get_max_model_len`.
        self._max_model_len = max_model_len
        self.get_tokenizer_for_seq = get_tokenizer_for_seq
        self.scheduler = scheduler
    
    def maybe_stop_sequence(
        self,
        seq: Sequence,
        new_char_count: int,
        sampling_params: SamplingParams,
        lora_req: Optional[LoRARequest] = None,
    ) -> None:
        if len(self.scheduler.waiting) == 0:
            # logger.info(f"No waiting groups, use spec stop checker")
            return self.maybe_stop_sequence_spec(seq, new_char_count, sampling_params, lora_req)
        else:
            return super().maybe_stop_sequence(seq, new_char_count, sampling_params, lora_req)
    
    def maybe_stop_sequence_spec(
        self,
        seq: Sequence,
        new_char_count: int,
        sampling_params: SamplingParams,
        lora_req: Optional[LoRARequest] = None,
    ) -> None:
        """Stop the finished sequences.

       new_char_count is the number of chars added to the
           sequence's output text for the newly generated token
        """

        # Check if the minimum number of tokens has been generated yet;
        # skip the stop string/token checks if not
        if seq.get_output_len() < sampling_params.min_tokens:
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == seq.eos_token_id):
            # Remove the last EOS token unless explicitly specified
            # This prevents unintended exposure of the EOS token
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                seq.output_text = seq.output_text[:-new_char_count]
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if a stop token was encountered.
        # This assumes a single token produced per step.
        last_token_id = seq.get_last_token_id()
        if last_token_id in (sampling_params.stop_token_ids or ()):
            if new_char_count and (
                    not sampling_params.include_stop_str_in_output):
                # Remove last token
                seq.output_text = seq.output_text[:-new_char_count]
            # seq.status = SequenceStatus.FINISHED_STOPPED # do not change the status here to enable speculative beam extension
            # TODO: Design choice we stop the sequence the second time it is stopped
            # if seq.stop_reason is not None:
            #     seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = last_token_id
            return

        # Check if any stop strings are matched.
        stop = self.check_stop_strings(
            seq.output_text, new_char_count, sampling_params.stop,
            sampling_params.include_stop_str_in_output)
        if stop is not None:
            stop_str, truncate_to = stop
            # if truncate_to != -1:
            #     seq.output_text = seq.output_text[:truncate_to]
            # seq.status = SequenceStatus.FINISHED_STOPPED
            # TODO: Design choice we stop the sequence the second time it is stopped
            # if seq.stop_reason is not None:
            #     seq.status = SequenceStatus.FINISHED_STOPPED
            seq.stop_reason = stop_str
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() >= self._get_max_model_len(lora_req):
            seq.stop_reason = None
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.stop_reason = None
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return