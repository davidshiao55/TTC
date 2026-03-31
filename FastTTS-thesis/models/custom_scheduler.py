from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.logger import init_logger
from vllm.sequence import SequenceStatus
from .spec_stopchecker import is_finished_stopped_with_stop
from .numbers import SPEC_BEAM_CANDIDATE_PRIORITY, WAITING_DEFAULT_PRIORITY

logger = init_logger(__name__)

class CustomScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        logger.info("Using CustomScheduler")
        super().__init__(*args, **kwargs)
        logger.info("CustomScheduler initialized with config: %s", self.scheduler_config)
        # self.prefix_aware_scheduling = False

    # def enable_prefix_aware_scheduling(self):
    #     self.prefix_aware_scheduling = True
    #     self.scheduler_config.policy = "priority"
    
    def schedule(self):
        # if self.prefix_aware_scheduling:
            # Re-sort waiting/running queues by priority if present
            # self.update_running_priorities_with_prefix()
        aborted_seq_groups = []
        for state_queue in [self.waiting, self.swapped]:
            for sg in state_queue:
                if is_finished_stopped_with_stop(sg.first_seq):
                    aborted_seq_groups.append(sg)
        for sg in aborted_seq_groups:
            logger.info(f"Freeing sequence {sg.first_seq.seq_id} from the waiting queue")
            sg.first_seq.status = SequenceStatus.FINISHED_STOPPED
            self.abort_seq_group(sg.request_id)
            self.free_seq(sg.first_seq)
            self._finished_requests_ids.append(sg.request_id)
        # logger.info("CustomScheduler schedule with policy: %s", self.scheduler_config.policy)
        return super().schedule()

    # def update_running_priorities_with_prefix(self):
    #     # Only update running group priorities based on current tokens
    #     running = [sg for sg in self.running if not sg.is_finished()]
    #     if not running:
    #         return
        # Get current tokens for each running group
        # tokenized_seqs = []
        # for sg in running:
        #     tokens = sg.first_seq.get_output_token_ids()
        #     tokenized_seqs.append(tokens)
        # from search.utils import assign_prefix_priorities
        # prefix_priorities = assign_prefix_priorities(tokenized_seqs)
        # # Find min waiting priority (or set to a large value if none)
        # waiting_priorities = [sg.priority for sg in self.waiting if sg.priority is not None]
        # min_waiting_priority = min(waiting_priorities) if waiting_priorities else WAITING_DEFAULT_PRIORITY
        # # Offset running priorities to always be higher than waiting
        # for i, sg in enumerate(running):
        #     sg.priority = min_waiting_priority - len(running) + prefix_priorities[i]