import random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from simulator.objects import Request, GPU


class Scheduler(ABC):
    def __init__(self, **kwargs):
        # Whether to allow continuous batching
        self.allow_continuous_batching = kwargs.get('allow_continuous_batching', False)
        # Whether to allow cross-model continuous batching
        # If True, requests from different models can be batched together
        # If False, tenant-B's requests cannot launch to a GPU until all tenant-A's requests on it are finished
        self.allow_cross_model_continuous_batching = kwargs.get('allow_cross_model_continuous_batching', False)
        # allow_cross_model_continuous_batching is True -> spatial-multiplexing, no context switching penalty
        # False -> time-multiplexing, we need to penalize context switching
        if self.allow_cross_model_continuous_batching:
            assert self.allow_continuous_batching, "If we allow cross-model continuous batching, we must allow continuous batching"
            self.context_switch_penalty = 0
        else:
            self.context_switch_penalty = kwargs.get('context_switch_penalty', 0)
        # Migration penanlty is proportional to the number of already-generated tokens
        # We simulate it as `cost=k*number_of_tokens_generated`
        self.migration_penalty_k = kwargs.get('migration_penalty_k', 0)

    @abstractmethod
    def migrate(self,
                gpus: List[GPU],
                now: int,
                history: Dict[int,List[Dict]]
               ) -> List[Tuple[Request,int,int]]:
        """
        Migrate requests between GPUs, return a list of (req, from_gpu_id, to_gpu_id)
        """
        pass

    @abstractmethod
    def try_launch(self,
                   all_req_queues: Dict[int, List[Request]],
                   gpus: List[GPU],
                   now: int,
                   history: Dict[int,List[Dict]]
                  ) -> List[Tuple[int, GPU]]:
        """
        Try to launch requests from the queues to the GPUs.
        Return a list of (req_id, gpu) pairs.
        Raises:
            NotImplementedError: this method should be implemented by subclasses.
        """
        raise NotImplementedError("`try_launch` should be implemented by subclasses")


class SequentialScheduler(Scheduler):
    def __init__(self, **kwargs):
        super(SequentialScheduler, self).__init__(**kwargs)

    def migrate(self,
                gpus: List[GPU],
                now: int,
                history: Dict[int,List[Dict]]
               ) -> List[Tuple[Request,int,int]]:
        return []  # No migration in sequential scheduler
    
    def try_launch(self,
                   all_req_queues: Dict[int, List[Request]],
                   gpus: List[GPU],
                   now: int,
                   history: Dict[int,List[Dict]]
                  ) -> List[Tuple[int, GPU]]:
        # TODO: implement the logic to randomly select requests from the current tenant's queue
        assignments: List[Tuple[int, GPU]] = []
        if self.allow_continuous_batching:
            # Can use any available GPU, even if it has some requests already assigned
            avail_gpus = {gpu: gpu.free_slots() for gpu in gpus if gpu.free_slots() > 0}
        else:
            # Must use an totally empty GPU, i.e., no requests assigned to it
            avail_gpus = {gpu: gpu.free_slots() for gpu in gpus if gpu.free_slots() == gpu.capacity}
        total_free_slots = sum(avail_gpus.values())
        if total_free_slots == 0:
            return []
        # For Case 1 (no continuous batching is allowed) and Case 2 (both cross-model and continuous batching are allowed)
        # The logic can be simplified to sequentially assign a request to any available GPU
        if self.allow_cross_model_continuous_batching or (not self.allow_continuous_batching):
            for tenant, req_queue in all_req_queues.items():
                for r in req_queue: # sequentially select requests
                    if r.id in history:
                        continue    
                    total_free_slots -= 1
                    for gpu, free_slots in avail_gpus.items(): # sequentially select assigned gpus
                        if free_slots > 0:
                            # Assign the request to the GPU.
                            assignments.append((r.id, gpu))
                            avail_gpus[gpu] -= 1
                            # if not allow cross-model continuous batching, then this tenant's batch monopolizes this GPU
                            if not self.allow_cross_model_continuous_batching:
                                gpu.cur_idx = tenant # Set the current tenant for this GPU
                            break
                    assert assignments[-1][0] == r.id
                    if total_free_slots == 0:
                        return assignments
        # Case 3 (continuous batching is allowed but cross-model is not)
        else:
            for tenant, req_queue in all_req_queues.items():
                unlaunched_reqs = [r for r in req_queue if r.remaining != 0 and r.id not in history]
                if len(unlaunched_reqs) == 0:
                    continue
                for gpu, free_slots in avail_gpus.items():
                    # Empty GPU, can assign any request from this tenant
                    if avail_gpus[gpu] == gpu.capacity:
                        req_num = min(gpu.capacity, len(unlaunched_reqs))
                        assignments += [(r.id, gpu) for r in unlaunched_reqs[:req_num]]
                        unlaunched_reqs = unlaunched_reqs[req_num:]
                        avail_gpus[gpu] -= req_num
                        gpu.cur_idx = tenant
                    # This GPU is currently belonging to this tenant, can use continuous batching
                    elif gpu.cur_idx == tenant and avail_gpus[gpu] > 0:
                        req_num = min(gpu.free_slots(), len(unlaunched_reqs))
                        assignments += [(r.id, gpu) for r in unlaunched_reqs[:req_num]]
                        unlaunched_reqs = unlaunched_reqs[req_num:]
                        avail_gpus[gpu] -= req_num
                    # This GPU is full or has other tenant's requests still running
                    else:
                        continue
        return assignments