import random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from simulator.objects import Request, GPU

class Scheduler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def initial_placement(self,
                          reqs: List[Request],
                          gpus: List[GPU],
                          now: int,
                          history: Dict[int,List[Dict]]):
        """Initially fill the cluster with all reqs in a batch"""
        pass

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
        """
        pass

class RandomScheduler(Scheduler):
    def initial_placement(self,
                          reqs: List[Request],
                          gpus: List[GPU],
                          now: int,
                          history: Dict[int,List[Dict]]):
        """Randomly assign each req to a GPU"""
        random.shuffle(reqs)
        for req in reqs:
            gpu = gpus[req.id % len(gpus)]  # Simple round-robin assignment
            slot = gpu.assign(req, now)
            history[req.id] = [{
                'tenant': req.tenant,
                'gpu': gpu.gid,
                'slot': slot,
                'start': now,
                'end': None
            }]

    def migrate(self,
                gpus: List[GPU],
                now: int,
                history: Dict[int,List[Dict]]
               ) -> List[Tuple[Request,int,int]]:
        return []  # No migration in random scheduler
    
    def try_launch(self,
                   all_req_queues: Dict[int, List[Request]],
                   gpus: List[GPU],
                   now: int,
                   history: Dict[int,List[Dict]]
                  ) -> List[Tuple[int, GPU]]:
        # TODO: implement the logic to randomly select requests from the current tenant's queue
        return []