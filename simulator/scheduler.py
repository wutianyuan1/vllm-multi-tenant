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
        assignments: List[Tuple[int, GPU]] = []
        total_free_slots = sum([gpu.free_slots() for gpu in gpus])
        if total_free_slots == 0:
            return []
        avail_gpus = {gpu: gpu.free_slots() for gpu in gpus if gpu.free_slots() > 0}
        for tenant, req_queue in all_req_queues.items():
            for r in random.sample(req_queue, len(req_queue)): # randomly select requests
                if r.id not in history:
                    total_free_slots -= 1
                    for gpu, free_slots in avail_gpus.items(): # sequentially select assigned gpus
                        if free_slots > 0:
                            assignments.append((r.id, gpu))
                            avail_gpus[gpu] -= 1
                            break
                    assert assignments[-1][0] == r.id
                    if total_free_slots == 0:
                        return assignments
        return assignments

class NaiveSequentialSelectScheduler(Scheduler):
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
        return []
    
    def try_launch(self,
                   all_req_queues: Dict[int, List[Request]],
                   gpus: List[GPU],
                   now: int,
                   history: Dict[int,List[Dict]]
                  ) -> List[Tuple[int, GPU]]:
        assignments: List[Tuple[int, GPU]] = []
        total_free_slots = sum([gpu.free_slots() for gpu in gpus])
        if total_free_slots == 0:
            return []
        avail_gpus = {gpu: gpu.free_slots() for gpu in gpus if gpu.free_slots() > 0}
        for tenant, req_queue in all_req_queues.items():
            for r in req_queue: # sequentially select requests
                if r.id not in history:
                    total_free_slots -= 1
                    for gpu, free_slots in avail_gpus.items(): # sequentially select assigned gpus
                        if free_slots > 0:
                            assignments.append((r.id, gpu))
                            avail_gpus[gpu] -= 1
                            break
                    assert assignments[-1][0] == r.id
                    if total_free_slots == 0:
                        return assignments
        return assignments

class LongestFirstScheduler(Scheduler):
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
        return []
    
    def try_launch(self,
                   all_req_queues: Dict[int, List[Request]],
                   gpus: List[GPU],
                   now: int,
                   history: Dict[int,List[Dict]]
                  ) -> List[Tuple[int, GPU]]:
        assignments: List[Tuple[int, GPU]] = []
        total_free_slots = sum([gpu.free_slots() for gpu in gpus])
        if total_free_slots == 0:
            return []
        avail_gpus = {gpu: gpu.free_slots() for gpu in gpus if gpu.free_slots() > 0}
        for tenant, req_queue in all_req_queues.items():
            # select request with the maximum length first
            for r in sorted([r for r in req_queue if r.id not in history], key=lambda r: r.length, reverse=True):
                if r.id not in history:
                    total_free_slots -= 1
                    for gpu, free_slots in avail_gpus.items(): # sequentially select assigned gpus
                        if free_slots > 0:
                            assignments.append((r.id, gpu))
                            avail_gpus[gpu] -= 1
                            break
                    assert assignments[-1][0] == r.id
                    if total_free_slots == 0:
                        return assignments
        return assignments