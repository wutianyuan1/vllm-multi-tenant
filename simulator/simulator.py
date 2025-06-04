import heapq
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict, Tuple
from simulator.objects import Request, GPU
from simulator.scheduler import Scheduler, RandomScheduler, NaiveSequentialSelectScheduler, LongestFirstScheduler


class Simulator:
    def __init__(self,
                 tenants: Dict[int,List[int]],
                 ngpus: int,
                 per_gpu_max_bsz: int,
                 scheduler: Scheduler):
        self.tenant_order = list(tenants.keys())
        # The launch queues for each tenant, key: tenant id, value: this tenant's requests
        self.all_req_queues: Dict[int, List[Request]] = {
            t: [Request(t, L) for L in lengths]
            for t, lengths in tenants.items()
        }

        self.gpus = [GPU(i, per_gpu_max_bsz) for i in range(ngpus)]
        self.scheduler = scheduler

        # Event queue for finish times, List[(req_id, finish_time)]
        self.event_q: List[Tuple[int,int]] = []
        # The finish time for each request. Key: request id, value: its finish time
        self.req_finish: Dict[int,int] = {}

        # History of request execution, key: request id, value: List[{tenant, gpu.gid, slot, start, end}]
        # Since there may be migrations, each request id may have multiple segments
        self.history: Dict[int,List[Dict]] = {}

        # Tenant finish time, key: tenant id, value: its finish time
        self.finish_time: Dict[int,int] = {}

        self.id2req: Dict[int,Request] = {}
        for q in self.all_req_queues.values():
            for r in q:
                self.id2req[r.id] = r

    def _schedule_finish(self, req: Request, now: int):
        ft = now + req.remaining
        self.req_finish[req.id] = ft
        heapq.heappush(self.event_q, (ft, req.id))

    def _try_launch(self, now: int):
        """Try to launch requests from the queues to the GPUs"""
        # TODO: Implement the scheduler's initial placement logic
        # It should calculate which requests can be launched to which GPUs
        assignments = self.scheduler.try_launch(self.all_req_queues, self.gpus, now, self.history)
        for (req_id, gpu) in assignments:
            req = self.id2req[req_id]
            slot = gpu.assign(req, now)
            assert req.id not in self.history, "Newly launched request should not have history"
            self.history.setdefault(req.id, []).append({
                        'tenant': req.tenant,
                        'gpu': gpu.gid,
                        'slot': slot,
                        'start': now,
                        'end': None
                    })
            # update the finish time for this request
            self._schedule_finish(req, now)

    def run(self):
        # Initial launch
        now = 0
        self._try_launch(now)

        # Event loop
        while self.event_q:
            ft, rid = heapq.heappop(self.event_q)
            if self.req_finish.get(rid) != ft:
                continue
            now = ft
            # Collect a batch of requests that dones at this time
            batch = [(ft, rid)]
            while self.event_q and self.event_q[0][0] == now:
                batch.append(heapq.heappop(self.event_q))

            # Process the done batch: 1) remove them from GPUs, 2) update history
            # 3) Record tenant finish time if all requests of this tenant done
            for _, rid in batch:
                req = self.id2req[rid]
                req.remaining = 0
                # Find the GPU that has this request
                for gpu in self.gpus:
                    if req in gpu.slots:
                        gpu.remove(req)
                        break
                # The request is done, update history
                segs = self.history[rid]
                segs[-1]['end'] = now
                # TODO: Update tenant finish time
                # if all done: finish_time[t] = xxxx
                finished = all([r.id in self.history and self.history[r.id][-1]['end'] is not None for r in self.all_req_queues[req.tenant]])
                if finished:
                    assert now == max([self.history[r.id][-1]['end'] for r in self.all_req_queues[req.tenant]])
                    self.finish_time[req.tenant] = now

            # Trigger migration
            ops = self.scheduler.migrate(self.gpus, now, self.history)
            for req, from_gpu, to_gpu in ops:
                # The previous segment is done, will run on the migrated GPU
                self.history[req.id][-1]['end'] = now
                self.gpus[from_gpu].remove(req)
                req.remaining -= now - self.history[req.id][-1]['start']
                # TODO: migration cost due to data transfer
                self.gpus[to_gpu].assign(req, now)
                self._schedule_finish(req, now)

            # After request done and migration, try to launch new requests as GPUs may have freed up
            self._try_launch(now)

        return self.finish_time, self.history


def plot_timeline(history: Dict[int,List[Dict]],
                  tenants: Dict[int,List[int]],
                  ngpus: int,
                  cap: int,
                  filename: str = 'timeline'):
    cmap = plt.get_cmap('tab10')
    tenant_list = sorted(tenants.keys())
    color_map = {t: cmap(i) for i, t in enumerate(tenant_list)}

    fig, ax = plt.subplots(figsize=(12, ngpus*cap*0.3+1))
    yticks, ylabels = [], []
    for req_id, segs in history.items():
        for seg in segs:
            start, end = seg['start'], seg['end']
            gpu, slot, t = seg['gpu'], seg['slot'], seg['tenant']
            y = gpu*(1+cap) + slot
            ax.barh(y, end-start, left=start,
                    color=color_map[t], edgecolor='k')
            if y not in yticks:
                yticks.append(y)
                ylabels.append(f"GPU{gpu}-S{slot}")

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time")
    ax.set_ylabel("GPUxslot")
    patches = [mpatches.Patch(color=color_map[t], label=f"T{t}") 
               for t in tenant_list]
    ax.legend(handles=patches, title="Tenant", 
              bbox_to_anchor=(1.05,1), loc="upper left")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"simulator/{filename}.png")


if __name__ == "__main__":
    random.seed(0)
    # tenants = {
    #     0: sorted([random.randint(20,100) for _ in range(12)]),
    #     1: sorted([random.randint(20,100) for _ in range(12)]),
    #     2: sorted([random.randint(20,100) for _ in range(12)]),
    # }
    tenants = {
        0: sorted([random.randint(20,40) for _ in range(11)] + [random.randint(80, 100) for _ in range(1)]),
        1: sorted([random.randint(20,40) for _ in range(11)] + [random.randint(80, 100) for _ in range(1)]),
        2: sorted([random.randint(20,40) for _ in range(11)] + [random.randint(80, 100) for _ in range(1)]),
    }
    for t_id, req_lengths in tenants.items():
        print(t_id, req_lengths)
    ngpus = 3
    per_gpu_max_bsz = 4
    lower_bound = {t_id: sum([sum(req_lengths) for req_lengths in list(tenants.values())[: i + 1]]) / (ngpus * per_gpu_max_bsz) for i, t_id in enumerate(tenants)}
    # Sequential
    best_history_and_finish = {t_id: None for t_id in tenants}
    best_history_and_finish['avg'] = None
    best_ft = {t_id: float('inf') for t_id in tenants}
    best_ft['avg'] = float('inf')
    sim = Simulator(tenants, ngpus=ngpus, per_gpu_max_bsz=per_gpu_max_bsz, scheduler=NaiveSequentialSelectScheduler())
    finish, history = sim.run()
    for t_id in tenants:
        if finish[t_id] < best_ft[t_id]:
            best_ft[t_id] = finish[t_id]
            best_history_and_finish[t_id] = (history, finish)
    if sum(finish.values()) / len(tenants) < best_ft['avg']:
        best_ft['avg'] = sum(finish.values()) / len(tenants)
        best_history_and_finish['avg'] = (history, finish)
    print(repr(NaiveSequentialSelectScheduler))
    print(lower_bound)
    last_t_id = list(tenants.keys())[-1]
    print(best_history_and_finish[last_t_id][1], best_ft['avg'])
    plot_timeline(best_history_and_finish[last_t_id][0], tenants, ngpus, per_gpu_max_bsz, 'sequential')

    # Random
    best_history_and_finish = {t_id: None for t_id in tenants}
    best_history_and_finish['avg'] = None
    best_ft = {t_id: float('inf') for t_id in tenants}
    best_ft['avg'] = float('inf')
    for i in range(100):
        random.seed(i)
        sim = Simulator(tenants, ngpus=ngpus, per_gpu_max_bsz=per_gpu_max_bsz, scheduler=RandomScheduler())
        finish, history = sim.run()
        for t_id in tenants:
            if finish[t_id] < best_ft[t_id]:
                best_ft[t_id] = finish[t_id]
                best_history_and_finish[t_id] = (history, finish)
        if sum(finish.values()) / len(tenants) < best_ft['avg']:
            best_ft['avg'] = sum(finish.values()) / len(tenants)
            best_history_and_finish['avg'] = (history, finish)
    print(repr(RandomScheduler))
    print(lower_bound)
    for t_id in list(tenants.keys())[1:] + ['avg']: # ignore the first tenant
        print(t_id, best_history_and_finish[t_id][1], sum(best_history_and_finish[t_id][1].values()) / len(tenants))
        plot_timeline(best_history_and_finish[t_id][0], tenants, ngpus, per_gpu_max_bsz, f'random_for_tenant_{t_id}')

    # Longest First
    best_history_and_finish = {t_id: None for t_id in tenants}
    best_history_and_finish['avg'] = None
    best_ft = {t_id: float('inf') for t_id in tenants}
    best_ft['avg'] = float('inf')
    sim = Simulator(tenants, ngpus=ngpus, per_gpu_max_bsz=per_gpu_max_bsz, scheduler=LongestFirstScheduler())
    finish, history = sim.run()
    for t_id in tenants:
        if finish[t_id] < best_ft[t_id]:
            best_ft[t_id] = finish[t_id]
            best_history_and_finish[t_id] = (history, finish)
    if sum(finish.values()) / len(tenants) < best_ft['avg']:
        best_ft['avg'] = sum(finish.values()) / len(tenants)
        best_history_and_finish['avg'] = (history, finish)
    print(repr(LongestFirstScheduler))
    print(lower_bound)
    print(best_history_and_finish[last_t_id][1], best_ft['avg'])
    plot_timeline(best_history_and_finish[last_t_id][0], tenants, ngpus, per_gpu_max_bsz, 'longest_first')
