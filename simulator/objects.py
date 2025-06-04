import itertools
from typing import List, Dict

class Request:
    """An inference request, needs `length` time-units to process"""
    _ids = itertools.count(0)
    def __init__(self, tenant: str, length: int):
        self.id = next(Request._ids)
        self.tenant = tenant
        self.length = length
        # Remaining tokens to process
        self.remaining = length
        # Record the start time of the current segment, for remaining calculation
        self._last_start = None

    def __repr__(self):
        return f"<Req{self.id} T={self.tenant} rem={self.remaining}>"


class GPU:
    def __init__(self, gid: int, capacity: int):
        self.gid = gid
        self.capacity = capacity
        self.slots: List[Request] = [None] * capacity
        self.cur_idx = 0  # a pointer to tenant_order

    def free_slots(self) -> int:
        return sum(1 for x in self.slots if x is None)

    def assign(self, req: Request, now: int) -> int:
        for i in range(self.capacity):
            if self.slots[i] is None:
                self.slots[i] = req
                req._last_start = now
                return i
        raise RuntimeError("No free slot")

    def remove(self, req: Request) -> int:
        for i, r in enumerate(self.slots):
            if r is not None and r.id == req.id:
                self.slots[i] = None
                return i
        raise RuntimeError("Request not found")

    def load(self) -> int:
        """Current load: sum of remaining on all slots"""
        return sum(r.remaining for r in self.slots if r)

    def __repr__(self):
        return f"<GPU{self.gid} T={self.cur_idx} free_slots={self.free_slots()}>"