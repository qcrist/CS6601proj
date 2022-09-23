# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from collections import namedtuple
from math import *
from typing import *

import networkx

if TYPE_CHECKING:
    from explorable_graph import ExplorableGraph


class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""

        self.queue = []
        self._ctr = 0
        self._size = 0

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """
        self._size -= 1

        prio, ctr, value = heapq.heappop(self.queue)
        return prio, value

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        prio, value = node
        heapq.heappush(self.queue, (prio, self._ctr, value))
        self._ctr += 1
        self._size += 1

    def __len__(self):
        return self._size

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """
        return self._size

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in the queue.
        """

        top = self.queue[0]

        return top[0], top[2]


def _unroll_src_bi(src, at, reverse=True):
    result = [at]
    while src[result[-1]][0]:
        result.append(src[result[-1]][0])
    if reverse:
        return list(reversed(result))
    else:
        return list(result)


Pushed = namedtuple('Pushed', ['node', 'dist', 'from_node'])
BestTouch = namedtuple('BestTouch', ['dist', 'path'])
NodeType = str


def _ensure_first(path: List[NodeType], first: NodeType):
    if path[-1] == first:
        return list(reversed(path))
    return path


def _list_contains_all(l, of_all):
    return all(x in l for x in of_all)


def return_your_name():
    """Return your name from this function"""
    return "Quintin Crist"


# from line_profiler_pycharm import profile

# Extra Credit: Your best search method for the race
# @profile
def custom_search(graph: ExplorableGraph, a, b, data: LoadDataData = None, with_p1b=False):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to Gradescope, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).

    #	Quintin0	366705	208.97496115326982	9.614548157000002

    """

    if a == b:
        return []

    # 'start', "queue", "src", "explored", "target"
    frontier_f = (a, PriorityQueue(), {}, set(), b)
    frontier_r = (b, PriorityQueue(), {}, set(), a)

    qpos = data.pos

    for start, queue, src, explored, target in [frontier_f, frontier_r]:
        queue.append((0, (start, 0, None)))
        src[start] = None, 0

    best_touch: Optional[BestTouch] = None

    def _make_path_with_w(node: NodeType, src_1, src_2):
        p1 = _unroll_src_bi(src_1, node, True)
        p2 = _unroll_src_bi(src_2, node, False)
        d1 = src_1[node][1]
        d2 = src_2[node][1]
        return [*p1, *p2[1:]], d1 + d2

    start: NodeType
    queue: PriorityQueue
    src: Dict[NodeType, Tuple[NodeType, float]]
    explored: Set[NodeType]
    target: NodeType

    while 1:  # YOLO

        if 0 < frontier_r[1]._size < frontier_f[1]._size:
            start, queue, src, explored, target = frontier_r
            _, _, o_src, o_explored, _ = frontier_f
        else:
            start, queue, src, explored, target = frontier_f
            _, _, o_src, o_explored, _ = frontier_r
        (prio, (node, dist, from_node)) = queue.pop()

        if node in explored:
            continue

        explored.add(node)

        if node in o_explored or node == target:
            break

        neighbors: List[NodeType] = list(graph.neighbors(node))
        nblen = len(neighbors)
        for nn in neighbors:
            if nn in explored:
                continue

            if data.is_de[(node, nn)]:
                # ignore deadends, I dont think its an issue if the goal is a deadend, as it starts from the deadend
                continue

            if nblen > 2:
                nn, skip_dist = data.skips[node, nn]
                distance = dist + skip_dist
            else:
                distance = graph.get_edge_weight(nn, node) + dist

            x1, y1 = qpos[nn]
            x2, y2 = qpos[target]
            h = hypot(x2 - x1, y2 - y1) * 20 + distance
            queue.append((h, (nn, distance, node)))

            get = src.get(nn, None)
            if get is None or get[1] > distance:
                src[nn] = node, distance

            if nn in o_src:
                p, w = _make_path_with_w(nn, src, o_src)
                if best_touch is None or w < best_touch.dist:
                    best_touch = BestTouch(path=p, dist=w)
    p = _ensure_first(best_touch.path, first=a)
    new_p = []
    for p1, p2 in zip(p, p[1:]):
        new_p.append(p1)
        lookup = data.skips_lookup.get((p1, p2), None)
        if lookup is not None:
            skip_dist, skip_nodes = lookup
            nb = set(graph.neighbors(p1))
            if p2 not in nb or graph.get_edge_weight(p1, p2) > skip_dist:
                new_p.extend(skip_nodes)
    new_p.append(p[-1])
    if with_p1b:
        return new_p, p
    return new_p


# @dataclass
class LoadDataData:
    __slots__ = ("skips", "skips_lookup", "pos", "is_de")
    skips: Dict[Tuple[NodeType, NodeType], Tuple[NodeType, float]]
    skips_lookup: Dict[Tuple[NodeType, NodeType], List[NodeType]]
    pos: Dict[NodeType, Tuple[float, float]]
    is_de: Dict[Tuple[NodeType, NodeType], bool]


def load_data(graph: ExplorableGraph, time_left):
    """
    Feel free to implement this method. We'll call it only once
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    ld = LoadDataData()
    ld.skips = {}
    ld.skips_lookup = {}
    ld.pos = {}
    ld.is_de = defaultdict(lambda: False)

    deadend_nodes = set()
    nodes = graph.nodes()

    for a in nodes:
        nb_a = list(graph.neighbors(a))
        if len(nb_a) == 1:
            deadend_nodes.add(a)

    i = 0
    todo = {
        y
        for x in deadend_nodes
        for y in graph.neighbors(x)
        if y not in deadend_nodes
    }
    while True:
        i += 1
        last_len = len(deadend_nodes)
        nxt = set()
        for a in todo:
            nb_a = list(graph.neighbors(a))
            nde = [x for x in nb_a if x not in deadend_nodes]
            if len(nde) <= 1:
                deadend_nodes.add(a)
                for x in nde:
                    nxt.add(x)
        todo = nxt
        if len(deadend_nodes) == last_len:
            break

    for a in nodes:
        nb_a = list(graph.neighbors(a))
        # ld.nbc[a] = len(nb_a)
        ld.pos[a] = graph.nodes[a]['pos']  # slow for some reason...? lets cache it
        a_is_de = a in deadend_nodes

        for n in nb_a:
            if len(list(graph.neighbors(n))) == 1:
                ld.is_de[(a, n)] = True

        if len(nb_a) > 2:
            for o in nb_a:
                o_is_de = o in deadend_nodes
                if not a_is_de and o_is_de:
                    ld.is_de[(a, o)] = True

                orig = o
                nb = list(graph.neighbors(o))
                prev = a
                dist = graph.get_edge_weight(o, a)
                nodes = []
                while len(nb) == 2:
                    l = [x for x in nb if x != prev][0]
                    nodes.append(o)
                    dist += graph.get_edge_weight(o, l)
                    prev, o = o, l
                    nb = list(graph.neighbors(o))
                ld.skips[(a, orig)] = (o, dist)
                # opt[(a,o)].append((dist, ctr, nodes))
                # ctr += 1
                if (a, o) in ld.skips_lookup:
                    old_dist, test = ld.skips_lookup[a, o]
                    if old_dist > dist:
                        ld.skips_lookup[a, o] = dist, nodes
                else:
                    ld.skips_lookup[(a, o)] = dist, nodes
                if len(nb) == 1:
                    #     if not ld.is_de[a, orig]:
                    #         print("?")
                    ld.is_de[(a, orig)] = True

    return ld
