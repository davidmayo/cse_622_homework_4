from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable
import random

import numpy as np


random.seed(40351)


@dataclass
class UniformInterval:
    min: float
    max: float

    def __repr__(self) -> str:
        if self.min == self.max:
            return repr(self.min)
        else:
            return f"[{self.min!r},{self.max!r}]"
        
    def __hash__(self) -> int:
        return hash((self.min, self.max))
    
    def best_case(self) -> float:
        return self.min
    
    def worst_case(self) -> float:
        return self.max

    def random_case(self) -> float:
        """Get a uniform random value in `[min,max]`"""
        return random.random() * (self.max - self.min) + self.min


@dataclass
class Node:
    id: int
    parents: dict[UniformInterval, "Node"] = field(default_factory=dict)
    children: dict[UniformInterval, "Node"] = field(default_factory=dict)

    instance_time: float | None = None
    instance_critical_path: tuple[int, ...] | None = None

    def reset(self) -> None:
        # Clear the cache
        self.instance_time = None
        self.instance_critical_path = None

    def __hash__(self) -> int:
        return hash(id(self))

    def add_child(self, child: "Node", weight: UniformInterval) -> None:
        self.children[weight] = child
        child.parents[weight] = self

    def analyze(
        self,
        method: Callable[[UniformInterval], float],
    ) -> tuple[float, tuple[int, ...]]:
        """Get a tuple of the total time for a simulation, and the critical path from that sim,
        in the form `(time, (node_id, ...))`

        `method` should be `UniformInterval.best_case`, `UniformInterval.worst_case`,
        or `UniformInterval.random_case`
        """
        # Return cached values, if already calculated for this node
        # Helps performance (memoization), but more importantly prevents
        # calculating different results in the random case
        if self.instance_critical_path:
            return (self.instance_time, self.instance_critical_path)

        # If not cached, do the calculations
        worst_time = 0.0
        critical_path = (self.id, )
        for parent_weight, parent_node in self.parents.items():
            parent_time, parent_critical = parent_node.analyze(method=method)
            time = parent_time + method(parent_weight)
            if time > worst_time:
                worst_time = time
                critical_path = parent_critical + (self.id,)

        # Cache results
        self.instance_time, self.instance_critical_path = (worst_time, critical_path)
        
        # Return
        return (self.instance_time, self.instance_critical_path)


if __name__ == "__main__":
    # Initialize the sim
    n1 = Node(1)
    n2 = Node(2)
    n3 = Node(3)
    n4 = Node(4)
    n5 = Node(5)
    n6 = Node(6)
    n7 = Node(7)
    n1.add_child(n2, UniformInterval(3, 5))
    n1.add_child(n5, UniformInterval(6, 6))
    n2.add_child(n3, UniformInterval(6, 6))
    n2.add_child(n4, UniformInterval(7, 9))
    n3.add_child(n4, UniformInterval(5, 8))
    n4.add_child(n7, UniformInterval(4, 4))
    n5.add_child(n3, UniformInterval(7, 7))
    n5.add_child(n4, UniformInterval(9, 9))
    n5.add_child(n6, UniformInterval(7, 10))
    # n6.add_child(n7, UniformInterval(8, 12))
    n6.add_child(n7, UniformInterval(8, 10))
    all_nodes = {n1, n2, n3, n4, n5, n6, n7}
    all_edges = []
    for node in all_nodes:
        for child in node.children.values():
            all_edges.append((node.id, child.id))
    all_edges.sort()

    # Do the analysis
    counter = Counter()
    times_by_path = defaultdict(list)
    number_of_sims = 10000

    def mean(lst: list[float]) -> float:
        return sum(lst) / len(lst)

    for sim in range(number_of_sims):
        for node in all_nodes:
            node.reset()
        time, critical_path = n7.analyze(UniformInterval.random_case)
        times_by_path[critical_path].append(time)
        counter[critical_path] += 1
        # print(f"{sim=}", time, critical_path)

    all_times = []
    for path, time in times_by_path.items():
        all_times.extend(time)
    print(counter)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax: plt.Axes
    bins = np.arange(22.0, 28.0001, 0.5)
    colors = ["#ff0000", "#00ff00"]
    path_keys = list(times_by_path)
    ax.hist(
        [times_by_path[key] for key in path_keys],
        bins=bins,
        edgecolor="black",
        label=[f"path={key}" for key in path_keys],
        color=[f"{colors[index]}88" for index in range(len(path_keys))],
        # rwidth=1.0,
    )
    for index, key in enumerate(path_keys):
        ax.axvline(
            mean(times_by_path[key]),
            label=f"mean path={key}={mean(times_by_path[key]):.2f}",
            color=colors[index],
            linewidth=3,
        )
    ax.axvline(
        mean(all_times),
        label=f"mean overall={mean(all_times):.2f}",
        color="black",
        linewidth=3,
    )
    ax.set_xticks(bins)
    ax.set_title(f"Total time distribution ({number_of_sims:,} sims)")
    ax.legend()
    plt.show()