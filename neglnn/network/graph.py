from collections import deque, defaultdict
from neglnn.layers.layer import Layer
from neglnn.utils.types import InputKey

class Graph:
    INPUT = 'input'

    def __init__(self):
        self.parents: dict[Layer, dict[InputKey, Layer]] = defaultdict(dict)
        self.children: dict[Layer, dict[InputKey, Layer]] = defaultdict(dict)
        self.layers: set[Layer] = set()

    def connect(self, parent: Layer, child: Layer, key: InputKey = INPUT):
        self.parents[child][key] = parent
        self.children[parent][key] = child
        self.layers.add(parent)
        self.layers.add(child)

    def get_ordered_dependencies(self, sources: list[Layer], sink: Layer) -> list[Layer]:
        seen: set[Layer] = set()
        path: list[Layer] = list()
        queue = deque(sources)
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            
            if any(
                p not in seen and node not in sources
                for p in self.parents[node].values()
            ):
                queue.insert(1, node)
                continue

            seen.add(node)
            path.append(node)
            
            if node == sink:
                break

            queue.extend(self.children[node].values())
        return path

    def get_sources(self) -> list[Layer]:
        return [layer for layer in self.layers if not self.parents[layer]]

    def get_sinks(self) -> list[Layer]:
        return [layer for layer in self.layers if not self.children[layer]]

    def copy(self, sources: list[Layer], sink: Layer) -> 'Graph':
        layers = self.query(sources, sink)
        graph = Graph()
        for parent in layers:
            for key, child in self.children[parent].items():
                graph.connect(parent, child, key)
        return graph