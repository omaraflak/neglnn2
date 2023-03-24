from collections import deque, defaultdict
from neglnn.layers.layer import Layer
from neglnn.utils.types import InputKey

Uid = str

class Graph:
    INPUT = 'input'

    def __init__(self):
        self.parents: dict[Layer, dict[Uid, dict[InputKey, Layer]]] = defaultdict(lambda: defaultdict(dict))
        self.children: dict[Layer, dict[Uid, dict[InputKey, Layer]]] = defaultdict(lambda: defaultdict(dict))
        self.layers: set[Layer] = set()

    def connect(self, parent: Layer, child: Layer, key: InputKey = INPUT):
        self.parents[child][parent.uid][key] = parent
        self.children[parent][child.uid][key] = child
        self.layers.add(parent)
        self.layers.add(child)

    def get_ordered_dependencies(self, source: Layer, sink: Layer) -> list[Layer]:
        seen: set[Layer] = set()
        path: list[Layer] = list()
        queue = deque([source])
        while queue:
            node = queue.popleft()
            if node in seen:
                continue
            
            if any(
                p not in seen and node != source
                for d in self.parents[node].values()
                for p in d.values()
            ):
                queue.insert(1, node)
                continue

            seen.add(node)
            path.append(node)
            
            if node == sink:
                break

            queue.extend([c for d in self.children[node].values() for c in d.values()])
        return path

    def get_sources(self) -> list[Layer]:
        return [layer for layer in self.layers if not self.parents[layer]]

    def get_sinks(self) -> list[Layer]:
        return [layer for layer in self.layers if not self.children[layer]]

    def copy(self, source: Layer, sink: Layer) -> 'Graph':
        layers = self.get_ordered_dependencies(source, sink)
        graph = Graph()
        for parent in layers:
            for _, child_dict in self.children[parent].items():
                for key, child in child_dict.items():
                    graph.connect(parent, child, key)
        return graph