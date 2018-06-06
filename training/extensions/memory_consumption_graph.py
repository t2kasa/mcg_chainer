import os

from chainer.training import extension
from chainer.training.triggers import ManualScheduleTrigger
from cupy.cuda.memory_hooks import LineProfileHook


class MemoryConsumptionGraph(extension.Extension):
    """Trainer extension to dump a memory consumption graph.

    This extension dumps a memory consumption graph. The graph is output in DOT
    language.

    It only dumps a graph at the first invocation.

    Args:
        out_name (str): Output file name.

    """

    def __init__(self, out_name='mcg.dot'):
        self._out_name = out_name
        self._hook = LineProfileHook()
        self.trigger = ManualScheduleTrigger(1, 'iteration')

    def initialize(self, _):
        self._hook.__enter__()

    def __call__(self, trainer):
        self._hook.__exit__()
        mcg = self._to_dot(self._hook._root)

        out_path = os.path.join(trainer.out, self._out_name)
        with open(out_path, 'w') as f:
            f.write(mcg)

        self._done = True

    def _to_dot(self, mf_root):
        dot = []
        dot += 'digraph mcg {\n'
        self._expand_nodes(_MemoryFrameNode(mf_root), dot)
        dot += '}\n'
        return ''.join(dot)

    def _expand_nodes(self, parent_node, dot):
        dot += parent_node.label
        for child in parent_node.mf.children:
            child_node = _MemoryFrameNode(child)
            dot += child_node.label
            dot += '{.id_} -> {.id_}\n'.format(parent_node, child_node)
            self._expand_nodes(child_node, dot)


class _MemoryFrameNode(object):
    def __init__(self, mf):
        self.mf = mf
        self.id_ = id(mf)

        if self._is_root(mf):
            self._attrs = {'label': 'root\n({}, {})>'.format(
                *mf.humanized_bytes())}
        else:
            st = mf.stackframe
            self._attrs = {'label': '{}:{}:{}\n({}, {})\n'.format(
                os.path.basename(st.filename), st.lineno, st.name,
                *mf.humanized_bytes())}

    def _is_root(self, mf):
        return mf.stackframe is None

    @property
    def label(self):
        attributes = ['{}="{}"'.format(k, v) for (k, v) in self._attrs.items()]
        return '{} [{}];\n'.format(self.id_, ','.join(attributes))
