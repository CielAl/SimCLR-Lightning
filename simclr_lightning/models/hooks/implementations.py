"""A handy piece of code derived from fastai.callback.hook package - simplified and decouple from fastai requirement
"""
from typing import Callable, List, Optional, Generator, Sequence
import torch
from torch import nn

TYPE_HK_FN = Callable[[nn.Module, Optional[torch.Tensor], Optional[torch.Tensor]], torch.Tensor | List[torch.Tensor]]


class HookSimple:
    """A simplified implementation of fastai.callback.hook.Hook

    """
    hook_func: TYPE_HK_FN
    stored: Optional[torch.Tensor]
    removed: bool

    def __init__(self, m, hook_func: TYPE_HK_FN, is_forward: bool = True):
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.stored = None
        self.removed = False
        self.hook_func = hook_func

    def hook_fn(self, module, in_data, out_data):
        self.stored = self.hook_func(module, in_data, out_data)

    def remove(self):
        """Remove the hook
        Returns:

        """
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def _hook_inner(m, i, o):
    """
    identity mapping to gather the output of the hook, derived from fastai.callback.hook.Hook._hook_inner
    Args:
        m:
        i:
        o:

    Returns:

    """
    if isinstance(o, (torch.Tensor, Sequence, Generator)):
        return o
    return list(o)


def register_output_hook(module: nn.Module, grad: bool = False):
    """Derived and simplified from fastai.callback.hook.Hook.hook_output.
    Args:
        module:
        grad:

    Returns:

    """
    return HookSimple(module, _hook_inner, is_forward=not grad)
