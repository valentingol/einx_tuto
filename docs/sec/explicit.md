# Explicit operations

It's always possible to get the explicit operations that are used to
compute the `einx` operations. Just add "graph=True" to the call of the function
and print the result.

```python
>>> x = torch.randn(2, 3, 4)
>>> einx.mean("a b c -> c a", x, graph=True)
import torch
def op0(i0):
    x0 = torch.mean(i0, axis=1)
    x1 = torch.permute(x0, (1, 0))
    return x1
```

As you can see, the operations are described by the code itself that is used to
compute the operation using the backend's API. This is useful to understand
what is going on under the hood in some cases.
