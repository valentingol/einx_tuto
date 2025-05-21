# Functionalities

## Rearrange, repeat, concatenation, split

- `einx.rearrange`: Rearrange a tensor according to a description. Can also be used to
  repeat, concatenate and split axis.

```python
>>> # Rearrange
>>> x = torch.randn(4, 10, 10, 3)
>>> einx.rearrange("(a1 a2) b c d -> a1 a2 d b c", x, a2=2).shape
Size([2, 2, 3, 10, 10])
>>> # Repeat
>>> einx.rearrange("a b c d -> a b c t d", x, t=10).shape
Size([4, 10, 10, 10, 3])
>>> # Concatenate
>>> x, y = torch.randn(7, 10), torch.randn(5, 10)
>>> einx.rearrange("a c, b c -> (a + b) c", x, y).shape
Size([12, 10])
>>> x, y = torch.randn(5, 4, 3), torch.randn(5, 4)
>>> einx.rearrange("a b c, a b -> a b (c + 1)", x, y).shape
Size([5, 4, 4])
>>> x = torch.randn(4, 10, 10, 3)
>>> # Split
>>> res = einx.rearrange("(a1 + a2) b c d -> a1 b c d, a2 b c d", x, a1=2)
>>> print(*(r.shape for r in res))
Size([2, 10, 10, 3]) Size([2, 10, 10, 3])
```

## Elementwise operations

Elementwise operations. When an operation is applied to each element

- `einx.elementwise`: Elementwise operation on one or multiple tensors.
  No need to pass the output shape if it's the same as input(s).
  "op" is a Callable or a supported function name as string.

```python
>>> x = torch.ones(2, 2) * 2.
>>> einx.elementwise("a b -> a b", x, op=torch.square)
tensor([[4., 4.],
        [4., 4.]])
>>> einx.elementwise("a b", x, op=torch.square)
tensor([[4., 4.],
        [4., 4.]])
>>> einx.elementwise("a b, a b -> a b", x, 2*x, op=torch.add)
tensor([[ 6., 6.],
        [ 6., 6.]])
>>> einx.elementwise("a b, a b, a b", x, 2*x, 3*x, op=lambda x, y, z: x*y - z)
tensor([[ 2., 2.],
        [ 2., 2.]])
```

Supported functions for op are:

- "add", similar to `einx.add`: Add two or more tensors elementwise.

```python
>>> x = torch.ones(2, 2) * 2.
>>> einx.add("a b, a b, a b -> a b", x, x, x)
tensor([[6., 6.],
        [6., 6.]])
```

- "substract", similar to `einx.subtract`: Subtract two tensors elementwise.

- "multiply", similar to `einx.multiply`: Multiply two or more tensors elementwise.

- "divide", similar to `einx.divide`: Divide two tensors elementwise.

- "true_divide", similar to `einx.true_divide`: Divide two tensors elementwise.

- "floor_divide", similar to `einx.floor_divide`: Floor divide two tensors elementwise.

- "logical_and", similar to `einx.logical_and`: Logical "and" on two or more tensors elementwise.

- "logical_or", similar to `einx.logical_or`: Logical or two or more tensors elementwise.

- "less", "greater", "less_equal", "greater_equal", "equal", "not_equal" similar to
  `einx.less`, `einx.greater`, ... : Operation "less than", "greater than", ...
  on two tensors elementwise.

- "maximum", "minimum", similar to `einx.maximum`, `einx.minimum`: Elementwise
  maximum/minimum of two or more tensors.

## Reduce operations

Reduce operations. When an operation is applied to the values of a tensor to reduce
its shape along one or more axis. The output shape is specified with "->" in the description or the axis where the operation is applied is marked with brackets.

- `einx.reduce`: Applies a reduction operation on axis. Marked axis (with `[]`)
  are usefull to make short notations. "op" can be a callable (with axis as parameter)
  or a supported string operation.

```python
>>> x = torch.randn(2, 3, 4)
>>> einx.reduce("a b c -> a c", x, op="mean").shape
Size([2, 4])
>>> einx.reduce("a [b] c", x, op="mean").shape
Size([2, 4])
>>> einx.reduce("a b c -> a 1 c", x, op="mean").shape  # keepdims
Size([2, 1, 4])
>>> einx.reduce("a [b] c", x, op="mean", keepdims=True).shape  # keepdims
Size([2, 1, 4])
>>> einx.reduce("a b c -> a c", x, op=lambda x, axis: x.mean(axis=axis)).shape
Size([2, 4])
```

Supporting string operations:

- `sum` (same as `einx.sum`): Sum over axis.

```python
>>> x = torch.randn(2, 3, 4)
>>> einx.sum("a [b] c", x).shape
Size([2, 4])
```

- "mean", similar to `einx.mean`: Mean over axis.

- "var", similar to `einx.var`: Variance over axis.

- "std", similar to `einx.std`: Standard deviation over axis.

- "prod", similar to `einx.prod`: Product over axis.

- "count_nonzero, similar to `einx.count_nonzero`: Count non-zero elements over axis.

- "any", similar to `einx.any`: Logical any over axis.

- "all", similar to `einx.all`: Logical all over axis.

- "max", similar to `einx.max`: Maximum over axis.

- "min", similar to `einx.min`: Minimum over axis.

- "logsumexp", similar to `einx.logsumexp`: Log-sum-exp over axis.

## Vectorized operations

Vectorized operations. When an operation is mapped over 1 or more axis of 1 or more
tensors. Reduced and element-wise operations are vectorized operations as well as
the dot product for instance.

- `einx.vmap`: Flexible operation to vectorized map over axis. It can be used
  to implement all element-wise and reduce operations.
  For reduce operations: the axis to map over should be marked with brackets and
  the output shape should be specified (with "->"). If the output has its own axis,
  they should be marked as well.
  These restrictions are not mandatory for element-wise operations.

```python
>>> x = torch.randn(5, 3, 4)
>>> y = torch.randn(4, 7)
>>> einx.vmap("a b c", x, op=torch.square).shape  # element-wise square
Size([5, 3, 4])
>>> einx.vmap("a b [c] -> a b", x, op=torch.mean).shape  # reduce mean
Size([5, 3])
>>> einx.vmap("a b [c], [c] d -> a b d", x, y, op=torch.dot).shape  # reduce dot
Size([5, 3, 7])
>>> def op(a, b):  # custom reduce op
...     return torch.stack([a.mean(), b.max()])
>>> einx.vmap("a [b] c, c [d] -> c [2] a", x, y, op=op).shape # all brackets and "->" are mandatory
Size([4, 2, 5])
```

- `einx.vmap_with_axis`: Same as `einx.vmap`, but the operation is
  called with the axis index(es) as an additional parameter.
  This is useful for operations that require the axis index as an argument.

```python
>>> x = torch.randn(5, 4)
>>> einx.vmap("a [b] -> a [b]", x, op=torch.softmax).shape  # dim parameter is missing
TypeError: softmax() received an invalid combination of arguments*
...
>>> einx.vmap_with_axis("a [b] -> a [b]", x, op=torch.softmax).shape
Size([5, 4])
```

- `einx.dot`: Dot product of two or more tensors. It's similar to "einsum" functions
  in other libraries.

```python
>>> x, y = torch.randn(5, 4), torch.randn(3, 4)
>>> einx.dot("a d, b d, c d -> a b c", x, y, y).shape
Size([5, 3, 3])
>>> einx.dot("a b, a b ->", x, x).shape  # Frobenius norm
Size([])
```

- `einx.roll`: Roll the tensor along specified axis. Need the shift value and
  the axis to roll over (that should be marked with brackets). Multiple shifts
  must be specified as a tuple when multiple axes are rolled over.

```python
>>> x = torch.randn(5, 4)
>>> einx.roll("a [b]", x, shift=1).shape
Size([5, 4])
>>> einx.roll("[a b]", x, shift=(1, 1)).shape
Size([5, 4])
```

- `einx.flip`: Flip the tensor along specified axis. Need the axis to roll
  over (that should be marked with brackets).

```python
>>> x = torch.randn(5, 4)
>>> einx.flip("a [b]", x).shape
Size([5, 4])
>>> einx.flip("[a b]", x).shape
Size([5, 4])
```

- `einx.softmax`: Softmax over specified axis.

```python
>>> x = torch.randn(5, 4)
>>> einx.softmax("a [b]", x).shape
Size([5, 4])
```

- `einx.log_softmax`: Log softmax over specified axis.

```python
>>> x = torch.randn(5, 4)
>>> einx.log_softmax("a [b]", x).shape
Size([5, 4])
```

## Modify/get values at specific coordinates

Functions to get/set/modify the values of a tensor but only on given coordinates.

- `einx.get_at`: Return values at coordinates ("gather").

```python
>>> x = torch.randn(4, 10, 10, 3)
>>> indices = torch.randint(0, 10, (4, 7, 2))
>>> einx.get_at("a [b c] d, a i [2] -> a i d", x, indices).shape
Size([4, 7, 3])
>>> coords1 = torch.randint(0, 10, (4, 7))
>>> coords2 = torch.randint(0, 10, (4, 7))
>>> einx.get_at("a [b c] d, a i, a i -> a i d", x, coords1, coords2).shape
Size([4, 7, 3])
```

- `einx.set_at`: Set values at coordinates ("scatter").

```python
>>> x1 = torch.randn(4, 10, 10, 3)
>>> indices = torch.randint(0, 10, (4, 7, 2))
>>> x2 = torch.randn(4, 7, 3)
>>> einx.set_at("a [b c] d, a i [2], a i d -> a [b c] d", x1, indices, x2).shape
```

- `einx.add_at`: Add values at coordinates.

- `einx.subtract_at`: Subtract values at coordinates.

## Create a tensor

Functions that create a tensor.

- `einx.arange`: Create a tensor where one axis is the coordinate of the other axis.
  The marked axis is always of size the number of other axis.

```python
>>> x = einx.arange("a b [2]", a=5, b=4, backend="numpy")
>>> x[2, 1]
[2, 1]
>>> x = einx.arange("a b [3] c", a=5, b=4, c=5, backend="numpy")
>>> x[2, 1, :, 4]
[2, 1, 4]
```

## Shape description utils

Functions/classes related to the shape description.

- `einx.solve`: Find axis dimensions of a description based on input tensor(s).
  Returns None if no solution found.

```python
>>> x = torch.randn(5, 4)
>>> einx.solve("a (b c)", x, b=2)
{'a': 5, 'b': 2, 'c': 2}
>>> einx.solve("a (b c)", x)
None
```

- `einx.matches`: Check if tensor(s) match a shape description. Returns True
  if the tensor matches the description, False otherwise.

```python
>>> x = torch.randn(5, 4)
>>> einx.matches("a (b c)", x, b=2)
True
>>> einx.matches("a (b c)", x, b=3)
False
>>> einx.matches("a b c", x)
False
>>> y = torch.randn(5, 6, 3)
>>> einx.matches("a b, a c d", x, y)
True
```

- `einx.check`: Same as `einx.matches`, but raises an exception if the tensor(s)
  do not match the description. Return None otherwise.

- `einx.expr.SolveException`: global exception for all exceptions raised by einx
  due to a wrong description or wrong input tensor(s).

```python
>>> x = torch.randn(5, 4)
>>> try:
...     einx.rearrange("a b c -> a c b", x)
... except einx.expr.SolveException as e:
...    print("Caught exception.")
Caught exception.
```
