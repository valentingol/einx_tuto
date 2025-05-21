# Einx Tuto

einx is a Python library that provides a universal interface to formulate tensor
operations in frameworks such as Numpy, PyTorch, Jax and Tensorflow.

Github project can be found here: https://github.com/fferflo/einx

## Installation

```bash
pip install einx
```

More information here for a specific installation
[here](https://einx.readthedocs.io/en/stable/gettingstarted/installation.html).

## Introduction to `einx` and `einops`

Einx is a library inspired by the popular [einops](https://github.com/arogozhnikov/einops)
library (with more functionalities). Their goal is to provide a simple and
intuitive way to manipulate tensors in a flexible and readable manner using the
same lines of code for all tensors backends (Numpy, Pytorch, Jax, Tensorflow, ...).

In this section, we are going to talk about features shared by `einops` and `einx` so
this section is mostly useful for people that are not familiar with `einops` yet.

For instance you want to apply a dense map in an image and rearranging the output
to a shape that fit a pytorch batch. You may want to do something like this:

```python
dense = torch.randn(3, 16)
img = torch.randn(224, 224, 3)
img = img @ dense
img = torch.permute(img, (2, 0, 1))
batch = torch.unsqueeze(img, dim=0)
```

This is a bit verbose and not very readable. Maybe you have to add some
comments to explain what you are doing, especially to track the shape of the
tensors. Like this:

```python
dense = torch.randn(3, 16)
img = torch.randn(224, 224, 3)
img = img @ dense  # shape: (224, 224, 16)
img = torch.permute(img, (2, 0, 1))  # shape: (16, 224, 224)
batch = torch.unsqueeze(img, dim=0)  # shape: (1, 16, 224, 224)
```

Now it's better but still verbose and not easy to maintain if the shape will
change in the future. Plus you have to remember all those pytorch functions that can be a pain.
Their names are not necessarily the same across all the backends (numpy, pytorch, jax, ...) leading to even more
functions to remember.

That's where `einx` (and `einops`) comes in. They add general functions to manipulate tensors
that take in input the full description of the inputs and outputs shapes as strings.
The comments you wrote to track the shapes are now directly in th code itself,
guaranteeing that the code is always up to date and readable. Plus the functions names and usage
are unified for all the backends. Here it is your code with `einx`:

```python
dense = torch.randn(3, 16)
img = torch.randn(224, 224, 3)
img = einx.dot("h w c, c dim -> h w dim", img, dense)
img = einx.rearrange("h w dim -> 1 dim h w", img)
```

Here you can see that full description of the input and output shapes.
You can even give more descriptive names to the axes. Once you get familiar with
the descriptions and the tensor manipulation, you can be even more efficient
and write the code directly like this:

```python
dense = torch.randn(3, 16)
img = torch.randn(224, 224, 3)
img = einx.dot("h w c, c dim -> 1 dim h w", img, dense)
```

Here you can see that the `dot` function not only does the dot product but also
rearranges the output to the shape you want.

`einx` and `einops` also add the possibility to specify the dimensions of
some axis when needed. For instance if you want to split a batch to multiple batches
of lower size (here 32), you can do it like this:

```python
batch = torch.randn(128, 32)
batches = einx.rearrange("(n_batch bsize) dim -> n_batch bsize dim", batch, bsize=32)
```

You can see that the parenthesis are used to decompose an axis into multiple axis with
row-major order. Here the input length must be a multiple of 32, so that
input_lenght = n_batch * bsize.

The number can also be used to check if the input size are corrects. For instance:

```python
batch = torch.randn(16, 32)
batches = einx.rearrange("(n_batch bsize) dim -> n_batch bsize dim", batch, n_batch=4, bsize=4)
```

Only passes because the input length is exactly 4 * 4 here.

Now it's time to present `einx` specifically. Most of the available functions
are presented in the *Functionalities* section. The next section will
present the specificity of `einx` compared to `einops` regarding the description string.

## Description tricks

`einx` add more tensor operations than `einops` but also add some functionalities
to make the description more readable and/or flexible.

### Marked axis with brackets

When you apply an operation on one or multiple axis, you can mark those axis with
brackets. The operations can be more readable with them. Sometimes you can omit
the output shape when it can be trivially inferred from the input shape to avoid
redundancy. For some advanced operations specific to `einx` the marked axis are
mandatory.

- Readability

```python
>>> x = torch.randn(2, 3, 4)
>>> y = torch.randn(5, 4)
>>> einx.dot("a b [c], [c] d -> a b d", x, y).shape
Size([2, 3, 5])
```

- Omitting output shape when it's trivial (avoid redundancy)

```python
>>> x = torch.randn(2, 3, 4)
>>> einx.mean("a b [c]", x).shape
Size([2, 3])
```

### Number in description

While `einops` only support letters and "1" in the description (to expand dims),
`einx` support all the numbers in the description. A number in the description means that
the axis is unnamed and its dimension is fixes. Then the two following lines are equivalent:

```python
einx.mean("(a [2]) b", x)
einx.mean("(a [_new_axis]) b", x, _new_axis=2)
```

This operation computes the mean over pairs of rows (1 output row per 2 input rows).

**Important**: Axis with same numbers in different places are not considered as the
same axis. For each number axis, a unique axis name is generated under the hood.
For instance this means that you can reuse the same axis between input and output shapes.

```python
>>> x = torch.randn(24, 10)
>>> einx.rearrange("(a 2) b -> a 2 b", x)
RuntimeError: Found input axes that are not in output expression
```

This is because the call is equivalent to:

```python
>>> einx.rearrange("(a _axis1) b -> a _axis2 b", x, _axis1=2, _axis2=2)
RuntimeError: Found input axes that are not in output expression
```

### Ellipsis

Sometimes you want to make a function that works when the shape is not determined.
It's often better to keep the track of the full shape along the way for clarity and
debugging but sometimes you just don't have the choice or simply prefer to allow
more flexibility.

The ellipsis can be used in two ways:

- Unnamed ellipsis: " ..." (the ellipsis is preceded by a space or the start of the string).
  This works like the ellipsis in `einops`.

In this case, the ellipsis replaces all the axis found in the input shape that are
not named. For instance:

```python
>>> x = torch.randn(2, 3, 4)
>>> y = torch.randn(5, 4)
>>> einx.mean("... [dim]", x).shape
Size([2, 3])
>>> einx.mean("... [dim]", y).shape
Size([5])
```

- Named ellipsis: "{expr}..." (the ellipsis is preceded by an expression without space)

In this case, the expression (name, parenthesis, ...) is used to all the axis in the
corresponding position but with different names from each other. For instance, the
two following calls are equivalent:

```python
>>> x = torch.randn(12, 8, 6, 3)
>>> einx.rearrange("(a b)... c -> a... two... c", x, b=2).shape
Size([6, 4, 3, 2, 2, 2, 3])
>>> einx.rearrange("(a1 b1) (a2 b2) (a3 b3) c -> (a1 a2 a3) (b1 b2 b3) c", x, b1=2, b2=2, b3=2)
Size([6, 4, 3, 2, 2, 2, 3])
```

### Factorize "->" in brackets (advanced)

"->" in brackets can be used to specify marked axis in input and output shapes when other
axis are shared. The brackets "factorize" the arrow. For instance the following calls
are equivalent:

```python
>>> x = torch.randn(2, 3, 4)
>>> einx.mean("a b [c -> 1]", x).shape
Size([2, 3, 1])
>>> einx.mean("a b [c] -> a b [1]", x).shape
Size([2, 3, 1])
```

It's also possible to use the "," to specify marked axis for multiple inputs.
For instance the following calls are equivalent:

```python
>>> x = torch.randn(2, 3, 4)
>>> y = torch.randn(2, 3)
>>> einx.dot("a b [c, ->]", x, y).shape
Size([2, 3, 3])
>>> einx.dot("a b [c], a b [] -> a b", x, y).shape
Size([2, 3, 3])
```

## Explicit operations

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

## Functionalities

### Rearrange, repeat, concatenation, split

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

### Elementwise operations

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

### Reduce operations

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

### Vectorized operations

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

### Modify/get values at specific coordinates

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

### Create a tensor

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

### Shape description utils

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
