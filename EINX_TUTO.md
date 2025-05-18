# Einx Tuto

## Introduction to `einx` and `einops`

In this section, we are going to talk about features shared by `einops` and `einx` so
this section is mostly useful for people that are not familiar with `einops` yet.

Einx is a library inspired by the popular [einops](https://github.com/arogozhnikov/einops)
library (with more functionalities). Their goal is to provide a simple and
intuitive way to manipulate tensors in a flexible and readable manner using the
same lines of code for all tensors backends (Numpy, Pytorch, Jax, Tensorflow, ...).

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
change. Plus you have to remember all those pytorch functions that can be a pain.
And their names are not necessarily the same in other backends leading to even more
functions to remember.

That's where `einx` (and `einops`) comes in. They add general functions to manipulate
that take in input the full description of the inputs and outputs shapes as strings.
So the comments you made to track the shapes are now directly in th code itself,
guaranteeing that the code is always up to date and readable. Plus the functions
are always the same across backends. Here it is your code with `einx`:

```python
dense = torch.randn(3, 16)
img = torch.randn(224, 224, 3)
img = einx.dot("h w c, c dim -> h w dim", img, dense)
img = einx.rearrange("h w dim -> 1 dim h w", img)
```

Here you can see that full description of the input and output shapes.
You can even give more descriptive names to the axes. Once you get familiar with
the descriptions and the tensor manipulation, you can be even more efficient
and write the code like this:

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

Only passes because the input length is exactly 4 * 4.

Now it's time to present `einx` specifically. Most of the available functions
are presented in the *Functionalities* section. The next section will
present the specificity of `einx` compared to `einops` regarding the description string.

## Description tricks

`einx` add more tensor operations than `einops` but also add some functionalities
to make the description more readable and flexible.

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

`einops` only allows strict input and output shapes. Sometimes you want to make a
function that also works when the shape is not determined. It's often better to keep
the track of the full shape along the way for clarity and debugging but sometimes
you just don't have the choice.

The ellipsis can be used in two ways:

- Unnamed ellipsis: ` ...` (the ellipsis is preceded by a space). This works like
  the ellipsis in `einops`.

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

- Named ellipsis: `<expr>...` (the ellipsis is preceded by an expression without space)

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

## Functionalities

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

- `mean` (same as `einx.mean`): Mean over axis.

- `var` (same as `einx.var`): Variance over axis.

- `std` (same as `einx.std`): Standard deviation over axis.

- `prod` (same as `einx.prod`): Product over axis.

- `count_nonzero` (same as `einx.count_nonzero`): Count non-zero elements over axis.

- `any` (same as `einx.any`): Logical any over axis.

- `all` (same as `einx.all`): Logical all over axis.

- `max` (same as `einx.max`): Maximum over axis.

- `min` (same as `einx.min`): Minimum over axis.

- `logsumexp` (same as `einx.logsumexp`): Log-sum-exp over axis.

### Vectorized operations

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
