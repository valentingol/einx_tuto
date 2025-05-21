# Description tricks

`einx` add more tensor operations than `einops` but also add some functionalities
to make the description more readable and/or flexible.

## Marked axis with brackets

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

## Number in description

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

## Ellipsis

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

## Factorize "->" in brackets (advanced)

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
