# Introduction to `einx` and `einops`

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
