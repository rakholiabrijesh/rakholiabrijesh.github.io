# From One Neuron to a Neural Network

In the previous post, our neuron had one input: $y = wx + b$. But a real neuron in a network receives many inputs. Think about MNIST - each image is 784 pixels. One neuron needs to take all 784 values and produce one output.

$$\hat{y} = (w_1 \cdot x_1 + w_2 \cdot x_2 + w_3 \cdot x_3 + ... + w_{784} \cdot x_{784}) + b$$

$$\hat{y} = \sum_{i=1}^{784} w_i \cdot x_i + b$$

In Python (using NumPy), this can be represented as:

```python
y = np.dot(w, x) + b
# or
y = w @ x + b
```

Yes, that's 784 multiplications and additions in one operation. That's the power of NumPy.

But wait, what does this computation of a neuron actually represent? Let's think about this concretely.
A neuron takes 784 pixel values, multiplies each by a weight and sums them up. Some weights are big, some are small, some might be negative.

If a weight for pixel #200 is large and positive, what does that mean? The neuron **cares a lot** about that pixel being bright.
If a weight for pixel #500 is large and negative? The neuron wants that pixel to be **dark**.
If a weight is near zero? The neuron **doesn't care** about that pixel.

So essentially, the output of a neuron really represents the following: **how much does this input match the pattern this neuron is looking for?**

A neuron is a _pattern detector_. Its weights define what pattern it's sensitive to. Given all this, one neuron detects one pattern. To classify 10 different digits, you need mulitple neurons, each looking for a different pattern. That's why we need something called a **layer** of neurons.

## Multiple Neurons

A single neuron has 784 weights and produces 1 output. If we want 10 neurons (one per digit), each has its own 784 weights.

```
Neuron 0: 784 weights -> 1 output
Neuron 1: 784 weights -> 1 output
...
Neuron 9: 784 weights -> 1 output
```

Instead of running 10 separate dot products, how can we organize all these weights so that we can compute 10 outputs at once?

We can stack all the weights into a single **matrix** and compute all 10 outputs at once. Here's how:

$$\begin{bmatrix} y_0 & y_1 & \cdots & y_9 \end{bmatrix} = \begin{bmatrix} x_0 & x_1 & \cdots & x_{783} \end{bmatrix} \cdot \begin{bmatrix} w_{0,0} & w_{0,1} & \cdots & w_{0,9} \\ w_{1,0} & w_{1,1} & \cdots & w_{1,9} \\ \vdots & \vdots & \ddots & \vdots \\ w_{783,0} & w_{783,1} & \cdots & w_{783,9} \end{bmatrix} + \begin{bmatrix} b_0 & b_1 & \cdots & b_9 \end{bmatrix}$$

In NumPy, this entire operation is just:

```python
y = x @ W + b
# (784,) @ (784, 10) + (10,) → (10,)
# 784 inputs, 784 weights, 10 neurons and 10 outputs.
```

10 neurons, 784 weights each, all computed in one line.

![Network so far](/images/arch_1.png)

<div class="note">
<strong>📝 NumPy Shapes</strong><br>
<p style="italic">Every array in NumPy has a shape. A shape of <code>(3,)</code> means 3 numbers in a row. A shape of <code>(2, 3)</code> means 2 rows and 3 columns. When you multiply arrays with <code>@</code>, the inner dimensions must match and they disappear — the outer dimensions survive. So <code>(2, 3) @ (3, 4)</code> gives <code>(2, 4)</code> because the 3s match and vanish. If the inner dimensions don't match, NumPy throws an error. This one rule governs every matrix operation in a neural network. </p>
</div>

Now that we have something that allows us to spin up 10 neurons that gives us 10 outputs, is that enough to classify digits? We'll call this a **layer**. This one layer does `x @ W + b`, and this is a linear function. It can only draw straight lines to separate things. Let's plot some 3's and 8's on a graph and see if a straight line can separate them.

![Can a straight line separate "3s" from 8s"?](/images/linear_separation.png)

The graph above shows 500 handwritten digits plotted as points in 2D space. Blue dots are 3s, red dots are 8s. Try drawing a single straight line that puts all the blue on one side and all the red on the other. It's impossible. They're mixed together.

A single linear layer can only draw straight boundaries, which means it will never perfectly separate these digits. You might think, what if we just stack more layers?

Let's try two layers:

$$\text{Layer 1: } z = W_1 \cdot x + b_1$$
$$\text{Layer 2: } y = W_2 \cdot z + b_2$$

Substituting layer 1 into layer 2:

$$y = W_2 \cdot (W_1 \cdot x + b_1) + b_2$$
$$y = (W_2 \cdot W_1) \cdot x + (W_2 \cdot b_1 + b_2)$$
$$y = W_{combined} \cdot x + b_{combined}$$

It collapsed back into a single linear function. No matter how many linear layers you stack, the result is always just one straight line. Depth is useless without something to break the linearity.

We need something between the layers that **bends** the output. Something that makes a straight line into a curve. This is called an **activation function**, and the simplest one is called **ReLU**.

## What is an activation function?

An activation function sits between layers. It takes the output from the previous linear layer, transforms it in a non-linear way, and passes the result to the next layer. As we saw above, this transformation from linear to non-linear is necessary, otherwise the layers will collapse. The activation function prevents that collapse.

One such example of an activation function that's commonly used is called **ReLU**. It's a pretty convenient function. All it does is it if the value is positive, it _keeps_ it, and if the value is negative, it returns _zero_.

$$ ReLU(x) = max(0, x) $$

![ReLU](/images/relu.png)

Let's add ReLU to our ongoing network!

![add activation function](/images/arch_2.png)
