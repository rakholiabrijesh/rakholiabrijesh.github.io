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

## Stacking Layers

At this point we'll start thinking of our network in terms of **layers**. Our network currently takes 784 inputs, passes them through a linear layer with 10 neurons (one per digit), and each output goes through ReLU. The hope is that each neuron has learned to detect a specific digit.

But is that realistic? Can a single neuron really learn to recognize every possible way someone writes a "3"?

One neuron per digit means each must detect the entire pattern by itself. What if instead we had 128 neurons detecting parts such as curves, edges and loops. Then add another layer which combines parts into digits.

Think about how you recognize an "8". You don't memorize every possible pixel arrangement. You see two loops stacked on top of each other. Your brain detects **parts** first, then combines them.

That's exactly what stacking layers does:

- **Layer 1** (784 → 128): 128 neurons each learn to detect simple patterns such as edges, curves, strokes, loops.
- **Layer 2** (128 → 64): 64 neurons combine those simple patterns into higher-level features like loops, corners, intersections.
- **Layer 3** (64 → 10): 10 neurons combine those features into final digit classifications

Each layer compresses the information further, from raw pixels to parts to patterns to answers.

![Full architecture](/images/arch_full.png)

There's no formula for choosing the right number of neurons. The output layer is fixed by your problem: 10 classes means 10 neurons. For hidden layers, start with something reasonable and experiment. Too few neurons and the network can't learn enough patterns. Too many and it might **memorize** the training data instead of learning general patterns (this is called **overfitting** — like a student who memorizes exam answers but can't solve new problems). A common starting point is to shrink gradually from input to output, giving early layers more neurons to capture many simple patterns and later layers fewer neurons to combine them. Numbers like 128 and 64 are popular partly because powers of 2 run slightly faster on hardware, but there's nothing magical about them. The real answer is: try something, train it, see if it works, adjust, repeat. Choosing layer sizes is more art than science.

## Backpropagation through Layers

We have the architecture. Now, how does it learn?

We already understand the core idea from our previous post. For a single neuron, we used the loss as a starting point and measured how much the loss changes when we wiggle $\hat{y}$. Then, using the chain rule, we figured out how to update the weights and bias to minimize the loss. This is **backpropagation**.

The same idea applies here — but instead of a single neuron with single values, we now have **layers** with **matrices** of weights and activation functions in between. Let's walk through it step by step.

- Start from the loss, work backward
- Trace the gradient backward through the last linear layer
- Each layer receives a gradient, and computes three things:
  1. Gradient for its weights (to update them)
  2. Gradient for its bias (to update it)
  3. Graident to pass backward(to keep the chain going)
- ReLU just passes gradient through where input was positive, blocks where negative.

#### How does a gradient flow through a Linear layer?

Let's start by comparing what we already know with what we need to learn.

For our single neuron, the forward pass was:

$$\hat{y} = w \cdot x + b$$

And we computed the gradients using the chain rule:

$$\frac{dL}{dw} = \frac{dL}{d\hat{y}} \cdot x \quad \quad \frac{dL}{db} = \frac{dL}{d\hat{y}} \cdot 1$$

For a layer of neurons, the forward pass looks almost identical but just with matrices:

$$Y = X \cdot W + b$$

And the gradients follow the exact same logic:

|                     | Single Neuron                                 | Layer of Neurons                          |
| ------------------- | --------------------------------------------- | ----------------------------------------- |
| **Forward**         | $\hat{y} = w \cdot x + b$                     | $Y = X \cdot W + b$                       |
| **Weight gradient** | $\frac{dL}{dw} = \frac{dL}{d\hat{y}} \cdot x$ | $\frac{dL}{dW} = X^T \cdot \frac{dL}{dY}$ |
| **Bias gradient**   | $\frac{dL}{db} = \frac{dL}{d\hat{y}}$         | $\frac{dL}{db} = \sum \frac{dL}{dY}$      |
| **Pass backward**   | $\frac{dL}{dx} = \frac{dL}{d\hat{y}} \cdot w$ | $\frac{dL}{dX} = \frac{dL}{dY} \cdot W^T$ |

It's the same idea — just scaled up. Let's break down each one.

**Weight gradient:** $\frac{dL}{dW} = X^T \cdot \frac{dL}{dY}$

In the single neuron case, the weight gradient was the input multiplied by the incoming gradient. Same thing here — $X^T$ is our input (transposed to make the shapes work) multiplied by the gradient flowing in from the next layer.

**Bias gradient:** $\frac{dL}{db} = \sum \frac{dL}{dY}$

For a single neuron, the bias gradient was just the incoming gradient (multiplied by 1). For a layer processing a batch of samples, each sample contributes its own gradient. We sum them up because there's only one bias shared across all samples.

**Gradient to pass backward:** $\frac{dL}{dX} = \frac{dL}{dY} \cdot W^T$

This is the gradient that gets sent to the previous layer. In the single neuron case, it was the incoming gradient multiplied by the weight. Same here, we multiply by $W^T$ (transposed to make the shapes work). This is what keeps the chain going.

#### How does a gradient flow through ReLU?

ReLU's forward pass was simple — keep positives, zero out negatives. Its backward pass is just as simple.

Think about it: if ReLU kept a value (it was positive), then a small change in the input causes the same small change in the output. The gradient passes through unchanged.

If ReLU zeroed a value (it was negative), then the output is stuck at zero no matter what. The gradient is blocked and it becomes zero.

$$\frac{dL}{dX} = \frac{dL}{dY} \cdot \begin{cases} 1 & \text{if input was positive} \\ 0 & \text{if input was negative} \end{cases}$$

ReLU acts like a gate. It lets the gradient through where the neuron was active, and blocks it where the neuron was dead.

#### Putting it all together

Let's trace the full backward pass through our network:

$$\text{Input} \rightarrow \text{Linear}_1 \rightarrow \text{ReLU} \rightarrow \text{Linear}_2 \rightarrow \text{ReLU} \rightarrow \text{Linear}_3 \rightarrow \text{Loss}$$

Starting from the loss, the gradient flows backward:

1. **Loss** computes $\frac{dL}{dY_3}$ — how much the loss changes with respect to the final output. This is our first gradient.
2. **Linear₃** receives this gradient and computes:
   - Its weight and bias gradients (to update its parameters)
   - $\frac{dL}{dX_3}$ to pass backward
3. **ReLU** receives $\frac{dL}{dX_3}$ and lets it through where neurons were active, blocks where they were dead
4. **Linear₂** receives the surviving gradient, computes its own weight and bias gradients, and passes another gradient backward
5. **ReLU** gates the gradient again
6. **Linear₁** receives the gradient and computes its weight and bias gradients

At the end of this process, every layer has its own weight and bias gradients which is the exact information needed to update the parameters and reduce the loss.
