# A Single Neuron: The Building Block of Neural Networks

Let's consider a singular Neuron that has weight($w$), bias($b$) and input($x$):

$$w = 0.5$$

$$b = 0.1$$

$$x = 2.0$$

Neuron's output (aka forward pass) is calculated by:

$$\hat{y} = wx + b $$

$$\hat{y} = (0.5) \cdot (2.0) + (0.1)$$

$$\hat{y} = 1.0 + 0.1 $$

$$\hat{y} = 1.1 $$

Neuron's output ($\hat{y}$) is considered to be the predicted value.

Lets say the correct value ($y$) is $1.5$, then how should the loss be calculated?

### What is a loss function?

A loss function measures how far the network's prediction are from the correct answers. A lower loss means better predictions.

As an example, you could say why not just subtract correct values from predictions in order to calculate the loss.

$(predicted - correct)$ doesn't work because, for values of correct that are higher than predicitions, the answer will be negative. We don't care about the negative sign. Positive or negative, we just want to know how bad is our network performing.

Therefore, we need a loss function that gets rid of the negative sign. Let's look at Mean-Squared Error loss function. MSE with multiple values, it's the _mean_ of all squared errors.

$$L = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^{2}$$

Let's compute the loss for our example now:

$$L = (1.1 - 1.5)^{2} = (-0.4)^{2} = 0.16$$

### We got a loss (looks bad!). Now what?

Simply taking a look at the information we have with this one neuron, we can see that there are some levers that we can turn. There's $weights$ and $bias$ that seems tweakable. We shouldn't be tweaking the input features ($x$) because they are constants.

Though the question now becomes, in what direction should these weights and bias change towards, and more importantly, by how much?

We need some way to measure each lever's influence on the loss. If we nudge a weight by a tiny amount, how much does the loss change? In mathematics, this is exactly what a **derivative** gives us.

### What is a derivative?

A derivative, by formal definition represents the instantaneous rate of change of a function with respect to one of its variables, defined as the slope of the tangent line to the function's graph at a specific point.

In our example, we are interested in seeing how does our _loss_ change if we **nudge** our prediction by a tiny amount. It can be represented as:

$$\frac{dL}{d\hat{y}} = \frac{d}{d\hat{y}}(\hat{y} - y)^{2} = 2(\hat{y} - y)$$

$$\frac{dL}{d\hat{y}} = 2(1.1 - 1.5) = 2(-0.4) = -0.8$$

Here, $-0.8$ means that, if we increase the prediction by a tiny amount, the loss decreases. That tells us the prediction should go **up** to get closer to 1.5.

OK, but we can't directly change the prediction. The prediction comes from $wx + b$. So how do we figure out which lever to adjust, $w$ or $b$?

As you might have predicted, this is where the chain rule comes in.

### Why do we need the chain rule?

The loss doesn't directly depend on $w$. It depends on $\hat{y}$, which depends on $w$. There's a chain: $$w \rightarrow \hat{y} \rightarrow L$$

Due to this chain, we can't compute $\frac{dL}{dw}$ directly. We need to traverse backwards, in chain, to the point where we can calculate $\frac{dL}{dw}$.

The chain rule says that we need to multiply derivatives along the chain.

$$\frac{dL}{dw} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{dw}$$

In English, this reads as the following. To compute how much the weights influence the loss, we need to take the derivative of the loss w.r.t predicted output ($\hat{y}$), and chain it with the derivative of the predicted output ($\hat{y}$) w.r.t to the weight ($w$).

Therefore,

$$\frac{dL}{dw} = -0.8 (computed\ above) \cdot \frac{d(wx+b)}{dw} = -0.8 \cdot x$$
$$\frac{dL}{dw} = -0.8 \cdot 2.0  = -1.6$$

The gradient we got here represents the sensitivity of weights towards the total loss when nudged by a tiny amount. The direction towards which it's sensitive is $negative$ and the magnitude by which it's sensitive is $1.6$ (which is significant). Simply put, the gradient tells you the **direction of increase** of the loss. If $\frac{dL}{dw} = -1.6$, that means increasing $w$ decreases loss. We want to **minimize** the loss, so we go in the **opposite direction** of the gradient. To the update the weight such that we minimize loss, we use:

$$ w\_{new} = w - learning_rate(\alpha) \cdot gradient $$

This is a self-correcting formula. Let's understand it with some example values before we move on.

##### Negative Gradient ($-1.6$)

- Means increasing $w$ decreases loss
- $w - lr \cdot (-1.6) = w + something \rightarrow w $ increases

##### Positive Gradient (say $+2.0$)

- Means: increasing $w$ increases loss (bad direction)
- $ w - lr \cdot (2.0) = w - something \rightarrow w$ decreases

Now let's use this formula for our weights.

$$ w\_{new} = 0.5 - 0.1 \cdot (-1.6) = 0.66 $$

But remember, weights aren't the only parameters that needs this update. We need to do the same thing with our $bias$.

$$ \frac{dL}{db} = \frac{dL}{d\hat{y}} \cdot \frac{d(wx + b)}{db} = -0.8 \cdot (0 + 1) = -0.8 $$

To update the bias given the gradient $-0.8$

$$ b\_{new} = b - \alpha \cdot gradient $$

$$ b\_{new} = 0.1 - 0.1 \cdot (-0.8) = 0.1 + 0.08 = 0.18$$

Now that we've got out new $weight$ and $bias$, let's run the forward prediction pass on our neuron to see how close we're to the correct answer.

$$w_{new} = 0.66, x = 2.0, b_{new} = 0.18, y (correct) = 1.5 $$

$$ \hat{y} = w*{new} \cdot x + b*{new} $$

$$ \hat{y} = 0.66 \cdot 2.0 + 0.18 = 1.5 $$

$$ \hat{y} = y $$

Wait, that's exactly 1.5! With one neuron and one input, it converged in a single step. That's too easy, but real networks need _many_ iterations. This whole process of forward pass, loss computation, backward pass, and parameter update is repeated until the loss converges to a minimum. This is what's called the **training loop**.

$$\text{For each iteration:}$$

$$\text{1. Forward Pass: } \hat{y} = wx + b$$

$$\text{2. Compute Loss: } L = (\hat{y} - y)^{2}$$

$$\text{3. Compute Gradients: } \frac{dL}{dw} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{dw}, \quad \frac{dL}{db} = \frac{dL}{d\hat{y}} \cdot \frac{d\hat{y}}{db}$$

$$\text{4. Update Parameters: } w_{new} = w - \alpha \cdot \frac{dL}{dw}, \quad b_{new} = b - \alpha \cdot \frac{dL}{db}$$

$$\text{5. Repeat until } L \approx 0$$

A single neuron with a single input has only 2 parameters to adjust. A real network classifying something like handwritten digits might have 100,000+ parameters and 60,000 training examples where each of them is pulling the parameters in slightly different directions. That's why the training loop runs for thousands of iterations. This is where **layers** and **matrix multiplication** come in. We are going to explore all this next!

## Appendix: Key Definitions

| Term                | Definition                                                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Weight**          | A factor that represents an influence over some input feature.                                                                                                                  |
| **Bias**            | A learnable shift that allows the neuron to activate even when inputs are zero.                                                                                                 |
| **Derivative**      | How much one thing changes when you nudge another thing by a tiny amount.                                                                                                       |
| **Backpropagation** | An algorithm that traverses the entire network backwards, calculating how much every single parameter is contributing to the total loss and what can be changed to minimize it. |
| **Loss Function**   | Measures how far the network's predictions are from the correct answers. Lower loss means better predictions.                                                                   |
| **Learning Rate**   | A small number that controls how big of a step we take when updating weights.                                                                                                   |
| **Forward Pass**    | Data flowing through the network from input to output.                                                                                                                          |
