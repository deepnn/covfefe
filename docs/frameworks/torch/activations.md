
# activations Module
> Activation functions common API wrappers for torch frameowrk.



## Functions

##### `elu(x)` 

> Exponential Linear Unit :math:`\varphi(x) = (x > 0) ? x : e^x - 1`
>     The Exponential Linear Unit (ELU) was introduced in [1]_. Compared to the
>     linear rectifier :func:`rectify`, it has a mean activation closer to zero
>     and nonzero gradient for negative input, which can help convergence.
>     Compared to the leaky rectifier :class:`LeakyRectify`, it saturates for
>     highly negative inputs.
>     Parameters
>     ----------
>     x : float32
>         The activation (the summed, weighed input of a neuron).
>     Returns
>     -------
>     float32
>         The output of the exponential linear unit for the activation.
>     Notes
>     -----
>     In [1]_, an additional parameter :math:`\alpha` controls the (negative)
>     saturation value for negative inputs, but is set to 1 for all experiments.
>     It is omitted here.
>     References
>     ----------
>     .. [1] Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter (2015):
>        Fast and Accurate Deep Network Learning by Exponential Linear Units
>        (ELUs), http://arxiv.org/abs/1511.07289



##### `identity(x)` 

> Linear activation function :math:`\varphi(x) = x`
>     Parameters
>     ----------
>     x : float32
>         The activation (the summed, weighted input of a neuron).
>     Returns
>     -------
>     float32
>         The output of the identity applied to the activation.



##### `linear(x)` 

> Linear activation function :math:`\varphi(x) = x`
>     Parameters
>     ----------
>     x : float32
>         The activation (the summed, weighted input of a neuron).
>     Returns
>     -------
>     float32
>         The output of the identity applied to the activation.



##### `relu(x)` 

> Rectify activation function :math:`\varphi(x) = \max(0, x)`
>     Parameters
>     ----------
>     x : float32
>         The activation (the summed, weighted input of a neuron).
>     Returns
>     -------
>     float32
>         The output of the rectify function applied to the activation.



##### `sigmoid(x)` 

> Sigmoid activation function :math:`\varphi(x) = \frac{1}{1 + e^{-x}}`
>     Parameters
>     ----------
>     x : float32
>         The activation (the summed, weighted input of a neuron).
>     Returns
>     -------
>     float32 in [0, 1]
>         The output of the sigmoid function applied to the activation.



##### `softmax(x)` 

> Softmax activation function
>     :math:`\varphi(\mathbf{x})_j =
>     \frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
>     where :math:`K` is the total number of neurons in the layer. This
>     activation function gets applied row-wise.
>     Parameters
>     ----------
>     x : float32
>         The activation (the summed, weighted input of a neuron).
>     Returns
>     -------
>     float32 where the sum of the row is 1 and each single value is in [0, 1]
>         The output of the softmax function applied to the activation.



##### `softmax_with_loss(x, labels)` 

> Softmax activation function
>     :math:`\varphi(\mathbf{x})_j =
>     \frac{e^{\mathbf{x}_j}}{\sum_{k=1}^K e^{\mathbf{x}_k}}`
>     where :math:`K` is the total number of neurons in the layer. This
>     activation function gets applied row-wise.
>     Parameters
>     ----------
>     x : float32
>         The activation (the summed, weighted input of a neuron).
>     labels : float32
>         The labels (the target input to a loss layer).
>     Returns
>     -------
>     float32 where the sum of the row is 1 and each single value is in [0, 1]
>         The output of the softmax_with_loss function applied to the activation.



##### `softplus(x)` 

> Softplus activation function :math:`\varphi(x) = \log(1 + e^x)`
>     Parameters
>     ----------
>     x : float32
>         The activation (the summed, weighted input of a neuron).
>     Returns
>     -------
>     float32
>         The output of the softplus function applied to the activation.



##### `tanh(x)` 

> Tanh activation function :math:`\varphi(x) = \tanh(x)`
>     Parameters
>     ----------
>     x : float32
>         The activation (the summed, weighted input of a neuron).
>     Returns
>     -------
>     float32 in [-1, 1]
>         The output of the tanh function applied to the activation.


