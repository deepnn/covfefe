# init Module
> Initialization functions common API wrappers for theano frameowrk.



## Functions

##### `constant(val=0.0)` 

> Initialize weights with constant value.
>     Parameters
>     ----------
>      val : float
>         Constant value for weights.



##### `glorot_normal(gain=1.0, c01b=False)` 

> Glorot with weights sampled from the Normal distribution.
>     This is also known as Xavier initialization [1]_.
> 
>     Parameters
>     ----------
>     gain : float or 'relu'
>         Scaling factor for the weights. Set this to ``1.0`` for linear and
>         sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
>         to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
>         leakiness ``alpha``. Other transfer functions may need different
>         factors.
>     c01b : bool
>         For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
>         with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
>         the correct fan-in and fan-out.
>     References
>     ----------
>     .. [1] Xavier Glorot and Yoshua Bengio (2010):
>            Understanding the difficulty of training deep feedforward neural
>            networks. International conference on artificial intelligence and
>            statistics.
>     Notes
>     -----
>     For a :class:`DenseLayer <lasagne.layers.DenseLayer>`, 
>     If ``gain=1``, the weights are initialized as
>     .. math::
>        \sigma &= \sqrt{\frac{2}{fan_{in}+fan_{out}}}\\
>        W &\sim N(0, \sigma)



##### `glorot_uniform(gain=1.0, c01b=False)` 

> Glorot with weights sampled from the Uniform distribution.
>     This is also known as Xavier initialization [1]_.
> 
>     Parameters
>     ----------
>     gain : float or 'relu'
>         Scaling factor for the weights. Set this to ``1.0`` for linear and
>         sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
>         to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
>         leakiness ``alpha``. Other transfer functions may need different
>         factors.
>     c01b : bool
>         For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
>         with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
>         the correct fan-in and fan-out.
>     References
>     ----------
>     .. [1] Xavier Glorot and Yoshua Bengio (2010):
>            Understanding the difficulty of training deep feedforward neural
>            networks. International conference on artificial intelligence and
>            statistics.
>     Notes
>     -----
>     For a :class:`DenseLayer <lasagne.layers.DenseLayer>`, 
>     If ``gain='relu'``, the weights are initialized as
>     .. math::
>        a &= \sqrt{\frac{12}{fan_{in}+fan_{out}}}\\
>        W &\sim U[-a, a]



##### `he_normal(gain=1.0, c01b=False)` 

> He initializer with weights sampled from the Normal distribution.
>     Weights are initialized with a standard deviation of
>     :math:`\sigma = gain \sqrt{\frac{1}{fan_{in}}}` [1]_.
>     Parameters
>     ----------
>     gain : float or 'relu'
>         Scaling factor for the weights. Set this to ``1.0`` for linear and
>         sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
>         to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
>         leakiness ``alpha``. Other transfer functions may need different
>         factors.
>     c01b : bool
>         For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
>         with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
>         the correct fan-in and fan-out.
>     References
>     ----------
>     .. [1] Kaiming He et al. (2015):
>            Delving deep into rectifiers: Surpassing human-level performance on
>            imagenet classification. arXiv preprint arXiv:1502.01852.



##### `he_uniform(gain=1.0, c01b=False)` 

> He initializer with weights sampled from the Uniform distribution.
>     Weights are initialized with a standard deviation of
>     :math:`\sigma = gain \sqrt{\frac{1}{fan_{in}}}` [1]_.
>     Parameters
>     ----------
>     gain : float or 'relu'
>         Scaling factor for the weights. Set this to ``1.0`` for linear and
>         sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
>         to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
>         leakiness ``alpha``. Other transfer functions may need different
>         factors.
>     c01b : bool
>         For a :class:`lasagne.layers.cuda_convnet.Conv2DCCLayer` constructed
>         with ``dimshuffle=False``, `c01b` must be set to ``True`` to compute
>         the correct fan-in and fan-out.
>     References
>     ----------
>     .. [1] Kaiming He et al. (2015):
>            Delving deep into rectifiers: Surpassing human-level performance on
>            imagenet classification. arXiv preprint arXiv:1502.01852.



##### `normal(std=0.01, mean=0.0)` 

> Sample initial weights from the Gaussian distribution.
>     Initial weight parameters are sampled from N(mean, std).
>     Parameters
>     ----------
>     std : float
>         Std of initial parameters.
>     mean : float
>         Mean of initial parameters.



##### `orthogonal(gain)` 

> Intialize weights as Orthogonal matrix.
>     Orthogonal matrix initialization [1]_. For n-dimensional shapes where
>     n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
>     corresponds to the fan-in, so this makes the initialization usable for
>     both dense and convolutional layers.
>     Parameters
>     ----------
>     gain : float or 'relu'
>         Scaling factor for the weights. Set this to ``1.0`` for linear and
>         sigmoid units, to 'relu' or ``sqrt(2)`` for rectified linear units, and
>         to ``sqrt(2/(1+alpha**2))`` for leaky rectified linear units with
>         leakiness ``alpha``. Other transfer functions may need different
>         factors.
>     References
>     ----------
>     .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
>            "Exact solutions to the nonlinear dynamics of learning in deep
>            linear neural networks." arXiv preprint arXiv:1312.6120 (2013).



##### `sparse(sparsity=0.1, std=0.01)` 

> Initialize weights as sparse matrix.
>     Parameters
>     ----------
>     sparsity : float
>         Exact fraction of non-zero values per column. Larger values give less
>         sparsity.
>     std : float
>         Non-zero weights are sampled from N(0, std).



##### `uniform(range=0.01, std=None, mean=0.0)` 

> Sample initial weights from the uniform distribution.
>     Parameters are sampled from U(a, b).
>     Parameters
>     ----------
>     range : float or tuple
>         When std is None then range determines a, b. If range is a float the
>         weights are sampled from U(-range, range). If range is a tuple the
>         weights are sampled from U(range[0], range[1]).
>     std : float or None
>         If std is a float then the weights are sampled from
>         U(mean - np.sqrt(3) * std, mean + np.sqrt(3) * std).
>     mean : float
>         see std for description.


