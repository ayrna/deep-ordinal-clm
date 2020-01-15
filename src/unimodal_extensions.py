# -*- coding: utf-8 -*-

import numpy as np

from keras.layers import Layer, Input, Activation, Dense, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Lambda
from keras.initializers import Ones, constant
from keras import backend as K
import numpy as np

class TauLayer(Layer):
    """Capa que contiene el parámetro tau a ser entrenado. Se divide la entrada por b + g(tau), donde b es
    sesgo especificado y g es un función de transferencia.

    :param tau: Valor de tau por el que dividr la entrada.
    :param tau_learn: Booleano para especificar si aprender tau o mantener el valor recibido constante.
    :param bias: Valor del sesgo.
    :param nonlinearity: No linealidad para aplicar a tau.
    """
    def __init__(self, tau, bias=1.0, nonlinearity='linear', tau_learn='True',  **kwargs):
        self.tau_ = constant(tau)
        self.tau_learn_ = tau_learn
        self.bias_ = constant(bias)
        self.nonlinearity = nonlinearity
        super(TauLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.tau = self.add_weight(name='tau', 
                                   shape=(1,),
                                   initializer=self.tau_,
                                   trainable=self.tau_learn_)
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                    shape=(1,),
                                    initializer=self.bias_)
        super(TauLayer, self).build(input_shape)  # Be sure to call this at the end

    def get_config(self):
        pass

    def call(self, x):
        return x / (self.bias + Activation(self.nonlinearity)(self.tau))

    def compute_output_shape(self, input_shape):
        return input_shape


def _add_pois(layer, num_classes, end_nonlinearity, tau, tau_mode, extra_depth=None):
    """Añade el bloque final de la red que contiene la implementación de la distribución de Poisson.
    
    :param end_nonlinearity: La no linealidad que se desea aplicar en la primera capa Dense de un nodo.
    :param tau: Valor inicial de tau
    :param tau_mode:
     non_learnable = No se aprende el valor de tau, se escoge el valor pasado como parámetro.
     sigm_learnable = Se aprende como un peso mediante entrenamiento el valor de tau dentro de una función sigmoide (estabiliza el entrenamiento.)
    """
    assert tau_mode in ["non_learnable", "sigm_learnable"]
    if extra_depth != None:
        layer = Dense(units=extra_depth, activation='linear')(layer)
    l_fx = Dense(units=1, activation=end_nonlinearity)(layer)
    l_copy = Dense(units=num_classes, activation='linear', kernel_initializer='ones', trainable=False)(l_fx)
    c = np.asarray([[(i+1) for i in range(0, num_classes)]], dtype="float32")
    from scipy.misc import factorial
    cf = factorial(c)

    tau_learn = True

    if tau_mode == "non_learnable":
        l_pois = Lambda(lambda x: ((c*K.log(x)) - x - K.log(cf)) / tau )(l_copy)
        tau_learn = False
    elif tau_mode in ["learnable", "sigm_learnable"]:
        l_pois = Lambda(lambda x: ((c*K.log(x)) - x - K.log(cf)))(l_copy)

        if tau_mode == "learnable":
            fn = 'linear'
        elif tau_mode == "sigm_learnable":
            fn = 'sigmoid'
        l_pois = TauLayer(tau, bias=0., nonlinearity=fn, tau_learn=tau_learn)(l_pois)

    l_softmax = Activation('softmax')(l_pois)

    return l_softmax

def _add_binom(layer, num_classes, tau, tau_mode, extra_depth=None, extra_depth_nonlinearity='relu'):
    """Añade el bloque final de la red que contiene la implementación de la distribución Binomial.
    
    :param end_nonlinearity: La no linealidad que se desea aplicar en la primera capa Dense de un nodo.
    :param tau: Valor inicial de tau
    :param tau_mode:
     non_learnable = No se aprende el valor de tau, se escoge el valor pasado como parámetro.
     sigm_learnable = Se aprende como un peso mediante entrenamiento el valor de tau dentro de una función sigmoide (estabiliza el entrenamiento.)
    """
    assert tau_mode in ["non_learnable", "sigm_learnable"]
    # NOTE: Weird Bug. This is numerically unstable when
    # deterministic=True with the article's resnet
    # so: Added eps, and added clip
    k = num_classes
    if extra_depth != None:
        layer = Dense(units=extra_depth, activation=extra_depth_nonlinearity)(layer)
    l_sigm = Dense(units=1, activation='sigmoid', kernel_initializer='he_normal', bias_initializer=constant(0.))(layer)
    l_copy = Dense(units=k, activation='linear', kernel_initializer='ones', trainable=False)(l_sigm)

    c = np.asarray([[(i) for i in range(0, k)]], dtype="float32")
    from scipy.special import binom
    binom_coef = binom(k-1, c).astype("float32")

    ### NOTE: NUMERICALLY UNSTABLE ###
    eps = 1e-6
    l_logf = Lambda(lambda px: (K.log(binom_coef) + (c*K.log(px+eps)) + ((k-1-c)*K.log(1.-px+eps))) )(l_copy)
    if tau_mode == "non_learnable":
        if tau != 1:
            l_logf = Lambda(lambda px: px / tau)(l_logf)
    else:
        l_logf = TauLayer(tau, bias=0., nonlinearity='sigmoid')(l_logf)
    l_logf = Activation('softmax')(l_logf)
    return l_logf


def _add_binom_m(model, num_classes, tau, tau_mode, extra_depth=None, extra_depth_nonlinearity='relu'):
    """Añade el bloque final de la red que contiene la implementación de la distribución Binomial.

    :param end_nonlinearity: La no linealidad que se desea aplicar en la primera capa Dense de un nodo.
    :param tau: Valor inicial de tau
    :param tau_mode:
     non_learnable = No se aprende el valor de tau, se escoge el valor pasado como parámetro.
     sigm_learnable = Se aprende como un peso mediante entrenamiento el valor de tau dentro de una función sigmoide (estabiliza el entrenamiento.)
    """
    assert tau_mode in ["non_learnable", "sigm_learnable"]
    # NOTE: Weird Bug. This is numerically unstable when
    # deterministic=True with the article's resnet
    # so: Added eps, and added clip
    k = num_classes
    if extra_depth != None:
        model.add(Dense(units=extra_depth, activation=extra_depth_nonlinearity))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='he_normal', bias_initializer=constant(0.)))
    model.add(Dense(units=k, activation='linear', kernel_initializer='ones', trainable=False))

    c = np.asarray([[(i) for i in range(0, k)]], dtype="float32")
    from scipy.special import binom
    binom_coef = binom(k - 1, c).astype("float32")

    ### NOTE: NUMERICALLY UNSTABLE ###
    eps = 1e-6
    model.add(Lambda(lambda px: (K.log(binom_coef) + (c * K.log(px + eps)) + ((k - 1 - c) * K.log(1. - px + eps)))))
    if tau_mode == "non_learnable":
        if tau != 1:
            model.add(Lambda(lambda px: px / tau))
    else:
        model.add(TauLayer(tau, bias=0., nonlinearity='sigmoid'))
    model.add(Activation('softmax'))
    # return model