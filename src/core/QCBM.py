import jax 
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", True)



class QCBM:
    def __init__(self, circuit, mmd, prior):
        self.circuit = circuit
        self.mmd = mmd
        self.prior = prior

    @partial(jax.jit, static_argnums=0)
    def mmd_loss(self, params):
        px = self.circuit(params)
        print(px.shape)
        return self.mmd(px, self.prior), px



