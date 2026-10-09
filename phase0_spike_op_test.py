import pjrt_plugin_tt  # harmless if unused — jax auto-registers 'tt' either way
import jax
import jax.numpy as jnp

tt_device = jax.devices('tt')[0]
print("Targeting:", tt_device)


@jax.custom_vjp
def spike(x):
    return jnp.where(x >= 0.0, 1.0, 0.0)

def spike_fwd(x):
    return spike(x), x

def spike_bwd(x, g):
    alpha = 4.0
    sg = 1.0 / (1.0 + alpha * jnp.abs(x)) ** 2
    return (g * sg,)

spike.defvjp(spike_fwd, spike_bwd)


@jax.jit
def toy_lif_step(v, I, beta=0.9, v_th=1.0):
    v_new = beta * v + I
    s = spike(v_new - v_th)
    return v_new * (1.0 - s), s


v0 = jax.device_put(jnp.zeros(8), tt_device)
I0 = jax.device_put(jnp.linspace(0.0, 2.0, 8), tt_device)

v1, s1 = toy_lif_step(v0, I0)
print("RESULT spikes:", s1)

def loss_fn(I):
    _, s = toy_lif_step(v0, I)
    return jnp.sum(s)

grad = jax.grad(loss_fn)(I0)
print("RESULT grad:", grad)