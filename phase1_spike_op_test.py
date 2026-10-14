import jax
import jax.numpy as jnp

tt_device = jax.devices('tt')[0]
print("Targeting:", tt_device)

# --- Reuse the Phase 0 spike primitive unchanged, it's already validated ---

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

# --- Fixed-parameter PLIF: beta is learnable, same value every timestep ---

def init_plif_params(n_channels, beta_init=0.9, v_th=1.0):
    raw = jnp.log(beta_init / (1.0 - beta_init)) * jnp.ones(n_channels)
    return {"raw_beta": raw, "v_th": jnp.full((n_channels,), v_th)}

def plif_step(params, v, I):
    beta = jax.nn.sigmoid(params["raw_beta"])
    v_new = beta * v + I
    s = spike(v_new - params["v_th"])
    return v_new * (1.0 - s), s

@jax.jit
def run_sequence(params, v0, I_seq):
    def step(v, I_t):
        return plif_step(params, v, I_t)
    return jax.lax.scan(step, v0, I_seq)

n_channels, T = 4, 10
params = {k: jax.device_put(v, tt_device) for k, v in init_plif_params(n_channels).items()}
v0 = jax.device_put(jnp.zeros(n_channels), tt_device)
I_seq = jax.device_put(jnp.where(jnp.arange(T)[:, None] >= 2, 0.3, 0.0) * jnp.ones((T, n_channels)), tt_device)

_, spikes = run_sequence(params, v0, I_seq)
print("RESULT spikes over time:\n", spikes)

def loss_fn(params, v0, I_seq):
    _, s = run_sequence(params, v0, I_seq)
    return jnp.sum(s)

grads = jax.grad(loss_fn)(params, v0, I_seq)
print("RESULT grad wrt raw_beta:", grads["raw_beta"])

slow = {**params, "raw_beta": jnp.full((n_channels,), 4.0)}
fast = {**params, "raw_beta": jnp.full((n_channels,), -4.0)}
_, s_slow = run_sequence(slow, v0, I_seq)
_, s_fast = run_sequence(fast, v0, I_seq)
print("RESULT total spikes, slow leak:", jnp.sum(s_slow))
print("RESULT total spikes, fast leak:", jnp.sum(s_fast))