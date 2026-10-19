"""
Reference PyTorch implementation of a single Spiking-SSM block
for the Tenstorrent Wormhole dataflow architecture project.

This file is designed as a CPU/Golden reference before porting
to TT-Metalium custom kernels.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Surrogate gradient for the Heaviside spike
# ---------------------------------------------------------------------------

class SurrogateSpike(torch.autograd.Function):
    """
    Forward : hard threshold  (v >= 0) -> 1 else 0
    Backward: fast sigmoid surrogate gradient

    grad = alpha / (1 + alpha * |v|)^2
    """
    @staticmethod
    def forward(ctx, v: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(v)
        ctx.alpha = alpha
        return (v >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        v, = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = alpha / (1.0 + alpha * v.abs()).pow(2)
        return grad_input * grad_output, None


def spike_activation(v: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return SurrogateSpike.apply(v, alpha)


# ---------------------------------------------------------------------------
# Spiking neurons
# ---------------------------------------------------------------------------

class PLIFNeuron(nn.Module):
    """
    Parametric Leaky Integrate-and-Fire (PLIF).
    One learnable decay parameter per feature.

    The membrane potential is carried across time steps:
        v_t = beta * v_{t-1} + (1 - beta) * x_t
        s_t = H(v_t - v_th)
        v_t = v_t * (1 - s_t) + reset_val * s_t     (reset on spike)
    """
    def __init__(self, dim: int, v_th: float = 1.0, beta_init: float = 0.9, alpha: float = 2.0):
        super().__init__()
        assert 0.0 < beta_init < 1.0, "beta_init must be in (0, 1)"
        self.dim = dim
        self.v_th = v_th
        self.alpha = alpha
        # Learnable decay stored as a logit: beta = sigmoid(self.beta) in (0, 1).
        # Initialised so that sigmoid(self.beta) == beta_init.
        self.beta = nn.Parameter(torch.full((dim,), math.log(beta_init / (1.0 - beta_init))))
        self.reset_val = 0.0

    def _step(
        self, x_t: torch.Tensor, v_prev: torch.Tensor, beta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One time step. x_t, v_prev: [B, D]; beta: [D]."""
        v = beta * v_prev + (1.0 - beta) * x_t
        s = spike_activation(v - self.v_th, self.alpha)
        v = v * (1.0 - s) + self.reset_val * s   # reset on spike
        return s, v

    def forward(
        self, x: torch.Tensor, v_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x       : [B, D] (one step) or [B, T, D] (sequence) input current
        v_prev  : [B, D] membrane potential carried in (None -> zeros)
        Returns : spikes (same shape as x), v_last [B, D] membrane potential
                  after the final step (feed it back in to continue the sequence)
        """
        beta = torch.sigmoid(self.beta)          # [D]

        if x.dim() == 2:
            if v_prev is None:
                v_prev = torch.zeros_like(x)
            return self._step(x, v_prev, beta)

        v = torch.zeros_like(x[:, 0, :]) if v_prev is None else v_prev
        spikes = []
        for t in range(x.shape[1]):
            s_t, v = self._step(x[:, t, :], v, beta)
            spikes.append(s_t)
        return torch.stack(spikes, dim=1), v


class ALIFNeuron(nn.Module):
    """
    Adaptive LIF neuron.
    Threshold adapts based on past spike history.
    State = (membrane potential v, adaptation current a)
    """
    def __init__(
        self,
        dim: int,
        v_th: float = 0.1,
        beta_v: float = 0.9,
        beta_a: float = 0.9,
        a_init: float = 0.1,
        alpha: float = 2.0,
    ):
        super().__init__()
        self.dim = dim
        self.v_th_base = v_th
        self.beta_v = beta_v
        self.beta_a = beta_a
        self.alpha = alpha
        # Per-feature learnable adaptation strength
        self.a = nn.Parameter(torch.full((dim,), a_init))

    def _step(
        self,
        x_t: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """One time step. x_t and both state tensors: [B, D]."""
        v_prev, a_prev = state
        v = self.beta_v * v_prev + x_t
        th = self.v_th_base + self.a * a_prev
        s = spike_activation(v - th, self.alpha)
        a = self.beta_a * a_prev + s
        v = v * (1.0 - s)  # reset to 0 on spike
        return s, (v, a)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x     : [B, D] (one step) or [B, T, D] (sequence)
        state : (v_prev, a_prev) each [B, D]  (None -> zeros)
        Returns: spikes (same shape as x), (v_last, a_last) each [B, D]
                 after the final step (feed back in to continue the sequence)
        """
        if state is None:
            ref = x[:, 0, :] if x.dim() == 3 else x
            state = (torch.zeros_like(ref), torch.zeros_like(ref))

        if x.dim() == 2:
            return self._step(x, state)

        spikes = []
        for t in range(x.shape[1]):
            s_t, state = self._step(x[:, t, :], state)
            spikes.append(s_t)
        return torch.stack(spikes, dim=1), state


# ---------------------------------------------------------------------------
# Structured SSM (diagonal / S4D-like)
# ---------------------------------------------------------------------------

class DiagonalSSM(nn.Module):
    """
    Diagonal structured SSM:
        h_{t} = A_d * h_{t-1} + B_d * (B_proj @ x_t)
        y_t   = C_proj @ h_t + D * x_t

    A is parameterized as complex diagonal for stability.
    For this reference we keep a real-only diagonal to avoid
    complex tensor issues in the baseline PyTorch path.
    """
    def __init__(self, dim: int, state_dim: Optional[int] = None, dt: float = 0.01):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim or dim
        self.dt = dt

        # Negative real part ensures stability: A = -exp(log_A_real)
        self.log_A_real = nn.Parameter(torch.randn(self.state_dim) * 0.01)
        # Imaginary component (kept for completeness; set to 0 if real-only desired)
        self.A_imag = nn.Parameter(torch.zeros(self.state_dim))

        self.B_proj = nn.Linear(dim, self.state_dim, bias=False)
        self.C_proj = nn.Linear(self.state_dim, dim, bias=False)
        self.D = nn.Parameter(torch.randn(dim) * 0.01)

    def discretize(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zero-order hold (ZOH) discretization for diagonal A.
        A_d = exp(A * dt)
        B_d = (A_d - 1) / A   (element-wise, guard against A≈0)
        """
        A_real = -torch.exp(self.log_A_real)
        A = A_real + 1j * self.A_imag
        A_d = torch.exp(A * self.dt)
        # Safe division
        denom = A + 1e-8
        B_d = (A_d - 1.0) / denom
        return A_d.real, B_d.real

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : [B, T, D]
        Returns y : [B, T, D], h : [B, T, state_dim]
        """
        B, T, D = x.shape
        A_d, B_d = self.discretize()  # [state_dim]

        Bx = self.B_proj(x)            # [B, T, state_dim]

        h = torch.zeros(B, T, self.state_dim, device=x.device, dtype=x.dtype)
        h_t = torch.zeros(B, self.state_dim, device=x.device, dtype=x.dtype)

        for t in range(T):
            h_t = A_d * h_t + B_d * Bx[:, t, :]
            h[:, t, :] = h_t

        y = self.C_proj(h) + self.D * x
        return y, h


# ---------------------------------------------------------------------------
# Spiking-SSM Block
# ---------------------------------------------------------------------------

class SpikingSSMBlock(nn.Module):
    """
    One block:
        Input -> LayerNorm -> Diagonal SSM -> Spiking Neuron -> Sparse spikes
    """
    def __init__(
        self,
        dim: int,
        neuron_type: str = "plif",
        state_dim: Optional[int] = None,
        dt: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.neuron_type = neuron_type
        self.ssm = DiagonalSSM(dim, state_dim=state_dim, dt=dt)
        self.norm = nn.LayerNorm(dim)

        if neuron_type == "plif":
            self.neuron = PLIFNeuron(dim)
        elif neuron_type == "alif":
            self.neuron = ALIFNeuron(dim)
        else:
            raise ValueError(f"Unsupported neuron_type={neuron_type}")

    def forward(
        self,
        x: torch.Tensor,
        neuron_state: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        x           : [B, T, D]
        neuron_state: previous neuron state, each tensor [B, D]:
                      v for PLIF, (v, a) for ALIF (None -> zeros)
        Returns:
            spikes      : [B, T, D]
            new_state   : neuron state after the last time step (same layout as
                          neuron_state; feed it back in to continue a sequence)
            h_ssm       : [B, T, state_dim] SSM hidden states

        Note: the SSM scan still starts from h = 0 on every call; only the
        neuron state is carried between calls.
        """
        x = self.norm(x)
        y, h_ssm = self.ssm(x)  # continuous SSM output

        if self.neuron_type == "plif":
            if neuron_state is None:
                neuron_state = torch.zeros_like(y[:, 0, :])
            s, new_state = self.neuron(y, neuron_state)
        elif self.neuron_type == "alif":
            s, new_state = self.neuron(y, neuron_state)
        else:
            raise RuntimeError("unreachable")

        return s, new_state, h_ssm


# ---------------------------------------------------------------------------
# Thin attention (simplified single-head for insertion every few blocks)
# ---------------------------------------------------------------------------

class ThinAttention(nn.Module):
    """
    Lightweight single-head self-attention for sparse spike representations.
    """
    def __init__(self, dim: int, num_heads: int = 1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, D]
        """
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Minimal stack (for testing / profiling)
# ---------------------------------------------------------------------------

class TinySpikingSSMStack(nn.Module):
    """
    Minimal stack of N Spiking-SSM blocks with thin attention every few blocks.
    """
    def __init__(
        self,
        dim: int,
        num_blocks: int = 4,
        attention_every: int = 2,
        neuron_type: str = "plif",
        state_dim: Optional[int] = None,
        dt: float = 0.01,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.attn = nn.ModuleDict()
        for i in range(num_blocks):
            self.blocks.append(
                SpikingSSMBlock(dim, neuron_type=neuron_type, state_dim=state_dim, dt=dt)
            )
            if (i + 1) % attention_every == 0:
                self.attn[str(i)] = ThinAttention(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        states = [None] * len(self.blocks)
        for i, block in enumerate(self.blocks):
            s, states[i], _ = block(x, states[i])
            x = s  # spikes feed forward
            if str(i) in self.attn:
                x = x + self.attn[str(i)](x)  # residual attention
        return x


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, D = 2, 16, 32
    x = torch.randn(B, T, D)

    block = SpikingSSMBlock(D, neuron_type="plif", state_dim=D)
    s, v_new, h = block(x)
    print("Input shape :", x.shape)
    print("Spike shape :", s.shape)
    print("Spike mean  :", s.mean().item())
    print("v_new shape :", v_new.shape)
    print("h shape     :", h.shape)

    loss = s.sum()
    loss.backward()
    print("Backward OK")
    print("Beta grad sample:", block.neuron.beta.grad[:5].tolist())

    # -- Regression checks for the stateful-neuron fixes ----------------------
    # 1) PLIF decay starts at beta_init (0.9), not sigmoid(0.9) ~= 0.71
    beta0 = torch.sigmoid(PLIFNeuron(D).beta)
    assert torch.allclose(beta0, torch.full((D,), 0.9), atol=1e-6), beta0[:3]

    # 2) Neuron state is [B, D], so it can be fed straight back in
    assert v_new.shape == (B, D), v_new.shape
    _, v_new2, _ = block(x, v_new.detach())
    assert v_new2.shape == (B, D), v_new2.shape

    # 3) Membrane potential integrates over time: a constant drive of 2.0 is
    #    below threshold on step 1 but must fire later (it never did before)
    drive = torch.full((1, 20, D), 2.0)
    s_const, _ = PLIFNeuron(D)(drive)
    first = s_const[0, :, 0].argmax().item()
    print("Const-drive first spike at t =", first)
    assert s_const.sum() > 0 and first > 0, first

    # 4) Chunked run == full run when the state is carried across calls
    y = torch.randn(B, T, D) * 2.0 + 3.0     # strong drive so spikes occur
    for name, neuron in (("plif", PLIFNeuron(D)), ("alif", ALIFNeuron(D))):
        s_full, _ = neuron(y)
        s_a, st = neuron(y[:, : T // 2])
        s_b, st = neuron(y[:, T // 2 :], st)
        assert s_full.sum() > 0, name
        assert torch.equal(torch.cat([s_a, s_b], dim=1), s_full), name
        print(f"Chunked == full ({name}) OK")

    print("All checks passed")
