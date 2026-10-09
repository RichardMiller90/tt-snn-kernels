# tt-snn-kernels

Spiking neural network (SNN) kernels written in [TT-Metalium](https://github.com/tenstorrent/tt-metal) for Tenstorrent hardware (Blackhole and Wormhole).

> **Status:** early development. Interfaces, layout, and kernel coverage will change.

## Why

Spiking neurons communicate with binary events, so their activations are often sparse. Standard dense kernels do not take advantage of that. TT-Metalium is Tenstorrent's low-level SDK for writing kernels that run on the chip's Tensix cores, which makes it possible to write custom kernels for neuron dynamics and to test whether spike sparsity can be turned into saved work on this hardware. That is an open question, and answering it is the point of this repository.

The kernels support ongoing research on a hybrid spiking state-space model with a thin attention block.

## Planned kernels

| Kernel | What it computes | Status |
|---|---|---|
| Spiking neuron (LIF, PLIF) | Membrane update, threshold, and reset over time steps | Planned |
| Adaptive-threshold neuron (ALIF, PALIF) | Neuron update with a spike-history-driven threshold | Planned |
| Diagonal SSM recurrence | Linear state-space scan over a sequence | Planned |
| Spike readout | Projection of binary spikes, accumulate only | Planned |

## Neuron models

Reference equations the kernels are meant to implement. Per channel, with input current `I_t`:

```
a_t     = rho * a_(t-1) + s_(t-1)                          # adaptation trace (ALIF, PALIF)
theta_t = theta0 + beta * a_t                              # firing threshold (beta = 0 for LIF, PLIF)
u_t     = alpha * u_(t-1) + I_t - theta_(t-1) * s_(t-1)    # leak plus soft reset
s_t     = 1 if u_t > theta_t else 0                        # spike
```

- **LIF:** fixed decay `alpha`, fixed threshold.
- **PLIF:** learnable decay `alpha`.
- **ALIF:** fixed decay, adaptive threshold.
- **PALIF:** learnable decay and adaptive threshold.

## Targets

- Tenstorrent Blackhole
- Tenstorrent Wormhole

## Getting started

You need a Tenstorrent device and a working TT-Metalium installation. Follow the official TT-Metalium documentation to set it up. Build and run instructions for this repository will be added with the first kernel.

## Disclaimer

Independent project, not affiliated with or endorsed by Tenstorrent.
