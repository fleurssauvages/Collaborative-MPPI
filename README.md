# Collaborative MPPI

This project studies **collaborative manipulation with two Franka arms and Model Predictive Path Integral control (MPPI)** in MuJoCo.

The two arms face each other and are controlled either by one **centralized** MPPI over both arms, or by one **decentralized** MPPI per arm that rolls out the other arm along its previous plan. MPPI can sample with the Standard, Pink, Poisson or Spike proposal (see [Franka_MPPI](https://github.com/fleurssauvages/Franka_MPPI)).

## Demo

<h3 align="center">Reaching</h3>

<table align="center" width="100%">
  <tr>
    <th width="50%">Centralized Spike-MPPI (32 rollouts)</th>
    <th width="50%">Decentralized Spike-MPPI (32 rollouts per arm)</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/spike_reaching_centralized.gif"
           width="100%"
           alt="Centralized Spike-MPPI reaching">
    </td>
    <td align="center">
      <img src="assets/spike_reaching_decentralized.gif"
           width="100%"
           alt="Decentralized Spike-MPPI reaching">
    </td>
  </tr>
</table>

<br>

<h3 align="center">Ball Pass</h3>

<table align="center" width="100%">
  <tr>
    <th width="50%">Centralized Spike-MPPI (32 rollouts)</th>
    <th width="50%">Decentralized Spike-MPPI (32 rollouts per arm)</th>
  </tr>
  <tr>
    <td align="center">
      <img src="assets/spike_ball_pass_centralized.gif"
           width="100%"
           alt="Centralized Spike-MPPI ball pass">
    </td>
    <td align="center">
      <img src="assets/spike_ball_pass_decentralized.gif"
           width="100%"
           alt="Decentralized Spike-MPPI ball pass">
    </td>
  </tr>
</table>

## Install

Requires Python 3.10+ and MuJoCo 3.3+.

```bash
cd "Collaborative MPPI"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python -m collab_mppi.reaching.run --sampler spike --layout centralized --viewer
python -m collab_mppi.ball_pass.run --sampler spike --layout decentralized --viewer
```

Use `--sampler standard`, `pink` or `poisson` for comparison, and `--help` for all options.
