# Multi-stride neural operators for long-time turbulence prediction

Code for the paper [*Long-time turbulence prediction at fine time steps using multi-stride neural operators*](https://arxiv.org/pdf/2604.12794) by G. Pan, H. Yang, Y. Wang, Z. Xu, N. Yi and J. Wang.

This repository contains the Channel-DT10 case: turbulent channel flow with velocity fields stored every 10 DNS time steps. Training has two stages:

- **Stage 1** trains the predictors on two-step autoregressive rollouts. IFF is the fixed-stride baseline. SC-IFF predicts at strides 1, 2, 4, 8, 16 and 32 with one of three stride-conditioning schemes: Concat, Indep or FiLM.
- **Stage 2** freezes the selected SC-IFF predictors and trains only FA-CSF, which generates the intermediate fields of each 32-frame interval and feeds its corrected last field into the next interval.

## Environment

```bash
python -m pip install -r requirements.txt
```

## Data

The dataset is not included because of its size. The code reads one NumPy file, set by `data_path` in `config.py`:

- type `float32`, shape `(20, 4000, 32, 33, 16, C)` with axes trajectory, time, x, y, z and component;
- `C >= 3`, where the first three components are u, v and w and any others are ignored;
- trajectories 0–11 are TRAIN, 12–15 are VALID and 16–19 are TEST.

The array holds velocities only. `channel_grid.json` holds the grid coordinates x, y and z.

## Running

```bash
python train.py --model iff
python train.py --model concat
python train.py --model indep
python train.py --model film
python compare.py
```

## Citation

```bibtex
@article{pan2026long,
  title   = {Long-time turbulence prediction at fine time steps using multi-stride neural operators},
  author  = {Pan, Guanyu and Yang, Huiyu and Wang, Yunpeng and Xu, Zikun and Yi, Nianyu and Wang, Jianchun},
  journal = {arXiv preprint arXiv:2604.12794},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
