# Experiments

This directory contains the scripts, saved results, and figures used to run
``pychop`` experiment workflows for benchmarking, numerical comparisons, image
classification, and object detection.

## Experiment purpose

The goal of these experiments is to provide a practical validation suite for
``pychop`` beyond small unit tests. The scripts exercise ``pychop`` on workloads that
represent common use cases for low-precision simulation: dense numerical
kernels, backend-level performance comparisons, iterative refinement, image
classification, and object detection. Together, they help evaluate whether
``pychop`` can emulate different floating-point formats and rounding modes
consistently across NumPy, PyTorch, JAX, and MATLAB-style workflows while
keeping the workflow transparent enough for users to rerun and adapt.

These experiments should therefore be read as a repository-level reproducibility
and validation workflow. They are intended to check implementation behavior,
measure runtime overhead, compare ``pychop`` against related tooling such as
`gfloat`, and show how low-precision choices affect representative machine
learning tasks. They are not intended to be a bit-for-bit archival copy of the
exact experiments used to prepare the paper.

## Environment setup

Run experiments from this directory so that relative paths such as `data/`,
`results/`, `figures/`, and model checkpoints resolve correctly.

The file `python_packages.txt` records the Python packages used for these
experiments. To recreate a similar Python environment, create and activate a
virtual environment, then run:

```bash
./install_python_packages.sh python_packages.txt
```

`cuda_cudnn.txt` records the CUDA/cuDNN environment used for GPU runs. GPU
benchmarks and deep-learning experiments can vary across driver, CUDA, cuDNN,
PyTorch, JAX, and hardware versions.

## Experiment workflow

### Runtime and backend benchmarks

1. Generate random matrix data with `data_gen.m`.
2. Run MATLAB, NumPy, PyTorch, and GPU timing scripts:
   - `speed_n.m`, `speed_n_b.m`
   - `speed_n.py`, `speed_n_b.py`
   - `speed_n_gpu.py`, `speed_n_gpu_b.py`
3. Draw runtime-ratio plots with `speed_draw.py` and `speed_draw_b.py`.
4. Run backend breakdown scripts:
   - `backend_breakdown_cpu.py`
   - `backend_breakdown_gpu.py`
   - `backend_breakdown_plot.py`
5. Compare ``pychop`` with `gfloat` using `compare_gfloat.py` and
   `compare_gfloat_plot.py`.

Most benchmark outputs are written under `results/`, `results_a/`, and
`figures/`.

### Iterative refinement overhead

Use `emulation_overhead_on_ir.py` to benchmark emulated FP32 overhead in an
iterative-refinement workflow. Use `emulation_overhead_on_ir_plot.py` to
generate the corresponding plots.

### Image classification experiments

1. Train baseline models with `image_class_train.py`.
2. Evaluate full-precision checkpoints with `image_class_val.py`.
3. Evaluate quantized checkpoints and collect quantization metrics with
   `image_class_val_chop.py`.
4. Generate summary plots with `image_class_analysis.py`.

The scripts cover MNIST, FashionMNIST, Caltech101, and Oxford-IIIT Pet. Saved
checkpoints, CSV files, and visualization images are stored in this directory
and its subdirectories.

### Object detection experiments

Run Faster R-CNN object-detection experiments with:

- `r_cnn_obj_ft.py`
- `r_cnn_obj_ft_st.py`

These scripts download and evaluate COCO val2017 data, then save CSV metrics
and visualization outputs.

## Notes on randomness and reproducibility

The latest scripts fix local seeds, 
However, the random seeds and random states in this directory are not
guaranteed to be identical to those used for the experiments reported in
[arXiv:2504.07835](https://arxiv.org/abs/2504.07835).

The difference is expected because the purpose of this directory is to preserve
an executable and extensible experiment workflow, not to freeze every hidden
state from the paper runs. Paper results can depend on details that are easy to
change or difficult to recover exactly, including random matrix generation,
stochastic rounding streams, train/validation/test splits, data augmentation
order, neural-network initialization, downloaded dataset versions,
nondeterministic GPU kernels, and the exact software and hardware stack.
