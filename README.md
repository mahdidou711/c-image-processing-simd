# C Image Processing SIMD

[![C CI](https://github.com/mahdidou711/c-image-processing-simd/actions/workflows/ci.yml/badge.svg)](https://github.com/mahdidou711/c-image-processing-simd/actions/workflows/ci.yml)

This repository contains a cleaned C project around image processing and performance-oriented optimization.

## What it demonstrates

- BMP image loading and writing in C
- scalar image-processing implementations
- OpenMP parallelization
- ARM NEON SIMD-oriented code paths when compiled on a NEON-capable target
- comparison of baseline and optimized implementations
- low-level C programming for performance-oriented workloads

## Repository structure

```text
include/              Header files
src/                  C source files
.github/workflows/    Continuous-integration workflow
Makefile              Local build entry point
README.md             Project documentation
```

## Current source files

- `include/lib_bmp.h`
- `src/lib_bmp.c`
- `src/tp8_etudiants.c`
- `src/main_tp10.c`

`src/tp8_etudiants.c` contains the optimized image-processing work, including OpenMP parallel sections and an ARM NEON implementation guarded by `__ARM_NEON`.

## Build

Requirements:

- GCC or a compatible C compiler
- OpenMP support (`-fopenmp`)
- `make`

Build the default TP8 executable:

```bash
make
```

Equivalent explicit target:

```bash
make tp8
```

Build the TP10 executable:

```bash
make tp10
```

Remove generated binaries:

```bash
make clean
```

The Makefile currently produces:

- `image_processing_tp8`
- `image_processing_tp10`

The GitHub Actions workflow builds the default `make` target on Ubuntu, then cleans the generated binary. This validates the portable scalar/OpenMP build path. The ARM NEON path requires a compatible ARM target and is not exercised by the current x86 GitHub-hosted runner.

## Platform note

The SIMD-specific code is conditionally compiled under `__ARM_NEON`. On non-ARM/NEON systems, the project remains buildable through the scalar/OpenMP path without requiring NEON compiler flags.

## Scope

This repository focuses on the implementation and optimization work itself. Course PDFs, reports, binaries, assistant configuration files, and generated build artifacts are intentionally excluded.
