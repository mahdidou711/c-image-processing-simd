# C Image Processing SIMD

[![C CI](https://github.com/mahdidou711/c-image-processing-simd/actions/workflows/ci.yml/badge.svg)](https://github.com/mahdidou711/c-image-processing-simd/actions/workflows/ci.yml)


This repository contains a cleaned C project around image processing and SIMD-oriented optimization.

## Goals

The project is intended to show:

- BMP image loading and writing in C
- scalar image-processing implementations
- SIMD-oriented image-processing code
- comparison between baseline and optimized versions
- low-level C programming for performance-oriented applications

## Repository structure

- include/ contains header files
- src/ contains C source files
- examples/ is reserved for small input and output examples
- docs/ is reserved for notes, benchmarks, and implementation explanations

## Current source files

- include/lib_bmp.h
- src/lib_bmp.c
- src/tp8_etudiants.c
- src/main_tp10.c

## Build

A Makefile will be added after normalizing the entry points and function names.

For now, compilation can be tested manually with:

gcc -Wall -Wextra -O2 -fopenmp -Iinclude src/lib_bmp.c src/tp8_etudiants.c -o image_processing

This command uses OpenMP, so the compiler must support `-fopenmp`. Depending on the target platform, SIMD-specific flags may also be required.

## Notes

This repository contains only cleaned source files. Course PDFs, reports, binaries, assistant configuration files, and generated files were excluded.
