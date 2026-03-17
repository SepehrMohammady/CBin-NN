<div align="center">

# CBin-NN
**Inference Engine for Binarized Neural Networks on Resource-Constrained Devices**

[![Paper](https://img.shields.io/badge/Paper-MDPI-blue.svg)](https://www.mdpi.com/2079-9292/13/9/1624)
[![Website](https://img.shields.io/badge/Website-Live-green.svg)](https://edge-learning-machine.github.io/CBin-NN/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## Overview
CBin-NN is an open-source, highly optimized C-based inference engine designed specifically for deploying Binarized Neural Networks (BNNs) onto resource-constrained edge devices (such as microcontrollers and FPGAs). By exploiting bitwise operations (XOR and POPCOUNT) and eliminating the need for floating-point arithmetic overhead where possible, CBin-NN achieves extreme efficiency in both memory footprint and execution latency.

## Key Features
- **Extensive Operator Support**: Multiple permutations of fully-connected and convolutional operators depending on variable datatypes:
  - 1: Float/Int Input &rarr; Binary Output
  - 2: Binary Input &rarr; Binary Output
  - 3: Binary Input &rarr; Float/Int Output
  - 4: Float/Int Input &rarr; Float/Int Output
- **Hardware-Aware Optimizations**: Loop unrolling mechanisms, fused Batch Normalization, optional PReLU activation routines, and dedicated bit-packing structures.
- **Work-Flow Integration**: Specially designed to sit downstream of BNN Neural Architecture Search (NAS) frameworks: 
  *(Train w/ NAS &rarr; Python Generation &rarr; C Code Build &rarr; Edge Deploy)*.

## Architecture & Visuals
The performance gap of edge-deployed BNNs leveraging the CBin-NN memory footprint vs traditional frameworks.

![Result 1](result1.png)

![Result 2](result2.png)

## Getting Started

### Prerequisites
- Python 3.x for Code Generation.
- C99/C11 compatible compiler (e.g., modern GCC/Clang) for edge inference.

### Code Generation Pipeline
The Python scripts situated in the `Code Generation/` directory consume pre-trained network parameters and generate optimized, ready-to-flash C headers containing bit-packed weights and the C execution tree.

Example implementations are currently being expanded. 

## License
Distributed under the MIT License. See `LICENSE` for more information.
