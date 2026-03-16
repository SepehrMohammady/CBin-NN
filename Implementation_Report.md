# CBin-NN Implementation Report

## Operator 3: Binary Input => Float/Int Output (BBQConv2D)

### 1. Initial Setup and Git Configuration
- **What:** Forked the repository from `Edge-Learning-Machine/CBin-NN` to `SepehrMohammady/CBin-NN` and configured Git remotes.
- **Why:** To enable independent development and tracking of changes without affecting the mainline, while retaining the ability to pull future updates.
- **How:** Set `origin` to the personal fork and `upstream` to the original repository.

### 2. Operator Architectural Planning
- **What:** Decided on the structure and naming conventions for Operator 3. Following the existing QB, BB, and QQ naming schemes, this new operator family will be named **BBQConv2D**.
- **Why:** To maintain consistency with the existing CBin-NN codebase where `B` indicates Binary and `Q` indicates Quantized/Float representations. `BBQ` inherently represents (Binary Input, Binary Weight, Quantized/Float Output).
- **How:** We will create `BBQConv2D.c`, `BBQConv2D_Optimized.c`, and `BBQConv2D_Optimized_PReLU.c` inside the `Operators/` folder, update the `CBin-NN.h` header, and modify the Python code generators (`model_converter.py` and `save_params.py`).

### 3. C Implementation for BBQConv2D Variants
- **What:** Implemented the core C functions to take Binary input arrays (int[]), perform bitwise XOR & POPCOUNT dot products, and output to unbinarized float arrays (loat[]). Created standard, loop-unrolled (optimized to packs of 32 filters), and PReLU activation versions.
- **Why:** To fulfill the explicit requirement for 'Operator 3: BINARY INPUT => FLOAT/INT OUTPUT' and support multiple operator variations as requested.
- **How:** Added BBQConv2D.c, BBQConv2D_Optimized.c, and BBQConv2D_Optimized_PReLU.c to the Operators directory. Updated CBin-NN.h with their function declarations.

