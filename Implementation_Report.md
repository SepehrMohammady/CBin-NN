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
