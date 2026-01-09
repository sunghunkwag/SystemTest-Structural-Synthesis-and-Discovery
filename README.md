# SystemTest: Neuro-Genetic Recursive Self-Improvement

This repository implements a verified Recursive Self-Improvement (RSI) system combining MLP-based neural guidance with evolutionary algorithms for program synthesis. It features a closed-loop architecture where discovered primitives are autonomously compiled, registered, and reused, enabling the system to solve increasingly complex tasks without hardcoded heuristics or pre-defined solutions. The synthesis engine strictly adheres to input-output constraints, ensuring verifiable honesty in capability acquisition.

## Usage
Run the infinite RSI loop:
```bash
python Systemtest.py hrm-life
```