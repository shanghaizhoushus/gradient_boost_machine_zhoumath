# Gradient Boost Machine Zhoumath Implementation - Zhoushus (v0.1.7)

## Overview
This project provides a basic implementation of a decision tree and tree-based ensemble learning algorithms like random forest and gradient boosting machines from scratch, aimed at helping developers understand the concepts of decision tree-based models in machine learning.

### Features
- Custom decision tree model implementation (`DecisionTreeZhoumath`).
- Calculation of entropy and data splitting based on feature thresholds.
- Example scripts for training, testing, and evaluating the model.
- Performance evaluation using ROC-AUC metrics.
- Utility functions for data ranking based on frequency bins.
- **New in v0.1.1**:
  - Improved tree generation with entropy-gain-ratio to enhance accuracy on test sets and prevent overfitting.
  - Updated example scripts to demonstrate new features.
  
- **New in v0.1.2**:
  - Added BFS (Breadth-First Search) functionality for building decision trees, allowing for more flexible and efficient tree construction.
  
- **New in v0.1.3**:
  - Optimized dataset splitting using index slicing to minimize unnecessary data copies.
  - Replaced recursive DFS implementation with an iterative stack-based approach to prevent stack overflow and enhance memory efficiency.

- **New in v0.1.4**:
  - Improved numerical stability by adding perturbations to feature values to handle duplicate thresholds.
  - Enhanced entropy and intrinsic value calculation with safeguards against division by zero using small constant additions.
  - Updated the prediction method to use more efficient tree traversal logic for batch processing.

- **New in v0.1.5**:
  - Integrated Numba JIT compilation to speed up key operations such as entropy calculation and threshold selection.
  - Optimized memory usage by ensuring contiguous arrays are used where necessary.
  - Improved tree traversal efficiency during predictions by reducing redundant calculations.

- **New in v0.1.6**:
  - Enhanced decision tree construction logic by refining BFS and DFS methods for better depth control and efficiency.
  - Improved feature perturbation logic to ensure better handling of data with low variance.
  - Added more detailed comments to code for easier readability and maintenance.

- **New in v0.1.7**:
  - Added support for handling missing values (`NaN`) in feature data during decision tree construction.
  - Enhanced the decision tree algorithm (`DecisionTreeWithNullZhoumath`) to make informed decisions on how to split data with missing values.
  - Introduced new logic for assigning missing values to either left or right branches during node splitting, ensuring better performance with incomplete datasets.
  - Updated prediction methods to handle missing values effectively during inference.

## Installation
### Prerequisites
The project requires Python 3.6 or higher. To install the necessary dependencies, run the following command:

```sh
pip install -r requirements.txt
```

You can clone the project from GitHub:

```bash
git clone https://github.com/shanghaizhoushus/gradient_boost_machine_zhoumath.git
```

or download the compressed document on GitHub.

### Dependencies
- `scikit-learn`: For dataset loading and evaluation metrics.
- `numpy`: For efficient numerical operations.
- `pandas`: For data handling and manipulation.
- **New in v0.1.1**:
  - `matplotlib`: For enhanced data visualization of decision boundaries.
- **New in v0.1.5**:
  - `numba`: To accelerate numerical computations for improved performance.

## File Structure
```
.
├── LICENSE
├── README.md
├── decision_tree.py
├── requirements.txt
├── scripts
│   └── decision_tree_zhoumath
│       └── decision_tree_zhoumath.py
└── examples
    ├── cal_ranking_by_freq.py
    └── decision_tree_zhoumath_examples
        ├── decision_tree_zhoumath_example_script.py
```

### Key Files
- **`decision_tree_zhoumath.py`**: Implements the `DecisionTreeZhoumath` class for custom decision tree modeling.
- **`examples/`**: Example scripts demonstrating usage of the decision tree model, including evaluation with ROC-AUC, data ranking, and visualization of decision boundaries.
- **`decision_tree_visualization.py`**: A script for visualizing decision tree decision boundaries.

## Usage
### Example: Training and Evaluating the Decision Tree
To see the decision tree in action, you can run the example script:

```sh
python examples/decision_tree_zhoumath_examples/decision_tree_zhoumath_example_script.py
```
This script demonstrates how to:
1. Load the Breast Cancer dataset using `scikit-learn`.
2. Split the data into training and testing sets.
3. Train the custom decision tree model (`DecisionTreeZhoumath`).
4. Evaluate the model using metrics such as ROC-AUC.

### Example: Calculating Frequency Bins
The `cal_ranking_by_freq.py` script provides an example of how to calculate model score ranking using frequency-based bins. This is useful for understanding the distribution of predictions and assessing model calibration.

```sh
python examples/cal_ranking_by_freq.py
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues if you find bugs or have suggestions for improvements.

## Version
Current version: **0.1.7**

## Author

- **Zhoushus**
  - Email: [zhoushus@foxmail.com](mailto:zhoushus@foxmail.com)
  - GitHub: [https://github.com/shanghaizhoushus](https://github.com/shanghaizhoushus)

## Acknowledgements
- `scikit-learn` for providing easy access to datasets and evaluation tools.
- The open-source community for ongoing inspiration and support.

