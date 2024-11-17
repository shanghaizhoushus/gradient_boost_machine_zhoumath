# Gradient Boost Machine Zhoumath Implementation - Zhoushus (v0.1.2)

## Overview
This project provides a basic implementation of a decision tree and tree based ensemble learning algorithms like random forest and gradient boosting machines from scratch, aimed at helping developers understand the concepts of decision tree based models in machine learning.

### Features
- Custom decision tree model implementation (`DecisionTreeZhoumath`).
- Calculation of entropy and data splitting based on feature thresholds.
- Example scripts for training, testing, and evaluating the model.
- Performance evaluation using ROC-AUC metrics.
- Utility functions for data ranking based on frequency bins.
- **New in v0.1.1**:
  - Improved tree pruning capabilities using entropy-gain and entropy-gain-ratio to enhence accurancy on test set and prevent overfitting.
  - Updated example scripts to demonstrate new features.
- **New in v0.1.2**:
  - Added BFS (Breadth-First Search) functionality for building decision trees, allowing for more flexible and efficient tree construction.


## Installation
### Prerequisites
The project requires Python 3.6 or higher. To install the necessary dependencies, run the following command:

```sh
pip install -r requirements.txt
```

### Dependencies
- `scikit-learn`: For dataset loading and evaluation metrics.
- `numpy`: For efficient numerical operations.
- `pandas`: For data handling and manipulation.
- **New in v0.1.1**:
  - `matplotlib`: For enhanced data visualization of decision boundaries.

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
        └── decision_tree_visualization.py

```

### Key Files
- **`decision_tree_zhoumath.py`**: Implements the `DecisionTreeZhoumath` class for custom decision tree modeling.
- **`examples/`**: Example scripts demonstrating usage of the decision tree model, including evaluation with ROC-AUC, data ranking, and visualization of decision boundaries.
- **`decision_tree_visualization.py`**: A new script added in v0.1.1 for visualizing decision tree decision boundaries.

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

### Example: Visualizing Decision Boundaries
The `decision_tree_visualization.py` script provides an example of how to visualize the decision boundaries of the trained model:

```sh
python examples/decision_tree_zhoumath_examples/decision_tree_visualization.py
```
This visualization can help in understanding how the decision tree splits the feature space.

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
Current version: **0.1.2**

## Author

- **Zhoushus**
  - Email: [zhoushus@foxmail.com](mailto:zhoushus@foxmail.com)
  - GitHub: [https://github.com/shanghaizhoushus](https://github.com/shanghaizhoushus)

## Acknowledgements
- `scikit-learn` for providing easy access to datasets and evaluation tools.
- The open-source community for ongoing inspiration and support.
