# Decision Tree Implementation - Zhoumath (v0.1.0)

## Overview
This project provides a basic implementation of a decision tree from scratch, aimed at helping developers and learners understand the foundational concepts of decision trees in machine learning. The project includes a custom-built `DecisionTreeZhoumath` class, example scripts demonstrating usage, and tools for evaluating model performance.

### Features
- Custom decision tree model implementation (`DecisionTreeZhoumath`).
- Calculation of entropy and data splitting based on feature thresholds.
- Example scripts for training, testing, and evaluating the model.
- Performance evaluation using ROC-AUC metrics.
- Utility functions for data ranking based on frequency bins.

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
├── examples
│   ├── cal_ranking_by_freq.py
│   └── decision_tree_zhoumath_examples
│       └── decision_tree_zhoumath_example_script.py
└── __pycache__
```

### Key Files
- **`decision_tree.py`**: Contains helper functions such as entropy calculation and data splitting.
- **`decision_tree_zhoumath.py`**: Implements the `DecisionTreeZhoumath` class for custom decision tree modeling.
- **`examples/`**: Example scripts demonstrating usage of the decision tree model, including evaluation with ROC-AUC and data ranking.

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
Current version: **0.1.0**

## Author

- **Zhoushus**
  - Email: [zhoushus@foxmail.com](mailto:zhoushus@foxmail.com)
  - GitHub: [https://github.com/shanghaizhoushus](https://github.com/shanghaizhoushus)

## Acknowledgements
- `scikit-learn` for providing easy access to datasets and evaluation tools.
- The open-source community for ongoing inspiration and support.
