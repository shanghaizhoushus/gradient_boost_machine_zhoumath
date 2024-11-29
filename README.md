# Decision Tree Zhoumath Implementation - Zhoushus (v0.2.1)

## Overview
This project provides a basic implementation of a decision tree and tree-based ensemble learning algorithms like random forest and gradient boosting machines from scratch, aimed at helping developers understand the concepts of decision tree-based models in machine learning.

### Features
- **Core Features**:
  - Custom decision tree model implementation (`DecisionTreeZhoumath`).
  - Calculation of entropy, Gini impurity, and data splitting based on feature thresholds.
  - Example scripts for training, testing, and evaluating the model.
  - Performance evaluation using ROC-AUC metrics.
  - Utility functions for data ranking based on frequency bins.

- **Version Updates**:
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

  - **New in v0.1.8**:
    - Refactored `DecisionTreeWithNullZhoumath` to extend `DecisionTreeZhoumath` for better code reuse and maintainability.
    - Added optimized feature sorting and filtering methods to improve efficiency when handling data with missing values.
    - Updated training methods to automatically select the appropriate decision tree version (with or without null support) based on the input data.
    - Enhanced batch prediction process to accommodate missing values, ensuring consistent performance during inference.

  - **New in v0.1.9**:
    - Added support for early stopping using validation data based on AUC score, allowing the model to stop training when there is no improvement, thus preventing overfitting.
    - Introduced `pos_weight` parameter to handle imbalanced datasets, adjusting the importance of positive samples during tree construction.
    - Enhanced missing value handling by improving the logic for dynamically assigning missing values to either left or right branches during node splitting.
    - Improved overall training efficiency by incorporating real-time validation AUC evaluation and saving the best model during early stopping.
    - Added a new method `to_pkl` to save the trained model as a `.pkl` file for easy reuse.

  - **New in v0.1.10**:
    - Introduced modular classes (`TreeNode`, `CollectionNode`, `BestStatus`, `EarlyStopper`) to enhance code organization, modularity, and readability.
    - Improved missing value handling by refining node split strategies and dynamically selecting left or right branches based on feature values.
    - Enhanced early stopping mechanism through a dedicated `EarlyStopper` class for better control and transparency during model training.
    - Added support for Gini impurity as a new split criterion (`gini`), providing additional flexibility for classification tasks.
    - Improved performance and readability by refactoring key components and introducing more efficient tree traversal and node processing logic.

  - **New in v0.1.11**:
    - Refactored key functions to improve modularity and reusability, enhancing the readability and maintainability of the code.
    - Added detailed docstrings for all key functions and methods to provide better understanding and easier onboarding for new developers.
    - Improved feature importance tracking with a dedicated `FeatureImportances` class, making it easier to analyze and interpret model behavior.
    - Enhanced `_iterate_features` function to better handle missing values and optimize feature iteration for more efficient decision tree building.
    - Improved `_init_root_collection_node` to simplify root node initialization, reducing code redundancy.
    - Made early stopping evaluation more transparent by refining output messages and evaluation logic.

  - **New in v0.1.12**:
    - Introduced `random_column_rate` parameter, enabling random feature selection during node splitting to prevent overfitting and add randomness, similar to random forest.
    - Added `CatgorialModule` to support categorical features, including the ability to handle mixed data types and perform automatic encoding for categorical variables.
    - Improved missing value handling with enhanced logic for assigning missing values during splitting, ensuring better performance with mixed-type datasets.
    - Updated `_choose_best_split` and `_iterate_features` functions to integrate random feature selection and categorical feature handling, improving efficiency and robustness.
    - Refactored prediction logic into a new `Predictor` class to make the prediction process modular and easily extendable.
    - Enhanced model explainability by adding feature names to nodes with `replace_features_with_column_names`, allowing easier interpretation of decision paths.
	
  - **New in v0.2.0**:
    - Random Forest Implementation: Introduced `RandomForestZhoumath`, an implementation of Random Forest based on `DecisionTreeZhoumath`, with customizable parameters like the number of base trees, feature and sample rates, and early stopping based on AUC score.
    - Early Stopping for Random Forest: Added support for early stopping in random forest training, saving the best models and preventing overfitting during ensemble training.
    - Feature Importance for Random Forest: Added `FeatureImportancesRF` to track feature importance across multiple trees in the random forest model.
    - Enhanced Model Flexibility: Improved flexibility of the Random Forest model with options for column-wise feature selection and more advanced handling of feature splits.
    - Performance Optimizations: Optimized training and prediction logic for Random Forest to support large datasets and improve overall performance.
	
  - **New in v0.2.1**:  
	- MSE as Split Criterion: Added support for MSE (Mean Squared Error) as a split criterion, enabling more effective handling of regression tasks.  
	- Regression Tree Support: Enhanced decision tree with support for regression tasks, allowing for more accurate predictions of continuous variables.  
	- Early Stopping with MSE: Introduced early stopping functionality for regression tasks based on MSE, preventing overfitting and improving model performance during training.   
	- Optimization of Training Process: Optimized the decision tree fitting process, improving efficiency and scalability for large datasets, especially in regression tasks.


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
│       ├── decision_tree_zhoumath.py
│       ├── decision_tree_with_null_zhoumath.py
│       └── decision_tree_helper_zhoumath.py
│   └── random_forest_zhoumath
│       ├── random_forest_zhoumath.py
│       └── random_forest_helper_zhoumath.py
└── examples
    ├── cal_ranking_by_freq.py
    ├── decision_tree_zhoumath_examples
	│    	└── decision_tree_zhoumath_example_script.py
	├── random_forest_zhoumath_examples
	└── random_forest_zhoumath_example_script.py
```

### Key Files
- **`decision_tree_zhoumath.py`**: Implements the `DecisionTreeZhoumath` class for custom decision tree modeling.
- **`decision_tree_with_null_zhoumath.py`**: Implements the `DecisionTreeWithNullZhoumath` class, extending `DecisionTreeZhoumath` to handle datasets with missing values (`NaN`).
- **`decision_tree_helper_zhoumath.py`**: Contains utility functions and helper methods for building and training decision trees, including feature selection and splitting logic.

- **`random_forest_zhoumath.py`**: Implements the `RandomForestZhoumath` class, which builds an ensemble of decision trees for improved classification tasks.
- **`random_forest_helper_zhoumath.py`**: Contains utility functions and helper methods specific to random forest, such as bootstrap sampling, model training, and feature importance calculation.

- **`examples/`**: Example scripts demonstrating usage of the decision tree and random forest models.
  - **`cal_ranking_by_freq.py`**: An example script for calculating model score ranking using frequency-based bins. Useful for understanding prediction distribution and model calibration.
  - **`decision_tree_zhoumath_examples/`**: Contains examples for using `DecisionTreeZhoumath`, including a script for training and evaluating the decision tree model.
    - **`decision_tree_zhoumath_example_script.py`**: A script demonstrating how to train and evaluate a decision tree using a dataset (e.g., Breast Cancer dataset).
  - **`random_forest_zhoumath_examples/`**: Contains examples for using `RandomForestZhoumath`, including a script for training and evaluating a random forest model.
    - **`random_forest_zhoumath_example_script.py`**: A script demonstrating how to train and evaluate a random forest using a dataset.



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
Contributions are welcome! Please feel free to submit pull requests or open issues if you find bugs or have suggestions for improvements. Contributions could include code improvements, new features, or documentation enhancements.

## Version
Current version: **0.2.1**

## Author
- **Zhoushus**
  - Email: [zhoushus@foxmail.com](mailto:zhoushus@foxmail.com)
  - GitHub: [https://github.com/shanghaizhoushus](https://github.com/shanghaizhoushus)

## Acknowledgements
- `scikit-learn` for providing easy access to datasets and evaluation tools.
- The open-source community for ongoing inspiration and support.

