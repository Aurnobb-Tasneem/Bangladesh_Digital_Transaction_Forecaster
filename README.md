# Linear Regression Analysis of Bangladeshi Transaction Data

## 📜 Project Overview

This project explores the application of **Linear Regression**, a fundamental machine learning algorithm, to analyze real-world financial transaction data from Bangladesh. The primary goal is to understand and model the relationship between different transactional variables, such as the number of issued Credit cards, the number of  Credit Card transactions, and the total transaction amounts.

This project was undertaken as a practical exercise for the first course in Andrew Ng's Machine Learning Specialization on Coursera, demonstrating the application of learned concepts to an authentic dataset.

## 📊 The Data

The dataset used in this project contains anonymized transaction data from Bangladesh. It provides insights into various card types and their associated transaction volumes and values over specific periods.

**Key features in the dataset (example):**
* `Period`: The time period of the data (e.g., March'25).
* `IC_Credit`: Number/Value related to issued credit cards.
* `CCT_Number`: Number of transactions for Credit card.
* `CCT_Amount`: Total amount of transactions for credit Card.

The data is sourced authentically and represents real transaction patterns.

## 🎯 Problem Statement

The primary objective is to apply a custom-built linear regression model to understand and predict a target variable (e.g., `CCT_Amount`) based on one or more input features (e.g., `CCT_Number`, `IC_Credit`). 

This involves:
1.  Preprocessing and understanding the data.
2.  Implementing the linear regression algorithm using vectorized operations in NumPy, including:
    * Defining a hypothesis function.
    * Implementing a cost function (e.g., Mean Squared Error).
    * Implementing an optimization algorithm (e.g., gradient descent) to find the optimal parameters (theta).
3.  Evaluating the model's performance.
4.  Interpreting the results to understand the relationships within the data.
## 🛠️ Methodology

1.  **Data Loading and Exploration:** The `Transaction_Data.csv` file is loaded using Pandas for initial inspection and understanding.
2.  **Data Preprocessing:** This may include handling missing values (if any), feature scaling (e.g., normalization or standardization, crucial for gradient descent), and selecting relevant features for the model.
3.  **Model Implementation (Custom):**
    * **Hypothesis Function:** Defined as `h(x) = θ₀ + θ₁x₁ + ... + θnxn` (or `Xθ` in vectorized form).
    * **Cost Function (MSE):** Implemented to measure the difference between predicted and actual values. Vectorized form: `J(θ) = (1/(2m)) * (Xθ - y)ᵀ(Xθ - y)`.
    * **Gradient Descent:** Implemented iteratively to update `θ` values and minimize `J(θ)`. Vectorized update rule: `θ := θ - α * (1/m) * Xᵀ(Xθ - y)`.
4.  **Model Training:** The custom model is trained on a subset of the data by running the gradient descent algorithm until convergence or for a fixed number of iterations.
5.  **Prediction & Evaluation:** The trained model (optimal `θ` values) is used to make predictions. Performance is evaluated using appropriate metrics (e.g., R-squared, Mean Squared Error, which might be calculated manually or using basic NumPy functions).
6.  **Visualization:** `matplotlib` and `seaborn` are used to visualize the data, the cost function's convergence, and the model's results (e.g., scatter plots of actual vs. predicted values, regression line).

## 💻 Technologies & Libraries Used

* **Python 3.x**
* **Jupyter Notebook**
* **Pandas:** For data manipulation and analysis (loading CSV, data structuring).
* **NumPy:** **Core library for numerical operations, vectorization, and implementing the linear regression algorithm from scratch (hypothesis, cost function, gradient descent).**
* **Matplotlib & Seaborn:** For data visualization.

## 📁 Project Structure


├── data/ 
    
    └── Transaction_Data.csv  # The dataset
├── notebooks/                 #Main directory

       └── main.ipynb            # Jupyter Notebook with the analysis and custom model
└── README.md                 



## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Aurnobb-Tasneem/Bangladesh_Digital_Transaction_Forecaster
    cd Bangladesh_Digital_Transaction_Forecaster
    ```
2.  **Ensure you have Python and the necessary libraries installed.** You can install them using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn jupyter
    ```
3.  **Place the `Transaction_Data.csv` file** in the designated `data/` directory (or update the path in the notebook if it's located elsewhere).
4.  **Open and run the `main.ipynb` Jupyter Notebook:**
    ```bash
    jupyter notebook main.ipynb
    ```
    Or, if you use JupyterLab:
    ```bash
    jupyter lab main.ipynb
    ```
5.  Follow the steps within the notebook to execute the data loading, preprocessing, custom model training, and evaluation.


## 👤 Author

* **Tasneem Abdullah Aurnobb**
    * https://www.linkedin.com/in/aurnobb/
    * https://github.com/Aurnobb-Tasneem

## 🙏 Acknowledgements

* Andrew Ng and the Coursera Machine Learning Specialization for the foundational knowledge and inspiration to implement algorithms from scratch.
