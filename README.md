# 🐚 Prediction of Abalone Age Using Regression Models

![Project Banner - Place Your Image Here](images/abalone-banner.png)

---

## 📘 Introduction

Abalone is a marine shellfish, rich in protein and omega-3 fatty acids. These sea snails have a distinctive flattened, ear-shaped shell and muscular foot. As abalones grow, their shells form **concentric rings**, similar to tree rings, which can be used to estimate age.

However, **manual ring counting** under a microscope is time-consuming and prone to error. Moreover, ring formation is influenced not only by age but also by **environmental and feeding factors**. Since **abalone value is age-dependent**, it becomes important to develop efficient methods to estimate age.

This project leverages **machine learning regression models** to predict the number of rings — and hence the age — using various physical measurements of the abalone, offering a faster and more scalable alternative.

---

## 📦 Dependencies and Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`
- `xgboost`, `linear_models`, `ensemble`
- `google.colab.drive` (for accessing data)

---

## 📂 Dataset

**Source**:  
[UCI Machine Learning Repository - Abalone Dataset](https://archive.ics.uci.edu/ml/datasets/Abalone)  
**Donated by**: Warwick J. Nash et al., 1995  
**Total Observations**: 4177  
**Total Features**: 9  
**Missing Values**: None

**Original Study Reference**:  
*The Population Biology of Abalone (Haliotis species) in Tasmania* — Sea Fisheries Division, Technical Report No. 48.

---

## 🧼 Data Cleaning and Transformation

- Among the 9 features, **1 is categorical**: `Sex` (M: Male, F: Female, I: Infant); the rest are numeric.
- Verified the dataset has **no missing values**.
- Found 2 records with **`Height = 0.0`**, which is unrealistic. Both were Infants. These were replaced by the **average height of all Infant abalones**.

---

## 📊 Data Exploration

### ➤ Data Description
![Data description of abalone dataset](./images/data-statistics.png)

### ➤ Pair Plot Analysis
![Pair scatter plot of abalone dataset](./images/sns-pair-plot.png)

- `Length` and `Diameter` are **highly correlated**.
- `Whole Weight` is also highly correlated with other weight-based features (`Shucked Weight`, `Viscera Weight`, `Shell Weight`).

### ➤ Correlation Heatmap
![Correlation heatmap of abalone dataset](./images/correlation_matrix.png)

### ➤ Box Plot for Outlier Detection
![Box plot of abalone dataset](./images/box-plot.png)

- Visualized all numerical features for skewness and outliers using box plots.

---

## 🧮 Preprocessing

- **Label Encoding** was used to convert `Sex` (categorical) into numeric values.
- The target variable is **`Rings`**, while the remaining columns are used as independent variables (**X**).
- **StandardScaler** was applied to standardize numeric features (mean = 0, std = 1).
- The dataset was **split into training (80%) and testing (20%)** using a random seed of 42 for reproducibility.

---

## 🚧 Work in Progress

Model training, evaluation, hyperparameter tuning, and final results will be covered in the next stages.

---

## 📌 Project Goals

- Build multiple regression models (Linear, Ridge, Lasso, Random Forest, XGBoost)
- Compare performance using metrics: **MAE**, **RMSE**, and **R²**
- Tune best-performing models for optimal generalization
- Visualize predictions and errors

---

## 🧠 Future Improvements

- Add SHAP value interpretation for feature importance
- Deploy the model via a web interface or API
- Incorporate time-based or environmental features if available

---

## 📷 Image Gallery (Replace with your visuals)

| Visual | Description |
|--------|-------------|
| ![](images/data-describe.png) | Dataset Summary |
| ![](images/pairplot.png) | Pairplot of Numerical Features |
| ![](images/heatmap.png) | Feature Correlation Heatmap |
| ![](images/boxplot.png) | Boxplot for Outliers |

---

## 👨‍💻 Author

**Sandesh Paudel**  
Master of Data Science  
Charles Darwin University  
GitHub: [@sandeshp77](https://github.com/sandeshp77)

---

## 📄 License

This project is licensed under the MIT License.

