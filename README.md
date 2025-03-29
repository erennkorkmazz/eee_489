 k-NN Classifier on the Wine Dataset 

This project implements the k-Nearest Neighbors (k-NN) algorithm from scratch** (without using `sklearn.neighbors`) to classify wines from the UCI Wine dataset.



 Project Files

- `knn.py` - Pure Python implementation of k-NN with Euclidean, Manhattan, and Minkowski distance support.
- `analysis.ipynb` - Jupyter notebook with data visualization, evaluation, plots, and final model testing.
- `README.md` 


Dataset

- Source: [UCI Wine Dataset](https://archive.ics.uci.edu/dataset/109/wine)
- 178 instances, 13 features, and 3 classes (target values: 0, 1, 2)



 How to Run

1. Clone this repo or download the files.
2. Open `analysis.ipynb` in [Google Colab]
3. (https://colab.research.google.com/drive/1v57tDPFPmR-2euPUH53pYXjZeYr2uH27?usp=sharing)) or Jupyter.
4. Run all cells in order to:
   - Load and preprocess data
   - Test different `k` values and distance metrics
   - View accuracy graphs and evaluation metrics



 Final Model Summary

- Best performing model: `k = 5`, Manhattan distance
- Accuracy: 100% on the test set (36 samples)
- Reason: Reaches perfect classification with fewer neighbors than other metrics (more efficient)


Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn`


 Author

- GitHub: erennkorkmazz
- Project created for  Fundamentals of Machine Learning homework I

---
