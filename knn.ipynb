{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "url = \"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data\"\n",
    "\n",
    "# Assign colum names to the dataset\n",
    "names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']\n",
    "\n",
    "# Read dataset to pandas dataframe\n",
    "dataset = pd.read_csv(url, names=names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Setting up the dependent and independent variables.\n",
    "X = dataset.iloc[:, :-1].values\n",
    "y = dataset.iloc[:, 4].values\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training and testing split\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "\n",
    "X_train = scaler.transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
       "           metric_params=None, n_jobs=None, n_neighbors=5, p=2,\n",
       "           weights='uniform')"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "classifier = KNeighborsClassifier(n_neighbors=5)\n",
    "classifier.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['Iris-virginica', 'Iris-setosa', 'Iris-versicolor',\n",
       "       'Iris-versicolor', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',\n",
       "       'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-versicolor',\n",
       "       'Iris-virginica', 'Iris-virginica', 'Iris-virginica',\n",
       "       'Iris-versicolor', 'Iris-versicolor', 'Iris-setosa',\n",
       "       'Iris-virginica', 'Iris-virginica', 'Iris-setosa',\n",
       "       'Iris-virginica', 'Iris-setosa', 'Iris-setosa', 'Iris-setosa',\n",
       "       'Iris-versicolor', 'Iris-versicolor', 'Iris-versicolor',\n",
       "       'Iris-setosa', 'Iris-virginica', 'Iris-virginica'], dtype=object)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred = classifier.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[12  1  0]\n",
      " [ 0  7  1]\n",
      " [ 0  0  9]]\n",
      "                 precision    recall  f1-score   support\n",
      "\n",
      "    Iris-setosa       1.00      0.92      0.96        13\n",
      "Iris-versicolor       0.88      0.88      0.88         8\n",
      " Iris-virginica       0.90      1.00      0.95         9\n",
      "\n",
      "      micro avg       0.93      0.93      0.93        30\n",
      "      macro avg       0.92      0.93      0.93        30\n",
      "   weighted avg       0.94      0.93      0.93        30\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "print(confusion_matrix(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
