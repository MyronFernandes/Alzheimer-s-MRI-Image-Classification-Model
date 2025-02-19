{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Alzheimer's MRI Image Classification Model**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Introduction**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Alzheimer's is a progressive disease that results in mild memory loss but can result in the inability to respond to an environment.Alzheimer's is the most common form of dementia. This results in a huge problem for the aging population that needs to be addressed. Changes in the brain can appear years before symptoms present themselves. Thus, catching these changes and addressing the problem early would be incredibly useful. Artificial intelligence can be used in this way to predict whether a patient will experience Alzheimer's.\n",
    "\n",
    "This project will use a custom convolutional net from RadImageNet to predict level of dementia of a patient, classifying them into 'Non-demented', 'Very mild demented', 'Mild demented' and 'Moderate Demented'. The Alzheimer MRI Disease Classification Dataset from author Falah.G.Salieh will be used to train and validate the models."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "****"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_kg_hide-input": true,
    "_kg_hide-output": true,
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "C:/Users/USER/Desktop/DataScience/AlzheimerMRIDiseaseClassification/AlzheimerMRIDiseaseClassificationDataset/Data/test.parquet\n",
      "C:/Users/USER/Desktop/DataScience/AlzheimerMRIDiseaseClassification/AlzheimerMRIDiseaseClassificationDataset/Data/train.parquet\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "count=0\n",
    "for dirname, _, filenames in os.walk('C:/Users/USER/Desktop/DataScience/AlzheimerMRIDiseaseClassification/AlzheimerMRIDiseaseClassificationDataset/Data/'):\n",
    "    for filename in filenames:\n",
    "        if count==0:\n",
    "            train_data_file_path=os.path.join(dirname, filename)\n",
    "            count+=1\n",
    "        else:\n",
    "            test_data_file_path=os.path.join(dirname, filename)\n",
    "        print(os.path.join(dirname, filename))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Import Data and Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "_kg_hide-input": true
   },
   "outputs": [],
   "source": [
    "import math, re, os, warnings\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow import keras\n",
    "from tensorflow.keras import layers\n",
    "from tensorflow.keras import preprocessing\n",
    "from tensorflow.keras.models import Model\n",
    "from tensorflow.keras.models import load_model\n",
    "\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import cv2\n",
    "import seaborn as sns\n",
    "sns.set_style('whitegrid')\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "from tensorflow.keras.preprocessing import image_dataset_from_directory\n",
    "\n",
    "import time\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "TensorFlow version: 2.17.0\n"
     ]
    }
   ],
   "source": [
    "print(\"TensorFlow version:\", tf.__version__)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Import data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Set matplotlib defaults**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set Matplotlib defaults\n",
    "plt.rc('figure', autolayout=True)\n",
    "plt.rc('axes', labelweight='bold', labelsize='large',\n",
    "       titleweight='bold', titlesize=18, titlepad=10)\n",
    "plt.rc('image', cmap='magma')\n",
    "warnings.filterwarnings(\"ignore\") # to clean up output cells"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Specify file paths\n",
    "\n",
    "##test_data_file_path ='C:\\\\Users\\\\USER\\\\Desktop\\\\DataScience\\\\Sem7projects\\\\AlzheimerMRIDiseaseClassificationDataset\\\\Data\\\\test.parquet'\n",
    "\n",
    "# Read data in\n",
    "\n",
    "train_data = pd.read_parquet(train_data_file_path)\n",
    "test_data = pd.read_parquet(test_data_file_path)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Turning byte format into img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def dict_to_image(image_dict):\n",
    "    if isinstance(image_dict, dict) and 'bytes' in image_dict:\n",
    "        byte_string = image_dict['bytes']\n",
    "        nparr = np.frombuffer(byte_string, np.uint8)\n",
    "        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)\n",
    "        return img"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data['image'] = train_data['image'].apply(dict_to_image)\n",
    "test_data['image'] = test_data['image'].apply(dict_to_image)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>image</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               image  label\n",
       "0  [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      3\n",
       "1  [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      0\n",
       "2  [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      2\n",
       "3  [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      3\n",
       "4  [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      0"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[[0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        ...,\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0]],\n",
       "\n",
       "       [[0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        ...,\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0]],\n",
       "\n",
       "       [[0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        ...,\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0]],\n",
       "\n",
       "       ...,\n",
       "\n",
       "       [[0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        ...,\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0]],\n",
       "\n",
       "       [[0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        ...,\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0]],\n",
       "\n",
       "       [[0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        ...,\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0],\n",
       "        [0, 0, 0]]], dtype=uint8)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data['image'][0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Split training data to get validation set**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = train_data['image']\n",
    "y = train_data['label']\n",
    "\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.75, random_state=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = test_data['image']\n",
    "y_test = test_data['label']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Rejoin data into one dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training data\n",
    "\n",
    "X_train_df = pd.DataFrame(X_train)\n",
    "y_train_df = pd.DataFrame(y_train)\n",
    "train_df = pd.concat([X_train_df, y_train_df], axis=1)\n",
    "\n",
    "# Validation data\n",
    "\n",
    "X_val_df = pd.DataFrame(X_val)\n",
    "y_val_df = pd.DataFrame(y_val)\n",
    "val_df = pd.concat([X_val_df, y_val_df], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test data\n",
    "X_test_df = pd.DataFrame(X_test)\n",
    "y_test_df = pd.DataFrame(y_test)\n",
    "test_df = pd.concat([X_test_df, y_test_df], axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "****"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Exploratory Data Analysis**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>image</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1200</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1237</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>454</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1207</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>399</th>\n",
       "      <td>[[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                  image  label\n",
       "1200  [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      3\n",
       "1237  [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      2\n",
       "454   [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      2\n",
       "1207  [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      2\n",
       "399   [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], ...      3"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**We can explore some images**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1000x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Training images\n",
    "mri_images = train_df['image']\n",
    "\n",
    "fig = plt.figure(figsize=(10, 7)) \n",
    "\n",
    "for i in range(0, len(mri_images), 1000):\n",
    "    n = int((i / 1000) + 1)\n",
    "    \n",
    "    fig.add_subplot(2,2,n)\n",
    "    plt.imshow(mri_images.iloc[i])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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wObcuXOTGoXC6H734IMPRsTTx/OHf/iH8dBDD2FAbfz48Ult+vTpSY3mOP28oGu9XvgrNxhO76f7W1fmVlecTWE0v7mVJElSabi4lSRJUml06+K2ra0tPvzhD8fznve8eP7znx+f+tSnal+Dr169Ol7/+tdHU1NTXHXVVfifHSRJkqSu6NbF7Uc+8pG444474qtf/Wp88pOfjH//93+P733ve9Ha2hpXX311XHjhhXHzzTfHwoUL4+1vf3u0trZ25+YlSZJ0luu2QNnevXtj2bJl8fWvfz3mz58fERFvfetbY+XKldHY2Bh9+/aN973vfVGpVOKDH/xg/OpXv4of//jHsXTp0u7ahdOOHuanB9FzO5bQw98UZKFgAXWlolACBcUoREDdv4oAV3HcTU1NdTtVUajh3HPPzdoOjSsFReh1uQEmOk80/nR89N7du3cntV27diU1CrKtX78+qd17771JjQIWDQ0N0djYGHPmzIlNmzbF0aNHcaxGjRqV1OjYaP9yg0W5Hd7oeqB9obARXQ+5XcYogEjHkXu90rgUXcuKuTlr1iwMhVFXJZpD9QJXuaHB3HBLbtgrNzyW2z0sd/8oNETbpddRhzKaR7nnnbZB85e+zNm5c2dSo65bdD5ou9R5jOYbhR/pet20aVPWdikkSeFgCoAV2yjO37x582LlypXJ65YvX57UKIhJHd5oX+j+SeMSweczd67S/Si3K2AumqtnU3iMdNs3t/fdd18MGjQolixZUqtdffXVcf3118fKlStj8eLFtRNQqVRi0aJF2J5PkiRJeq667ZvbLVu2xPjx4+OHP/xhfPnLX44jR47E0qVL46//+q+jubk5ZsyYcczrR4wYEWvXrn3W2ym+VaRvF0+3U/HNbW5/d/qbPv0KF/q2I/cbmuJ4O/9Z79fq5P7NMvdvvrm/miX3G9nurhEaA6rR+Nf7lUb03uL9xZ90TnLHuSvno7u/Ocgdv9xartxfQXai93b+k95L55eudfrWLILnDO137rHQ607FWHe3evP3+G9uc48j93qg99L1mnufrndfzdGVe1buHOpK7fj7VWNjI45L7q9668p/dXg2P79yr6Xc+zn9+r2urCm6ev/tqWut3P2pVLvpJ9AXv/jF+OpXvxozZ86M97///dHc3Bwf+tCH4uqrr45f/vKXsXjx4rjmmmtqr//sZz8b999/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l14aX/va1073Lp11du7cGddcc00sWbIkLrvssrj++uvj8OHDERGxZcuWeMtb3hILFiyIV7/61fGb3/zmNO/t2eXqq6+Ov/3bv639/9WrV8frX//6aGpqiquuuipWrVp1Gvfu7NHW1hYf/vCH43nPe148//nPj0996lO1Puiek1Nv+/bt8fa3vz0WLVoUL33pS+PGG2+s/TvPx6nV1tYWV1xxRdx999212jP93LjjjjviiiuuiKampnjTm94UW7ZsOdW7XVp0PlasWBFveMMbYuHChfHKV74yvv/97x/znjPlfJxRi9uPf/zjsWrVqvjGN74R/+t//a/4/Oc/Hz/+8Y9P926dNarValxzzTVx8ODB+Pa3vx2f/vSn4xe/+EV85jOfiWq1Gu985ztj5MiRsWzZsrjyyivjXe96V2zbtu107/ZZ4ZZbbolf/vKXtf/f2toaV199dVx44YVx8803x8KFC+Ptb397tLa2nsa9PDt85CMfiTvuuCO++tWvxic/+cn493//9/je977nOTlN3vOe98SAAQPi5ptvjg984APxmc98Jn72s595Pk6xw4cPx3vf+95Yu3ZtrfZMPze2bdsW73znO2Pp0qVx0003xfDhw+Md73hH2Fi16+h8NDc3x9ve9rZYsmRJ/OAHP4hrrrkmrrvuurj99tsj4gw7H9UzxIEDB6rz5s2r3nXXXbXaF77wheqf//mfn8a9OrusW7euOmvWrGpzc3Ot9qMf/ah66aWXVu+4447qggULqgcOHKj9uze/+c3V//2///fp2NWzyp49e6ovfOELq1dddVX1/e9/f7VarVa///3vV1/60pdWOzo6qtVqtdrR0VH9gz/4g+qyZctO566W3p49e6qzZ8+u3n333bXaV77ylerf/u3fek5Og71791ZnzZpVfeSRR2q1d73rXdUPf/jDno9TaO3atdX/9t/+W/UP//APq7Nmzar9HH+mnxuf+cxnjvkZ39raWl24cOEx6wA9e/XOx3e+853qq171qmNe+/d///fV9773vdVq9cw6H2fMN7dr1qyJo0ePxsKFC2u1xYsXx8qVK6Ojo+M07tnZY9SoUfGv//qvMXLkyGPqLS0tsXLlypg9e3YMGDCgVl+8eHGsWLHiFO/l2edjH/tYXHnllTFjxoxabeXKlbF48eKoVCoREVGpVGLRokWej5Psvvvui0GDBsWSJUtqtauvvjquv/56z8lp0K9fv+jfv3/cfPPNceTIkdiwYUMsX748LrjgAs/HKXTPPffERRddFN/73veOqT/Tz42VK1fGhRdeWPt3/fv3jzlz5niOuqje+SgeNTxeS0tLRJxZ5+OMWdw2NzfHsGHDok+fPrXayJEj4/Dhw7F3797Tt2NnkSFDhsRll11W+/8dHR3xrW99Ky6++OJobm6O0aNHH/P6ESNGxI4dO071bp5V7rzzzvjd734X73jHO46pez5Ojy1btsT48ePjhz/8YbzqVa+Kl73sZfGFL3whOjo6PCenQd++feNDH/pQfO9734umpqa4/PLL44UvfGG8/vWv93ycQn/6p38aH/jAB6J///7H1J/pHHiOTo5652PChAmxYMGC2v9/4okn4pZbbolLLrkkIs6s89F4uncg18GDB49Z2EZE7f+3tbWdjl06633iE5+I1atXx0033RQ33ngjnh/Pzclz+PDh+F//63/Fhz70oejXr98x/67e9eL5OLlaW1tj06ZN8d3vfjeuv/76aG5ujg996EPRv39/z8lpsn79+njJS14Sf/EXfxFr166N6667Li655BLPRw/wTOfAc3T6HDp0KN797nfHyJEj44//+I8j4sw6H2fM4rZv377JABb///gf7Dr5PvGJT8Q3vvGN+PSnPx2zZs2Kvn37Jt+gt7W1eW5Oos9//vMxd+7cY75NL9S7XjwfJ1djY2O0tLTEJz/5yRg/fnxEPB3C+Ld/+7eYPHmy5+QUu/POO+Omm26KX/7yl9GvX7+YN29e7Ny5M770pS/FxIkTPR+n2TP93Kh3HxsyZMip2sWz0oEDB+Id73hHbNy4Mb7zne/UvuE9k87HGfNYwpgxY2LPnj1x9OjRWq25uTn69evXIwe2zK677rr4+te/Hp/4xCfila98ZUQ8fX527959zOt2796d/CcMdZ9bbrklfv7zn8fChQtj4cKF8aMf/Sh+9KMfxcKFCz0fp8moUaOib9++tYVtRMTUqVNj+/btnpPTYNWqVTF58uRjFqyzZ8+Obdu2eT56gGc6B/X+/ahRo07ZPp5tWlpa4i//8i9j7dq18Y1vfCOmTJlS+3dn0vk4Yxa3F1xwQTQ2Nh7z4PJ9990X8+bNi4aGM+Ywznif//zn47vf/W586lOfite85jW1elNTUzz00ENx6NChWu2+++6Lpqam07GbZ4VvfvOb8aMf/Sh++MMfxg9/+MN46UtfGi996Uvjhz/8YTQ1NcX9999f+xUt1Wo1li9f7vk4yZqamuLw4cPx2GOP1WobNmyI8ePHe05Og9GjR8emTZuO+bZpw4YNMWHCBM9HD/BMPzeamprivvvuq/27gwcPxurVqz1HJ0lHR0e8613viq1bt8Y3v/nNmDlz5jH//kw6H2fMqrB///7x2te+Nv7hH/4hHnjggfj5z38eX/va1+JNb3rT6d61s8b69evji1/8YrztbW+LxYsXR3Nzc+1/S5YsibFjx8a1114ba9eujRtuuCEeeOCBeN3rXne6d7u0xo8fH5MnT679b+DAgTFw4MCYPHlyvOpVr4r9+/fHRz/60Vi3bl189KMfjYMHD8bll19+une71KZNmxYvfvGL49prr401a9bEr3/967jhhhviT/7kTzwnp8FLX/rS6N27d/zd3/1dPPbYY3HbbbfFl7/85XjjG9/o+egBnunnxlVXXRXLly+PG264IdauXRvXXnttTJgwIS666KLTvOfldNNNN8Xdd98dH/nIR2LIkCG1n+/FoyNn1Pk4nb+H7NlqbW2tvu9976suWLCgeumll1a//vWvn+5dOqt85Stfqc6aNQv/V61Wqxs3bqz+2Z/9WXXu3LnV17zmNdXf/va3p3mPzy7vf//7a7/ntlqtVleuXFl97WtfW503b171da97XfWhhx46jXt39ti/f3/1//v//r/qggULqpdcckn1c5/7XO13qXpOTr21a9dW3/KWt1QXLVpUffnLX179+te/7vk4jTr/XtVq9Zl/btx+++3VV7ziFdX58+dX3/zmN1c3b958qne51Dqfj7e+9a34873z77Y9U85HpVrtia0lJEmSpGfvjHksQZIkSXomLm4lSZJUGi5uJUmSVBoubiVJklQaLm4lSZJUGi5uJUmSVBoubiVJklQaLm4lSZJUGi5uJUmSVBoubiVJklQaLm4lSZJUGv8/72E7uopZLGIAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1000x700 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Test images\n",
    "mri_images_test = test_df['image']\n",
    "\n",
    "fig = plt.figure(figsize=(10, 7)) \n",
    "\n",
    "for i in range(0, len(mri_images_test), 1000):\n",
    "    n = int((i / 1000) + 1)\n",
    "    \n",
    "    #fig.add_subplot(2,2)\n",
    "    plt.imshow(mri_images_test.iloc[i])\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Lets check the count of each label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>Non Demented</th>\n",
       "      <td>634</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Very Mild Demented</th>\n",
       "      <td>459</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Mild Demented</th>\n",
       "      <td>172</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Moderate Demented</th>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                    label\n",
       "Non Demented          634\n",
       "Very Mild Demented    459\n",
       "Mild Demented         172\n",
       "Moderate Demented      15"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "labels = train_data['label']\n",
    "label_counts = pd.DataFrame(labels.value_counts())\n",
    "label_counts.index = ['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented']\n",
    "label_counts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>634</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>459</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>172</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   label\n",
       "2    634\n",
       "3    459\n",
       "0    172\n",
       "1     15"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pd.DataFrame(labels.value_counts())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Y8eUk5OjoKAgderUSV5efJjoCYqdyWVmZurEiRPKzc1VnTp1VL58eaMjoYTiUhUASophw4Zp7ty5CggIyLP8woULeu6557RmzRqDkpkflzsxqezsbM2aNUuff/65HA6HnE6nvL299cQTT+i1117jGkDI5z8vVVEQLpeDWxk6dGihx8fixYvvchqYzXfffaf9+/dL+vWC6R9++KH8/f3zbHPixAmdPn3aiHiWQbEzqTfffFObN2/W/Pnz1apVK+Xm5mrv3r2aPn263nnnHU2YMMHoiChhlixZYnQEmBwHtcMTQUFB+vjjj+V0OuV0OrVnz548x4fbbDb5+/trxowZBqY0Pz6KNakHH3xQ7733Xr6Jdvv27Ro3bpy2bt1qUDKUVF988YV69eolHx8fffHFFwVu269fv2LJBKB0Cg8P16RJkzh86C5gj51J3Tgb9r9VqlRJV69eNSARSrr3339fnTt3lo+Pj95///1bbmez2Sh2uKn//GUcHh5e4LYRERHFlApmFBERoZycHJ09e1YOh0PSr7/XsrKy9OOPP6pXr14GJzQvip1JPfjgg3rrrbf01ltvuf7iuXz5MtcAwi3FxMTc9L8BoLhFR0dr8uTJunjxYr51VatWpdh5gI9iTers2bMaNmyYzp07p6CgIEnS8ePHFRgYqPnz56tWrVoGJ0RJs2vXrkJtZ7PZ1KZNm7ucBkBp1rNnT7Vt21bDhw/X4MGDtWDBAl28eFGvv/66XnjhBYWGhhod0bTYY2dS1atX11dffaXvvvtOx44dk6+vr4KCgtSxY0euAYSb+u8zGm/1N53NZtOPP/5YXLFgIlwyB0UlKSlJH330kerUqaP7779fqamp6t69u7y8vBQZGUmx8wDFzsTKlCmjbt26qVu3bkZHgQn07NlTW7duVf369fX444+re/fuXEgWbpk7d668vLzUtGlTlStXrsA/DoCCBAQEKDMzU9KvZ8seOnRI3bt3V7169XTq1CmD05kbH8WaSNeuXQs1YdpsNm3atKkYEsFssrOzFRsbq02bNikmJkYVK1bUY489pscff9x1WyjgVpYvX65NmzZp3759atu2resPy0qVKhkdDSYTHh6uEydOaNq0aTp+/LgiIyP17rvv6l//+pfrH+4Mxc5ECrqNWEZGhj755BOdPn1arVq10vLly4sxGczI6XRq7969io6O1qZNm5STk6Pu3bure/fuatu2rdHxUIKlp6dr8+bN2rhxo7Zt26ZGjRqpe/fueuyxxzi+F4WSnp6uGTNmqH379urbt69eeuklff311/L399esWbPUtWtXoyOaFsXOAqKjozVjxgxlZGRo3LhxGjhwoNGRYDLXrl3TsmXL9MEHHygjI4Nj7FBoWVlZio2NVXR0tL799ltVqVJF3bt315gxY4yOhhLsq6++UseOHVWxYkXXsvT0dPn6+ua5aDHcR7EzsdOnT2v69OnavHmzQkNDNW7cON17771Gx4JJXLhwQTExMYqJiVFsbKwqVqyorl27qlu3burQoYPR8WAiubm5iouLU3R0tFatWiWHw6F9+/YZHQslWNu2bbVy5UrVq1fP6CiWQ7EzoZycHC1cuFDz589X3bp1NXXqVLVq1croWDCBxMRExcTEKDo6WgkJCWrcuLHrOKmmTZsaHQ8mcvXqVW3ZskUxMTH67rvvJEldunRR165d1alTp3z3AAX+U1hYmBo1aqTRo0dzb/MiRrEzmR07dmjatGk6e/aswsLCNGzYMC5vgkJ57LHHdObMGddB7127dlWNGjWMjgUTOXPmjKKjoxUTE6Ndu3apevXqrr28rVu3lt1uNzoiTGLw4MHau3evvLy8VKlSJfn6+uZZHx0dbVAy86PYmci4ceP09ddfq1atWvrrX/+q6tWr33JbDn7Hf/vPs15vd3Y1x9jhZpo2bSpvb2/XHweNGjW65bbMQShIQScDSlL//v2LKYn1UOxMpLCXo+ACs7iZnTt3Fnrbdu3a3cUkMCvmINwNly5dUoUKFWSz2bgGYhGg2AEAgGLldDr14Ycf6tNPP9WVK1f0r3/9S++99578/f01efJkjrvzAAdnAQCAYjVv3jytW7dOM2fOdJW4/v376/vvv1dkZKTB6cyNYgcAAIrV2rVrNW3aND366KOuj187duyoN998U998843B6cyNYgcAAIrV+fPnVa1atXzLAwIClJGRYUAi66DYAaXQ+++/r6NHjxodA0Ap9eCDD2rhwoV5lqWnp2v27Nlq3769QamsgZMngFJo9OjR+v777xUUFKTevXurV69eCgwMNDoWgFLizJkzCgsLU0pKin755RfVr19fycnJqlmzpubPn6/atWsbHdG0KHZAKZWenq6NGzdq/fr12rZtm5o0aaLevXurZ8+eBV4jEQCKSmxsrI4dO6acnBwFBQWpU6dOXHTfQxQ7ALpy5YoWLlyoRYsWKTs7W61bt9bvf/979enTx+hoAAA3UOyAUmzv3r1av369NmzYoEuXLqlbt27q1auXUlNT9eGHH6pNmzZcegBAkWjSpEmhL0DMBa7vnLfRAQAUv+nTp2vTpk06f/68HnnkEb300kvq1q1bnvs1litXTpMnTzYwJQArWbx4seu/ExIStGjRIr3wwgsKDg5WmTJldPDgQc2dO1fDhg0zMKX5sccOKIVGjBih3r176/HHH1eFChVuus3Jkyd16tQpPfTQQ8WcDoDV9ejRQ1OmTFHHjh3zLN+xY4fCw8MVExNjUDLzY48dUApdvHhR999//y1LnSTVqVNHderUKcZUAEqLc+fOqXLlyvmW+/n56fLlywYksg5OPQFKoXPnzslutxsdA0Ap1aVLF02cOFF79uxRRkaGrl69qu3bt2vixInq2bOn0fFMjY9igVLorbfe0ldffaUnn3xStWrVynNsnST169fPmGAASoX09HS9+uqrWr9+vXJzcyVJdrtd/fr105QpU/LNSSg8ih1QCnXt2vWW62w2m6Kjo4sxDYDSKj09XcePH5ckBQUFqXz58gYnMj+KHQAAKHbnzp3TsmXLdPToUTkcDtWrV09PPfWUfvOb3xgdzdQ4xg4opa5cuaJly5ZpxowZunDhgr799lslJSUZHQtAKbB792799re/1Y4dO1S7dm3Vrl1bu3btUt++fRUXF2d0PFNjjx1QCv3000965plnVKNGDf3000/65ptv9MEHH2j9+vX66KOP1K5dO6MjArCwgQMHqkOHDvr73/+eZ/lbb72l3bt3a8WKFQYlMz/22AGl0PTp0zV48GCtWbNGZcqUkSRFRERoyJAh3GkCwF135MgRDRgwIN/ygQMHctcJD1HsgFIoISHhpme+Dho0SImJicUfCECpUqtWLe3fvz/f8vj4eFWpUsWARNbBBYqBUqhSpUo6fvx4vgsQ79mz56YXDQWAovTcc8/p1Vdf1bFjx9SiRQtJv5a6JUuW6G9/+5vB6cyNYgeUQiNHjtTkyZM1evRoOZ1Obd++XWvXrtVnn32mF1980eh4ACwuNDRUkrR06VItWrRIvr6+CgoK0owZM7hAsYc4eQIoJbKzs13H00lSTEyMFi5c6LrUQFBQkIYPH65evXoZmBIA4AmKHVBKtGvXTj169FCfPn046xVAsZs7d26htw0LC7uLSayNYgeUEuvWrdP69eu1detWVaxYUb169dKTTz6ppk2bGh0NQCnQpEkTeXl5qWnTpipXrpxuVT9sNpsWL15czOmsg2IHlDLp6enatGmT1q9fr++//16BgYHq3bu3nnjiiXwnUwBAUVm+fLk2bdqkffv2qW3bturWrZu6deumSpUqGR3NUih2QCmWnp6ujRs3av369YqNjVXjxo31xBNPaNiwYUZHA2BR6enp2rx5szZu3Kht27apUaNG6t69ux577DHVqlXL6HimR7EDIEnavn273nzzTR06dIgLhAIoFllZWYqNjVV0dLS+/fZbValSRd27d9eYMWOMjmZaFDuglHI6ndq1a5c2bNigTZs2KSMjQ927d1efPn300EMPGR0PQCmRm5uruLg4RUdHa9WqVXI4HNq3b5/RsUyLYgeUIjk5Odq2bZs2btyo6OhoZWRkqHPnznriiSf0yCOPyMfHx+iIAEqBq1evasuWLYqJidF3330nSerSpYu6du2qTp06yd/f3+CE5kWxA0qJl156SZs3b1ZGRoY6dOig3r1767HHHlO5cuWMjgagFDhz5oyio6MVExOjXbt2qXr16uratau6deum1q1by263Gx3REih2QCnxhz/8Qb1791aPHj04Cw1AsWvatKm8vb1dZ8Q2atToltu2bdu2GJNZC8UOAADcdU2aNCnUdjabjRO4PECxAwAAsAgvowMAAACgaFDsAAAALIJiBwAAYBEUOwAAAIug2AEAAFgExQ4AAMAiKHYAAAAWQbEDAACwCIodAACARfw/cSgw/ELtXnYAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "label_plot = label_counts.plot.bar()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There is a very clear class imbalance. With hardly any Moderate Demented and Mild Demented images, it will be well worth it to perform data augmentation to increase the class sizes."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Create scatterplot to view the classification plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#plt.scatter(train_df.image[:,0], train_df.image[:,1], c='label', cmap=plt.cm.RdYlBu)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "****"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Preprocessing**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Reshape images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(960, 128, 128, 3) (960, 4)\n",
      "(320, 128, 128, 3) (320, 4)\n",
      "(5120, 128, 128, 3) (5120, 4)\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.utils import to_categorical\n",
    "\n",
    "# Define reshape function\n",
    "def reshape_X_y(X, y):\n",
    "    X_array = []\n",
    "    for x in X:\n",
    "        X_array.append(x)\n",
    "    \n",
    "    y = to_categorical(y)\n",
    "\n",
    "    # Normalisation\n",
    "    X_array = np.array(X_array)\n",
    "    X_array = X_array / 255.0\n",
    "    \n",
    "    y = np.array(y)\n",
    "    \n",
    "    print(X_array.shape, y.shape)\n",
    "    return X_array, y\n",
    "\n",
    "# Training data\n",
    "X_train_ds, y_train_ds = reshape_X_y(X_train, y_train)\n",
    "\n",
    "# Validation data\n",
    "X_val_ds, y_val_ds = reshape_X_y(X_val, y_val)\n",
    "\n",
    "# Test data\n",
    "X_test_ds, y_test_ds = reshape_X_y(X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Define class names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Non Demented' 'Very Mild Demented' 'Mild Demented' 'Moderate Demented']\n"
     ]
    }
   ],
   "source": [
    "class_names = np.array(['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented'])\n",
    "print(class_names)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "****"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Model building**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Defining Baseline Model Architectures"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "*We are going to create a custom convnet model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 1: *Custom convolutional net*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "model1 = keras.Sequential([\n",
    "    \n",
    "    # First Block\n",
    "    layers.Conv2D(kernel_size=3, filters=32, input_shape=([128, 128, 3]), activation='relu', padding='same'),\n",
    "    layers.MaxPool2D(),\n",
    "    \n",
    "    # Second Block\n",
    "    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),\n",
    "    layers.MaxPool2D(),\n",
    "    \n",
    "    # Classifier Head\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(units=64, activation='relu'),\n",
    "    layers.Dense(units=4, activation='softmax')\n",
    "])\n",
    "\n",
    "model1.compile(optimizer=tf.keras.optimizers.Adam(epsilon=0.01),\n",
    "              loss='categorical_crossentropy',\n",
    "              metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 10ms/step - accuracy: 0.4574 - loss: 1.0372 - val_accuracy: 0.4517 - val_loss: 0.9322\n",
      "Epoch 2/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.4835 - loss: 1.0190 - val_accuracy: 0.4517 - val_loss: 0.9244\n",
      "Epoch 3/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.4801 - loss: 0.9947 - val_accuracy: 0.3267 - val_loss: 0.9510\n",
      "Epoch 4/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.4853 - loss: 1.0114 - val_accuracy: 0.4517 - val_loss: 0.9224\n",
      "Epoch 5/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.4683 - loss: 0.9919 - val_accuracy: 0.4915 - val_loss: 0.8841\n",
      "Epoch 6/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.5165 - loss: 0.9687 - val_accuracy: 0.4545 - val_loss: 0.8778\n",
      "Epoch 7/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.5103 - loss: 0.9462 - val_accuracy: 0.4545 - val_loss: 0.8598\n",
      "Epoch 8/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.5477 - loss: 0.9280 - val_accuracy: 0.4545 - val_loss: 0.9026\n",
      "Epoch 9/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.5399 - loss: 0.9301 - val_accuracy: 0.4830 - val_loss: 0.8247\n",
      "Epoch 10/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.5662 - loss: 0.8773 - val_accuracy: 0.5426 - val_loss: 0.7866\n",
      "Epoch 11/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.5754 - loss: 0.8516 - val_accuracy: 0.5057 - val_loss: 0.8037\n",
      "Epoch 12/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.6069 - loss: 0.8160 - val_accuracy: 0.5284 - val_loss: 0.7804\n",
      "Epoch 13/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 11ms/step - accuracy: 0.6502 - loss: 0.7489 - val_accuracy: 0.5739 - val_loss: 0.7552\n",
      "Epoch 14/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.6764 - loss: 0.7274 - val_accuracy: 0.5682 - val_loss: 0.7132\n",
      "Epoch 15/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.7011 - loss: 0.6488 - val_accuracy: 0.5710 - val_loss: 0.6923\n",
      "Epoch 16/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 11ms/step - accuracy: 0.7206 - loss: 0.6028 - val_accuracy: 0.6023 - val_loss: 0.6778\n",
      "Epoch 17/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.7836 - loss: 0.5077 - val_accuracy: 0.5994 - val_loss: 0.6319\n",
      "Epoch 18/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.8108 - loss: 0.4323 - val_accuracy: 0.6080 - val_loss: 0.6801\n",
      "Epoch 19/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.8072 - loss: 0.4165 - val_accuracy: 0.6165 - val_loss: 0.6777\n",
      "Epoch 20/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.8053 - loss: 0.4073 - val_accuracy: 0.6307 - val_loss: 0.6779\n",
      "Epoch 21/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 12ms/step - accuracy: 0.8860 - loss: 0.2816 - val_accuracy: 0.6449 - val_loss: 0.6407\n",
      "Epoch 22/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9214 - loss: 0.1779 - val_accuracy: 0.6591 - val_loss: 0.6686\n",
      "Epoch 23/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9334 - loss: 0.1467 - val_accuracy: 0.6619 - val_loss: 0.6331\n",
      "Epoch 24/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9559 - loss: 0.1016 - val_accuracy: 0.6705 - val_loss: 0.6628\n",
      "Epoch 25/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.9537 - loss: 0.0909 - val_accuracy: 0.6392 - val_loss: 0.8290\n",
      "Epoch 26/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 11ms/step - accuracy: 0.9564 - loss: 0.0717 - val_accuracy: 0.6761 - val_loss: 0.6896\n",
      "Epoch 27/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.9658 - loss: 0.0394 - val_accuracy: 0.6676 - val_loss: 0.8729\n",
      "Epoch 28/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.9677 - loss: 0.0463 - val_accuracy: 0.6818 - val_loss: 0.8404\n",
      "Epoch 29/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9372 - loss: 0.0903 - val_accuracy: 0.6761 - val_loss: 0.7047\n",
      "Epoch 30/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9567 - loss: 0.0570 - val_accuracy: 0.6392 - val_loss: 1.0437\n",
      "Epoch 31/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9583 - loss: 0.0446 - val_accuracy: 0.6676 - val_loss: 0.7830\n",
      "Epoch 32/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9688 - loss: 0.0136 - val_accuracy: 0.6818 - val_loss: 0.8406\n",
      "Epoch 33/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9688 - loss: 0.0097 - val_accuracy: 0.6705 - val_loss: 0.8639\n",
      "Epoch 34/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.9688 - loss: 0.0076 - val_accuracy: 0.6818 - val_loss: 0.8731\n",
      "Epoch 35/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 12ms/step - accuracy: 0.9688 - loss: 0.0073 - val_accuracy: 0.6875 - val_loss: 0.8832\n",
      "Epoch 36/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 12ms/step - accuracy: 0.9688 - loss: 0.0060 - val_accuracy: 0.6847 - val_loss: 0.9046\n",
      "Epoch 37/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - accuracy: 0.9688 - loss: 0.0050 - val_accuracy: 0.6790 - val_loss: 0.9094\n",
      "Epoch 38/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9688 - loss: 0.0042 - val_accuracy: 0.6818 - val_loss: 0.8933\n",
      "Epoch 39/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - accuracy: 0.9688 - loss: 0.0037 - val_accuracy: 0.6790 - val_loss: 0.9372\n",
      "Epoch 40/40\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 12ms/step - accuracy: 0.9688 - loss: 0.0034 - val_accuracy: 0.6847 - val_loss: 0.9451\n",
      "It takes 7.091901870568593 minutes\n"
     ]
    }
   ],
   "source": [
    "start = time.time()\n",
    "\n",
    "history1 = model1.fit(X_train_ds, y_train_ds,\n",
    "                      steps_per_epoch=len(X_train_ds),\n",
    "                      batch_size=32,\n",
    "                      validation_data=(X_val_ds, y_val_ds),\n",
    "                      validation_steps=len(X_val_ds),\n",
    "                      epochs=40\n",
    ")\n",
    "\n",
    "print('It takes %s minutes' % ((time.time() - start)/60))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                         </span>┃<span style=\"font-weight: bold\"> Output Shape                </span>┃<span style=\"font-weight: bold\">         Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">128</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)        │             <span style=\"color: #00af00; text-decoration-color: #00af00\">896</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)         │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Conv2D</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)          │          <span style=\"color: #00af00; text-decoration-color: #00af00\">18,496</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">MaxPooling2D</span>)       │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">32</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)          │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)                    │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">65536</span>)               │               <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                        │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">64</span>)                  │       <span style=\"color: #00af00; text-decoration-color: #00af00\">4,194,368</span> │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">4</span>)                   │             <span style=\"color: #00af00; text-decoration-color: #00af00\">260</span> │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                        \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape               \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m        Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ conv2d (\u001b[38;5;33mConv2D\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m128\u001b[0m, \u001b[38;5;34m128\u001b[0m, \u001b[38;5;34m32\u001b[0m)        │             \u001b[38;5;34m896\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d (\u001b[38;5;33mMaxPooling2D\u001b[0m)         │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m64\u001b[0m, \u001b[38;5;34m64\u001b[0m, \u001b[38;5;34m32\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ conv2d_1 (\u001b[38;5;33mConv2D\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m64\u001b[0m, \u001b[38;5;34m64\u001b[0m, \u001b[38;5;34m64\u001b[0m)          │          \u001b[38;5;34m18,496\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ max_pooling2d_1 (\u001b[38;5;33mMaxPooling2D\u001b[0m)       │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m32\u001b[0m, \u001b[38;5;34m64\u001b[0m)          │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)                    │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m65536\u001b[0m)               │               \u001b[38;5;34m0\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                        │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m64\u001b[0m)                  │       \u001b[38;5;34m4,194,368\u001b[0m │\n",
       "├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m4\u001b[0m)                   │             \u001b[38;5;34m260\u001b[0m │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">12,642,062</span> (48.23 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m12,642,062\u001b[0m (48.23 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">4,214,020</span> (16.08 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m4,214,020\u001b[0m (16.08 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Optimizer params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">8,428,042</span> (32.15 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Optimizer params: \u001b[0m\u001b[38;5;34m8,428,042\u001b[0m (32.15 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model1.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>accuracy</th>\n",
       "      <th>loss</th>\n",
       "      <th>val_accuracy</th>\n",
       "      <th>val_loss</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.456653</td>\n",
       "      <td>1.034294</td>\n",
       "      <td>0.451705</td>\n",
       "      <td>0.932164</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.482863</td>\n",
       "      <td>1.017778</td>\n",
       "      <td>0.451705</td>\n",
       "      <td>0.924383</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.478831</td>\n",
       "      <td>0.994369</td>\n",
       "      <td>0.326705</td>\n",
       "      <td>0.950955</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.485887</td>\n",
       "      <td>1.010604</td>\n",
       "      <td>0.451705</td>\n",
       "      <td>0.922408</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.468750</td>\n",
       "      <td>0.989697</td>\n",
       "      <td>0.491477</td>\n",
       "      <td>0.884128</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0.516129</td>\n",
       "      <td>0.968327</td>\n",
       "      <td>0.454545</td>\n",
       "      <td>0.877786</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0.511089</td>\n",
       "      <td>0.944436</td>\n",
       "      <td>0.454545</td>\n",
       "      <td>0.859772</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0.547379</td>\n",
       "      <td>0.926975</td>\n",
       "      <td>0.454545</td>\n",
       "      <td>0.902592</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>0.539315</td>\n",
       "      <td>0.928935</td>\n",
       "      <td>0.482955</td>\n",
       "      <td>0.824672</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>0.565524</td>\n",
       "      <td>0.875608</td>\n",
       "      <td>0.542614</td>\n",
       "      <td>0.786615</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   accuracy      loss  val_accuracy  val_loss\n",
       "0  0.456653  1.034294      0.451705  0.932164\n",
       "1  0.482863  1.017778      0.451705  0.924383\n",
       "2  0.478831  0.994369      0.326705  0.950955\n",
       "3  0.485887  1.010604      0.451705  0.922408\n",
       "4  0.468750  0.989697      0.491477  0.884128\n",
       "5  0.516129  0.968327      0.454545  0.877786\n",
       "6  0.511089  0.944436      0.454545  0.859772\n",
       "7  0.547379  0.926975      0.454545  0.902592\n",
       "8  0.539315  0.928935      0.482955  0.824672\n",
       "9  0.565524  0.875608      0.542614  0.786615"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "history1_plot = pd.DataFrame(history1.history)\n",
    "history1_plot.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## **Evaluation of baseline models**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.rcParams['figure.figsize'] = [4, 4]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model 1: *Custom Convnet*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Validation loss**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.0, 1.0)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "history1_plot.loc[:,['loss', 'val_loss']].plot()\n",
    "plt.xlim(0, 40)\n",
    "plt.ylim(0, 1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Validation Accuracy**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(0.0, 1.0)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "history1_plot.loc[:,['accuracy', 'val_accuracy']].plot()\n",
    "plt.xlim(0, 40)\n",
    "plt.ylim(0, 1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 63ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 16ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 18ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 31ms/step\n",
      "                Image Name    Predicted Class\n",
      "0          mild_sample.jpg  Moderate Demented\n",
      "1      moderate_sample.jpg  Moderate Demented\n",
      "2  non_demented_sample.jpg  Moderate Demented\n",
      "3   very_mild_demented.jpg       Non Demented\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing.image import img_to_array, load_img\n",
    "\n",
    "# Set your folder path here\n",
    "folder_path = 'C:/Users/USER/Desktop/DataScience/AlzheimerMRIDiseaseClassification/AlzheimerMRIDiseaseClassificationDataset/predict_data'  # Replace with your folder path\n",
    "\n",
    "# List to store predictions\n",
    "predictions_list = []\n",
    "\n",
    "# Loop over each image file in the folder\n",
    "for image_name in os.listdir(folder_path):\n",
    "    if image_name.endswith('.jpg'):\n",
    "        # Load and preprocess the image\n",
    "        img_path = os.path.join(folder_path, image_name)\n",
    "        img = load_img(img_path, target_size=(128, 128))  # Resize to match model input\n",
    "        img_array = img_to_array(img)\n",
    "        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension\n",
    "        img_array = img_array / 255.0  # Normalize if done during training\n",
    "\n",
    "        # Make a prediction\n",
    "        predictions = model1.predict(img_array)  # Use model1 or your chosen model\n",
    "        predicted_class = np.argmax(predictions, axis=1)  # Get index of highest probability\n",
    "\n",
    "        # Map index to class name\n",
    "        class_names = np.array(['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented'])\n",
    "        predicted_label = class_names[predicted_class][0]\n",
    "\n",
    "        # Store result\n",
    "        predictions_list.append((image_name, predicted_label))\n",
    "\n",
    "# Convert to a DataFrame for easier viewing\n",
    "predictions_df = pd.DataFrame(predictions_list, columns=['Image Name', 'Predicted Class'])\n",
    "\n",
    "# Display the predictions\n",
    "print(predictions_df)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "****"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Experiment with improving the models**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Both had very obvious flaws. The pretrained bases need to be finetuned to fit our specific data. There is an accuracy of around 0.9688 . The custom convnet is underfitting as shown by the large space between the training and validation curves."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Experimentation: *Custom convolutional net*"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Experiment 1: EarlyStopping and Precision"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We are going to experiment with using EarlyStopping. Due to the unbalanced classes, it is also a bad idea to use accuracy, thus we will use precision as our metric. As a general rule, changing more than one parameter at a time is not advised, but the use of accuracy as a metric is unlikely to continue, regardless."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m12s\u001b[0m 10ms/step - loss: 1.3648 - precision: 0.4700 - val_loss: 0.9366 - val_precision: 0.0000e+00\n",
      "Epoch 2/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 11ms/step - loss: 1.0100 - precision: 0.5313 - val_loss: 0.9117 - val_precision: 0.5105\n",
      "Epoch 3/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 1.0000 - precision: 0.5583 - val_loss: 0.8941 - val_precision: 0.7287\n",
      "Epoch 4/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 11ms/step - loss: 0.9589 - precision: 0.6159 - val_loss: 0.8855 - val_precision: 0.5186\n",
      "Epoch 5/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 11ms/step - loss: 0.8855 - precision: 0.6841 - val_loss: 0.8010 - val_precision: 0.7143\n",
      "Epoch 6/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.7535 - precision: 0.7702 - val_loss: 0.7339 - val_precision: 0.7070\n",
      "Epoch 7/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.6098 - precision: 0.8090 - val_loss: 0.6168 - val_precision: 0.7603\n",
      "Epoch 8/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 11ms/step - loss: 0.4762 - precision: 0.8511 - val_loss: 0.5842 - val_precision: 0.7678\n",
      "Epoch 9/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.3587 - precision: 0.9000 - val_loss: 0.6428 - val_precision: 0.7525\n",
      "Epoch 10/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.2259 - precision: 0.9478 - val_loss: 0.5658 - val_precision: 0.7685\n",
      "Epoch 11/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.1438 - precision: 0.9671 - val_loss: 0.5488 - val_precision: 0.7919\n",
      "Epoch 12/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0791 - precision: 0.9927 - val_loss: 0.5384 - val_precision: 0.7781\n",
      "Epoch 13/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0464 - precision: 0.9979 - val_loss: 0.6011 - val_precision: 0.7810\n",
      "Epoch 14/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0338 - precision: 0.9979 - val_loss: 0.5911 - val_precision: 0.7634\n",
      "Epoch 15/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0142 - precision: 1.0000 - val_loss: 0.6557 - val_precision: 0.7753\n",
      "Epoch 16/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0145 - precision: 1.0000 - val_loss: 0.6543 - val_precision: 0.7704\n",
      "Epoch 17/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0078 - precision: 1.0000 - val_loss: 0.6840 - val_precision: 0.7799\n",
      "Epoch 18/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0033 - precision: 1.0000 - val_loss: 0.7147 - val_precision: 0.7736\n",
      "Epoch 19/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0024 - precision: 1.0000 - val_loss: 0.7222 - val_precision: 0.7830\n",
      "Epoch 20/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0020 - precision: 1.0000 - val_loss: 0.7431 - val_precision: 0.7736\n",
      "Epoch 21/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 11ms/step - loss: 0.0017 - precision: 1.0000 - val_loss: 0.7291 - val_precision: 0.7925\n",
      "Epoch 22/100\n",
      "\u001b[1m960/960\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 10ms/step - loss: 0.0015 - precision: 1.0000 - val_loss: 0.7656 - val_precision: 0.7743\n"
     ]
    }
   ],
   "source": [
    "# 1. Build experiment 1\n",
    "\n",
    "model1_1 = keras.Sequential([\n",
    "    \n",
    "    # First Block\n",
    "    layers.Conv2D(kernel_size=3, filters=32, input_shape=([128, 128, 3]), activation='relu', padding='same'),\n",
    "    layers.MaxPool2D(),\n",
    "    \n",
    "    \n",
    "    # Second Block\n",
    "    layers.Conv2D(filters=64, activation='relu', padding='same', kernel_size=3),\n",
    "    layers.MaxPool2D(),\n",
    "    \n",
    "    # Output Layers\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(units=64, activation='relu'),\n",
    "    layers.Dense(units=4, activation='softmax')\n",
    "])\n",
    "\n",
    "# 2. Compile experiment 1\n",
    "\n",
    "model1_1.compile(optimizer=tf.keras.optimizers.Adam(),\n",
    "                loss='categorical_crossentropy',\n",
    "                metrics=['precision'])\n",
    "\n",
    "# - Define Early Stopping\n",
    "\n",
    "earlystop1_1 = keras.callbacks.EarlyStopping(monitor='val_loss',\n",
    "                                           patience=10)\n",
    "\n",
    "# 3. Fit experiment 1\n",
    "\n",
    "history1_1 = model1_1.fit(X_train_ds, y_train_ds,\n",
    "                steps_per_epoch=len(X_train_ds),\n",
    "                validation_data=(X_val_ds, y_val_ds),\n",
    "                validation_steps=len(X_val_ds),\n",
    "                batch_size=32,\n",
    "                epochs=100,\n",
    "                callbacks=[earlystop1_1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot: >"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 400x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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/fr1Y+jQoSxdutThNT///HOGDRtGXFwcY8aMYfv27YDjqaSTJ08yceJEkpKSuOiii3j88cc5ffq0fb2qp5J++OEHRo4cSb9+/Rg+fDhr1661P3fHHXfw3nvvce+999rL88svv7hcD65qvsFgv1xVgkF4IUWB8uKmvbk4UPN1113HDz/8QEVFBQDl5eX88MMPDB48mAkTJnDppZfy1Vdf8e9//5v9+/czf/78OlXFunXrUBSFVatWcfPNNzNp0iR2795tf/7LL7/k3//+N6+88gomk4n777+fxMREvvjiC5588kneffdd/ve//wHqXNDPPPMM48aN44svvqBv375MmDCB8vJyh32+9dZbGI1Gli1bxpIlS8jKyrIP413V77//zsSJE7nxxhv5/PPPGTVqFI888giZmZn2dd5//32uu+46vvrqK3r37s1zzz1XbY7qhubyqSSTycTzzz/P2rVrMRgM3HPPPdxzzz01rrt9+3amTZvGzp076dGjB88//zx9+/atd6HPy2o5EwzS+Cy8jaLAwqGQt/H86zakzhfD31c6vfrll1+O1Wpl48aNJCcn8+uvv2IwGIiNjeWf//wnd999NxqNhs6dOzNkyBC2bt1ap2K1bNmSF154AV9fX7p3787PP//MypUr7UcrN9xwg32inhUrVhAaGsrDDz8MQNeuXTl48CBLliwhJSWFTz75hOuvv55bb70VgMmTJ+Pj48OJEycc9nnw4EECAwPp1KkT/v7+vPnmmzWWbenSpQwdOpS77roLUGeO27p1KwsXLuT1118H4IorruCmm24C4IEHHuDGG2/EaDQSHt547acuB8Ps2bPJzMxk8eLFHDp0iCeffJIOHTowbNgwh/VKSkoYP348I0aM4JVXXmHZsmVMmDCBdevWVZuztcGVHEejWFHQQGBY4+5LCI/UsOP8NwZfX1/+9re/sXbtWpKTk1m7di1Dhw4lPDyclJQUFi1axI4dO9i9ezfZ2dn079+/Tvvp27evw/zMffv2JScnx/5zx44d7Y/37NlDVlYWCQkJ9mUWi8XeJrB3717GjBnj8DucfToM4M477+Sf//wngwYNYtCgQQwdOpQRI0ZUWy8nJ8fh9QASEhJYufJMwHbt2tX+OCgoCACz2Xze37s+XAqGkpISVqxYwYIFC4iJiSEmJoZdu3axdOnSasHwzTff4Ofnx+TJ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",
      "text/plain": [
       "<Figure size 400x400 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "history1_1_plot = pd.DataFrame(history1_1.history)\n",
    "history1_1_plot.loc[: , ['loss', 'val_loss']].plot()\n",
    "history1_1_plot.loc[:, ['precision', 'val_precision']].plot()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 43ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 14ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 26ms/step\n",
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 35ms/step\n",
      "                Image Name    Predicted Class\n",
      "0          mild_sample.jpg  Moderate Demented\n",
      "1      moderate_sample.jpg  Moderate Demented\n",
      "2  non_demented_sample.jpg  Moderate Demented\n",
      "3   very_mild_demented.jpg  Moderate Demented\n"
     ]
    }
   ],
   "source": [
    "from tensorflow.keras.preprocessing.image import img_to_array, load_img\n",
    "\n",
    "# Set your folder path here\n",
    "folder_path = 'C:/Users/USER/Desktop/DataScience/AlzheimerMRIDiseaseClassification/AlzheimerMRIDiseaseClassificationDataset/predict_data'  # Replace with your folder path\n",
    "\n",
    "# List to store predictions\n",
    "predictions_list = []\n",
    "\n",
    "# Loop over each image file in the folder\n",
    "for image_name in os.listdir(folder_path):\n",
    "    if image_name.endswith('.jpg'):\n",
    "        # Load and preprocess the image\n",
    "        img_path = os.path.join(folder_path, image_name)\n",
    "        img = load_img(img_path, target_size=(128, 128))  # Resize to match model input\n",
    "        img_array = img_to_array(img)\n",
    "        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension\n",
    "        img_array = img_array / 255.0  # Normalize if done during training\n",
    "\n",
    "        # Make a prediction\n",
    "        predictions = model1_1.predict(img_array)  # Use model1 or your chosen model\n",
    "        predicted_class = np.argmax(predictions, axis=1)  # Get index of highest probability\n",
    "\n",
    "        # Map index to class name\n",
    "        class_names = np.array(['Non Demented', 'Very Mild Demented', 'Mild Demented', 'Moderate Demented'])\n",
    "        predicted_label = class_names[predicted_class][0]\n",
    "\n",
    "        # Store result\n",
    "        predictions_list.append((image_name, predicted_label))\n",
    "\n",
    "# Convert to a DataFrame for easier viewing\n",
    "predictions_df = pd.DataFrame(predictions_list, columns=['Image Name', 'Predicted Class'])\n",
    "\n",
    "# Display the predictions\n",
    "print(predictions_df)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "****"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **Analyses and Conclusion**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We found that the custom convolutional net outperformed. However, the lack of 'Moderate Demented' images proved to be troublesome. Augmentation was considered, however due to the nature of Alzheimers, altering the images could change the class and was considered too risky to achieve. Regardless, the custom convolutional net achieved fairly satisfactory results in good time, but overfit a bit. Further optimisation would be required before deployment. Future steps would include gathering more data, gaining more domain knowledge to learn ways to make more use of our current data as well as hyperpareameter tuning. A user-friendly interface for deployment could be considered as an extension as well."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "****"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# **References**"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "https://www.analyticsvidhya.com/blog/2021/05/tuning-the-hyperparameters-and-layers-of-neural-network-deep-learning/\n",
    "\n",
    "https://medium.com/geekculture/eda-for-image-classification-dcada9f2567a\n",
    "\n",
    "https://insightsimaging.springeropen.com/articles/10.1007/s13244-018-0639-9\n",
    "\n",
    "https://towardsdatascience.com/loss-functions-and-their-use-in-neural-networks-a470e703f1e9\n",
    "\n",
    "https://insightsimaging.springeropen.com/articles/10.1007/s13244-018-0639-9\n",
    "\n",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8321322/\n",
    "\n",
    "https://medium.com/@thecybermarty/multi-class-activation-functions-df969651d4c5#:~:text=Softmax%20is%20the%20most%20commonly,make%20decisions%20based%20on%20them.\n"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [
    {
     "datasetId": 5152637,
     "sourceId": 8610355,
     "sourceType": "datasetVersion"
    },
    {
     "datasetId": 5334145,
     "sourceId": 8860039,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30732,
   "isGpuEnabled": true,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
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
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
