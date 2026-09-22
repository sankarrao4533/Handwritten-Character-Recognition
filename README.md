# EMNIST Handwritten Character Recognition

This project trains a CNN on EMNIST Balanced and provides a Streamlit interface for uploaded or drawn characters.

## Structure

```text
data/emnist-balanced.mat
model/emnist_balanced_cnn.keras
notebooks/EMNIST_CNN_Training.ipynb
src/data_loader.py
src/preprocessing.py
src/predict.py
app.py
requirements.txt
README.md
```

The notebook uses exactly 30,000 rows for training and 10,000 rows for testing. The same upright orientation is used during training and app inference.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Train

Run all sections in `notebooks/EMNIST_CNN_Training.ipynb`. The model is saved to `model/emnist_balanced_cnn.keras`.

## Run the app

```powershell
streamlit run app.py
```

The app accepts PNG, BMP, and WebP uploads and includes a drawing canvas.

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Select the repository, branch `main`, and main file `app.py`.
4. Click **Deploy**.

The committed model and label mapping are sufficient to run the app. The large
`data/emnist-balanced.mat` file is used only for training and is excluded from
the deployment repository.
