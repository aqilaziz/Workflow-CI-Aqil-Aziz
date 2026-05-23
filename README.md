# Workflow CI - Aqil Aziz

Repository ini berisi MLflow Project dan GitHub Actions untuk menjalankan retraining model secara otomatis saat perubahan kode atau data dipush.

`MLProject/modelling.py` memakai `mlflow.sklearn.autolog()` saja untuk baseline training. Logging manual dan tuning dipisahkan di submission lokal pada file `modelling_tuning.py`.

## Cara menjalankan lokal

```bash
cd MLProject
pip install -r requirements.txt
mlflow run . --env-manager=local
```

Workflow `.workflow/train-model.yml` disimpan sebagai salinan struktur kriteria. Workflow GitHub Actions yang aktif berada di `.github/workflows/train-model.yml` dan menyimpan artefak MLflow sebagai GitHub Actions artifact.
