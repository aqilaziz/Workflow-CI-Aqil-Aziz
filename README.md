# Workflow CI - Aqil Aziz

Repository ini berisi MLflow Project dan GitHub Actions untuk menjalankan retraining model secara otomatis saat perubahan kode atau data dipush.

## Cara menjalankan lokal

```bash
cd MLProject
pip install -r requirements.txt
mlflow run . --env-manager=local
```

Workflow `.workflow/train-model.yml` menyimpan artefak MLflow sebagai GitHub Actions artifact.
