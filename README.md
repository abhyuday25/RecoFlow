# Netflix-Style Recommendation System

Production-ready Python recommendation service with:

- User-based and item-based collaborative filtering using cosine similarity
- Matrix factorization with truncated SVD
- Neural collaborative filtering with PyTorch embeddings plus MLP layers
- Evaluation using RMSE, Precision@K, and Recall@K
- FastAPI endpoint at `/recommend/{user_id}`
- Docker packaging for deployment

## Project Structure

```text
.
├── app
│   ├── api
│   ├── data
│   ├── models
│   └── utils
├── artifacts
├── data
│   └── raw
├── Dockerfile
├── requirements.txt
└── train.py
```

## Run Locally

```bash
pip install -r requirements.txt
python train.py
uvicorn app.api.main:app --reload
```

## API

```bash
curl "http://localhost:8000/recommend/1?model=hybrid"
curl "http://localhost:8000/recommend/1?model=svd"
curl "http://localhost:8000/recommend/1?model=ncf"
```