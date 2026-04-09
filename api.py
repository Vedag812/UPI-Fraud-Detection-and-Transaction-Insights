"""
FastAPI gateway for the UPI fraud detection platform.

serves the entire system through a clean REST API:
  - real-time fraud scoring (sync + async via celery)
  - FAISS similarity search
  - causal inference results (DiD analysis)

designed to be production-ready:
  - proper error handling and HTTP status codes
  - pydantic models for request/response validation
  - CORS enabled for frontend integration
  - health check endpoint for monitoring
  - structured JSON responses throughout

start with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# make sure src modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# -- pydantic models --
# these define exactly what the API accepts and returns.
# FastAPI auto-generates OpenAPI docs from these.

class TransactionInput(BaseModel):
    """what a single transaction looks like coming in"""
    transaction_id: str = Field(..., example="UPI20240615000001")
    timestamp: str = Field(..., example="2024-06-15 14:30:00")
    sender_upi_id: str = Field(..., example="rahul123@ybl")
    receiver_upi_id: str = Field(..., example="quickmart42@paytm")
    sender_bank: str = Field(default="PhonePe", example="PhonePe")
    receiver_bank: str = Field(default="SBI", example="SBI")
    amount: float = Field(..., gt=0, example=2499.00)
    transaction_type: str = Field(default="P2M", example="P2M")
    city: str = Field(default="Mumbai", example="Mumbai")
    device_os: str = Field(default="Android", example="Android")
    # these are optional - if not provided, we compute them server-side
    txn_velocity: int = Field(default=0, ge=0)
    amount_zscore: float = Field(default=0.0)
    is_night: int = Field(default=0, ge=0, le=1)
    is_weekend: int = Field(default=0, ge=0, le=1)
    hour: Optional[int] = Field(default=None, ge=0, le=23)

class BatchInput(BaseModel):
    """for scoring multiple transactions at once"""
    transactions: List[TransactionInput]
    async_mode: bool = Field(default=False, description="use celery async processing if available")

class FraudScoreResponse(BaseModel):
    """what you get back after scoring a transaction"""
    scores: dict
    is_fraud_predicted: bool
    rules_triggered: str = ""
    amount_zscore: float = 0.0
    similar_fraud_patterns: Optional[list] = None

class CausalEstimateInput(BaseModel):
    """filters for a custom causal estimate on a subset of data"""
    city: Optional[str] = Field(default=None, example="Mumbai")
    transaction_type: Optional[str] = Field(default=None, example="P2M")
    start_date: Optional[str] = Field(default=None, example="2024-01-01")
    end_date: Optional[str] = Field(default=None, example="2024-12-31")


# -- app setup --

app = FastAPI(
    title="UPI Fraud Detection Platform",
    description=(
        "Real-time fraud scoring with ensemble ML models, FAISS vector similarity search, "
        "and causal inference (Difference-in-Differences) measuring the impact of 2FA on fraud rates."
    ),
    version="2.0.0",
    docs_url="/docs",  # swagger UI at /docs
)

# allow requests from any origin (for dashboards, frontends, etc)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# base directory for file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# -- helper: load cached causal results --
def _load_causal_results():
    """
    loads pre-computed causal results from JSON.
    the causal pipeline runs during data processing (step 7 of run_pipeline.py),
    and results are cached as JSON. the API serves these cached results rather
    than re-running the regression on every request.
    """
    results_path = os.path.join(BASE_DIR, 'data', 'processed', 'causal_results.json')
    if not os.path.exists(results_path):
        return None
    with open(results_path, 'r') as f:
        return json.load(f)


# -- endpoints --

@app.get("/health")
def health_check():
    """
    health check endpoint - standard practice for containerized services.
    load balancers (ELB, nginx) hit this to know if the service is alive.
    returns system status + which components are available.
    """
    # check what's available
    faiss_available = False
    faiss_index_exists = False
    celery_available = False
    causal_results_exist = False

    try:
        import faiss
        faiss_available = True
        faiss_index_exists = os.path.exists(
            os.path.join(BASE_DIR, 'data', 'faiss_index', 'faiss_fraud_index.bin')
        )
    except ImportError:
        pass

    try:
        from tasks import CELERY_AVAILABLE
        celery_available = CELERY_AVAILABLE
    except ImportError:
        pass

    causal_results_exist = os.path.exists(
        os.path.join(BASE_DIR, 'data', 'processed', 'causal_results.json')
    )

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "fraud_scoring": True,
            "faiss_similarity": faiss_available and faiss_index_exists,
            "causal_inference": causal_results_exist,
            "celery_async": celery_available,
        }
    }


@app.post("/fraud/score", response_model=FraudScoreResponse)
def score_transaction(txn: TransactionInput):
    """
    score a single transaction for fraud.

    runs the full ensemble pipeline:
      - Z-Score anomaly detection
      - Rule-based pattern matching
      - FAISS similarity search (if available)
      - Ensemble vote (flags if 2+ methods agree)

    returns all individual method scores plus the ensemble verdict.
    """
    try:
        from tasks import score_transaction_sync

        # convert pydantic model to dict for the scoring pipeline
        txn_dict = txn.model_dump()

        result = score_transaction_sync(txn_dict)
        return FraudScoreResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"scoring failed: {str(e)}")


@app.post("/fraud/batch")
def score_batch(batch: BatchInput):
    """
    score multiple transactions at once.

    if async_mode=true and celery is available, tasks are queued and you get
    back task IDs to poll for results. otherwise, everything runs synchronously.
    """
    try:
        txn_list = [t.model_dump() for t in batch.transactions]

        if batch.async_mode:
            # try async via celery
            try:
                from tasks import score_batch_async, CELERY_AVAILABLE
                if CELERY_AVAILABLE:
                    task = score_batch_async.delay(txn_list)
                    return {
                        "mode": "async",
                        "task_id": task.id,
                        "status": "queued",
                        "message": f"{len(txn_list)} transactions queued for processing",
                        "poll_url": f"/task/{task.id}/status"
                    }
            except Exception:
                pass  # fall through to sync mode

        # synchronous fallback
        from tasks import score_batch_sync
        results = score_batch_sync(txn_list)
        return {
            "mode": "sync",
            "results": results,
            "total": len(results)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"batch scoring failed: {str(e)}")


@app.get("/fraud/similar/{txn_id}")
def find_similar_fraud(txn_id: str, k: int = Query(default=5, ge=1, le=20)):
    """
    given a transaction ID, find the K most similar historical fraud patterns.

    uses FAISS vector similarity search over confirmed fraud embeddings.
    useful for fraud investigators who want to see "what does this look like?"
    """
    try:
        import pandas as pd
        from fraud_similarity import FraudSimilarityEngine

        # load the FAISS index
        index_dir = os.path.join(BASE_DIR, 'data', 'faiss_index')
        engine = FraudSimilarityEngine.load(index_dir)

        if not engine.is_fitted:
            raise HTTPException(status_code=503, detail="FAISS index not built yet. run the pipeline first.")

        # find the transaction in our processed data
        processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'upi_transactions_flagged.csv')
        if not os.path.exists(processed_path):
            raise HTTPException(status_code=404, detail="processed data not found")

        # read just enough to find the transaction (chunked for memory)
        for chunk in pd.read_csv(processed_path, chunksize=100_000, parse_dates=['timestamp']):
            match = chunk[chunk['transaction_id'] == txn_id]
            if len(match) > 0:
                results = engine.query(match, k=k)
                return {
                    "transaction_id": txn_id,
                    "query_amount": float(match.iloc[0]['amount']),
                    "query_is_fraud": bool(match.iloc[0].get('is_fraud', 0)),
                    "results": results[0] if results else {},
                }

        raise HTTPException(status_code=404, detail=f"transaction {txn_id} not found")

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=503, detail="faiss-cpu not installed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"similarity search failed: {str(e)}")


@app.get("/causal/summary")
def causal_summary():
    """
    returns the full Difference-in-Differences analysis results.

    this is the flagship endpoint - answers "what is the causal effect of
    2FA on UPI fraud rates?" with proper statistical rigor:
      - ATE (Average Treatment Effect) with confidence intervals
      - p-value and significance test
      - parallel trends validation result
      - placebo test result
      - sensitivity analysis across model specifications
    """
    results = _load_causal_results()
    if not results:
        raise HTTPException(
            status_code=404,
            detail="causal analysis not yet run. execute run_pipeline.py first."
        )
    return results


@app.post("/causal/estimate")
def causal_estimate(filters: CausalEstimateInput):
    """
    re-estimates the causal effect on a filtered subset of data.

    use this to answer questions like:
      - "did 2FA reduce fraud more in Mumbai than Bangalore?"
      - "was 2FA more effective for P2M transactions?"
      - "what if we only look at Q3-Q4?"

    runs the DiD regression from scratch on the filtered subset.
    """
    try:
        import pandas as pd
        from causal_inference import augment_data_for_causal, run_did_analysis

        # load processed data
        processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'upi_transactions_processed.csv')
        if not os.path.exists(processed_path):
            raise HTTPException(status_code=404, detail="processed data not found")

        df = pd.read_csv(processed_path, parse_dates=['timestamp'])

        # apply filters
        if filters.city:
            df = df[df['city'] == filters.city]
        if filters.transaction_type:
            df = df[df['transaction_type'] == filters.transaction_type]
        if filters.start_date:
            df = df[df['timestamp'] >= pd.to_datetime(filters.start_date)]
        if filters.end_date:
            df = df[df['timestamp'] <= pd.to_datetime(filters.end_date)]

        if len(df) < 1000:
            raise HTTPException(
                status_code=400,
                detail=f"only {len(df)} rows after filtering — need at least 1,000 for meaningful analysis"
            )

        # sample if too large
        if len(df) > 100_000:
            df = df.sample(n=100_000, random_state=42)

        # run the analysis
        df_augmented = augment_data_for_causal(df)
        model, ate, ci, p_value = run_did_analysis(df_augmented)

        return {
            "filters_applied": filters.model_dump(exclude_none=True),
            "n_observations": len(df_augmented),
            "ate": round(float(ate), 6),
            "confidence_interval": [round(float(ci[0]), 6), round(float(ci[1]), 6)],
            "p_value": round(float(p_value), 8),
            "is_significant": bool(p_value < 0.05),
            "interpretation": (
                f"2FA {'reduces' if ate < 0 else 'increases'} fraud probability "
                f"by {abs(ate)*100:.2f} percentage points in this subset"
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"causal estimate failed: {str(e)}")


@app.get("/causal/parallel-trends")
def parallel_trends():
    """
    returns the monthly fraud rate time series for treatment vs control groups
    during the pre-treatment period (Jan-Jun 2024).

    this is the data behind the parallel trends plot. clients can use this to
    build their own visualizations or verify the assumption independently.
    """
    try:
        import pandas as pd
        from causal_inference import augment_data_for_causal, TREATMENT_DATE

        processed_path = os.path.join(BASE_DIR, 'data', 'processed', 'upi_transactions_processed.csv')
        if not os.path.exists(processed_path):
            raise HTTPException(status_code=404, detail="processed data not found")

        # load and sample
        df = pd.read_csv(processed_path, parse_dates=['timestamp'])
        if len(df) > 100_000:
            df = df.sample(n=100_000, random_state=42)

        df_augmented = augment_data_for_causal(df)

        # get pre-treatment monthly fraud rates by group
        pre_df = df_augmented[df_augmented['timestamp'] < TREATMENT_DATE]
        monthly = pre_df.groupby(
            [pre_df['timestamp'].dt.month, 'has_2fa']
        )['is_fraud'].agg(['mean', 'count']).reset_index()
        monthly.columns = ['month', 'has_2fa', 'fraud_rate', 'n_transactions']
        monthly['group'] = monthly['has_2fa'].map({0: 'Control (No 2FA)', 1: 'Treatment (2FA)'})
        monthly['month_name'] = monthly['month'].map({
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun'
        })

        return {
            "period": "Pre-treatment (Jan-Jun 2024)",
            "treatment_date": str(TREATMENT_DATE.date()),
            "data": monthly[['month', 'month_name', 'group', 'fraud_rate', 'n_transactions']].to_dict('records'),
            "interpretation": (
                "If the lines are roughly parallel, the key DiD assumption holds. "
                "Diverging lines would invalidate the causal estimate."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"parallel trends query failed: {str(e)}")


@app.get("/task/{task_id}/status")
def task_status(task_id: str):
    """
    check the status of an async celery task.
    only works if celery is running with a redis backend.
    """
    try:
        from tasks import CELERY_AVAILABLE
        if not CELERY_AVAILABLE:
            raise HTTPException(status_code=503, detail="celery not available")

        from celery.result import AsyncResult
        from tasks import app as celery_app

        result = AsyncResult(task_id, app=celery_app)
        response = {
            "task_id": task_id,
            "status": result.status,
        }

        if result.ready():
            response["result"] = result.result
        elif result.failed():
            response["error"] = str(result.result)

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"status check failed: {str(e)}")


# -- run directly for development --
if __name__ == '__main__':
    import uvicorn
    print("starting UPI Fraud Detection API...")
    print("docs available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
