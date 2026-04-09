"""
celery async task definitions for real-time fraud scoring.

in a real production setup, transactions come in at thousands per second.
you can't block the API while running ensemble + FAISS + causal models.
celery lets us queue these scoring jobs and process them asynchronously
with dedicated workers, while the API immediately returns a task ID.

architecture:
  1. FastAPI receives a transaction → creates a Celery task
  2. task goes into Redis queue
  3. Celery worker picks it up, runs the scoring pipeline
  4. result stored in Redis backend
  5. client polls /task/{id}/status to get the result

graceful degradation: if Redis isn't running, everything falls back to
synchronous execution. the project works standalone without any external
services - the celery integration is opt-in.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# make sure src is on the path
sys.path.insert(0, os.path.dirname(__file__))

# try to import celery - if not available, we'll stub it out
try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

# redis connection config - defaults to local redis
# override with environment variables in production
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# -- celery app setup --
if CELERY_AVAILABLE:
    app = Celery(
        'upi_fraud_tasks',
        broker=REDIS_URL,
        backend=REDIS_URL,
    )

    # reasonable defaults for a fraud scoring workload
    app.conf.update(
        task_serializer='json',
        result_serializer='json',
        accept_content=['json'],
        timezone='Asia/Kolkata',  # IST since this is a UPI (India) project
        enable_utc=True,
        # tasks shouldn't take more than 30 seconds even on slow hardware
        task_soft_time_limit=25,
        task_time_limit=30,
        # retry failed tasks once with a 5-second delay
        task_acks_late=True,
        task_reject_on_worker_lost=True,
    )
else:
    app = None


def _get_fraud_detector():
    """lazy-loads the fraud detection pipeline to avoid importing everything at module level"""
    from fraud_detector import zscore_detection, isolation_forest_detection, rule_based_detection, ensemble_detection
    return {
        'zscore': zscore_detection,
        'isolation': isolation_forest_detection,
        'rules': rule_based_detection,
        'ensemble': ensemble_detection,
    }


def _get_similarity_engine():
    """lazy-loads the FAISS engine - only if the index exists on disk"""
    try:
        from fraud_similarity import FraudSimilarityEngine
        base_dir = os.path.dirname(os.path.dirname(__file__))
        index_dir = os.path.join(base_dir, 'data', 'faiss_index')
        if os.path.exists(os.path.join(index_dir, 'faiss_fraud_index.bin')):
            return FraudSimilarityEngine.load(index_dir)
    except Exception as e:
        print(f"could not load FAISS engine: {e}")
    return None


def score_transaction_sync(txn_data):
    """
    synchronous fraud scoring - used when celery/redis aren't available.

    runs the full ensemble pipeline on a single transaction:
      1. z-score: is this amount unusual for this user?
      2. isolation forest: does it look like an anomaly in feature space?
      3. rule-based: does it match known UPI fraud patterns?
      4. FAISS: is it similar to historical confirmed fraud?
      5. ensemble: do 2+ methods agree it's suspicious?

    returns a dict with all scores and flags.
    """
    import pandas as pd
    import numpy as np

    # convert single transaction to a dataframe (our pipeline expects dataframes)
    if isinstance(txn_data, dict):
        df = pd.DataFrame([txn_data])
    else:
        df = txn_data.copy()

    # ensure required columns exist with sane defaults
    # (in a real system these would come from the transaction stream)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = df['hour'].between(1, 5).astype(int)

    for col in ['txn_velocity', 'amount_zscore', 'time_since_last']:
        if col not in df.columns:
            df[col] = 0

    if 'amount_log' not in df.columns and 'amount' in df.columns:
        df['amount_log'] = np.log1p(df['amount'])

    # run each detection method
    pipeline = _get_fraud_detector()

    df = pipeline['zscore'](df)
    # skip isolation forest for single transactions (needs more data to be meaningful)
    df['isolation_flag'] = 0
    df['isolation_score'] = 0.0
    df = pipeline['rules'](df)
    df = pipeline['ensemble'](df)

    # FAISS similarity check
    engine = _get_similarity_engine()
    similarity_result = None
    if engine and engine.is_fitted:
        try:
            results = engine.query(df, k=3)
            if results:
                similarity_result = results[0]
                df['faiss_flag'] = 1 if similarity_result['is_suspicious'] else 0
                df['fraud_similarity_score'] = similarity_result['fraud_similarity_score']
        except Exception:
            pass

    if 'faiss_flag' not in df.columns:
        df['faiss_flag'] = 0
        df['fraud_similarity_score'] = 0.0

    # build the response
    row = df.iloc[0]
    result = {
        'scores': {
            'zscore_flag': int(row.get('zscore_flag', 0)),
            'isolation_flag': int(row.get('isolation_flag', 0)),
            'rule_flag': int(row.get('rule_flag', 0)),
            'faiss_flag': int(row.get('faiss_flag', 0)),
            'ensemble_flag': int(row.get('ensemble_flag', 0)),
            'fraud_similarity_score': float(row.get('fraud_similarity_score', 0.0)),
        },
        'is_fraud_predicted': bool(row.get('ensemble_flag', 0) == 1),
        'rules_triggered': str(row.get('rules_triggered', '')).strip(','),
        'amount_zscore': round(float(row.get('amount_zscore', 0)), 4),
    }

    if similarity_result:
        result['similar_fraud_patterns'] = similarity_result['similar_frauds'][:3]

    return result


def score_batch_sync(txn_list):
    """
    scores multiple transactions at once - useful for batch processing.
    each transaction gets its own independent score.
    """
    results = []
    for txn in txn_list:
        try:
            result = score_transaction_sync(txn)
            results.append({'status': 'success', 'result': result})
        except Exception as e:
            results.append({'status': 'error', 'error': str(e)})
    return results


# -- celery task wrappers --
# these are the async versions that go through the redis queue.
# they just call the sync functions but with celery's retry and monitoring

if CELERY_AVAILABLE and app:

    @app.task(bind=True, max_retries=2, default_retry_delay=5)
    def score_transaction_async(self, txn_data):
        """
        async version of score_transaction.
        celery handles queuing, retries, and result storage automatically.
        """
        try:
            return score_transaction_sync(txn_data)
        except Exception as exc:
            # retry once on failure, then give up
            raise self.retry(exc=exc)

    @app.task(bind=True, max_retries=1)
    def score_batch_async(self, txn_list):
        """async batch scoring - each item scored independently"""
        try:
            return score_batch_sync(txn_list)
        except Exception as exc:
            raise self.retry(exc=exc)

    @app.task
    def refresh_faiss_index():
        """
        rebuilds the FAISS index from the latest processed data.
        useful as a periodic task (e.g., run nightly via celery beat)
        to keep the similarity search up to date with new fraud patterns.
        """
        import pandas as pd
        from fraud_similarity import build_and_save_index

        base_dir = os.path.dirname(os.path.dirname(__file__))
        flagged_path = os.path.join(base_dir, 'data', 'processed', 'upi_transactions_flagged.csv')
        index_dir = os.path.join(base_dir, 'data', 'faiss_index')

        if os.path.exists(flagged_path):
            df = pd.read_csv(flagged_path, parse_dates=['timestamp'])
            build_and_save_index(df, index_dir)
            return {'status': 'success', 'indexed_frauds': int(df['is_fraud'].sum())}
        return {'status': 'error', 'message': 'flagged data not found'}
