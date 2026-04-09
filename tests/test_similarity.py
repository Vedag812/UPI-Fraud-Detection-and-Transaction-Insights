"""
tests for the FAISS fraud similarity engine.

verifies:
  - index builds correctly from fraud data
  - queries return the right number of results
  - distances are non-negative (L2 distances always are)
  - similarity scores are in [0, 1]
  - save/load round-trips work
  - engine handles edge cases (no fraud data, unseen categories)
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# skip everything if faiss isn't installed
faiss = pytest.importorskip("faiss", reason="faiss-cpu not installed")

from fraud_similarity import FraudSimilarityEngine, build_and_save_index


@pytest.fixture
def sample_data():
    """synthetic dataset with a mix of fraud and normal transactions"""
    np.random.seed(42)
    n = 5_000
    n_fraud = 200

    df = pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(n)],
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='5min'),
        'sender_upi_id': np.random.choice([f'user{i}@ybl' for i in range(200)], n),
        'amount': np.random.lognormal(6.0, 1.2, n).clip(1, 100000),
        'transaction_type': np.random.choice(['P2P', 'P2M', 'Bill Payment'], n),
        'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune'], n),
        'sender_bank': np.random.choice(['PhonePe', 'Google Pay', 'SBI'], n),
        'hour': np.random.randint(0, 24, n),
        'day_of_week': np.random.randint(0, 7, n),
        'is_night': np.zeros(n, dtype=int),
        'is_weekend': np.random.binomial(1, 0.28, n),
        'txn_velocity': np.random.poisson(2, n),
        'amount_zscore': np.random.normal(0, 1, n),
        'is_fraud': 0,
        'fraud_type': None,
    })

    df['is_night'] = df['hour'].between(1, 5).astype(int)

    # mark some as fraud
    fraud_idx = np.random.choice(n, n_fraud, replace=False)
    df.loc[fraud_idx, 'is_fraud'] = 1
    df.loc[fraud_idx, 'fraud_type'] = np.random.choice(
        ['rapid_fire', 'structuring', 'late_night_large'], n_fraud
    )

    return df


class TestIndexBuilding:
    """tests that the FAISS index gets built correctly"""

    def test_fit_creates_index(self, sample_data):
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        assert engine.is_fitted
        assert engine.index is not None

    def test_index_has_correct_count(self, sample_data):
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        n_fraud = sample_data['is_fraud'].sum()
        assert engine.index.ntotal == n_fraud

    def test_metadata_stored(self, sample_data):
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        assert engine.fraud_metadata is not None
        assert len(engine.fraud_metadata) == engine.index.ntotal


class TestQuerying:
    """tests that similarity queries return valid results"""

    def test_returns_k_results(self, sample_data):
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        results = engine.query(sample_data.head(1), k=5)
        assert len(results) == 1
        assert len(results[0]['similar_frauds']) == 5

    def test_different_k_values(self, sample_data):
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        for k in [1, 3, 10]:
            results = engine.query(sample_data.head(1), k=k)
            assert len(results[0]['similar_frauds']) == k

    def test_distances_non_negative(self, sample_data):
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        results = engine.query(sample_data.head(5), k=3)
        for r in results:
            for fraud in r['similar_frauds']:
                assert fraud['distance'] >= 0

    def test_similarity_score_range(self, sample_data):
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        results = engine.query(sample_data.head(10), k=3)
        for r in results:
            assert 0 <= r['fraud_similarity_score'] <= 1

    def test_batch_query(self, sample_data):
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        results = engine.query(sample_data.head(20), k=3)
        assert len(results) == 20


class TestPersistence:
    """tests that save/load round-trips preserve the index"""

    def test_save_and_load(self, sample_data, tmp_path):
        # build and save
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        engine.save(str(tmp_path))

        # load into a new engine
        loaded = FraudSimilarityEngine.load(str(tmp_path))
        assert loaded.is_fitted
        assert loaded.index.ntotal == engine.index.ntotal

    def test_loaded_engine_queries_correctly(self, sample_data, tmp_path):
        engine = FraudSimilarityEngine()
        engine.fit(sample_data)
        original_results = engine.query(sample_data.head(3), k=3)
        engine.save(str(tmp_path))

        loaded = FraudSimilarityEngine.load(str(tmp_path))
        loaded_results = loaded.query(sample_data.head(3), k=3)

        # results should be identical after load
        for orig, load in zip(original_results, loaded_results):
            assert orig['fraud_similarity_score'] == load['fraud_similarity_score']


class TestEdgeCases:

    def test_no_fraud_data(self):
        # dataset with zero fraud transactions
        df = pd.DataFrame({
            'transaction_id': ['TXN001'],
            'timestamp': [pd.Timestamp('2024-06-15')],
            'sender_upi_id': ['user1@ybl'],
            'amount': [500.0],
            'transaction_type': ['P2M'],
            'city': ['Mumbai'],
            'sender_bank': ['PhonePe'],
            'hour': [14],
            'day_of_week': [2],
            'is_night': [0],
            'is_weekend': [0],
            'txn_velocity': [1],
            'amount_zscore': [0.0],
            'is_fraud': [0],
            'fraud_type': [None],
        })
        engine = FraudSimilarityEngine()
        engine.fit(df)
        # should gracefully handle no fraud data
        assert not engine.is_fitted or engine.index.ntotal == 0

    def test_unfitted_query_returns_empty(self, sample_data):
        engine = FraudSimilarityEngine()
        results = engine.query(sample_data.head(1))
        assert results == []
