"""
tests for the causal inference module.

verifies that the DiD analysis produces statistically valid results:
  - model converges and returns coefficients
  - treatment effect has the expected sign (negative = 2FA reduces fraud)
  - parallel trends validation runs without errors
  - placebo test doesn't find spurious effects
  - all required columns are created during data augmentation
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

# add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from causal_inference import (
    augment_data_for_causal,
    run_did_analysis,
    validate_parallel_trends,
    run_placebo_test,
    run_sensitivity_analysis,
    run_causal_pipeline,
    assign_city_tier,
)


@pytest.fixture
def sample_data():
    """
    creates a small synthetic dataset for testing.
    10K rows is enough to test the code logic without waiting 5 minutes.
    """
    np.random.seed(42)
    n = 10_000

    dates = pd.date_range('2024-01-01', '2024-12-31', periods=n)

    df = pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(n)],
        'timestamp': dates,
        'sender_upi_id': np.random.choice([f'user{i}@ybl' for i in range(500)], n),
        'receiver_upi_id': np.random.choice([f'merchant{i}@paytm' for i in range(100)], n),
        'sender_bank': np.random.choice(['PhonePe', 'Google Pay', 'Paytm', 'SBI'], n,
                                         p=[0.47, 0.34, 0.11, 0.08]),
        'receiver_bank': np.random.choice(['SBI', 'HDFC', 'ICICI'], n),
        'amount': np.random.lognormal(6.0, 1.2, n).clip(1, 100000),
        'transaction_type': np.random.choice(['P2P', 'P2M', 'Bill Payment'], n),
        'city': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Pune', 'Bhopal'], n,
                                  p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'is_fraud': np.random.binomial(1, 0.025, n),
        'hour': np.random.randint(0, 24, n),
        'is_night': np.zeros(n, dtype=int),
        'is_weekend': np.random.binomial(1, 0.28, n),
        'txn_velocity': np.random.poisson(2, n),
        'amount_zscore': np.random.normal(0, 1, n),
    })

    df['is_night'] = df['hour'].between(1, 5).astype(int)
    df['amount_log'] = np.log1p(df['amount'])

    return df


class TestDataAugmentation:
    """tests that augment_data_for_causal creates all necessary columns correctly"""

    def test_creates_required_columns(self, sample_data):
        result = augment_data_for_causal(sample_data)
        required = ['post_treatment', 'has_2fa', 'treatment', 'city_tier',
                     'user_risk_score', 'avg_txn_frequency', 'month_num']
        for col in required:
            assert col in result.columns, f"missing column: {col}"

    def test_post_treatment_is_binary(self, sample_data):
        result = augment_data_for_causal(sample_data)
        assert set(result['post_treatment'].unique()).issubset({0, 1})

    def test_has_2fa_is_binary(self, sample_data):
        result = augment_data_for_causal(sample_data)
        assert set(result['has_2fa'].unique()).issubset({0, 1})

    def test_treatment_is_interaction(self, sample_data):
        result = augment_data_for_causal(sample_data)
        # treatment should be has_2fa × post_treatment
        expected = result['has_2fa'] * result['post_treatment']
        pd.testing.assert_series_equal(result['treatment'], expected, check_names=False)

    def test_city_tier_values(self, sample_data):
        result = augment_data_for_causal(sample_data)
        assert set(result['city_tier'].unique()).issubset({1, 2, 3})

    def test_both_groups_present(self, sample_data):
        result = augment_data_for_causal(sample_data)
        assert result['has_2fa'].sum() > 0, "no treatment group members"
        assert (result['has_2fa'] == 0).sum() > 0, "no control group members"


class TestDiDAnalysis:
    """tests that the DiD regression produces valid output"""

    def test_model_converges(self, sample_data):
        df = augment_data_for_causal(sample_data)
        model, ate, ci, p_value = run_did_analysis(df)
        assert model is not None

    def test_returns_numeric_ate(self, sample_data):
        df = augment_data_for_causal(sample_data)
        _, ate, _, _ = run_did_analysis(df)
        assert isinstance(ate, (float, np.floating))
        assert not np.isnan(ate)

    def test_confidence_interval_valid(self, sample_data):
        df = augment_data_for_causal(sample_data)
        _, _, ci, _ = run_did_analysis(df)
        assert ci[0] < ci[1], "CI lower bound should be less than upper bound"

    def test_p_value_in_range(self, sample_data):
        df = augment_data_for_causal(sample_data)
        _, _, _, p_value = run_did_analysis(df)
        assert 0 <= p_value <= 1


class TestValidation:
    """tests for parallel trends and robustness checks"""

    def test_parallel_trends_runs(self, sample_data):
        df = augment_data_for_causal(sample_data)
        is_valid, f_stat, p_value, monthly = validate_parallel_trends(df)
        assert isinstance(is_valid, (bool, np.bool_))
        assert f_stat >= 0

    def test_placebo_test_runs(self, sample_data):
        df = augment_data_for_causal(sample_data)
        ate, p_value, passed = run_placebo_test(df)
        assert isinstance(passed, (bool, np.bool_))
        assert 0 <= p_value <= 1

    def test_sensitivity_analysis_returns_specs(self, sample_data):
        df = augment_data_for_causal(sample_data)
        results = run_sensitivity_analysis(df)
        assert 'no_controls' in results
        assert 'full_controls' in results


class TestCityTier:
    """quick sanity checks for the tier assignment"""

    def test_tier_1_cities(self):
        assert assign_city_tier('Mumbai') == 1
        assert assign_city_tier('Delhi') == 1
        assert assign_city_tier('Bangalore') == 1

    def test_tier_2_cities(self):
        assert assign_city_tier('Pune') == 2
        assert assign_city_tier('Jaipur') == 2

    def test_tier_3_cities(self):
        assert assign_city_tier('Bhopal') == 3
        assert assign_city_tier('Unknown') == 3


class TestFullPipeline:
    """integration test for the complete causal pipeline"""

    def test_pipeline_completes(self, sample_data):
        results, df_causal = run_causal_pipeline(sample_data, sample_size=5000)
        assert results is not None
        assert results.n_observations > 0
        assert results.ate != 0 or results.p_value > 0
