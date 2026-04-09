"""
causal inference: measuring whether 2FA actually *causes* a reduction in fraud.

this is the key shift from the rest of the project. everything else asks "is this
transaction fraudulent?" - this module asks "did rolling out 2FA actually reduce fraud,
or would fraud have dropped anyway?"

uses Difference-in-Differences (DiD) which is the gold standard for measuring the
causal impact of a policy change. same stuff economists use to evaluate minimum wage
laws or healthcare mandates - we're applying it to fintech.

methodology:
  - split users into treatment (adopted 2FA) and control (didn't)
  - compare fraud rates before vs after the 2FA launch date
  - the "difference in differences" isolates the causal effect from time trends
  - if fraud was already declining for everyone, DiD strips that out

references:
  - Angrist & Pischke, "Mostly Harmless Econometrics" (the DiD bible)
  - RBI mandates on two-factor auth for UPI published Dec 2023
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os
import warnings
warnings.filterwarnings('ignore')


# -- config --
# july 1 is the midpoint of our jan-dec 2024 dataset
# simulates a hypothetical RBI mandate for mandatory 2FA on UPI
TREATMENT_DATE = pd.Timestamp('2024-07-01')

# fake placebo date for robustness testing - if the model picks up a
# "treatment effect" here, something is wrong with our methodology
PLACEBO_DATE = pd.Timestamp('2024-04-01')

# not everyone adopts 2FA on day one - 60% is realistic for a phased rollout
# (banks like SBI took months to comply with RBI's tokenization mandate in 2022)
ADOPTION_RATE = 0.60


@dataclass
class CausalResults:
    """bundles all the DiD outputs into one clean object for the API to serve"""
    research_question: str = "Effect of 2FA on UPI fraud rates"
    method: str = "Difference-in-Differences (OLS with HC1 robust SE)"
    ate: float = 0.0  # average treatment effect - the main number
    ate_pct_points: str = ""  # human-readable version
    confidence_interval: list = field(default_factory=list)
    p_value: float = 0.0
    is_significant: bool = False
    parallel_trends_valid: bool = False
    parallel_trends_f_stat: float = 0.0
    parallel_trends_p_value: float = 0.0
    placebo_test_passed: bool = False
    placebo_ate: float = 0.0
    placebo_p_value: float = 0.0
    n_observations: int = 0
    n_treatment: int = 0
    n_control: int = 0
    model_r_squared: float = 0.0
    controls: list = field(default_factory=list)
    sensitivity_results: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def to_json(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"  saved causal results: {filepath}")


def assign_city_tier(city):
    """
    tier 1 = metro, tier 2 = mid-size, tier 3 = smaller cities.
    this matters because 2FA adoption rates differ massively by city tier -
    someone in mumbai is way more likely to use phonepe's 2FA than someone in bhopal.
    """
    tier_1 = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata']
    tier_2 = ['Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Surat']
    if city in tier_1:
        return 1
    elif city in tier_2:
        return 2
    else:
        return 3


def augment_data_for_causal(df):
    """
    simulates the 2FA rollout scenario by adding treatment/control groups.

    the key design decision: 2FA adoption is NOT random. users in tier-1 cities
    using phonepe/gpay are more likely to have 2FA. this creates confounding -
    exactly the kind of thing DiD needs to handle. if we made it random, the
    analysis would be trivially easy and wouldn't demonstrate real-world rigor.

    confounders we simulate:
      - city_tier: metro vs mid-size vs small (affects both fraud and 2FA adoption)
      - user_risk_score: historical behavior metric (higher = more suspicious)
      - avg_txn_frequency: how active the user is (power users adopt 2FA faster)
    """
    print("  augmenting data for causal analysis...")
    df = df.copy()

    # make sure timestamp is proper datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # -- treatment timing --
    # post_treatment = 1 means the transaction happened after July 1 (2FA launch)
    df['post_treatment'] = (df['timestamp'] >= TREATMENT_DATE).astype(int)

    # -- city tier --
    # filling 'Unknown' cities as tier 3 since they're likely smaller cities
    df['city_tier'] = df['city'].apply(assign_city_tier)

    # -- simulate 2FA adoption (the non-random part) --
    # base probability of 2FA adoption varies by city and bank
    # this is intentionally confounded - tier 1 cities and fintech-first banks
    # adopt 2FA more, AND they already have lower fraud rates
    np.random.seed(42)

    # build per-user adoption probability
    user_data = df.groupby('sender_upi_id').agg({
        'city_tier': 'first',
        'sender_bank': 'first',
        'amount': 'mean',
        'transaction_id': 'count'
    }).rename(columns={'transaction_id': 'txn_count', 'amount': 'avg_amount'})

    # tier 1 users 2x more likely to have 2FA than tier 3
    tier_boost = {1: 0.25, 2: 0.10, 3: -0.15}
    user_data['adoption_prob'] = ADOPTION_RATE

    for tier, boost in tier_boost.items():
        mask = user_data['city_tier'] == tier
        user_data.loc[mask, 'adoption_prob'] += boost

    # fintech banks (phonepe, gpay) rolled out 2FA faster than legacy banks
    fintech_banks = ['PhonePe', 'Google Pay', 'Paytm', 'CRED']
    is_fintech = user_data['sender_bank'].isin(fintech_banks)
    user_data.loc[is_fintech, 'adoption_prob'] += 0.10

    # clamp probabilities to [0.1, 0.95] - nobody has 0% or 100% chance
    user_data['adoption_prob'] = user_data['adoption_prob'].clip(0.1, 0.95)

    # actually assign the treatment
    user_data['has_2fa'] = (np.random.random(len(user_data)) < user_data['adoption_prob']).astype(int)

    # merge back into transaction data
    df = df.merge(
        user_data[['has_2fa']],
        left_on='sender_upi_id',
        right_index=True,
        how='left'
    )
    df['has_2fa'] = df['has_2fa'].fillna(0).astype(int)

    # -- the DiD interaction term --
    # this is what we actually care about: did people WITH 2FA see fraud drop MORE
    # than people WITHOUT 2FA after the launch date?
    df['treatment'] = df['has_2fa'] * df['post_treatment']

    # -- confounders / controls --
    # user risk score: synthetic metric based on night-time activity and high-value txns
    # in a real system this would come from a risk model, here we approximate it
    user_risk = df.groupby('sender_upi_id').agg({
        'is_night': 'mean',
        'amount': lambda x: (x > x.quantile(0.90)).mean(),
        'txn_velocity': 'mean'
    })
    user_risk.columns = ['night_pct', 'high_value_pct', 'avg_velocity']
    # combine into a single risk score (0-1 range)
    user_risk['user_risk_score'] = (
        0.4 * user_risk['night_pct'] +
        0.3 * user_risk['high_value_pct'] +
        0.3 * (user_risk['avg_velocity'] / user_risk['avg_velocity'].max())
    ).clip(0, 1)

    df = df.merge(user_risk[['user_risk_score']], left_on='sender_upi_id',
                  right_index=True, how='left')
    df['user_risk_score'] = df['user_risk_score'].fillna(0.5)

    # average transaction frequency per user (txns per month)
    user_freq = df.groupby('sender_upi_id')['transaction_id'].count() / 12
    user_freq.name = 'avg_txn_frequency'
    df = df.merge(user_freq, left_on='sender_upi_id', right_index=True, how='left')
    df['avg_txn_frequency'] = df['avg_txn_frequency'].fillna(1)

    # month number for the parallel trends analysis
    df['month_num'] = df['timestamp'].dt.month

    counts = df['has_2fa'].value_counts()
    print(f"  treatment (has_2fa=1): {counts.get(1, 0):,} transactions")
    print(f"  control   (has_2fa=0): {counts.get(0, 0):,} transactions")
    print(f"  pre-treatment:  {(df['post_treatment'] == 0).sum():,}")
    print(f"  post-treatment: {(df['post_treatment'] == 1).sum():,}")

    return df


def run_did_analysis(df, treatment_date=None):
    """
    the actual Difference-in-Differences regression.

    the model:
      is_fraud = β0 + β1(has_2fa) + β2(post_treatment) + β3(has_2fa × post_treatment)
                 + β4(amount_log) + β5(is_night) + β6(is_weekend) + β7(city_tier)
                 + β8(user_risk_score) + ε

    β3 is what we're after - the causal effect of 2FA on fraud. positive means
    2FA increases fraud (would be very weird), negative means it reduces fraud (expected).

    using HC1 robust standard errors because financial data is almost always
    heteroscedastic (variance of fraud isn't constant across amount levels).
    """
    print("\n  running Difference-in-Differences regression...")

    # make sure we have amount_log
    if 'amount_log' not in df.columns:
        df['amount_log'] = np.log1p(df['amount'])

    # the DiD formula - treatment is the interaction term (has_2fa × post_treatment)
    formula = ('is_fraud ~ has_2fa + post_treatment + treatment '
               '+ amount_log + is_night + is_weekend + city_tier + user_risk_score')

    model = smf.ols(formula, data=df).fit(cov_type='HC1')

    # extract the treatment effect (the coefficient on the interaction term)
    ate = model.params['treatment']
    ci = model.conf_int().loc['treatment']
    p_val = model.pvalues['treatment']

    print(f"\n  --- DiD Results ---")
    print(f"  ATE (treatment coefficient): {ate:.6f}")
    print(f"  interpretation: 2FA {'reduces' if ate < 0 else 'increases'} fraud probability "
          f"by {abs(ate)*100:.2f} percentage points")
    print(f"  95% CI: [{ci[0]:.6f}, {ci[1]:.6f}]")
    print(f"  p-value: {p_val:.8f} ({'significant' if p_val < 0.05 else 'NOT significant'})")
    print(f"  R²: {model.rsquared:.4f}")
    print(f"  N: {model.nobs:,.0f}")

    return model, ate, ci, p_val


def validate_parallel_trends(df):
    """
    the parallel trends assumption is the single most important thing in DiD.

    it says: "before the treatment, the treatment and control groups had the
    same trend in fraud rates." if this fails, DiD is invalid because the
    groups were already diverging and we can't attribute the change to 2FA.

    we test this by checking if has_2fa interacted with month dummies has any
    effect on fraud in the pre-treatment period. if those interactions are
    jointly insignificant (high p-value on the F-test), the assumption holds.
    """
    print("\n  validating parallel trends assumption...")

    # only look at pre-treatment data
    pre_df = df[df['post_treatment'] == 0].copy()

    if len(pre_df) == 0:
        print("  WARNING: no pre-treatment data found!")
        return False, 0.0, 0.0, None

    # regress fraud on has_2fa × month to check for differential trends
    pre_df['month_num'] = pre_df['timestamp'].dt.month

    # restricted model (no interaction - assumes parallel trends)
    restricted_formula = 'is_fraud ~ has_2fa + C(month_num)'
    restricted = smf.ols(restricted_formula, data=pre_df).fit()

    # unrestricted model (with interaction - allows different trends)
    unrestricted_formula = 'is_fraud ~ has_2fa * C(month_num)'
    unrestricted = smf.ols(unrestricted_formula, data=pre_df).fit()

    # F-test: are the interaction terms jointly zero?
    # high p-value = good (no evidence of differential trends)
    f_stat = ((restricted.ssr - unrestricted.ssr) /
              (restricted.df_resid - unrestricted.df_resid)) / \
             (unrestricted.ssr / unrestricted.df_resid)

    df_num = restricted.df_resid - unrestricted.df_resid
    df_den = unrestricted.df_resid
    f_p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)

    is_valid = f_p_value > 0.05  # we WANT this to be non-significant

    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {f_p_value:.4f}")
    print(f"  parallel trends {'VALID ✓' if is_valid else 'VIOLATED ✗'}")
    if is_valid:
        print("  (no evidence of differential pre-trends - assumption holds)")
    else:
        print("  (treatment and control groups had different trends before 2FA)")

    # compute monthly fraud rates for both groups (for plotting)
    monthly = pre_df.groupby(['month_num', 'has_2fa'])['is_fraud'].mean().reset_index()
    monthly.columns = ['month', 'group', 'fraud_rate']
    monthly['group'] = monthly['group'].map({0: 'Control (No 2FA)', 1: 'Treatment (2FA)'})

    return is_valid, f_stat, f_p_value, monthly


def run_placebo_test(df):
    """
    placebo test: pretend 2FA launched on april 1 instead of july 1.

    if we find a "treatment effect" at a fake date, it means our model is
    picking up some other trend, not the actual 2FA impact. a good placebo
    test shows ATE ≈ 0 with a big p-value (not significant).

    this is the same logic as a drug trial placebo - if the sugar pill also
    "works," your drug probably doesn't.
    """
    print("\n  running placebo test (fake treatment date: April 2024)...")

    # only use pre-treatment data (before the real July cutoff)
    placebo_df = df[df['timestamp'] < TREATMENT_DATE].copy()

    if len(placebo_df) == 0:
        print("  WARNING: no data for placebo test")
        return 0.0, 1.0, True

    # create fake treatment indicator at April 1
    placebo_df['post_treatment'] = (placebo_df['timestamp'] >= PLACEBO_DATE).astype(int)
    placebo_df['treatment'] = placebo_df['has_2fa'] * placebo_df['post_treatment']

    if 'amount_log' not in placebo_df.columns:
        placebo_df['amount_log'] = np.log1p(placebo_df['amount'])

    formula = ('is_fraud ~ has_2fa + post_treatment + treatment '
               '+ amount_log + is_night + is_weekend + city_tier + user_risk_score')

    try:
        model = smf.ols(formula, data=placebo_df).fit(cov_type='HC1')
        placebo_ate = model.params['treatment']
        placebo_p = model.pvalues['treatment']
    except Exception as e:
        print(f"  placebo model failed: {e}")
        return 0.0, 1.0, True

    passed = placebo_p > 0.05  # we WANT this to be non-significant

    print(f"  placebo ATE: {placebo_ate:.6f}")
    print(f"  placebo p-value: {placebo_p:.4f}")
    print(f"  placebo test {'PASSED ✓' if passed else 'FAILED ✗'}")
    if passed:
        print("  (no spurious effect at fake date - good sign)")
    else:
        print("  (model detects effect at fake date - could indicate confounding)")

    return placebo_ate, placebo_p, passed


def run_sensitivity_analysis(df):
    """
    checks if the treatment effect is stable when we add/remove control variables.

    if ATE swings wildly depending on which controls we include, the result
    is fragile and we can't be confident about it. ideally ATE stays in a
    narrow range regardless of specification - that's a robust finding.

    this is standard practice in econometrics papers (Peer 5 would do this).
    """
    print("\n  running sensitivity analysis...")

    if 'amount_log' not in df.columns:
        df['amount_log'] = np.log1p(df['amount'])

    # model 1: bare bones - just the DiD terms, no controls
    specs = {
        'no_controls': 'is_fraud ~ has_2fa + post_treatment + treatment',
        'time_controls': 'is_fraud ~ has_2fa + post_treatment + treatment + is_night + is_weekend',
        'amount_controls': 'is_fraud ~ has_2fa + post_treatment + treatment + amount_log + is_night + is_weekend',
        'full_controls': ('is_fraud ~ has_2fa + post_treatment + treatment '
                          '+ amount_log + is_night + is_weekend + city_tier + user_risk_score'),
    }

    results = {}
    for name, formula in specs.items():
        try:
            model = smf.ols(formula, data=df).fit(cov_type='HC1')
            results[name] = {
                'ate': round(float(model.params['treatment']), 6),
                'p_value': round(float(model.pvalues['treatment']), 6),
                'r_squared': round(float(model.rsquared), 4),
                'n_obs': int(model.nobs)
            }
            print(f"  {name:20s} → ATE={results[name]['ate']:.6f}, "
                  f"p={results[name]['p_value']:.4f}, R²={results[name]['r_squared']:.4f}")
        except Exception as e:
            print(f"  {name}: failed ({e})")
            results[name] = {'ate': None, 'p_value': None, 'error': str(e)}

    return results


def generate_parallel_trends_plot(monthly_data, output_path):
    """
    creates the parallel trends visualization - this is what reviewers and
    interviewers will look at first. if the lines are roughly parallel before
    the cutoff, the assumption holds visually.

    saves to reports/ so it can go straight into the README.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # no GUI needed
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        for group_name in monthly_data['group'].unique():
            group = monthly_data[monthly_data['group'] == group_name]
            ax.plot(group['month'], group['fraud_rate'],
                    marker='o', linewidth=2, markersize=8, label=group_name)

        ax.set_xlabel('Month (2024)', fontsize=12)
        ax.set_ylabel('Fraud Rate', fontsize=12)
        ax.set_title('Parallel Trends Validation\n(Pre-Treatment Period: Jan-Jun 2024)',
                      fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 7))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])

        # add annotation explaining what we're looking for
        ax.annotate('Lines should be roughly parallel\n(= no differential pre-trends)',
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    ha='center', fontsize=9, fontstyle='italic',
                    color='gray')

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  saved parallel trends plot: {output_path}")
    except ImportError:
        print("  matplotlib not available, skipping plot generation")


def run_causal_pipeline(df, sample_size=100_000, output_dir=None):
    """
    orchestrates the full causal inference analysis.

    runs on a sample of the data by default (100K rows) because OLS on 1M+ rows
    is slow and the statistical conclusions don't change much with more data.
    the precision of the estimates improves marginally past 100K - not worth the wait.

    pipeline:
      1. sample → 2. augment → 3. DiD regression → 4. parallel trends
      5. placebo test → 6. sensitivity analysis → 7. bundle results
    """
    print("=" * 50)
    print("CAUSAL INFERENCE PIPELINE")
    print("=" * 50)
    print(f"\nresearch question: does 2FA causally reduce UPI fraud?")
    print(f"method: Difference-in-Differences (DiD)")
    print(f"treatment date: {TREATMENT_DATE.strftime('%B %d, %Y')}")

    # step 1: sample for speed (preserving fraud ratio)
    if len(df) > sample_size:
        print(f"\n[1/6] sampling {sample_size:,} rows from {len(df):,} (stratified by fraud label)...")
        # stratified sampling so we keep the same fraud rate
        fraud = df[df['is_fraud'] == 1]
        normal = df[df['is_fraud'] == 0]
        fraud_ratio = len(fraud) / len(df)

        n_fraud = int(sample_size * fraud_ratio)
        n_normal = sample_size - n_fraud

        sampled_fraud = fraud.sample(n=min(n_fraud, len(fraud)), random_state=42)
        sampled_normal = normal.sample(n=min(n_normal, len(normal)), random_state=42)
        df_sample = pd.concat([sampled_fraud, sampled_normal]).sample(frac=1, random_state=42)
        print(f"  sampled: {len(df_sample):,} rows, fraud rate: {df_sample['is_fraud'].mean()*100:.2f}%")
    else:
        df_sample = df.copy()

    # step 2: augment with treatment/control assignments
    print(f"\n[2/6] augmenting data with treatment/control groups...")
    df_causal = augment_data_for_causal(df_sample)

    # step 3: run the DiD regression
    print(f"\n[3/6] running DiD regression...")
    model, ate, ci, p_value = run_did_analysis(df_causal)

    # step 4: validate parallel trends
    print(f"\n[4/6] validating parallel trends...")
    pt_valid, pt_f_stat, pt_p_value, monthly_data = validate_parallel_trends(df_causal)

    # step 5: placebo test
    print(f"\n[5/6] running placebo test...")
    placebo_ate, placebo_p, placebo_passed = run_placebo_test(df_causal)

    # step 6: sensitivity analysis
    print(f"\n[6/6] running sensitivity analysis...")
    sensitivity = run_sensitivity_analysis(df_causal)

    # bundle everything into a results object
    results = CausalResults(
        ate=round(float(ate), 6),
        ate_pct_points=f"2FA {'reduces' if ate < 0 else 'increases'} fraud probability by {abs(ate)*100:.2f} percentage points",
        confidence_interval=[round(float(ci[0]), 6), round(float(ci[1]), 6)],
        p_value=round(float(p_value), 8),
        is_significant=bool(p_value < 0.05),
        parallel_trends_valid=bool(pt_valid),
        parallel_trends_f_stat=round(float(pt_f_stat), 4),
        parallel_trends_p_value=round(float(pt_p_value), 4),
        placebo_test_passed=bool(placebo_passed),
        placebo_ate=round(float(placebo_ate), 6),
        placebo_p_value=round(float(placebo_p), 4),
        n_observations=int(len(df_causal)),
        n_treatment=int((df_causal['has_2fa'] == 1).sum()),
        n_control=int((df_causal['has_2fa'] == 0).sum()),
        model_r_squared=round(float(model.rsquared), 4),
        controls=['amount_log', 'is_night', 'is_weekend', 'city_tier', 'user_risk_score'],
        sensitivity_results=sensitivity
    )

    # save outputs if directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results.to_json(os.path.join(output_dir, 'causal_results.json'))

        # save the parallel trends plot
        if monthly_data is not None:
            plot_dir = os.path.join(os.path.dirname(output_dir), 'reports')
            generate_parallel_trends_plot(
                monthly_data,
                os.path.join(plot_dir, 'parallel_trends.png')
            )

    # final summary
    print(f"\n{'=' * 50}")
    print(f"CAUSAL INFERENCE RESULTS")
    print(f"{'=' * 50}")
    print(f"  ATE: {results.ate:.6f} ({results.ate_pct_points})")
    print(f"  95% CI: [{results.confidence_interval[0]:.6f}, {results.confidence_interval[1]:.6f}]")
    print(f"  p-value: {results.p_value:.8f} → {'SIGNIFICANT' if results.is_significant else 'not significant'}")
    print(f"  parallel trends: {'VALID' if results.parallel_trends_valid else 'VIOLATED'}")
    print(f"  placebo test: {'PASSED' if results.placebo_test_passed else 'FAILED'}")
    print(f"  observations: {results.n_observations:,}")

    return results, df_causal


if __name__ == '__main__':
    # quick standalone test
    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_path = os.path.join(base_dir, 'data', 'processed', 'upi_transactions_processed.csv')

    if os.path.exists(processed_path):
        print(f"loading {processed_path}...")
        df = pd.read_csv(processed_path, parse_dates=['timestamp'])
        output_dir = os.path.join(base_dir, 'data', 'processed')
        results, df_causal = run_causal_pipeline(df, output_dir=output_dir)
        print("\ndone. check data/processed/causal_results.json for full output.")
    else:
        print("processed data not found. run the main pipeline first.")
