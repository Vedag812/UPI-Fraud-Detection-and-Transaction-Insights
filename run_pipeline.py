"""
runs the full pipeline in order. just execute this and it handles everything.
expect around 5-10 minutes for 1M rows depending on your machine.

pipeline:
  step 1: generate synthetic UPI transaction data (1M rows)
  step 2: load into SQLite and run analytical queries
  step 3: clean data and engineer features
  step 4: run fraud detection (z-score, isolation forest, rules, ensemble)
  step 5: generate Excel report
  step 6: build FAISS similarity index over fraud patterns (NEW)
  step 7: run causal inference - DiD analysis of 2FA impact (NEW)
"""

import os
import sys
import time

# put src on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_generator import generate_transactions, add_missing_values, save_data
from data_cleaning import load_raw_data, check_data_quality, clean_data, add_features, save_processed
from db_utils import create_database, run_all_queries
from fraud_detector import run_full_pipeline
from report_generator import create_excel_report


def main():
    base_dir = os.path.dirname(__file__)
    raw_dir = os.path.join(base_dir, 'data', 'raw')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    db_path = os.path.join(base_dir, 'database', 'upi_transactions.db')
    report_path = os.path.join(base_dir, 'reports', 'fraud_report.xlsx')

    start = time.time()

    # Step 1: Generate data
    print("\n" + "="*60)
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print("="*60)
    df = generate_transactions()
    df = add_missing_values(df)
    raw_csv = save_data(df, raw_dir)

    # Step 2: Load into SQL
    print("\n" + "="*60)
    print("STEP 2: CREATING SQL DATABASE")
    print("="*60)
    create_database(raw_csv, db_path)
    print("\nRunning SQL queries...")
    results = run_all_queries(db_path)
    print(f"Executed {len(results)} queries successfully")

    # Step 3: Clean and engineer features
    print("\n" + "="*60)
    print("STEP 3: DATA CLEANING & FEATURE ENGINEERING")
    print("="*60)
    df = load_raw_data(raw_csv)
    check_data_quality(df)
    df = clean_data(df)
    df = add_features(df)
    save_processed(df, processed_dir)

    # Step 4: Run fraud detection
    print("\n" + "="*60)
    print("STEP 4: FRAUD DETECTION")
    print("="*60)
    df, comparison, hyp_results = run_full_pipeline(df)

    # saves all the outputs from this step
    flagged_path = os.path.join(processed_dir, 'upi_transactions_flagged.csv')
    df.to_csv(flagged_path, index=False)
    comparison.to_csv(os.path.join(processed_dir, 'method_comparison.csv'), index=False)
    hyp_results.to_csv(os.path.join(processed_dir, 'hypothesis_tests.csv'), index=False)

    # Step 5: Generate Excel report
    print("\n" + "="*60)
    print("STEP 5: GENERATING EXCEL REPORT")
    print("="*60)
    create_excel_report(df, comparison, hyp_results, report_path)

    # Step 6: Build FAISS similarity index
    # indexes all confirmed fraud transactions as vectors so we can do
    # "find me the 5 most similar historical fraud cases" in milliseconds
    print("\n" + "="*60)
    print("STEP 6: BUILDING FAISS SIMILARITY INDEX")
    print("="*60)
    try:
        from fraud_similarity import build_and_save_index
        faiss_dir = os.path.join(base_dir, 'data', 'faiss_index')
        engine = build_and_save_index(df, faiss_dir)

        # also add FAISS flags to the dataset
        if engine.is_fitted:
            df = engine.add_faiss_flags(df)
            # re-save with FAISS flags included
            df.to_csv(flagged_path, index=False)
            print(f"  updated flagged data with FAISS scores")
    except ImportError:
        print("  faiss-cpu not installed, skipping. install with: pip install faiss-cpu")
    except Exception as e:
        print(f"  FAISS index build failed: {e}")

    # Step 7: Causal Inference (Difference-in-Differences)
    # answers: "does 2FA actually cause a reduction in fraud, or is it just correlation?"
    # this is the key differentiator from standard ML-based fraud detection
    print("\n" + "="*60)
    print("STEP 7: CAUSAL INFERENCE (Difference-in-Differences)")
    print("="*60)
    try:
        from causal_inference import run_causal_pipeline
        causal_results, df_causal = run_causal_pipeline(
            df,
            sample_size=100_000,
            output_dir=processed_dir
        )
    except ImportError:
        print("  statsmodels not installed, skipping. install with: pip install statsmodels")
    except Exception as e:
        print(f"  causal analysis failed: {e}")

    elapsed = time.time() - start
    print("\n" + "="*60)
    print(f"ALL DONE. Total time: {elapsed/60:.1f} minutes")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  Raw data:        {raw_csv}")
    print(f"  Database:        {db_path}")
    print(f"  Processed:       {os.path.join(processed_dir, 'upi_transactions_processed.csv')}")
    print(f"  Flagged:         {flagged_path}")
    print(f"  Comparison:      {os.path.join(processed_dir, 'method_comparison.csv')}")
    print(f"  Causal results:  {os.path.join(processed_dir, 'causal_results.json')}")
    print(f"  FAISS index:     {os.path.join(base_dir, 'data', 'faiss_index')}")
    print(f"  Excel report:    {report_path}")
    print(f"\nTo launch the dashboard:")
    print(f"  streamlit run dashboard/app.py")
    print(f"\nTo start the API:")
    print(f"  uvicorn api:app --reload")
    print(f"  then visit http://localhost:8000/docs for interactive API docs")
    print(f"\nTo run with Docker:")
    print(f"  docker-compose up -d")


if __name__ == '__main__':
    main()
