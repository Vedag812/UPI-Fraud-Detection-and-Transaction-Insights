"""
FAISS-based fraud pattern similarity search.

instead of just asking "is this transaction fraudulent?", this module asks
"which known fraud patterns does this transaction look like?"

the idea: embed every confirmed fraud transaction into a vector space, build a
FAISS index over them, and when a new transaction comes in, find the K most
similar historical fraud cases. if the nearest fraud neighbors are really close,
this transaction is probably suspicious too.

why FAISS over a basic KNN? because FAISS (Facebook AI Similarity Search) is
built for production-scale vector search - it can handle millions of vectors
with sub-millisecond query times. even though our dataset is only 1M rows,
using FAISS demonstrates knowledge of production vector database patterns
(same tech behind recommendation systems at Meta, Spotify, Uber etc).

vector databases like FAISS/Pinecone/Weaviate are becoming essential for
RAG systems, semantic search, and recommendation engines - having hands-on
experience with one is a strong signal.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss-cpu not installed. run 'pip install faiss-cpu' for similarity search.")


# -- config --
# 11-dimensional feature vector per transaction
# carefully chosen to capture the key fraud signals without overfitting to noise
FEATURE_COLS = [
    'amount_normalized',
    'hour_sin', 'hour_cos',           # cyclical encoding so 23:00 is close to 00:00
    'dow_sin', 'dow_cos',             # same for day of week
    'is_night', 'is_weekend',
    'txn_velocity_norm',
    'amount_zscore',
    'city_encoded', 'txn_type_encoded'
]

# how many similar fraud cases to return by default
DEFAULT_K = 5

# if the fraud_similarity_score exceeds this, flag as suspicious.
# score = 1/(1+avg_distance), so 0.90 means avg L2 distance < 0.11
# calibrated so roughly 3-5% of transactions get flagged (matching real fraud rates).
# the old threshold of 2.5 was way too loose — with 29K fraud vectors in 11D space,
# almost everything has a fraud neighbor within distance 2.5.
SIMILARITY_THRESHOLD = 0.12


class FraudSimilarityEngine:
    """
    wraps the FAISS index with preprocessing, building, querying, and persistence.

    usage:
        engine = FraudSimilarityEngine()
        engine.fit(df)                           # build index from fraud data
        results = engine.query(new_txn_features) # find similar frauds
        engine.save('path/to/index')             # persist for API use
        engine = FraudSimilarityEngine.load('path/to/index')  # reload
    """

    def __init__(self):
        self.index = None
        self.scaler = StandardScaler()
        self.city_encoder = LabelEncoder()
        self.txn_type_encoder = LabelEncoder()
        self.fraud_metadata = None  # stores details about indexed fraud txns
        self.n_features = len(FEATURE_COLS)
        self.is_fitted = False

    def _engineer_features(self, df):
        """
        transforms raw transaction data into the embedding vector.

        a few design decisions worth noting:
        - hour and day_of_week use sin/cos encoding because they're cyclical
          (11pm is close to midnight, sunday is close to monday)
        - amount uses z-score instead of raw value because ₹500 on a ₹300/avg
          account is more suspicious than ₹500 on a ₹50K/avg account
        - city and txn_type are label-encoded (simple but effective for L2 distance)
        """
        features = pd.DataFrame(index=df.index)

        # amount - normalized so scale doesn't dominate the distance metric
        features['amount_normalized'] = df['amount'].values

        # cyclical time encoding - 23:00 should be "close" to 00:00 in vector space
        hours = df['hour'].values if 'hour' in df.columns else df['timestamp'].dt.hour.values
        features['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        # cyclical day-of-week encoding
        if 'day_of_week' in df.columns:
            dow = df['day_of_week'].values
        else:
            dow = df['timestamp'].dt.dayofweek.values
        features['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        features['dow_cos'] = np.cos(2 * np.pi * dow / 7)

        # binary flags
        features['is_night'] = df['is_night'].values if 'is_night' in df.columns else 0
        features['is_weekend'] = df['is_weekend'].values if 'is_weekend' in df.columns else 0

        # velocity - how many txns this user did in the same hour
        # high velocity = possible account takeover
        features['txn_velocity_norm'] = df['txn_velocity'].values if 'txn_velocity' in df.columns else 0

        # z-score - how unusual this amount is for this specific user
        features['amount_zscore'] = df['amount_zscore'].values if 'amount_zscore' in df.columns else 0

        # categorical features
        city_col = df['city'].fillna('Unknown').values
        txn_type_col = df['transaction_type'].fillna('Unknown').values

        if self.is_fitted:
            # use existing encoders for new data (might have unseen categories)
            features['city_encoded'] = self._safe_encode(self.city_encoder, city_col)
            features['txn_type_encoded'] = self._safe_encode(self.txn_type_encoder, txn_type_col)
        else:
            features['city_encoded'] = self.city_encoder.fit_transform(city_col)
            features['txn_type_encoded'] = self.txn_type_encoder.fit_transform(txn_type_col)

        return features

    def _safe_encode(self, encoder, values):
        """handles unseen categories gracefully instead of crashing"""
        known = set(encoder.classes_)
        encoded = np.zeros(len(values), dtype=int)
        for i, val in enumerate(values):
            if val in known:
                encoded[i] = encoder.transform([val])[0]
            else:
                # unseen category gets -1, which is fine for distance calc
                encoded[i] = -1
        return encoded

    def fit(self, df):
        """
        builds the FAISS index from confirmed fraud transactions.

        only indexes fraud transactions because we want to answer "how similar
        is this new transaction to known fraud?" - we don't care about similarity
        to normal transactions.
        """
        if not FAISS_AVAILABLE:
            print("  FAISS not available, skipping index build")
            return self

        print("  building FAISS fraud similarity index...")

        # extract only confirmed fraud transactions
        fraud_df = df[df['is_fraud'] == 1].copy()

        if len(fraud_df) == 0:
            print("  WARNING: no fraud transactions to index!")
            return self

        # engineer features
        features = self._engineer_features(fraud_df)

        # fit the scaler on fraud features and transform
        feature_matrix = self.scaler.fit_transform(features.values).astype(np.float32)

        # build the FAISS index
        # using IndexFlatL2 (exact search with L2 distance)
        # for our dataset size (~25K fraud txns), exact search is fast enough
        # for millions of vectors you'd use IndexIVFFlat or IndexHNSW
        self.index = faiss.IndexFlatL2(self.n_features)
        self.index.add(feature_matrix)

        # store metadata so we can return useful info with search results
        self.fraud_metadata = fraud_df[['transaction_id', 'amount', 'fraud_type',
                                        'city', 'sender_bank', 'timestamp',
                                        'transaction_type']].reset_index(drop=True)

        self.is_fitted = True
        print(f"  indexed {self.index.ntotal:,} fraud vectors ({self.n_features}D)")

        return self

    def query(self, df_or_features, k=DEFAULT_K):
        """
        finds the K most similar historical fraud patterns for each input transaction.

        returns a list of dicts, one per query transaction, containing:
          - similar fraud transaction IDs and details
          - L2 distances (lower = more similar)
          - fraud_similarity_score (0-1, higher = more suspicious)

        the fraud_similarity_score is computed as: 1 / (1 + avg_distance)
        so a transaction that's very close to known fraud gets a score near 1.
        """
        if not self.is_fitted or self.index is None:
            return []

        if isinstance(df_or_features, pd.DataFrame):
            features = self._engineer_features(df_or_features)
            feature_matrix = self.scaler.transform(features.values).astype(np.float32)
        else:
            feature_matrix = np.array(df_or_features, dtype=np.float32)
            if feature_matrix.ndim == 1:
                feature_matrix = feature_matrix.reshape(1, -1)

        # FAISS search - returns distances and indices of nearest neighbors
        distances, indices = self.index.search(feature_matrix, k)

        results = []
        for i in range(len(feature_matrix)):
            similar_frauds = []
            valid_distances = []

            for j in range(k):
                idx = indices[i][j]
                dist = distances[i][j]

                if idx < 0 or idx >= len(self.fraud_metadata):
                    continue

                meta = self.fraud_metadata.iloc[idx]
                similar_frauds.append({
                    'transaction_id': str(meta['transaction_id']),
                    'amount': float(meta['amount']),
                    'fraud_type': str(meta['fraud_type']),
                    'city': str(meta['city']),
                    'bank': str(meta['sender_bank']),
                    'distance': round(float(dist), 4)
                })
                valid_distances.append(dist)

            # compute similarity score: inverse of average distance, normalized to 0-1
            avg_dist = np.mean(valid_distances) if valid_distances else float('inf')
            similarity_score = round(1.0 / (1.0 + avg_dist), 4)

            results.append({
                'similar_frauds': similar_frauds,
                'fraud_similarity_score': similarity_score,
                'is_suspicious': similarity_score > (1.0 / (1.0 + SIMILARITY_THRESHOLD)),
                'avg_distance': round(float(avg_dist), 4)
            })

        return results

    def add_faiss_flags(self, df, k=DEFAULT_K):
        """
        runs similarity search on ALL transactions and adds a faiss_flag column.

        this integrates FAISS into the ensemble - transactions that are very
        similar to known fraud patterns get flagged, even if the other methods
        missed them. particularly useful for catching new variants of known
        fraud types that are slightly different from the training patterns.
        """
        if not self.is_fitted:
            df['faiss_flag'] = 0
            df['fraud_similarity_score'] = 0.0
            return df

        print("  scoring all transactions against FAISS index...")

        # process in batches to manage memory (1M rows × 11 features is fine but why not)
        batch_size = 50_000
        all_scores = []
        all_flags = []

        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            batch = df.iloc[start:end]
            results = self.query(batch, k=k)

            for r in results:
                all_scores.append(r['fraud_similarity_score'])
                all_flags.append(1 if r['is_suspicious'] else 0)

        df['fraud_similarity_score'] = all_scores
        df['faiss_flag'] = all_flags

        flagged = sum(all_flags)
        print(f"  FAISS flagged {flagged:,} transactions ({flagged/len(df)*100:.2f}%)")

        return df

    def save(self, directory):
        """persists the index and preprocessing artifacts to disk"""
        if not self.is_fitted:
            print("  nothing to save - engine not fitted")
            return

        os.makedirs(directory, exist_ok=True)

        # save FAISS index
        index_path = os.path.join(directory, 'faiss_fraud_index.bin')
        faiss.write_index(self.index, index_path)

        # save the scaler, encoders, and metadata
        artifacts_path = os.path.join(directory, 'similarity_artifacts.pkl')
        artifacts = {
            'scaler': self.scaler,
            'city_encoder': self.city_encoder,
            'txn_type_encoder': self.txn_type_encoder,
            'fraud_metadata': self.fraud_metadata,
            'n_features': self.n_features
        }
        with open(artifacts_path, 'wb') as f:
            pickle.dump(artifacts, f)

        print(f"  saved FAISS index: {index_path} ({self.index.ntotal:,} vectors)")
        print(f"  saved artifacts: {artifacts_path}")

    @classmethod
    def load(cls, directory):
        """loads a previously saved engine"""
        engine = cls()

        index_path = os.path.join(directory, 'faiss_fraud_index.bin')
        artifacts_path = os.path.join(directory, 'similarity_artifacts.pkl')

        if not os.path.exists(index_path) or not os.path.exists(artifacts_path):
            print(f"  FAISS index not found at {directory}")
            return engine

        if not FAISS_AVAILABLE:
            print("  FAISS not available, cannot load index")
            return engine

        engine.index = faiss.read_index(index_path)

        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)

        engine.scaler = artifacts['scaler']
        engine.city_encoder = artifacts['city_encoder']
        engine.txn_type_encoder = artifacts['txn_type_encoder']
        engine.fraud_metadata = artifacts['fraud_metadata']
        engine.n_features = artifacts['n_features']
        engine.is_fitted = True

        print(f"  loaded FAISS index: {engine.index.ntotal:,} vectors, {engine.n_features}D")
        return engine


def build_and_save_index(df, output_dir):
    """
    convenience function to build the index and save it in one go.
    called from run_pipeline.py as step 6.
    """
    engine = FraudSimilarityEngine()
    engine.fit(df)
    engine.save(output_dir)
    return engine


if __name__ == '__main__':
    # quick standalone test
    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_path = os.path.join(base_dir, 'data', 'processed', 'upi_transactions_processed.csv')

    if os.path.exists(processed_path):
        print(f"loading {processed_path}...")
        df = pd.read_csv(processed_path, parse_dates=['timestamp'])

        # build index
        index_dir = os.path.join(base_dir, 'data', 'faiss_index')
        engine = build_and_save_index(df, index_dir)

        # test a query
        if engine.is_fitted:
            test_txns = df.head(5)
            results = engine.query(test_txns, k=3)
            for i, r in enumerate(results):
                print(f"\ntxn {i}: similarity_score={r['fraud_similarity_score']:.4f}, "
                      f"suspicious={r['is_suspicious']}")
                for fraud in r['similar_frauds']:
                    print(f"  → {fraud['fraud_type']} | ₹{fraud['amount']:,.0f} | "
                          f"dist={fraud['distance']:.3f}")
        print("\ndone.")
    else:
        print("processed data not found. run the main pipeline first.")
