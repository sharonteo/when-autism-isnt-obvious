"""Discover reported autism profiles without using diagnosis age in clustering."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from sklearn.metrics import adjusted_rand_score, silhouette_score


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FILES = {
    "2022": ROOT / "data/multiyear/nsch_2022e_topical.dta",
    "2023": ROOT / "data/multiyear/nsch_2023e_topical.dta",
    "2024": ROOT / "data/nsch_2024e_topical.dta",
}
FEATURES = {
    "Severity": "k2q35c",
    "ADHD": "k2q31a",
    "Learning disability": "k2q30a",
    "Speech disorder": "k2q37a",
    "Intellectual disability": "k2q60a",
    "Developmental delay": "k2q36a",
    "Behavior problems": "k2q34a",
    "Depression": "k2q32a",
    "Anxiety": "k2q33a",
}


def weighted_mean(values, weights):
    return float(np.average(values, weights=weights))


def weighted_median(values, weights):
    z = pd.DataFrame({"value": values, "weight": weights}).dropna().sort_values("value")
    return float(z.loc[z.weight.cumsum() >= z.weight.sum() / 2, "value"].iloc[0])


def load_cohort():
    parts = []
    keep = ["k2q35a", "k2q35a_1_years", "sc_age_years", "fwc", *FEATURES.values()]
    for year, path in FILES.items():
        df = pd.read_stata(path, convert_categoricals=False)
        part = df.loc[
            (df.k2q35a == 1)
            & df.k2q35a_1_years.between(1, 15)
            & (df.sc_age_years >= 5), keep
        ].copy()
        part["survey_year"] = year
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def encode(cohort):
    cohort = cohort[cohort.k2q35c.isin([1, 2, 3])].reset_index(drop=True)
    encoded = pd.DataFrame(index=cohort.index)
    encoded["Severity"] = cohort.k2q35c.astype(int)
    for name, column in list(FEATURES.items())[1:]:
        encoded[name] = np.where(cohort[column] == 1, 1, np.where(cohort[column] == 2, 0, -1))
    return cohort, encoded


def name_profiles(cohort, encoded):
    prevalence = {}
    for cluster in sorted(cohort.cluster.unique()):
        rows = cohort.cluster == cluster
        prevalence[cluster] = {
            name: weighted_mean((encoded.loc[rows, name] == 1).astype(int), cohort.loc[rows, "fwc"])
            for name in list(FEATURES)[1:]
        }
    # Names use reported feature composition only; diagnosis age is never used.
    complex_cluster = max(prevalence, key=lambda c: np.mean(list(prevalence[c].values())))
    remaining = [c for c in prevalence if c != complex_cluster]
    apparent_cluster = max(
        remaining,
        key=lambda c: prevalence[c]["Speech disorder"] + prevalence[c]["Developmental delay"],
    )
    subtle_cluster = next(c for c in remaining if c != apparent_cluster)
    return {
        apparent_cluster: "Developmentally apparent",
        subtle_cluster: "Less-obvious overlapping",
        complex_cluster: "Complex multi-condition",
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    cohort, encoded = encode(load_cohort())
    model = KModes(n_clusters=3, init="Huang", n_init=20, random_state=42)
    cohort["cluster"] = model.fit_predict(encoded.to_numpy())
    names = name_profiles(cohort, encoded)
    cohort["profile"] = cohort.cluster.map(names)
    cohort["later_diagnosis"] = (cohort.k2q35a_1_years > 4).astype(int)

    profile_rows, feature_rows, year_rows = [], [], []
    for profile, part in cohort.groupby("profile", observed=True):
        profile_rows.append({
            "profile": profile,
            "sample_n": len(part),
            "weighted_later_pct": 100 * weighted_mean(part.later_diagnosis, part.fwc),
            "weighted_median_diagnosis_age": weighted_median(part.k2q35a_1_years, part.fwc),
        })
        for feature in list(FEATURES)[1:]:
            rows = part.index
            feature_rows.append({
                "profile": profile,
                "feature": feature,
                "weighted_prevalence_pct": 100 * weighted_mean(
                    (encoded.loc[rows, feature] == 1).astype(int), part.fwc
                ),
            })
        for year, year_part in part.groupby("survey_year", observed=True):
            year_rows.append({
                "profile": profile,
                "survey_year": year,
                "sample_n": len(year_part),
                "weighted_later_pct": 100 * weighted_mean(year_part.later_diagnosis, year_part.fwc),
            })

    pd.DataFrame(profile_rows).sort_values("weighted_median_diagnosis_age").to_csv(
        RESULTS / "cluster_profile_summary.csv", index=False
    )
    pd.DataFrame(feature_rows).to_csv(RESULTS / "cluster_feature_summary.csv", index=False)
    pd.DataFrame(year_rows).to_csv(RESULTS / "cluster_year_summary.csv", index=False)

    sample = encoded.sample(n=min(2000, len(encoded)), random_state=42)
    silhouette = silhouette_score(
        sample.to_numpy(), cohort.loc[sample.index, "cluster"], metric="hamming"
    )
    seed_labels = [
        KModes(n_clusters=3, init="Huang", n_init=5, random_state=seed).fit_predict(encoded.to_numpy())
        for seed in (7, 21, 42)
    ]
    stability = [
        adjusted_rand_score(seed_labels[i], seed_labels[j])
        for i in range(3) for j in range(i + 1, 3)
    ]
    audit = {
        "eligible_records_with_valid_severity": len(cohort),
        "clustering_method": "K-modes; three clusters; simple-matching dissimilarity",
        "diagnosis_age_used_to_form_clusters": False,
        "sample_silhouette_hamming": silhouette,
        "mean_seed_stability_adjusted_rand_index": float(np.mean(stability)),
        "interpretation": "Exploratory reported profiles, not validated clinical subtypes.",
    }
    (RESULTS / "cluster_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(pd.DataFrame(profile_rows).sort_values("weighted_median_diagnosis_age").to_string(index=False))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
