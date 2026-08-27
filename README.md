# When Autism Isn't Obvious

An explainable machine-learning exploration of reported autism profiles associated with earlier or later diagnosis, using the 2022–2024 National Survey of Children's Health (NSCH).

## Live dashboard

[Explore the interactive Streamlit dashboard](https://autism-signs.streamlit.app/)

## Central finding

K-modes clustering identified three reported profiles without using diagnosis age:

- Developmentally apparent: median diagnosis age 3; 28.7% after age 4
- Complex multi-condition: median age 4; 43.8% after age 4
- Less-obvious overlapping: median age 7; 73.9% after age 4

The less-obvious overlapping profile combined mild reported severity, relatively little reported speech/developmental delay, and more ADHD/anxiety. Its later-diagnosis pattern appeared in every survey year. These are exploratory associations, not validated clinical subtypes or a prospective screening tool.

## Cohort definition

- Caregiver reported that the child had ever been diagnosed with autism (`K2Q35A = Yes`)
- Valid reported age at first diagnosis (`K2Q35A_1_YEARS`)
- Child was at least age 5 at survey, ensuring an opportunity to experience the outcome
- Outcome: first reported diagnosis after age 4

Eligible clustering sample: **4,759 children** with a valid reported severity category.

## Run locally

1. Download the official 2022, 2023, and 2024 NSCH Topical Stata datasets from the Census Bureau NSCH data pages.
2. Extract the 2024 file into `data/` and the 2022–2023 files into `data/multiyear/`.
3. Install dependencies and run:

```bash
pip install -r requirements.txt
python cluster_analysis.py
streamlit run app.py
```

## Responsible interpretation

This project examines population-level patterns. It does not diagnose autism, estimate an individual child's needs, or establish causation. Reported severity and co-occurring conditions were measured at survey, not necessarily before autism diagnosis; the profiles therefore cannot be used as a prospective screening tool. Full NSCH variance estimation should use the complex survey design variables and appropriate survey-analysis software.

## Suggested LinkedIn story

Lead with the human finding: the children who are easiest to miss may wait the longest for answers. Show the three profiles and reveal that the less-obvious overlapping profile had a median diagnosis age four years later than the developmentally apparent profile.
