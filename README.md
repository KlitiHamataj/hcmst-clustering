# How Couples Meet and Stay Together — Clustering by Meeting Method

A data science project analyzing Stanford's HCMST 2017–2022 dataset to understand how American couples meet and whether meeting context predicts relationship outcomes. Built as part of a team project where each member explored the data through a different clustering lens.

This repo contains my contribution: clustering 3,295 partnered respondents by **how they met their partner**, then testing whether meeting method predicts relationship survival or COVID resilience.

## Dataset

The [How Couples Meet and Stay Together (HCMST)](https://data.stanford.edu/hcmst2017) study tracked 3,510 US adults across three waves:

- **Wave 1 (2017):** Full sample — demographics, how they met, relationship status and quality
- **Wave 2 (2020):** 2,107 respondents — COVID impact on relationships
- **Wave 3 (2022):** 1,722 respondents — long-term relationship outcomes

The dataset intentionally oversamples LGB respondents (11.4% vs ~5% in the US population). Survey weights are provided to correct for this when making population-level claims.

The raw data is in SPSS `.sav` format and is not included in this repo due to size. Download it from the Stanford link above and place it in `data/raw/`.

## Project Structure

```
├── data/
│   ├── raw/                     # Place .sav file here (gitignored)
│   └── processed/               # Generated CSV and value labels JSON
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   └── 02_clustering.ipynb      # Clustering + COVID + model + conclusions
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load, clean, and label the data
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/KlitiHamataj/hcmst-clustering.git
cd hcmst-clustering
pip install -r requirements.txt
```

Place the `.sav` file in `data/raw/`, then convert it once:

```bash
python src/data_loader.py --convert
```

This creates `data/processed/hcmst_clean.csv` and `data/processed/value_labels.json`.

## What the Code Does

### `src/data_loader.py`

Handles everything related to loading and cleaning the data:

- **`--convert`** — Reads the SPSS `.sav` file using `pyreadstat`, saves a CSV and extracts all value labels (the numeric code to text mappings like `1 = "Male"`) into a JSON file
- **`load_data()`** — Loads the CSV and replaces negative codes (`-1` = Refused, `-2` = Not asked) with NaN
- **`label_column(series, var_name)`** — Converts numeric codes to readable labels using the JSON dictionary

### `notebooks/01_eda_and_cleaning.ipynb`

Exploratory data analysis covering:

- Missing value analysis — 227 columns with >90% missing dropped, leaving ~498 usable columns
- Attrition analysis — who dropped out between waves (younger, less educated, unmarried partners drop more)
- Demographics — age, gender, education, race, region, marital status distributions
- How couples met — proportion for each of 24 meeting channels
- LGB oversample and survey weight distributions
- Summary of key findings for teammates

### `notebooks/02_clustering.ipynb`

The main analysis notebook, structured in four parts:

**Part 1 — Clustering**

- Selected 24 binary "how they met" features from Q24 (summary-level variables only, avoiding double-counting with the detailed R/P/I intermediary variables)
- Filtered to 3,295 respondents who had a partner and answered Q24
- Tested K-Means with k=2 through k=10, evaluated using silhouette scores
- Chose k=6 (silhouette score 0.28, balancing statistical quality with interpretability)
- Visualized with a heatmap of cluster profiles and PCA scatter plot
- Profiled each cluster demographically (age, gender, education, marriage rate, same-sex couple rate, relationship duration)

The six clusters:

| Cluster | Name | Size | Defining Feature | Avg Age | Married |
|---------|------|------|-----------------|---------|---------|
| 0 | School Sweethearts | 327 (10%) | Met at school | 45.5 | 64% |
| 1 | Bar & Social | 624 (19%) | Met at bar/restaurant | 51.1 | 59% |
| 2 | Party Circle | 292 (9%) | Met at private party | 48.7 | 62% |
| 3 | Friend Introductions | 406 (12%) | Introduced by friend | 50.1 | 68% |
| 4 | Workplace | 472 (14%) | Met through work | 51.5 | 71% |
| 5 | Other/Mixed | 1174 (36%) | No dominant channel | 49.0 | 56% |

Online daters (11.8% of the sample) did not form their own cluster. 62% landed in Other/Mixed and 32% in Bar & Social, suggesting online dating in 2017 was used alongside traditional meeting methods rather than as a standalone channel.

**COVID Analysis**

- Connected the 6 clusters to Wave 2 COVID impact data (1,590 respondents)
- 76% reported no change in their relationship, 18% said better, 6% said worse
- Chi-square test: p = 0.16 — not significant
- Meeting method does not predict how the pandemic affected relationships

**Part 2 — Predictive Model**

Research question: What Wave 1 factors predict whether a couple is still together in 2022?

- Target variable: binary survival (still partnered in Wave 3 vs not)
- 1,332 respondents (partnered at Wave 1 and responded to Wave 3)
- 17 features: 6 demographic, 5 relationship characteristics, 5 meeting methods, plus cluster label
- Random Forest classifier with balanced class weights
- Baseline: 91.5% (always predict "still together")
- Model accuracy: 91.4% — did not beat baseline
- The model struggled because only 8.5% of couples broke up, leaving too few examples to learn from
- Feature importance revealed: relationship duration, age, income, and marital status matter most. All "how they met" features scored below 0.02

**Part 3 — Conclusions**

Recommendations for OKCupid:

1. Online couples survive at the same rate as everyone else — use this in marketing
2. Meeting method doesn't predict outcomes — focus product effort on post-match relationship quality
3. Workplace and friend-intro clusters have the highest marriage rates but the lowest online presence — target users who recently lost access to these channels (job changes, city moves)
4. Online daters aren't a separate demographic — they look like everyone else

## Limitations

- **Survivorship bias:** only relationships existing in 2017 are studied
- **Attrition:** 51% dropped out by Wave 3, non-randomly — inflates survival rates
- **Historical data:** most relationships started years before dating apps — the 5.9% internet dating rate underestimates current rates
- **Correlation not causation:** relationship duration predicts survival, but that's partly tautological
- **Class imbalance:** only 117 breakups out of 1,371 cases made the prediction task very difficult

## Tools

- Python 3.10+
- pandas, numpy — data handling
- scikit-learn — K-Means, Random Forest, PCA, silhouette scores
- matplotlib, seaborn — visualization
- scipy — chi-square test
- pyreadstat — reading SPSS files