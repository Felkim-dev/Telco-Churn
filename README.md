# Telco Churn — retention assistant

Machine Learning, Task 6 (Yachay Tech): hyperparameter optimization, multi-objective optimization,
explainable AI and a business proposal on the IBM Telco Customer Churn dataset.

**Live app:** https://telco-churn-v5dc.onrender.com/
(free Render instance: it sleeps after 15 min without use, the first visit can take ~1 min to wake up)

## What the app does

The app only **loads** the exported pipeline (`model.joblib`) and the results in `artifacts/`; it never retrains.

| Tab | Content |
|---|---|
| EDA | Filters (contract, internet service, tenure), 3 KPIs and 3 charts of who stays and who leaves |
| Model performance | CV comparison of models, Grid / Random / TPE search, Pareto front with the chosen point, metrics and confusion matrix on test |
| Prediction & explainability | Form for one customer → probability of leaving, plus SHAP and LIME bars for that same customer |
| Business | The proposal and a scenario simulator (hypothetical numbers, declared as such) |

## Results (real, from the notebook)

**Model comparison** (5-fold stratified CV on train): Logistic Regression recall 0.543 vs Random Forest 0.479.
Logistic Regression was kept as the base model, because a missed churner (FN) costs the company more than an
unnecessary offer (FP), and the dataset is imbalanced (26.5 % churn), so accuracy is misleading.

**Hyperparameter search** (same pipeline, same folds, F1, 24 configurations each):

| Method | Best CV F1 | Time (s) |
|---|---|---|
| Grid | 0.633 | 27.9 |
| Random | 0.633 | 12.2 |
| TPE (Optuna) | 0.634 | 28.1 |

The three tie: the space has only one continuous hyperparameter (`C`).

**Multi-objective search** (NSGA-II, recall vs precision): the chosen point is row 31 of the Pareto front
(`class_weight` for churn ≈ 5.5, `C` ≈ 1.1, `l1`), the one with the lowest total cost `12·FN + FP`
(the 12× ratio is a **hypothesis**, see below).

**Final model on test** (test used once, after every decision):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.669 | 0.440 | 0.914 | 0.594 | 0.841 |

Recall on test equals recall in CV (0.914), so the detection rate holds for new customers.

**Explainability:** SHAP and LIME agree on the most relevant features (`tenure`, fiber-optic internet,
`MonthlyCharges`); only the order changes. These are associations learned by the model, not causes.

## Business proposal (summary)

- **Problem:** the 1,869 customers who left in the last month paid **$139,131/month** (real figure from the
  dataset); that revenue is lost every month after.
- **Product:** a monthly list of at-risk customers (detects 91 % of those who leave, precision 44 %) with the
  reasons behind each one, in an app anyone in the company can use.
- **Buyer:** subscription telecom companies with similar data (internet, contract, tenure, charges). Signed by
  whoever runs finance and the business.
- **Value:** offers only to a selected group, personalised to what affects each customer, instead of a
  discount for everyone.
- **Pricing:** **15 % (hypothetical)** of what is really saved, measured against a random half of the at-risk
  customers who receive no offer (control group).

**Hypothetical numbers** (not model results): an FN costs 12 times an FP; churn of 40 % without offer vs 25 % with
offer in the simulator; the 15 % fee.

## Limitations

- Trained on one telecom's data: another company needs its own data, and another kind of business (gym, streaming)
  needs a new model.
- A new service means retraining.
- SHAP shows what pushes a customer to leave, not what will fix their problem.

## Run locally

Python 3.13 (the version used to train `model.joblib`; another scikit-learn version may fail to load it).

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or with Docker: `docker build -t telco-churn . && docker run -p 7860:7860 telco-churn` → http://localhost:7860

## Files

| Path | What |
|---|---|
| `telco_churn.ipynb` | Full notebook: data prep, models, CV, Grid/Random/TPE, NSGA-II, SHAP, LIME, findings |
| `app.py` | Streamlit app |
| `model.joblib` | Final pipeline (preprocessing + Logistic Regression) |
| `artifacts/` | CV results, searches, Pareto front, test predictions |
| `data/` | Telco Customer Churn dataset ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) |
| `Dockerfile` | Image used for the Render deploy |
