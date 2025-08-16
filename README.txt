
# inas-collated-hotfix-pricing

This patch fixes TypeErrors caused by yfinance returning a DataFrame for `Close`.

Includes:
- `hotfix_ma.py` (helpers)
- `app.py` (patched to use `to_series_close` in `get_last_price` and alerts)

## Use
1) Replace your current `app.py` with the one in this zip and add `hotfix_ma.py` alongside it.
2) Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```
