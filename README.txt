
# inas-collated-hotfix-ma (v2)

Includes:
- `hotfix_ma.py` (forgiving MA helper)
- `app.py` (patched app wired to use the helper)

## Use
1) Replace your current `app.py` with the one in this bundle and add `hotfix_ma.py` alongside it.
2) Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```
