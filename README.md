# Ensemble Plumes

ECMWF ensemble forecast visualization scripts for snowfall prediction at North Peak, Keystone, CO.

## Scripts

**Production (App Environment):**
- `ecmwf_ensemble.py` - ECMWF IFS 0.25° model
- `ecmwf_aifs_ensemble.py` - ECMWF AIFS 0.25° model

**Testing:**
- `test_ifs.py` - Test IFS model
- `test_aifs.py` - Test AIFS model
- `test_ensemble_scripts.py` - Test both models

## Usage

```bash
# Production
python ecmwf_ensemble.py
python ecmwf_aifs_ensemble.py

# Testing
python test_ifs.py
python test_aifs.py
```

## Configuration

Copy `.env.example` to `.env` and update values if needed. Defaults work for North Peak location.

## Dependencies

- Python 3.9+
- pandas, numpy, matplotlib, seaborn, requests

## Output

Production scripts save images to `app/static/ecmwf_images/`. Test scripts save to `test_output/`.
