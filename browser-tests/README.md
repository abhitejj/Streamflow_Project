# Local browser checks

Run from the repository root with Python Playwright and its Chromium browser already installed.

```sh
python3 -m http.server 8000 --directory site
```

In a second terminal, run the checks against that server.

```sh
python3 browser-tests/verify_site.py --output browser-tests/artifacts/local
```

The test checks the served HTML and assets, scrolls through the charts, and exercises the linked controls on desktop and mobile. The output directory holds screenshots and `report.json`. `browser-tests/artifacts/` is ignored by Git.

To regenerate and check the comparison data from the existing audited CSVs, run these from the repository root.

```sh
python3 scripts/export_explorer_data.py
python3 scripts/export_explorer_data.py --out site/assets/explorer-data.js
python3 browser-tests/verify_explorer_data.py
```

Rebuild and check the historical replay from the saved forecast and threshold files.

```sh
python3 scripts/export_replay_data.py
python3 site/tools/build_site_data.py
python3 browser-tests/verify_replay_data.py
```

The comparison exporter also verifies and packages the saved uncertainty intervals and guided examples. It rejects station intervals whose rounded NSE difference or scored-week count differs from the current audit.

```sh
python3 browser-tests/verify_uncertainty.py
```
