# MGM Hackathon · 24–25 Oct 2025

## About the Project
Invoice Analyzer is a Gradio-based chat assistant that ingests uploaded PDF receipts, runs a LangChain/LangGraph agent over them, and returns concise, structured findings plus downloadable artifacts.

## Branch History
This branch continues development after the hackathon deadline; checkout `state-at-end-of-hackathon` to see exactly what shipped by the end of day 2.

## Data
Download the dataset from `https://mgmbox.mgm-tp.com/index.php/s/ZuDdW5C2J1za4HD?path=%2Fuse-case-1` or `https://www.kaggle.com/datasets/jenswalter/receipts`, then extract it into `./data` (ignored by git):

```bash
cd data
wget -q https://www.kaggle.com/api/v1/datasets/download/jenswalter/receipts
unzip pdf_receipt.zip
```

## Run
```bash
uv run mgm-hackathon
```

## Tests
```bash
uv sync --extra dev
uv run --extra dev pytest
```
