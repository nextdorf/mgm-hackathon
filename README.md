# MGM Hackathon 24. - 25. Okt, 2025

## Data
Download the data from `https://mgmbox.mgm-tp.com/index.php/s/ZuDdW5C2J1za4HD?path=%2Fuse-case-1` or `https://www.kaggle.com/datasets/jenswalter/receipts`. Prefearably extract to `./data` as that folder will be ignored by git.

```bash
cd data
wget -q https://www.kaggle.com/api/v1/datasets/download/jenswalter/receipts
unzip pdf_receipt.zip
```

## Run
```bash
uv run mgm-hackathon
```

