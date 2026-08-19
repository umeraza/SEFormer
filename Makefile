.PHONY: install test lint audit smoke clean

install:
	python -m pip install -e '.[face,dev]'

test:
	python -m unittest discover -s tests -v

lint:
	ruff check src scripts tests

audit:
	python scripts/audit_paper.py --config configs/daisee.yaml

smoke:
	python scripts/train.py --config configs/smoke.yaml

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('.').rglob('*.pyc')]"
