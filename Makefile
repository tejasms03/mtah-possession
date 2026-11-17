# Single-threaded numerics so the submission file is byte-identical everywhere.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

TEAM ?= team_name
OUT  := outputs/$(TEAM).csv

.PHONY: setup test submission validate eval verify

setup:            ## install the pinned environment from uv.lock
	uv sync --frozen

test:
	uv run pytest -q

submission:       ## run the full pipeline -> outputs/$(TEAM).csv
	uv run possession run --out $(OUT)
	uv run possession validate $(OUT)

validate:
	uv run possession validate $(OUT)

eval:             ## score against the public 5-minute GT with the organisers' scorer
	cp $(OUT) vendor/mtah/evaluation/results/$(TEAM).csv
	cd vendor/mtah && uv run --project ../.. python evaluation/run_evaluation.py

verify:           ## regenerate and compare against the recorded hash
	uv run possession run --out outputs/verify.csv
	shasum -a 256 outputs/verify.csv | cut -d' ' -f1 | diff - SUBMISSION_SHA256 && echo "byte-identical"
