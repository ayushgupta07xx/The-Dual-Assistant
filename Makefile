.PHONY: help install demo cli chat-oss eval eval-quick sample report test clean

help:
	@echo "Targets:"
	@echo "  install     pip install -r requirements.txt"
	@echo "  demo        run the Streamlit dual-assistant UI"
	@echo "  cli         chat in the terminal (frontier backend)"
	@echo "  chat-oss    chat in the terminal (oss backend)"
	@echo "  eval        run the full 3-arm evaluation (needs API key + OSS backend)"
	@echo "  eval-quick  run eval on a 3-item slice for a fast smoke test"
	@echo "  sample      regenerate the illustrative sample_results.json"
	@echo "  report      build the one-page PDF report + metrics.png"
	@echo "  test        run unit tests (no API key / model required)"
	@echo "  clean       remove caches and logs"

install:
	pip install -r requirements.txt

demo:
	streamlit run app/streamlit_app.py

cli:
	python cli.py --backend frontier

chat-oss:
	python cli.py --backend oss

# Real evaluation. Writes eval/results.json. Requires ANTHROPIC_API_KEY and a
# working OSS backend (transformers/ollama/endpoint).
eval:
	python -m eval.run

eval-quick:
	python -m eval.run --limit 3

sample:
	python -m eval.make_sample

# By default builds from the illustrative sample. Point at eval/results.json
# after a real run:  make report RESULTS=eval/results.json
RESULTS ?= eval/sample_results.json
report:
	python -m report.generate --results $(RESULTS)

test:
	pytest -q

clean:
	rm -rf **/__pycache__ .pytest_cache logs *.db
	find . -name '*.pyc' -delete
