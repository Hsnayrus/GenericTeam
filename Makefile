PYTHON ?= python3
PIP ?= pip
MLX_MODEL ?= mlx-community/gemma-3-1b-it-4bit
MLX_ADAPTER_PATH ?=
RAW_SESSION ?= data/raw_sessions/session.offline.json
CLEAN_SESSION ?= data/results/session.cleaned.json
SESSION_VIDEO ?= data/raw_videos/session.webm
COMPARE_OUTPUT ?= data/results/session_compare.json
BASELINE_OUTPUT ?= data/results/session_baseline.json
REP ?= 3
MIN_CONFIDENCE ?= 0.3
TEACHER_COUNT ?= 200
TEACHER_BATCH_SIZE ?= 50
TEACHER_OUTPUT ?= data/gemini_teacher_dataset.jsonl
TEACHER_RAW_OUTPUT ?= data/gemini_teacher_raw.json
GEMINI_MODEL ?= gemini-2.5-pro
GEMINI_REVIEW_OUTPUT ?= data/results/gemini_rep_review.json

.PHONY: install setup serve dirs clean-session baseline-summary eval-rules eval-mlx compare-session baseline teacher-dry-run teacher-generate review-rep

install:
	$(PIP) install -r requirements.txt
	npm install

setup:
	bash setup.sh

serve:
	$(PYTHON) server.py

dirs:
	mkdir -p data/raw_videos data/raw_sessions data/results data/extracted_frames

clean-session: dirs
	$(PYTHON) clean_offline_session.py \
		--input $(RAW_SESSION) \
		--output $(CLEAN_SESSION) \
		--min-confidence $(MIN_CONFIDENCE)

baseline-summary:
	$(PYTHON) summarize_session_baseline.py \
		--input $(CLEAN_SESSION) \
		--output $(BASELINE_OUTPUT)

eval-rules:
	$(PYTHON) evaluate_coach_dataset.py --backend rules

eval-mlx:
	$(PYTHON) evaluate_coach_dataset.py --backend mlx --model $(MLX_MODEL)

compare-session: dirs
	$(PYTHON) compare_session_models.py \
		--session-json $(CLEAN_SESSION) \
		--video $(SESSION_VIDEO) \
		--rep $(REP) \
		--models $(MLX_MODEL) \
		--mode pose \
		--output $(COMPARE_OUTPUT)

baseline: clean-session baseline-summary eval-rules eval-mlx compare-session

teacher-dry-run:
	$(PYTHON) generate_gemini_dataset.py \
		--model $(GEMINI_MODEL) \
		--count $(TEACHER_COUNT) \
		--batch-size $(TEACHER_BATCH_SIZE) \
		--output $(TEACHER_OUTPUT) \
		--raw-output $(TEACHER_RAW_OUTPUT) \
		--dry-run

teacher-generate:
	@test -n "$$GEMINI_API_KEY" || (echo "GEMINI_API_KEY is not set" && exit 1)
	$(PYTHON) generate_gemini_dataset.py \
		--model $(GEMINI_MODEL) \
		--count $(TEACHER_COUNT) \
		--batch-size $(TEACHER_BATCH_SIZE) \
		--output $(TEACHER_OUTPUT) \
		--raw-output $(TEACHER_RAW_OUTPUT)

review-rep:
	@test -n "$$GEMINI_API_KEY" || (echo "GEMINI_API_KEY is not set" && exit 1)
	$(PYTHON) review_rep_with_gemini.py \
		--model $(GEMINI_MODEL) \
		--session-json $(CLEAN_SESSION) \
		--video $(SESSION_VIDEO) \
		--rep $(REP) \
		--output $(GEMINI_REVIEW_OUTPUT)
