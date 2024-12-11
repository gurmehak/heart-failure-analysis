.PHONY: all clean

all: report/heart_failure_analysis.html report/heart_failure_analysis.pdf

# Download and convert data
data/raw/heart_failure_clinical_records_dataset.csv: scripts/download_and_convert.py
	python scripts/download_and_convert.py \
		--url="https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip" \
		--write_to=data/raw

# Process and analyze data
data/processed/heart_failure_train.csv data/processed/heart_failure_test.csv: scripts/process_and_analyze.py data/raw/heart_failure_clinical_records_dataset.csv
	python scripts/process_and_analyze.py \
		--file_path=data/raw/heart_failure_clinical_records_dataset.csv \
		--output_dir=data/processed

# Perform correlation analysis
results/figures/correlation_heatmap.png: scripts/correlation_analysis.py data/processed/heart_failure_train.csv data/processed/heart_failure_test.csv
	python scripts/correlation_analysis.py \
		--train_file=data/processed/heart_failure_train.csv \
		--test_file=data/processed/heart_failure_test.csv \
		--output_file=results/figures/correlation_heatmap.png

# Train and evaluate the model
results/models/pipeline.pickle results/figures/training_plots: scripts/modelling.py data/processed/heart_failure_train.csv
	python scripts/modelling.py \
		--training-data=data/processed/heart_failure_train.csv \
		--pipeline-to=results/models \
		--plot-to=results/figures \
		--seed=123

results/tables/test_evaluation.csv: scripts/model_evaluation.py data/processed/heart_failure_test.csv results/models/pipeline.pickle
	python scripts/model_evaluation.py \
		--scaled-test-data=data/processed/heart_failure_test.csv \
		--pipeline-from=results/models/pipeline.pickle \
		--results-to=results/tables

# Build HTML and PDF reports
report/heart_failure_analysis.html report/heart_failure_analysis.pdf: report/heart_failure_analysis.qmd \
results/models/pipeline.pickle \
results/figures/correlation_heatmap.png \
results/figures/training_plots \
results/tables/test_evaluation.csv
	quarto render report/heart_failure_analysis.qmd --to html
	quarto render report/heart_failure_analysis.qmd --to pdf

# Clean up analysis
clean:
	rm -rf data/raw/*
	rm -f data/processed/heart_failure_train.csv \
		data/processed/heart_failure_test.csv \
		results/models/pipeline.pickle \
		results/figures/correlation_heatmap.png \
		results/figures/training_plots \
		results/tables/test_evaluation.csv \
		report/heart_failure_analysis.html \
		report/heart_failure_analysis.pdf
