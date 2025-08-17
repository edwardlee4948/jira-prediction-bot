data:
	uv run /workspaces/jira-prediction-bot/experiments/generate_synthetic_jira.py --rows 1000 --feedback-ratio 0.05 --output experiments/jira_data/synthetic.csv --events-output experiments/jira_data/synthetic_events.csv --feedback-output experiments/jira_data/synthetic_feedback.csv --write-split --split-test-size 0.2 --split-seed 42 --split-dir experiments/jira_data
model: data
	uv run /workspaces/jira-prediction-bot/experiments/analyze_transactions.py --issues experiments/jira_data/synthetic.csv --events experiments/jira_data/synthetic_events.csv --feedback experiments/jira_data/synthetic_feedback.csv --save-model --split-index experiments/jira_data/split_index.csv 
vis: model
	uv run python /workspaces/jira-prediction-bot/experiments/visualize.py --plot-events --sample 50 --only-test --split-index experiments/jira_data/split_index.csv --annotate-pred
re:
	rm -rf experiments/jira_data/*.csv experiments/models/*.joblib experiments/plots/events/*.png experiments/plots/*.png	
.PHONY: data model vis re