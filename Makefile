data:
	python3 scripts/data_prep.py

train: data
	python3 scripts/train_model.py

evaluate: train
	python3 scripts/evaluate_model.py

deploy: train
	python3 scripts/deploy_model.py

