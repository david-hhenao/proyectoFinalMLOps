# Makefile
PY            ?= python
vPY           ?= .venv/bin/python
pPY           ?= .venv/bin/pip
_PY		   	  ?= .venv/bin/
MLRUNS_DIR    ?= mlruns
export MLFLOW_TRACKING_URI ?= file://$(PWD)/$(MLRUNS_DIR)

.PHONY: install format format-check train validate evaluate eval ci cd clean

.venv/pyvenv.cfg:
	$(PY) -m venv .venv

clean:
	rm -rf $(MLRUNS_DIR)
	rm -rf .venv
	rm data/validation.csv
	rm -rf pkl

install:
	$(PY) -m venv .venv
# 	source .venv/bin/activate
# 	.venv/bin/pip install --upgrade pip
	$(pPY) install --upgrade pip
	$(pPY) install -r requirements.txt

format:
	$(_PY)isort --profile black .
	$(_PY)black .

format-check:
	$(_PY)isort --profile black --check-only .
	$(_PY)black --check .

train:
	mkdir -p $(MLRUNS_DIR)
# 	.venv/bin/python src/train.py
	$(vPY) src/train.py


test:
# 	.venv/bin/python src/validate.py
	$(vPY) src/validate.py


evaluate: test
eval: evaluate

ci: clean install format format-check train test
ci_sc: install format format-check train test

cd:
	@echo "mlflow models serve -m $(MLRUNS_DIR)/<run>/artifacts/model -p 8000 --no-conda"

# Asi funciono:

# export MLFLOW_TRACKING_URI="file://$PWD/ruta/mlruns"
# mlflow models serve -m "runs:/36ac30daa98b4e289535a94bd41b8d06/model" -p 5001 --env-manager local #36ac30daa98b4e289535a94bd41b8d06: revisar el id del run

# PERO ANTES CORRER LOCAL :

# make clean
# make install
# make format-check
# make train
# make evaluate
