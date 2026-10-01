install:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

lint:
	flake8 src tests --max-line-length=100

test:
	pytest -v

train:
	python -m src.train

clean:
	python -c "import pathlib, shutil; [p.unlink() for d in ('src','tests') for p in pathlib.Path(d).rglob('*.pyc')]; [shutil.rmtree(p, ignore_errors=True) for d in ('src','tests') for p in pathlib.Path(d).rglob('__pycache__')]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
