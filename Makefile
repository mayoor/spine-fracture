TMPDIR=/tmp
TMPDIR:=/tmp

envset:
	conda activate spine-kaggle
lab:
	JUPYTER_DATA_DIR=$(TMPDIR) jupyter lab --NotebookApp.token='' --NotebookApp.password='' --no-browser
