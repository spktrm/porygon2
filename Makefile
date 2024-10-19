datas: 
	sh scripts/make_data.sh

protos:
	sh scripts/compile_protos.sh

lint:
	sh scripts/lint.sh

clean:
	find . -type d -name "__pycache__" -print -exec rm -r {} +


