datas: 
	sh scripts/make_data.sh

protos:
	sh scripts/compile_protos.sh

lint:
	sh scripts/lint.sh

ignore:
	cp .gitignore .dockerignore

clean:
	find . -type d -name "__pycache__" -print -exec rm -r {} +

kill:
	-tmux kill-server 2>/dev/null
	-killall -9 python 2>/dev/null
	-killall -9 node 2>/dev/null

build:
	sudo docker build -t porygon2:latest .