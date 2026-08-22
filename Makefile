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

build:
	npm --prefix service run compile-base

kill:
	-tmux kill-server 2>/dev/null
	-killall -9 python 2>/dev/null
	-killall -9 node 2>/dev/null
	-pkill -9 -f "dist/server/index.js" 2>/dev/null
	-for p in 8080 8081; do fuser -k -9 $$p/tcp 2>/dev/null; done

attach:
	tmux attach -t train

ensemble:
	for k in 0 1 2 3; do python -m rl.offline.train --ensemble-index $k --num-steps 10000; done