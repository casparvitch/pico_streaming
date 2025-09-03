


format:
	ruff format
	ruff check --fix --select F,B,I .
	ruff format

lint:
	prospector --with-tool mypy

complexity:
	complexipy --sort asc .

complexity-json:
	complexipy --sort desc --output-json .


