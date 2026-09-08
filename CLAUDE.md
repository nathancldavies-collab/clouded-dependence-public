# Running code
- The project uses `uv` for management. Always use `uv run <python-file>.py` to run scripts or `uv run python -c ...` to execute python. 
- The code should be linted using `just lint {{files}}`. The repo has legacy code, so we don't do it all at once - just the files you work on. 