# loopsmith

A CLI coding agent built from scratch in Python, with no dependencies
beyond the standard library.

Give it a task inside a project folder. It reads the code, edits files,
runs commands and tests, and keeps going until the task is done. It asks
before every change. A session looks like this (illustrative):

```
$ loopsmith "The tests are failing. Find the bug and fix it."
  -> list_files()  [48 chars]
  -> run_command(command='pytest')  [912 chars]
  -> read_file(path='stats.py')  [861 chars]

Edit file stats.py
--- a/stats.py
+++ b/stats.py
@@ -17,3 +17,3 @@
     if len(ordered) % 2 == 1:
         return ordered[middle]
-    return (ordered[middle] + ordered[middle + 1]) / 2
+    return (ordered[middle - 1] + ordered[middle]) / 2
Allow? [y/N] y
  -> edit_file(path='stats.py', ...)  [16 chars]
  -> run_command(command='pytest')  [164 chars]
Fixed the median calculation for even-length lists. All 5 tests pass.
```

## How it works

The model cannot run anything. It can only ask for a tool to be run.
The agent loop does the rest:

1. Send the conversation and the list of tools to the model.
2. If the model replies with text only, the task is done.
3. If it asks for tools, run them and add the results to the
   conversation.
4. Repeat.

| Tool | What it does | Needs approval |
|---|---|---|
| `list_files` | List a folder | No |
| `read_file` | Read a text file | No |
| `edit_file` | Replace one exact piece of text in a file | Yes |
| `write_file` | Create or overwrite a file | Yes |
| `run_command` | Run a shell command | Yes |

## Safety

- Every file change and command is shown and needs a `y` before it runs.
  `--yes` approves everything, for use inside a sandbox.
- File tools cannot reach outside the folder the agent was started in,
  and refuse `.env` files and the `.git` folder.
- Commands have a 60-second limit, cannot read input, and do not
  receive the API key.
- A run stops after 20 model calls (`--max-steps`).

An approved command still runs with your full permissions. Review what
you approve, and do not use `--yes` outside an isolated environment.

## Setup

You need Python 3.11 or newer and git.

**1. Get the code**

```
git clone https://github.com/Saketh1304/loopsmith.git
cd loopsmith
```

**2. Create a virtual environment and install**

Mac or Linux:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Windows (PowerShell):

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

**3. Add your API key**

Create a free key at https://aistudio.google.com. Copy `.env.example`
to a new file called `.env` and put the key after `GEMINI_API_KEY=`.
The `.env` file is ignored by git.

To use loopsmith in any folder on your machine, put the same line in a
file called `.loopsmith.env` in your home folder.

**4. Check that it works**

```
pytest
loopsmith "What does this project do?"
```

If you get a "model not found" error, run `loopsmith --list-models` and
set `GEMINI_MODEL` in `.env` to one of the names it prints.

## Try it on a real bug

```
cd examples/buggy_stats
pytest
loopsmith "The tests are failing. Find the bug and fix it."
```

Run `git restore .` in that folder to put the bug back.

## Project layout

```
src/loopsmith/
  agent.py       the agent loop and approval
  cli.py         command-line entry point
  config.py      API key and model settings
  llm/           the model layer: neutral types, Gemini client, retries
  tools/         the actions the agent can take
tests/           automated tests
examples/        a small buggy project to try the agent on
docs/            design decisions and interview notes
```

## Design notes

The reasoning behind each design choice is in
[docs/decisions.md](docs/decisions.md).
