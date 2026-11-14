# loopsmith

A CLI coding agent built from scratch in Python.

> Work in progress. It can talk to a model; the agent loop is being built.

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

Copy `.env.example` to a new file called `.env`, then put your key after
`GEMINI_API_KEY=`. The `.env` file is ignored by git.

**4. Check that it works**

```
pytest
loopsmith "say hello"
```

`pytest` should report all tests passing, and `loopsmith` should print
the model's reply.

If you get a "model not found" error, run `loopsmith --list-models` and
set `GEMINI_MODEL` in `.env` to one of the names it prints.

## Project layout

```
src/loopsmith/   the agent's code
tests/           automated tests
docs/            design decisions and notes
```
