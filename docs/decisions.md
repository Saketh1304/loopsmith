# Decision log

Each entry records a design choice, the alternatives, and the reason.

## 1. Python, not TypeScript

- **Choice:** build the agent in Python 3.11+.
- **Alternatives:** TypeScript.
- **Why:** AI engineering roles and evaluation tooling are mostly Python.
  The design is language-neutral, so the concepts carry over.

## 2. `src/` layout with `pyproject.toml`

- **Choice:** code lives in `src/loopsmith/`, described by one
  `pyproject.toml`.
- **Alternatives:** a flat folder of scripts; `setup.py`;
  `requirements.txt` only.
- **Why:** `pyproject.toml` is the current standard, similar in role to
  Maven's `pom.xml`. The `src/` layout forces tests to run against the
  installed package, which catches packaging mistakes early.

## 3. API key from the environment, with a `.env` file for development

- **Choice:** read `GEMINI_API_KEY` from environment variables. A local
  `.env` file is loaded at startup, and the real environment wins over it.
- **Alternatives:** hard-code the key; pass it as a command-line flag;
  a config file in the repo.
- **Why:** secrets must never reach git. Flags leak into shell history.
  Environment variables are how deployed services receive secrets, so
  the same code works locally and in production.

## 4. No dependencies yet, including for `.env` loading

- **Choice:** a small hand-written `.env` loader instead of the
  `python-dotenv` package.
- **Alternatives:** `python-dotenv`.
- **Why:** it is about fifteen lines, and each dependency is something to
  install, update and trust. Dependencies get added when they earn
  their place.

## 5. Tests and CI from the first commit

- **Choice:** `pytest` tests, run on every push by GitHub Actions on two
  Python versions.
- **Alternatives:** add tests later.
- **Why:** an agent edits files and runs commands, so regressions are
  costly. Starting with tests keeps every later step verifiable.
