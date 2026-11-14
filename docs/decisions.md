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

## 6. Plain HTTP calls, not the provider's SDK

- **Choice:** call the Gemini REST API with Python's standard library.
- **Alternatives:** Google's `google-genai` SDK.
- **Why:** the raw request and response are visible, which makes tool
  calling easier to understand and debug. There are no dependencies, and
  every provider is a similar HTTP call, so adding one is cheap.
- **Cost:** retries and streaming have to be written by hand.

## 7. A provider-neutral model interface

- **Choice:** the agent depends on `ModelClient`, a small interface with
  one method, and on neutral `Message` and `ModelReply` types.
  `GeminiClient` is one implementation.
- **Alternatives:** call Gemini directly from the agent code.
- **Why:** switching to another provider means writing one new class.
  It also allows a `FakeClient` in tests, so the agent's logic is tested
  without a network or an API key.

## 8. The network layer is injected

- **Choice:** `GeminiClient` receives a `transport` function that sends
  the request. The default one uses the network; tests pass a fake.
  `main()` accepts an optional `client` in the same way.
- **Alternatives:** patch `urllib` inside tests.
- **Why:** this is dependency injection, the idea behind Spring's
  `@Autowired`, done by hand. Tests stay fast and do not depend on the
  internals of a library.

## 9. Errors carry a status and a `retryable` flag

- **Choice:** every failure becomes a `ModelError` with the HTTP status,
  a message that says how to fix it, and whether a retry could succeed
  (rate limits, server errors and network failures can; a bad key or an
  unknown model cannot).
- **Alternatives:** let raw HTTP errors reach the user.
- **Why:** the retry logic added later only needs to check one flag.

## 10. The model name is configuration

- **Choice:** read the model from `GEMINI_MODEL`, with a default, and
  provide `loopsmith --list-models`.
- **Alternatives:** hard-code the model name.
- **Why:** providers rename and retire models often. A stale name should
  be a one-line fix in `.env`, with a command that shows valid names.

## 11. The `generateContent` endpoint

- **Choice:** use Gemini's `generateContent` REST method.
- **Alternatives:** Google's newer Interactions API, which its docs now
  recommend for new projects.
- **Why:** `generateContent` is stateless: the full conversation is sent
  on every call. That makes the agent's state explicit and keeps the
  design close to other providers' APIs. Worth revisiting later.

## 12. The loop stops when the model stops asking for tools

- **Choice:** each step calls the model once. A reply with tool calls
  means "keep going"; a reply with only text means "done".
- **Alternatives:** a special `finish` tool the model must call; asking
  the model "are you done?" after each step.
- **Why:** it is the simplest rule and the one most agents use. A
  `finish` tool gives a more structured result and may be worth adding
  for the benchmark.

## 13. A hard limit on steps

- **Choice:** stop after `max_steps` model calls (default 20) and exit
  with a failure code.
- **Alternatives:** no limit; a time or cost budget.
- **Why:** a model can loop forever, calling the same tool again and
  again. Every step costs money and quota, so a run must be bounded.
  A token or cost budget would be a good later addition.

## 14. Tool failures go back to the model

- **Choice:** an unknown tool, a bad argument or a crash inside a tool
  becomes an error result that the model reads. The loop never raises
  because of a tool.
- **Alternatives:** stop the run on the first tool error.
- **Why:** models often recover: they fix the path or try another tool.
  Error messages are written for the model, saying what to do next.
  Failures of the model API itself are different and do stop the run.

## 15. All file access goes through a workspace check

- **Choice:** every path is resolved (following `..` and symlinks) and
  refused unless it lands inside the project folder. `.env` files are
  refused as well.
- **Alternatives:** trust the model's paths.
- **Why:** tool arguments are untrusted input, like request parameters
  in a web API. The model could be steered by text inside a file it
  reads (prompt injection) into asking for private files. Whatever a
  tool returns is also sent to the model provider, so secrets must not
  be readable. This check is a first layer; the sandbox project adds
  real isolation.

## 16. Tool output is capped

- **Choice:** file contents are cut at 40,000 characters and folder
  listings at 200 entries, with a note saying what was left out.
- **Alternatives:** return everything.
- **Why:** the whole history is re-sent on every step, so one huge file
  would be paid for again and again and could overflow the context
  window. The note tells the model the output is incomplete.

## 17. The provider's raw reply is kept and replayed

- **Choice:** an assistant `Message` stores the provider's original
  parts in `raw`, and `GeminiClient` sends them back unchanged.
- **Alternatives:** rebuild the model's turn from the neutral fields.
- **Why:** Gemini attaches an encrypted `thoughtSignature` to function
  calls and rejects the next request with HTTP 400 if it is missing.
  Keeping the raw parts preserves it without the agent loop needing to
  know that signatures exist.
- **Cost:** a history recorded with one provider cannot be replayed
  through another without dropping `raw`.

## 18. Tool activity goes to stderr, the answer to stdout

- **Choice:** the trace of tool calls is printed to stderr and the final
  answer to stdout.
- **Alternatives:** print everything to stdout.
- **Why:** it is the Unix convention. `loopsmith "..." > answer.txt`
  captures only the answer, and the benchmark can read stdout cleanly.

## 19. Read-only tools first

- **Choice:** this step ships `list_files` and `read_file` only.
- **Alternatives:** add file editing and shell commands at once.
- **Why:** read-only tools cannot damage anything, so the loop can be
  tested on a real model safely. Tools that change things arrive with
  confirmation prompts in the next step.

## 20. Editing by exact text replacement

- **Choice:** `edit_file` replaces one exact piece of text, which must
  appear exactly once in the file. `write_file` handles new files and
  full rewrites.
- **Alternatives:** always rewrite the whole file; have the model
  produce a unified diff; edit by line numbers.
- **Why:** whole-file rewrites are slow and costly for large files, and
  the model can silently drop lines. Models often get diff syntax and
  line numbers wrong. Exact replacement is easy for a model to produce
  and fails loudly: zero matches or several matches are both errors,
  each with advice on how to fix the call.

## 21. Line endings are preserved

- **Choice:** files are matched with `\n` line endings and saved with
  whatever style they had, `\n` or `\r\n`.
- **Alternatives:** let Python translate line endings.
- **Why:** the model writes `\n`, while Windows files often use `\r\n`.
  Without this, edits would fail to match on Windows, or one small edit
  would show up in git as every line changed.

## 22. Changes need approval, and the loop enforces it

- **Choice:** a tool that changes things provides a `preview` function.
  Before running such a tool, the agent loop shows the preview (a diff
  or the command) and asks an `approve` callback. Read-only tools run at
  once.
- **Alternatives:** each tool asks the user itself; no approval; approve
  a whole plan up front.
- **Why:** the rule lives in one place, so a new tool cannot forget it.
  The loop does not know how approval is given: the CLI asks on the
  terminal, tests pass a function, and the benchmark will approve
  everything inside the sandbox.

## 23. Approval fails closed

- **Choice:** no approver configured means "no". No terminal to ask on
  means "no". Any answer other than `y` or `yes` means "no". `--yes`
  must be given to skip the questions.
- **Alternatives:** default to allowing.
- **Why:** when it is unclear whether a person agreed, the safe result
  is to do nothing. A declined action goes back to the model as an
  error telling it not to repeat the action.

## 24. Shell commands: time limit, no stdin, no API key

- **Choice:** `run_command` runs through the system shell in the project
  folder, with a 60-second limit, with stdin closed, and with the model
  API key removed from the environment.
- **Alternatives:** an allowlist of permitted commands; no shell.
- **Why:** a general shell is what makes the agent useful, since
  projects differ in how they build and test. The limits handle the
  common accidents: a hung command, a prompt waiting for input, and a
  command that prints the environment.
- **Known gaps:** a command can still do anything the user's account
  can, including reaching files outside the project folder. On timeout,
  processes started by the command may keep running. Approval is the
  only guard today. The sandbox project exists to close these gaps.

## 25. A failing command is a normal result

- **Choice:** `run_command` returns the exit code and output whatever
  the exit code is. Only a timeout is a tool error.
- **Alternatives:** treat non-zero exit codes as errors.
- **Why:** a failing test run is the most useful thing the agent can
  read. It is information, not a malfunction.

## 26. Long command output keeps both ends

- **Choice:** output over 20,000 characters keeps the first and last
  halves, with a note in the middle.
- **Alternatives:** keep only the start, as `read_file` does.
- **Why:** test summaries and error messages are usually at the end.

## 27. Retries live in a wrapper around the client

- **Choice:** `RetryingClient` wraps any `ModelClient` and retries
  errors marked `retryable`, up to four attempts. It waits for the time
  the server asks for, or else doubles the wait each time with a random
  part added (exponential backoff with jitter).
- **Alternatives:** retry inside `GeminiClient`; retry inside the agent
  loop.
- **Why:** this is the decorator pattern. Retrying works for every
  provider without being rewritten, the agent loop stays simple, and
  tests inject a fake `sleep` so they run instantly. Jitter stops many
  clients that failed together from retrying at the same moment.

## 28. The settings file is found by walking up the folders

- **Choice:** `.env` is looked for in the current folder, then each
  parent, then `~/.loopsmith.env`.
- **Alternatives:** only the current folder.
- **Why:** the agent is run inside other projects and their subfolders,
  and should not need a copy of the key in each one.
