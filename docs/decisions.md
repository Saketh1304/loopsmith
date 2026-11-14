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
