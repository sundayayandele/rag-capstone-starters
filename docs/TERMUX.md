# Running this repo with Claude Code on Termux (Android)

## 1. Install the basics

```bash
pkg update
pkg install -y git python python-numpy unzip nodejs
```

Termux's own `python-numpy` avoids compiling numpy on the phone. Do not run `pkg` as root.

## 2. Get the repo

```bash
cd ~
git clone https://github.com/sundayayandele/rag-capstone-starters.git   # or the repo you push this to
cd rag-capstone-starters
pip install -e . --no-deps
pip install pytest pyyaml
python -m ragkit eval        # should print five PASS lines
pytest -q
```

If you received this as a zip: `unzip -o /sdcard/Download/rag-capstone-starters.zip -d ~/` (run `termux-setup-storage` once if `/sdcard` is not readable).

## 3. Push to GitHub

```bash
git init -b main            # skip if you cloned
git add .
git commit -m "Add RAG capstone starters"
git remote add origin https://github.com/<you>/rag-capstone-starters.git
git push -u origin main
```

GitHub asks for a username and a personal access token (Settings, Developer settings, Tokens) instead of a password.

## 4. Use Claude Code in the repo

```bash
cd ~/rag-capstone-starters
claude
```

Claude Code reads `CLAUDE.md` automatically. Useful first prompts:

- "Run the evaluation and summarise the results table."
- "Add 15 harder questions to p02 and show how the metrics change."
- "Switch p03 to RAG_EMBEDDER=st and compare against the hashing embedder."

If Claude Code hangs on the phone, close other apps, restart Termux, and run it from the repo folder. Heavy installs (sentence-transformers) may not fit in phone memory; run those in GitHub Actions or a Codespace instead.

## 5. Turn on the GitHub parts (once)

1. Repository **Settings, Pages, Source: GitHub Actions**.
2. **Settings, Secrets and variables, Actions**: add `ANTHROPIC_API_KEY` (only needed for the Claude workflow or `llm=anthropic`).
3. Open the **Actions** tab, run **CI** with "Run workflow". The results table appears in the job summary and on your Pages site.
4. Open an issue that starts with `@claude ...` to trigger the Claude Code workflow.
