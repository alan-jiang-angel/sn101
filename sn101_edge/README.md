# sn101_edge — competitive miner for Tag101 (Bittensor SN101)

A drop-in replacement for the reference SN101 tag solver, built directly against
the validator's scoring code rather than against the README's description of it.

It lives **outside** the `tag101` checkout and is loaded through the supported
`--task.miner_module` hook, so `git pull --ff-only` during auto-update can never
clobber it.

---

## 1. Install

Python 3.12 is required (the `tag101` package pins `==3.12.*`).

```bash
# Start from an installed tag101 checkout
git clone <tag101-repo> ~/tag101 && cd ~/tag101
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Add this package alongside it
cd ~
unzip sn101_edge.zip -d ~/sn101_edge_pkg
cd ~/sn101_edge_pkg
pip install -r requirements.txt
```

Make the package importable. Either install it:

```bash
pip install -e ~/sn101_edge_pkg
```

or put it on the path:

```bash
export PYTHONPATH=$HOME/sn101_edge_pkg:$PYTHONPATH
```

Download the embedding checkpoint once, up front (~90 MB):

```bash
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

Also make sure spaCy's model is present — the validator uses it to build extra
scoring spans, and having it locally keeps your simulation faithful:

```bash
python -m spacy download en_core_web_sm
```

## 2. Configure

```bash
cp sn101_edge.env.example ~/sn101_edge.env
$EDITOR ~/sn101_edge.env      # set OPENROUTER_API_KEY at minimum
set -a && source ~/sn101_edge.env && set +a
```

Set `SN101_ASSUMED_MINERS` to the actual miner count on the subnet:

```bash
btcli subnet metagraph --netuid 101
```

## 3. Verify before you deploy

```bash
python -m sn101_edge.selftest          # offline: no network, no checkpoint
python -m sn101_edge.selftest --live   # adds real embeddings + a live LLM call
```

Everything must pass. The `--live` run prints pairwise tag similarities against
the diversity gate (0.55 / 0.85) — worth reading once so you have a feel for
which tag pairs the validator will treat as redundant.

## 4. Run

### PM2 (recommended while tuning)

```bash
pm2 start python --name sn101-miner --interpreter none -- \
  -m sn101_edge.run_miner \
  --netuid 101 \
  --subtensor.network finney \
  --wallet.name <WALLET> \
  --wallet.hotkey <HOTKEY> \
  --axon.port 8091 \
  --axon.external_ip <YOUR_PUBLIC_IP> \
  --task.miner_module sn101_edge.task \
  --logging.info

pm2 logs sn101-miner
```

### Using the repo's own launcher

If you prefer the stock entrypoint, append the module override to `NODE_ARGS`
in your `miner.env`:

```
NODE_ARGS=--netuid 101 --subtensor.network finney --wallet.name <W> --wallet.hotkey <H> --logging.info --task.miner_module sn101_edge.task
```

Then start it normally. Note this path uses the stock `forward`, which solves
**synchronously inside the server's event loop** — concurrent validator queries
will serialise. `sn101_edge.run_miner` exists specifically to fix that; prefer
it.

### Docker

Add to your `miner.env`:

```
NODE_ARGS=... --task.miner_module sn101_edge.task
```

and mount the package plus the model cache into the container:

```
DOCKER_EXTRA_ARGS=-v /home/user/sn101_edge_pkg:/opt/sn101_edge \
  -v /home/user/.cache/huggingface:/root/.cache/huggingface \
  -e PYTHONPATH=/opt/sn101_edge
```

Keeping the HuggingFace cache on a host volume matters: without it every
container restart re-downloads the checkpoint and the first few tasks time out.

## 5. Open the port

The validator reaches you over plain HTTP at `POST /TaskEnvelope`:

```bash
sudo ufw allow 8091/tcp
curl -sS -o /dev/null -w '%{http_code}\n' \
  -X POST http://<YOUR_PUBLIC_IP>:8091/TaskEnvelope
```

A `401` is the correct, healthy answer — it means the server is reachable and
rejecting your unsigned request. A timeout means the port is closed and you are
scoring zero on every task.

## 6. Confirm it is working

```bash
pm2 logs sn101-miner | grep MINER_SOLVED_TASK
```

You want `elapsed` comfortably under the wire timeout and a non-empty `answer`
on every line. Turn on `SN101_DEBUG=true` to see per-task candidate scoring.

---

## Configuration reference

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | — | Ensemble provider key |
| `SN101_MODELS` | 3 entries | Ensemble members; keep a gpt-4o-mini |
| `SN101_SAMPLES_PER_MODEL` | 3 | Samples per model — more = better crowd estimate, more cost |
| `SN101_TEMPERATURE` | 0.9 | High on purpose: we want the output *distribution*, not the mode of one call |
| `SN101_LLM_BUDGET_FRACTION` | 0.45 | Share of wire timeout for the LLM phase |
| `SN101_SELECT_BUDGET` | 2.5 | Wall-clock cap on the combinatorial search |
| `SN101_ASSUMED_MINERS` | 200 | Used to estimate duplicate-set collisions |
| `SN101_SELECT_TOP_K` | 6 | Candidate pool width (6 → 41 combinations) |
| `SN101_CACHE_TTL` | 3600 | Same-post cache; makes every validator after the first free |
| `SN101_DEBUG` | false | Per-task scoring logs |

## Module map

| File | Role |
|---|---|
| `constraints.py` | Bit-exact mirror of the validator's tokenizer, format gate, and verbatim-validity rule |
| `encoder.py` | Cached MiniLM wrapper shared by every local scorer |
| `crowd.py` | LLM ensemble that samples the distribution consensus is scored against |
| `selector.py` | Reconstructs the validator's scorer and searches tag sets against it |
| `tagger.py` | Deadline discipline, caching, guaranteed fallback |
| `task.py` | `TaskHandler` export for `--task.miner_module` |
| `run_miner.py` | Entrypoint with non-blocking `forward` and startup warmup |
| `selftest.py` | Pre-deployment verification |

## Operational notes

- **Uptime dominates.** The default score strategy averages your raw scores over
  a 24h window as `sum / max_count`, where `max_count` is the highest
  observation count of *any* miner. A missed task is a hard zero in that
  average for a full day.
- **Restart during a gap, not mid-round.** Validators poll every 15 minutes.
- **Warmup is not optional.** A cold start pays ~1.8s for sklearn plus several
  seconds for the checkpoint. `run_miner.py` does this before serving; if you
  use the stock entrypoint, the first task after a restart will likely miss.
- **Watch your OpenRouter spend.** Default settings issue 9 completions per
  unique post, roughly 96 posts/day per validator, largely cached across
  validators. Drop `SN101_SAMPLES_PER_MODEL` to 2 if cost matters more than the
  crowd estimate's precision.
