# Tool streaming interoperability

The server fixture uses the real Axum routes, model-output parser and SSE
serialization. Only token generation is scripted. It never loads a model,
executes a generated tool, or edits a harness's configuration.

From the repository root, start the fixture in a terminal:

```sh
cargo test --release -p higgs --lib streaming_compat_tests::serve_tool_streaming_fixture -- --ignored --nocapture
```

On the development Mac, prefix Cargo with the existing compiler shim when
the default Metal compiler cannot build MLX:

```sh
PATH="$HOME/Dev/toolchains/xcrun-shim:$PATH" cargo test --release -p higgs --lib streaming_compat_tests::serve_tool_streaming_fixture -- --ignored --nocapture
```

Then install the pinned client versions in this directory and run:

```sh
npm install --ignore-scripts
npm test
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python clients.py
```

Default origin is `http://127.0.0.1:19091`; set `HIGGS_COMPAT_URL` for a
different fixture origin. The model is `required-stream-script-long`.
Each request takes about two seconds; actual tool argument fragments must
arrive with gaps under one second. The pi-ai test uses the real harness
adapter's one-second model-event watchdog and real HTTP provider with explicit
fixture metadata. No retry is allowed. The other clients verify semantic
event spacing and exact argument reconstruction; they do not claim to have
the same watchdog as the harness.

These scaled tests avoid waiting five minutes in CI. A live-model test is
separate evidence and must report prompt/cache size, token speed and maximum
semantic gap; passing this fixture does not prove inference is fast.

Run the client suites sequentially: the fixture deliberately shares the real
serialized worker gate. The Node suite also tests an idle worker, requires the
harness watchdog to fail, and checks that a follow-up request can run promptly
after that disconnect. Concurrent fixture requests would confound that latency
check with normal queueing.

## Live model comparison

`python live.py baseline` and `python live.py candidate` probe a separately
started local server at port 19092 (`--origin` overrides it), using the test-only
key `streaming-local-validation` and model name `escha-35b-a3b`. Each label must
use a fresh server process with disk prefix caching disabled. Run the two model
processes sequentially to avoid memory and GPU contention. Use matching model,
KV and generation settings; the probe explicitly disables speculation.

The probe performs inert tool calls only. It grows the actual conversation for
cold, warm-continuation and multi-turn cases. Token calibration excludes tool
schema overhead; actual response usage is authoritative. Correlate client
results with the server's AR completion timings for prefill/decode throughput.
The candidate also disconnects during arguments and initial waiting, then checks
fresh-session follow-ups. Follow-up latency includes that request's own prefill;
initial-wait cancellation alone does not prove which GPU phase was active.

Post-cancellation follow-ups must finish within 30 seconds in the live probe;
this includes their own prompt processing. The final validation report records
measured timings and limitations rather than assuming HTTP disconnect means
the model worker has already stopped.
