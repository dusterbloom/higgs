# Streaming tool calls

For local inference, `/v1/chat/completions` sends a tool identity followed by
append-only `delta.tool_calls[].function.arguments` fragments. The index is
stable throughout a call, and the ID and name are sent at its start. Clients
concatenate the argument fragments and parse the resulting JSON after a
successful completion. Partial arguments are not ready for execution.

`/v1/messages` uses the corresponding Anthropic tool-use block and
`input_json_delta` events. Text and tool blocks preserve model-output order.

The incremental parser recognizes Qwen JSON envelopes, Qwen function/parameter
XML, and MiniCPM function XML with CDATA. Large schema-declared string arguments
(such as file contents) stream while they are generated. JSON values are
validated incrementally; XML values with absent or ambiguous types may need
bounded buffering because the eventual JSON type cannot be inferred from an
incomplete value.

Successful valid values retain their contents and JSON types. Malformed
schema-declared values in incremental streams fail explicitly rather than
falling back to strings after emitting a different type. Missing closers,
duplicate keys, excessive nesting, or exceeding the existing 1 MiB call limit
cannot produce a successful tool-call finish. Once a tentative call is exposed,
its raw bytes are never replayed as assistant prose. Clients must discard
partial calls on errors or interrupted streams.

Required and named tool choices still enforce their completion postconditions.
Streaming a tentative call does not assert that the entire response is valid;
only the successful terminal outcome does. Native Anthropic nonstream `any`
and named `tool` choices explicitly require `stream: true`; nonstream tool-use
responses are outside this change. A second call, wrong name or leaked
prose can fail a request after some tentative argument fragments were emitted.

SSE comments keep the transport alive but do not represent model progress.
Higgs does not emit fabricated assistant text to satisfy client watchdogs.
Long prefill, genuinely stalled generation, clients that ignore tool deltas,
and fixed total-response deadlines remain separate concerns.

Interrupted streaming requests are recorded with partial output counts;
client disconnects use metrics status 499. Retained prefill observes disconnects
at chunk boundaries, including DFlash before any cold retry; a running GPU chunk
may finish before the next request can proceed. Debug tracing reports generated
and emitted event timings without recording tool contents. Cache reuse and
token generation speed should be evaluated separately from visible streaming.

See [the interoperability suite](../tests/compat/tool_streaming/README.md) for
real HTTP tests against official SDKs and the deepseek-harness pi-ai adapter.
