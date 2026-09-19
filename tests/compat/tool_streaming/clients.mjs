import assert from 'node:assert/strict';
import OpenAI from 'openai';
import Anthropic from '@anthropic-ai/sdk';
import { streamSimple } from '@earendil-works/pi-ai/api/openai-completions';
import { PiAiAdapter } from '@deepseek-ai/dsh-llm-pi-ai';

const origin = process.env.HIGGS_COMPAT_URL ?? 'http://127.0.0.1:19091';
const model = 'required-stream-script-long';
const parameters = { type: 'object', properties: { content: { type: 'string' } }, required: ['content'] };
const tool = { name: 'write', description: 'Inert test tool; do not execute', parameters };
const request = { model, messages: [{ role: 'user', content: 'write the fixture' }],
  tools: [{ type: 'function', function: tool }], stream: true, max_tokens: 4096 };
const expected = { content: 'chunk '.repeat(16 * 80) };

// SDKs must report interrupted calls as failures, never completed tool uses.
async function verifyFailures() {
  for (const suffix of ['malformed', 'capacity']) {
    const failedModel = `required-stream-script-${suffix}`;
    for (const protocol of ['openai', 'anthropic']) {
      let failed = false; let successfulFinish = false; let argumentEvents = 0;
      try {
        const stream = protocol === 'openai'
          ? await new OpenAI({ baseURL: `${origin}/v1`, apiKey: 'fixture', maxRetries: 0 })
            .chat.completions.create({ ...request, model: failedModel })
          : await new Anthropic({ baseURL: origin, apiKey: 'fixture', maxRetries: 0 })
            .messages.create({ ...request, model: failedModel,
              tools: [{ name: 'write', input_schema: parameters }] });
        for await (const event of stream) {
          if (protocol === 'openai') {
            successfulFinish ||= event.choices?.some(c => c.finish_reason === 'tool_calls') ?? false;
            for (const choice of event.choices ?? []) {
              argumentEvents += choice.delta?.tool_calls?.filter(call => call.function?.arguments).length ?? 0;
            }
          } else {
            successfulFinish ||= event.type === 'message_stop' || event.delta?.stop_reason === 'tool_use';
            if (event.delta?.type === 'input_json_delta' && event.delta.partial_json) argumentEvents++;
          }
        }
      } catch { failed = true; }
      assert(failed, `${protocol}/${suffix}: SDK did not receive a failure`);
      assert(argumentEvents > 0, `${protocol}/${suffix}: test did not interrupt a streamed call`);
      assert(!successfulFinish, `${protocol}/${suffix}: emitted success before failure`);
    }
  }
  console.log(JSON.stringify({ client: 'sdk-interruption-semantics', result: 'pass' }));
}

function collector(label) {
  let argumentsText = '';
  const times = [];
  return {
    add(fragment) { if (fragment) { argumentsText += fragment; times.push(performance.now()); } },
    done() {
      assert.deepEqual(JSON.parse(argumentsText), expected, `${label}: exact arguments`);
      assert(times.length > 10, `${label}: received real incremental arguments`);
      const span = times.at(-1) - times[0];
      assert(span > 1000, `${label}: fixture must exceed idle deadline`);
      const maxGap = Math.max(...times.slice(1).map((time, i) => time - times[i]));
      assert(maxGap < 1000, `${label}: semantic progress gap ${maxGap}ms`);
      console.log(JSON.stringify({ client: label, argumentEvents: times.length,
        spanMs: Math.round(span), maxGapMs: Math.round(maxGap), result: 'pass' }));
    },
  };
}

// Discard all comments and reconstruct records across arbitrary network chunks.
{
  const result = collector('raw-sse');
  const response = await fetch(`${origin}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(request),
  });
  assert.equal(response.status, 200);
  let buffer = ''; let finish; let done = false;
  for await (const text of response.body.pipeThrough(new TextDecoderStream())) {
    buffer += text;
    let end;
    while ((end = buffer.indexOf('\n\n')) >= 0) {
      const record = buffer.slice(0, end); buffer = buffer.slice(end + 2);
      const data = record.split('\n').filter(line => line.startsWith('data: ')).map(line => line.slice(6)).join('\n');
      if (!data) continue;
      if (data === '[DONE]') { done = true; continue; }
      const event = JSON.parse(data); assert(!event.error, JSON.stringify(event));
      const choice = event.choices?.[0];
      finish = choice?.finish_reason ?? finish;
      for (const call of choice?.delta?.tool_calls ?? []) result.add(call.function?.arguments);
    }
  }
  assert(done); assert.equal(finish, 'tool_calls'); result.done();
}

{
  const result = collector('openai-node');
  const client = new OpenAI({ baseURL: `${origin}/v1`, apiKey: 'fixture', maxRetries: 0 });
  const stream = await client.chat.completions.create(request);
  let finish;
  for await (const event of stream) {
    finish = event.choices[0]?.finish_reason ?? finish;
    for (const call of event.choices[0]?.delta?.tool_calls ?? []) result.add(call.function?.arguments);
  }
  assert.equal(finish, 'tool_calls'); result.done();
}

{
  const result = collector('anthropic-node');
  const client = new Anthropic({ baseURL: origin, apiKey: 'fixture', maxRetries: 0 });
  const stream = await client.messages.create({ ...request, tools: [{ name: 'write', description: tool.description, input_schema: parameters }] });
  let finish;
  for await (const event of stream) {
    if (event.type === 'content_block_delta' && event.delta.type === 'input_json_delta') result.add(event.delta.partial_json);
    if (event.type === 'message_delta') finish = event.delta.stop_reason;
  }
  assert.equal(finish, 'tool_use'); result.done();
}

{
  // Exercise the installed harness adapter's actual model-event watchdog,
  // using real pi-ai HTTP streaming and explicit fixture model metadata.
  const result = collector('deepseek-harness-pi-ai');
  const descriptor = { id: model, name: model, api: 'openai-completions', provider: 'higgs',
    baseUrl: `${origin}/v1`, reasoning: false, input: ['text'],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 32768, maxTokens: 4096 };
  const profiles = new Map([['higgs', { streamIdleTimeoutMs: 1000, modelErrors: new Map(), piProvider: {} }]]);
  const adapter = new PiAiAdapter({ profiles: () => profiles, resolveApiKey: async () => 'fixture' });
  const snapshot = { profiles, models: { getModel: () => descriptor, streamSimple } };
  let finished = false;
  for await (const event of adapter.streamWithSnapshot({ provider: 'higgs', model,
    messages: [{ role: 'user', content: [{ type: 'text', text: 'write the fixture' }] }],
    tools: [tool], maxTokens: 4096 }, snapshot)) {
    if (event.type === 'tool-call-delta') result.add(event.argumentsDelta);
    if (event.type === 'finish') {
      assert.notEqual(event.reason?.kind, 'error', JSON.stringify(event)); finished = true;
    }
  }
  assert(finished); result.done();

  // The inverse case must still trip the same real watchdog. Comments or
  // transport activity must not disguise a worker with no semantic progress.
  const idleModel = 'required-stream-script-idle';
  const idleSnapshot = { profiles, models: {
    getModel: () => ({ ...descriptor, id: idleModel, name: idleModel }), streamSimple,
  } };
  const started = performance.now();
  let idleFailure;
  try {
    for await (const event of adapter.streamWithSnapshot({ provider: 'higgs', model: idleModel,
      messages: [{ role: 'user', content: [{ type: 'text', text: 'write the fixture' }] }],
      tools: [tool], maxTokens: 4096 }, idleSnapshot)) {
      if (event.type === 'finish' && event.reason?.kind === 'error') idleFailure = JSON.stringify(event);
    }
  } catch (error) { idleFailure = String(error); }
  assert(idleFailure?.includes('idle timeout'), `Expected idle timeout: ${idleFailure}`);
  assert(performance.now() - started < 4000, 'idle watchdog was masked');
  // A prompt follow-up also verifies that cancellation freed the worker gate.
  const followupStarted = performance.now();
  const followup = await fetch(`${origin}/v1/chat/completions`, {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ ...request, model: 'required-stream-script-xml' }),
  });
  const followupWire = await followup.text();
  assert(followupWire.includes('"finish_reason":"tool_calls"'));
  assert(performance.now() - followupStarted < 1000, 'cancelled worker blocked follow-up');
  console.log(JSON.stringify({ client: 'deepseek-harness-pi-ai-idle', result: 'pass' }));
}

await verifyFailures();
