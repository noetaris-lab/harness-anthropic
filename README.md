# @noetaris/harness-anthropic

Anthropic Claude adapter for [@noetaris/harness](../core).

## Overview

`@noetaris/harness-anthropic` provides a `Claude` class that implements the `LLM` and `ObserverAware` interfaces from `@noetaris/harness`. It handles translation between the harness message format and the Anthropic SDK format, and emits telemetry events (token usage, model ID) through an attached `Observer`.

## Installation

```sh
pnpm add @noetaris/harness-anthropic
```

Peer dependencies:

```sh
pnpm add @noetaris/harness @noetaris/harness-types
```

Requires Node.js ≥ 22.

## Usage

```ts
import { Claude } from '@noetaris/harness-anthropic'

// The model ID is the required first argument; options are optional.
const llm = new Claude('claude-3-5-haiku-20241022', {
  apiKey: process.env.ANTHROPIC_API_KEY, // defaults to the ANTHROPIC_API_KEY env var
})

// Wire into a harness provider slot
h.provide('model', runtime())

const agent = createAgent(h, { prompts: { system: '...' } })
const run = agent.run(initialState, { model: llm })
```

## API

### `Claude`

```ts
new Claude(model: string, options?: ClaudeOptions)
```

Implements `LLM` and `ObserverAware`. `ClaudeOptions` accepts `apiKey` and the
generation parameters `temperature`, `maxTokens` (default `4096`), `topP`, and
`thinking` (extended-thinking budget for supported models).

- **`invoke(messages, options?)`** — translates harness `Message[]` and `Tool[]` to Anthropic SDK format, calls `client.messages.create()`, and maps the response back to an `LLMResponse` (including a required `usage: { inputTokens, outputTokens, contextWindowSize? }` field).
- **`bindObserver(observer)`** — attaches an `Observer`. Each `invoke` emits an `"llm.request"` event (`{ modelId, providerName: 'anthropic' }`) before the call and an `"llm.response"` event (`{ tokens: { input, output }, modelId, stopReason, providerName, contextWindowSize? }`) after it.
- **`setStepContext(ctx)`** — sets the `StepContext` attached to emitted events; called by the harness before each step.

### `MockClaude`

A deterministic test double for use in tests and demos without a real API key.

## Related Packages

- [`@noetaris/harness`](https://github.com/noetaris-lab/harness) — core execution engine
- [`@noetaris/harness-types`](https://github.com/noetaris-lab/harness-types) — shared LLM type contract
- [`@noetaris/harness-openai`](https://github.com/noetaris-lab/harness-openai) — OpenAI adapter
- [`@noetaris/harness-google`](https://github.com/noetaris-lab/harness-google) — Google Gemini adapter
- [`@noetaris/harness-otel`](https://github.com/noetaris-lab/harness-otel) — OpenTelemetry observer bridge

## License

MIT
