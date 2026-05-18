# @noetaris/harness-anthropic

Anthropic Claude adapter for [@noetaris/harness](../core).

> **Status:** not yet released. Implementation tracked in F20.

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

const llm = new Claude({ apiKey: process.env.ANTHROPIC_API_KEY })

// Wire into a harness provider slot
h.provide('model', runtime())

const agent = createAgent(h, { prompts: { system: '...' } })
const run = agent.run(initialState, { model: llm })
```

## API

### `Claude`

Implements `LLM` and `ObserverAware`.

- **`invoke(messages, options?)`** — translates harness `Message[]` and `Tool[]` to Anthropic SDK format, calls `client.messages.create()`, and maps the response back to `LLMResponse`.
- **`bindObserver(observer)`** — attaches an `Observer`; after each `invoke`, emits an `"llm.response"` event with `{ tokens: { input, output }, modelId }`.

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
