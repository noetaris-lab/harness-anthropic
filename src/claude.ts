import type { LLM, Message, Tool, ToolCall, LLMResponse, LLMUsageEvent } from '@noetaris/harness-types'
import type { ObserverAware, Observer, StepContext } from '@noetaris/harness'
import Anthropic from '@anthropic-ai/sdk'
import type { Tool as AnthropicSDKTool } from '@anthropic-ai/sdk/resources/messages/messages.js'

/** Options for {@link Claude}. */
export interface ClaudeOptions {
  /** Anthropic API key. Defaults to the `ANTHROPIC_API_KEY` environment variable. */
  apiKey?: string
  /**
   * Sampling temperature in [0, 1]. Higher values produce more varied output.
   * When absent, the provider default applies.
   */
  temperature?: number
  /**
   * Maximum number of tokens to generate.
   * Defaults to `4096` when absent.
   */
  maxTokens?: number
  /**
   * Top-p nucleus sampling probability. When absent, the provider default applies.
   */
  topP?: number
  /**
   * Extended thinking configuration for supported models.
   * When present, enables extended thinking mode with the specified token budget.
   */
  thinking?: {
    type: 'enabled'
    budgetTokens: number
  }
}

type AnthropicContentBlock =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: unknown }
  | { type: 'tool_result'; tool_use_id: string; content: string }

type AnthropicMessage = {
  role: 'user' | 'assistant'
  content: string | AnthropicContentBlock[]
}

// Use SDK's Tool type for the translated tools array passed to client.messages.create
type AnthropicTool = AnthropicSDKTool

function translateMessages(messages: Message[]): AnthropicMessage[] {
  const result: AnthropicMessage[] = []

  for (const msg of messages) {
    if (msg.role === 'user') {
      result.push({ role: 'user', content: msg.content })
    } else if (msg.role === 'assistant') {
      if (msg.toolCalls && msg.toolCalls.length > 0) {
        const blocks: AnthropicContentBlock[] = []
        if (msg.content) {
          blocks.push({ type: 'text', text: msg.content })
        }
        for (const tc of msg.toolCalls) {
          blocks.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.input })
        }
        result.push({ role: 'assistant', content: blocks })
      } else {
        result.push({ role: 'assistant', content: msg.content ?? '' })
      }
    } else if (msg.role === 'tool') {
      const toolResultBlock: AnthropicContentBlock = {
        type: 'tool_result',
        tool_use_id: msg.toolCallId,
        content: msg.content,
      }

      const last = result[result.length - 1]
      if (last !== undefined && last.role === 'user') {
        // last message is a user message — merge into it
        if (typeof last.content === 'string') {
          // convert string content to array with text block + tool_result
          last.content = [
            { type: 'text', text: last.content },
            toolResultBlock,
          ]
        } else {
          last.content.push(toolResultBlock)
        }
      } else {
        // no preceding user message — wrap in a new user turn
        result.push({ role: 'user', content: [toolResultBlock] })
      }
    }
  }

  return result
}

function translateTools(tools: Tool[]): AnthropicTool[] {
  return tools.map((t) => ({
    name: t.name,
    description: t.description,
    // as: harness Tool.inputSchema is Record<string,unknown>; SDK requires InputSchema with type:'object'
    // which is always present at runtime — cannot be statically verified from the generic type
    input_schema: t.inputSchema as AnthropicSDKTool['input_schema'],
  }))
}

function mapStopReason(stopReason: string): LLMResponse['stopReason'] {
  if (stopReason === 'end_turn') return 'end'
  if (stopReason === 'tool_use') return 'tool_use'
  if (stopReason === 'max_tokens') return 'max_tokens'
  return 'end'
}

function normalizeResponse(response: Anthropic.Message): LLMResponse {
  let text = ''
  const toolCalls: ToolCall[] = []

  for (const block of response.content) {
    if (block.type === 'text') {
      text += block.text
    } else if (block.type === 'tool_use') {
      toolCalls.push({ id: block.id, name: block.name, input: block.input })
    }
  }

  return {
    text,
    toolCalls,
    stopReason: mapStopReason(response.stop_reason ?? ''),
  }
}

const ZEROED_STEP_CONTEXT: StepContext = { agentId: '', sessionId: '', stepName: '' }

/**
 * {@link LLM} adapter for the Anthropic Messages API.
 *
 * Implements {@link ObserverAware} — when an observer is bound the adapter
 * emits an `'llm.response'` event carrying an `LLMUsageEvent` payload after
 * each successful invocation.
 *
 * @example
 * ```ts
 * const llm = new Claude('claude-3-5-haiku-20241022')
 * const response = await llm.invoke(messages)
 * ```
 */
export class Claude implements LLM, ObserverAware {
  private readonly client: Anthropic
  private readonly model: string
  private readonly options?: ClaudeOptions
  private observer: Observer = {}
  private stepContext: StepContext = ZEROED_STEP_CONTEXT

  /**
   * @param model - Anthropic model ID, e.g. `'claude-3-5-haiku-20241022'`.
   * @param options - Optional configuration including API key and generation params.
   */
  constructor(model: string, options?: ClaudeOptions) {
    this.model = model
    this.options = options
    this.client = new Anthropic({ apiKey: options?.apiKey })
  }

  bindObserver(observer: Observer): void {
    this.observer = observer
  }

  setStepContext(ctx: StepContext): void {
    this.stepContext = ctx
  }

  async invoke(messages: Message[], options?: { tools?: Tool[] }): Promise<LLMResponse> {
    const translatedMessages = translateMessages(messages)
    const tools = options?.tools

    const response = await this.client.messages.create({
      model: this.model,
      messages: translatedMessages,
      max_tokens: this.options?.maxTokens ?? 4096,
      ...(this.options?.temperature !== undefined ? { temperature: this.options.temperature } : {}),
      ...(this.options?.topP !== undefined ? { top_p: this.options.topP } : {}),
      ...(this.options?.thinking !== undefined ? { thinking: { type: this.options.thinking.type, budget_tokens: this.options.thinking.budgetTokens } } : {}),
      ...(tools !== undefined ? { tools: translateTools(tools) } : {}),
    })

    const result = normalizeResponse(response)

    const event: LLMUsageEvent = {
      tokens:     { input: response.usage.input_tokens, output: response.usage.output_tokens },
      modelId:    this.model,
      stopReason: result.stopReason,
      providerName: 'anthropic',
    }
    this.observer.onEvent?.(this.stepContext, 'llm.response', event)

    return result
  }
}
