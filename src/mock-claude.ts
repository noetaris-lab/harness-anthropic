import type { LLM, Message, Tool, LLMResponse, LLMUsageEvent } from '@noetaris/harness-types'
import type { ObserverAware, Observer, StepContext } from '@noetaris/harness'

/** Thrown by {@link MockClaude} when `invoke` is called with no responses queued. */
export class MockClaudeEmptyQueueError extends Error {
  constructor() {
    super('MockClaude has no responses configured — call new MockClaude(response) or enqueue(response) before invoke')
    this.name = 'MockClaudeEmptyQueueError'
  }
}

const ZEROED_STEP_CONTEXT: StepContext = { agentId: '', sessionId: '', stepName: '' }

/**
 * In-memory test double for {@link Claude}.
 *
 * Pre-load one or more {@link LLMResponse} objects; each `invoke()` call
 * dequeues the next response.  The last response is **sticky** — once only
 * one remains it repeats indefinitely rather than throwing.
 *
 * `lastMessages` is populated after each `invoke()` call, allowing assertions
 * on the conversation history passed to the adapter.
 *
 * @example
 * ```ts
 * const llm = new MockClaude({ text: 'hello', toolCalls: [], stopReason: 'end' })
 * ```
 */
export class MockClaude implements LLM, ObserverAware {
  /** The message list from the most recent `invoke()` call. */
  lastMessages: Message[] = []

  private queue: LLMResponse[] = []
  private observer: Observer = {}
  private stepContext: StepContext = ZEROED_STEP_CONTEXT

  /**
   * @param responses - One or more responses to queue up front. May also be
   *   added later via {@link enqueue}.
   */
  constructor(responses?: LLMResponse | LLMResponse[]) {
    if (responses !== undefined) {
      this.enqueue(responses)
    }
  }

  /** Add one or more responses to the end of the queue. */
  enqueue(response: LLMResponse | LLMResponse[]): void {
    const items = Array.isArray(response) ? response : [response]
    this.queue.push(...items)
  }

  bindObserver(observer: Observer): void {
    this.observer = observer
  }

  setStepContext(ctx: StepContext): void {
    this.stepContext = ctx
  }

  async invoke(messages: Message[], options?: { tools?: Tool[] }): Promise<LLMResponse> {
    void options
    if (this.queue.length === 0) {
      throw new MockClaudeEmptyQueueError()
    }

    // sticky-last: dequeue only when more than one element remains
    const response: LLMResponse = this.queue.length > 1
      ? (this.queue.shift() as LLMResponse) // as: shift() on non-empty array is always defined; length > 1 is checked above
      : (this.queue[0] as LLMResponse) // as: queue.length === 1 guaranteed by the empty check; index 0 is always defined

    this.lastMessages = messages

    const event: LLMUsageEvent = {
      tokens:     { input: 0, output: 0 },
      modelId:    'mock',
      stopReason: response.stopReason,
      providerName: 'mock',
    }
    this.observer.onEvent?.(this.stepContext, 'llm.response', event)

    return response
  }
}
