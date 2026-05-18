import { describe, it, expect, vi, beforeEach } from 'vitest'
import { Claude } from './claude.js'

// minimal end-turn response used across multiple test cases
const minimalEndTurnResponse = {
  content: [{ type: 'text', text: 'ok' }],
  stop_reason: 'end_turn',
  usage: { input_tokens: 1, output_tokens: 1 },
}

const mockCreate = vi.fn()

vi.mock('@anthropic-ai/sdk', () => {
  // Anthropic is a class; the mock must be a constructor
  function MockAnthropic() {
    return { messages: { create: mockCreate } }
  }
  return { default: MockAnthropic }
})

beforeEach(() => {
  vi.clearAllMocks()
  mockCreate.mockResolvedValue(minimalEndTurnResponse)
})

describe('Claude', () => {

  describe('Group 1: Basic Invocation and Tool Translation', () => {

    it('passes model string and user message to SDK and returns normalized text response', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'Hello back' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 5 },
      })
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })

      // act
      const result = await claude.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          model: 'claude-sonnet-4-6',
          messages: [{ role: 'user', content: 'hello' }],
        })
      )
      expect(result.text).toBe('Hello back')
      expect(result.toolCalls).toEqual([])
      expect(result.stopReason).toBe('end')
    })

    it('translates Tool.inputSchema to input_schema in SDK call', async () => {
      // arrange
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      const harnessTool = {
        name: 'get_weather',
        description: 'Fetch weather',
        inputSchema: { type: 'object', properties: { city: { type: 'string' } } },
      }

      // act
      await claude.invoke(
        [{ role: 'user', content: "What's the weather?" }],
        { tools: [harnessTool] }
      )

      // assert
      expect(mockCreate).toHaveBeenCalledWith(
        expect.objectContaining({
          tools: [
            {
              name: 'get_weather',
              description: 'Fetch weather',
              input_schema: { type: 'object', properties: { city: { type: 'string' } } },
            },
          ],
        })
      )
    })

    it('omits tools field entirely from SDK call when no options provided', async () => {
      // arrange
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })

      // act
      await claude.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockCreate).toHaveBeenCalledOnce()
      const callArg = mockCreate.mock.calls[0]?.[0]
      expect(callArg).not.toHaveProperty('tools')
    })

  })

  describe('Group 2: Response Normalization', () => {

    it('returns populated toolCalls, empty text, and stopReason "tool_use" when response is tool-use only', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        content: [{ type: 'tool_use', id: 'tu_1', name: 'get_weather', input: { city: 'Paris' } }],
        stop_reason: 'tool_use',
        usage: { input_tokens: 20, output_tokens: 8 },
      })
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })

      // act
      const result = await claude.invoke([{ role: 'user', content: 'call a tool' }])

      // assert
      expect(result.text).toBe('')
      expect(result.toolCalls).toEqual([{ id: 'tu_1', name: 'get_weather', input: { city: 'Paris' } }])
      expect(result.stopReason).toBe('tool_use')
    })

    it('returns both text and toolCalls with stopReason "tool_use" for mixed response', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        content: [
          { type: 'text', text: "I'll call the tool" },
          { type: 'tool_use', id: 'tu_2', name: 'lookup', input: {} },
        ],
        stop_reason: 'tool_use',
        usage: { input_tokens: 15, output_tokens: 12 },
      })
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })

      // act
      const result = await claude.invoke([{ role: 'user', content: 'do it' }])

      // assert
      expect(result.text).toBe("I'll call the tool")
      expect(result.toolCalls).toHaveLength(1)
      expect(result.toolCalls[0]).toEqual({ id: 'tu_2', name: 'lookup', input: {} })
      expect(result.stopReason).toBe('tool_use')
    })

    it('concatenates multiple text blocks without separator', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        content: [
          { type: 'text', text: 'Part A' },
          { type: 'text', text: 'Part B' },
        ],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 10 },
      })
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })

      // act
      const result = await claude.invoke([{ role: 'user', content: 'multi' }])

      // assert
      expect(result.text).toBe('Part APart B')
      expect(result.toolCalls).toEqual([])
    })

    it('maps stop_reason "max_tokens" to stopReason "max_tokens"', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'truncated' }],
        stop_reason: 'max_tokens',
        usage: { input_tokens: 100, output_tokens: 200 },
      })
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })

      // act
      const result = await claude.invoke([{ role: 'user', content: 'long prompt' }])

      // assert
      expect(result.stopReason).toBe('max_tokens')
    })

    it('maps unrecognized stop_reason to "end" as safe fallback', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'done' }],
        stop_reason: 'stop_sequence',
        usage: { input_tokens: 5, output_tokens: 3 },
      })
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })

      // act
      const result = await claude.invoke([{ role: 'user', content: 'stop seq' }])

      // assert
      expect(result.stopReason).toBe('end')
    })

  })

  describe('Group 3: Observer Integration', () => {

    it('calls observer.onEvent with tokens and modelId after successful invoke', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'ok' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 25, output_tokens: 10 },
      })
      const mockObserver = { onEvent: vi.fn() }
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      claude.bindObserver(mockObserver)

      // act
      await claude.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledOnce()
      expect(mockObserver.onEvent).toHaveBeenCalledWith(
        expect.any(Object),
        'llm.response',
        { tokens: { input: 25, output: 10 }, modelId: 'claude-sonnet-4-6' }
      )
    })

    it('does not throw when observer is NOOP_OBSERVER ({})', async () => {
      // arrange
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      claude.bindObserver({})

      // act
      const act = () => claude.invoke([{ role: 'user', content: 'hi' }])

      // assert
      await expect(act()).resolves.toBeDefined()
    })

    it('uses zeroed StepContext when setStepContext was never called', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'ok' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 1, output_tokens: 1 },
      })
      const mockObserver = { onEvent: vi.fn() }
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      claude.bindObserver(mockObserver)

      // act
      await claude.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledWith(
        { agentId: '', sessionId: '', stepName: '' },
        'llm.response',
        expect.any(Object)
      )
    })

    it('uses stored StepContext when setStepContext was called before invoke', async () => {
      // arrange
      const mockObserver = { onEvent: vi.fn() }
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      claude.bindObserver(mockObserver)
      claude.setStepContext({ agentId: 'agent-1', sessionId: 'sess-abc', stepName: 'callModel' })

      // act
      await claude.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledWith(
        { agentId: 'agent-1', sessionId: 'sess-abc', stepName: 'callModel' },
        'llm.response',
        expect.any(Object)
      )
    })

  })

  describe('Group 4: Message Translation', () => {

    it('translates assistant message with toolCalls only to tool_use content blocks with no text block', async () => {
      // arrange
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      const messages = [
        { role: 'user' as const, content: 'call it' },
        { role: 'assistant' as const, toolCalls: [{ id: 'tc_1', name: 'lookup', input: { q: 'x' } }] },
      ]

      // act
      await claude.invoke(messages)

      // assert
      const sentMessages = mockCreate.mock.calls[0]?.[0].messages
      expect(sentMessages[1]).toEqual({
        role: 'assistant',
        content: [{ type: 'tool_use', id: 'tc_1', name: 'lookup', input: { q: 'x' } }],
      })
      expect(sentMessages[1].content).not.toContainEqual(expect.objectContaining({ type: 'text' }))
    })

    it('translates assistant message with both content and toolCalls to text block followed by tool_use blocks', async () => {
      // arrange
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      const messages = [
        { role: 'user' as const, content: 'go' },
        { role: 'assistant' as const, content: 'I will call', toolCalls: [{ id: 'tc_2', name: 'run', input: {} }] },
      ]

      // act
      await claude.invoke(messages)

      // assert
      const sentMessages = mockCreate.mock.calls[0]?.[0].messages
      expect(sentMessages[1]).toEqual({
        role: 'assistant',
        content: [
          { type: 'text', text: 'I will call' },
          { type: 'tool_use', id: 'tc_2', name: 'run', input: {} },
        ],
      })
    })

    it('groups consecutive tool messages after assistant into a single Anthropic user message', async () => {
      // arrange
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      const messages = [
        { role: 'user' as const, content: 'go' },
        {
          role: 'assistant' as const,
          toolCalls: [
            { id: 'tc_a', name: 'fnA', input: {} },
            { id: 'tc_b', name: 'fnB', input: {} },
          ],
        },
        { role: 'tool' as const, toolCallId: 'tc_a', content: 'result-A' },
        { role: 'tool' as const, toolCallId: 'tc_b', content: 'result-B' },
      ]

      // act
      await claude.invoke(messages)

      // assert
      const sentMessages = mockCreate.mock.calls[0]?.[0].messages
      expect(sentMessages).toHaveLength(3)
      expect(sentMessages[2]).toEqual({
        role: 'user',
        content: [
          { type: 'tool_result', tool_use_id: 'tc_a', content: 'result-A' },
          { type: 'tool_result', tool_use_id: 'tc_b', content: 'result-B' },
        ],
      })
    })

    it('merges user message immediately preceding tool messages into a single Anthropic user message', async () => {
      // arrange
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      const messages = [
        { role: 'user' as const, content: 'start' },
        { role: 'assistant' as const, toolCalls: [{ id: 'tc_c', name: 'fn', input: {} }] },
        { role: 'user' as const, content: 'here is context' },
        { role: 'tool' as const, toolCallId: 'tc_c', content: 'tool-result' },
      ]

      // act
      await claude.invoke(messages)

      // assert
      const sentMessages = mockCreate.mock.calls[0]?.[0].messages
      expect(sentMessages[sentMessages.length - 1]).toEqual({
        role: 'user',
        content: [
          { type: 'text', text: 'here is context' },
          { type: 'tool_result', tool_use_id: 'tc_c', content: 'tool-result' },
        ],
      })
    })

    it('wraps tool messages in a new user message when no user message precedes them', async () => {
      // arrange
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      const messages = [
        { role: 'user' as const, content: 'start' },
        { role: 'assistant' as const, toolCalls: [{ id: 'tc_d', name: 'fn', input: {} }] },
        { role: 'tool' as const, toolCallId: 'tc_d', content: 'result-D' },
      ]

      // act
      await claude.invoke(messages)

      // assert
      const sentMessages = mockCreate.mock.calls[0]?.[0].messages
      expect(sentMessages[sentMessages.length - 1]).toEqual({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 'tc_d', content: 'result-D' }],
      })
      expect(sentMessages[sentMessages.length - 1].content[0]).not.toHaveProperty('type', 'text')
    })

  })

  describe('Group 5: Error Propagation', () => {

    it('propagates Anthropic API error unchanged and does not call onEvent', async () => {
      // arrange
      const apiError = new Error('AuthenticationError')
      mockCreate.mockRejectedValue(apiError)
      const mockObserver = { onEvent: vi.fn() }
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'bad-key' })
      claude.bindObserver(mockObserver)

      // act
      const act = () => claude.invoke([{ role: 'user', content: 'hi' }])

      // assert
      await expect(act()).rejects.toThrow('AuthenticationError')
      expect(mockObserver.onEvent).not.toHaveBeenCalled()
    })

    it('propagates network error and returns no LLMResponse', async () => {
      // arrange
      const networkError = new Error('ECONNRESET')
      mockCreate.mockRejectedValue(networkError)
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })

      // act
      const act = () => claude.invoke([{ role: 'user', content: 'hi' }])

      // assert
      await expect(act()).rejects.toThrow('ECONNRESET')
    })

  })

  describe('Group 6: Edge Cases and Repeated Calls', () => {

    it('forwards empty messages array to SDK without modification', async () => {
      // arrange
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })

      // act
      await claude.invoke([]).catch(() => {})

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.objectContaining({ messages: [] }))
    })

    it('uses second observer when bindObserver is called twice', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'ok' }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 1, output_tokens: 1 },
      })
      const firstObserver = { onEvent: vi.fn() }
      const secondObserver = { onEvent: vi.fn() }
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      claude.bindObserver(firstObserver)
      claude.bindObserver(secondObserver)

      // act
      await claude.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(secondObserver.onEvent).toHaveBeenCalledOnce()
      expect(firstObserver.onEvent).not.toHaveBeenCalled()
    })

    it('uses most recently set StepContext when setStepContext is called twice', async () => {
      // arrange
      const mockObserver = { onEvent: vi.fn() }
      const claude = new Claude('claude-sonnet-4-6', { apiKey: 'test-key' })
      claude.bindObserver(mockObserver)
      claude.setStepContext({ agentId: 'a1', sessionId: 's1', stepName: 'step-one' })
      claude.setStepContext({ agentId: 'a1', sessionId: 's1', stepName: 'step-two' })

      // act
      await claude.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledWith(
        { agentId: 'a1', sessionId: 's1', stepName: 'step-two' },
        'llm.response',
        expect.any(Object)
      )
    })

  })

})
