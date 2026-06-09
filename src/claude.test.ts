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
      expect(mockObserver.onEvent).toHaveBeenCalledWith(
        expect.any(Object),
        'llm.response',
        { tokens: { input: 25, output: 10 }, modelId: 'claude-sonnet-4-6', stopReason: 'end', providerName: 'anthropic' }
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

    it('propagates Anthropic API error unchanged and does not emit llm.response', async () => {
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
      const eventTypes = mockObserver.onEvent.mock.calls.map((c: unknown[]) => c[1])
      expect(eventTypes).not.toContain('llm.response')
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

  describe('Group 7: Generation Params', () => {

    describe('default behavior with no generation params', () => {

      it('sends max_tokens: 4096 and omits optional params when no generation options are given', async () => {
        // arrange
        mockCreate.mockResolvedValue({
          id: 'msg_test',
          type: 'message',
          role: 'assistant',
          content: [{ type: 'text', text: 'hello' }],
          model: 'claude-3-5-haiku-20241022',
          stop_reason: 'end_turn',
          usage: { input_tokens: 10, output_tokens: 5 },
        })
        const claude = new Claude('claude-3-5-haiku-20241022')

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        expect(mockCreate).toHaveBeenCalledOnce()
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).toMatchObject({ max_tokens: 4096 })
        expect(callArg).not.toHaveProperty('temperature')
        expect(callArg).not.toHaveProperty('top_p')
        expect(callArg).not.toHaveProperty('thinking')
      })

    })

    describe('forwarding individual and combined generation params', () => {

      it('sends max_tokens: 8192 when maxTokens: 8192 is set', async () => {
        // arrange
        const claude = new Claude('claude-3-5-haiku-20241022', { maxTokens: 8192 })

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).toMatchObject({ max_tokens: 8192 })
      })

      it('sends temperature: 0.7 and preserves default max_tokens: 4096 when temperature is set', async () => {
        // arrange
        const claude = new Claude('claude-3-5-haiku-20241022', { temperature: 0.7 })

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).toMatchObject({ temperature: 0.7, max_tokens: 4096 })
      })

      it('sends top_p: 0.9 when topP: 0.9 is set', async () => {
        // arrange
        const claude = new Claude('claude-3-5-haiku-20241022', { topP: 0.9 })

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).toMatchObject({ top_p: 0.9 })
        expect(callArg).not.toHaveProperty('topP')
      })

      it('sends thinking with budget_tokens when thinking.budgetTokens is set', async () => {
        // arrange
        const claude = new Claude('claude-3-5-haiku-20241022', { thinking: { type: 'enabled', budgetTokens: 2000 } })

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).toMatchObject({ thinking: { type: 'enabled', budget_tokens: 2000 } })
        expect(callArg.thinking).not.toHaveProperty('budgetTokens')
      })

      it('sends all four generation params with correct provider-form field names when all options are set', async () => {
        // arrange
        const claude = new Claude('claude-3-5-haiku-20241022', {
          temperature: 0.5,
          maxTokens: 8192,
          topP: 0.8,
          thinking: { type: 'enabled', budgetTokens: 1500 },
        })

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).toMatchObject({
          temperature: 0.5,
          max_tokens: 8192,
          top_p: 0.8,
          thinking: { type: 'enabled', budget_tokens: 1500 },
        })
      })

    })

    describe('absent options use defaults or are omitted', () => {

      it('omits temperature from body when temperature is omitted', async () => {
        // arrange
        const claude = new Claude('claude-3-5-haiku-20241022', {})

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).not.toHaveProperty('temperature')
        expect(callArg).toMatchObject({ max_tokens: 4096 })
      })

      it('sends max_tokens: 4096 when maxTokens is omitted', async () => {
        // arrange
        const claude = new Claude('claude-3-5-haiku-20241022', {})

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).toMatchObject({ max_tokens: 4096 })
      })

      it('omits top_p from body when topP is omitted', async () => {
        // arrange
        const claude = new Claude('claude-3-5-haiku-20241022', {})

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).not.toHaveProperty('top_p')
      })

      it('omits thinking from body when thinking is omitted', async () => {
        // arrange
        const claude = new Claude('claude-3-5-haiku-20241022', {})

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).not.toHaveProperty('thinking')
      })

    })

    describe('generation params coexist with tools', () => {

      it('includes both thinking and tools in request body when both are provided', async () => {
        // arrange
        const tool = {
          name: 'get_weather',
          description: 'Returns weather for a city',
          inputSchema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        }
        const claude = new Claude('claude-3-5-haiku-20241022', { thinking: { type: 'enabled', budgetTokens: 1000 } })

        // act
        await claude.invoke([{ role: 'user', content: 'What is the weather?' }], { tools: [tool] })

        // assert
        const callArg = mockCreate.mock.calls[0]?.[0]
        expect(callArg).toMatchObject({ thinking: { type: 'enabled', budget_tokens: 1000 } })
        expect(callArg.tools).toHaveLength(1)
        expect(callArg.tools[0]).toMatchObject({ name: 'get_weather' })
      })

    })

    describe('observer event integrity', () => {

      it('emits llm.response event with correct token counts and metadata when generation params are set', async () => {
        // arrange
        mockCreate.mockResolvedValue({
          id: 'msg_obs',
          type: 'message',
          role: 'assistant',
          content: [{ type: 'text', text: 'result' }],
          model: 'claude-3-5-haiku-20241022',
          stop_reason: 'end_turn',
          usage: { input_tokens: 20, output_tokens: 15 },
        })
        const mockObserver = { onEvent: vi.fn() }
        const claude = new Claude('claude-3-5-haiku-20241022', { temperature: 0.9, maxTokens: 2048, topP: 0.7 })
        claude.bindObserver(mockObserver)

        // act
        await claude.invoke([{ role: 'user', content: 'hi' }])

        // assert
        expect(mockObserver.onEvent).toHaveBeenCalledWith(
          expect.anything(),
          'llm.response',
          expect.objectContaining({
            tokens: { input: 20, output: 15 },
            modelId: 'claude-3-5-haiku-20241022',
            stopReason: 'end',
            providerName: 'anthropic',
          })
        )
      })

    })

  })

  describe('Group 8: "llm.request" emission', () => {

    it('emits "llm.request" with modelId and providerName before client.messages.create', async () => {
      // arrange
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Claude('claude-3-5-haiku-20241022')
      adapter.bindObserver(mockObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }], { tools: [{ name: 'search', description: 'search', inputSchema: {} }] })

      // assert
      const calls = mockObserver.onEvent.mock.calls
      expect(calls[0]?.[1]).toBe('llm.request')
      expect(calls[0]?.[2]).toEqual({ modelId: 'claude-3-5-haiku-20241022', providerName: 'anthropic' })
      expect(mockCreate).toHaveBeenCalledOnce()
      expect(mockObserver.onEvent.mock.invocationCallOrder[0] ?? 0).toBeLessThan(mockCreate.mock.invocationCallOrder[0] ?? 0)
    })

    it('emits "llm.request" before "llm.response" on successful invoke', async () => {
      // arrange
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Claude('claude-3-5-haiku-20241022')
      adapter.bindObserver(mockObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledTimes(2)
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
      expect(mockObserver.onEvent.mock.calls[1]?.[1]).toBe('llm.response')
    })

    it('"llm.request" payload excludes messages and tools; "llm.response" payload excludes output', async () => {
      // arrange
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Claude('claude-3-5-haiku-20241022')
      adapter.bindObserver(mockObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      const requestPayload = mockObserver.onEvent.mock.calls[0]?.[2]
      expect(requestPayload).not.toHaveProperty('messages')
      expect(requestPayload).not.toHaveProperty('tools')
      const responsePayload = mockObserver.onEvent.mock.calls[1]?.[2]
      expect(responsePayload).not.toHaveProperty('output')
    })

    it('emits "llm.request" before SDK throw and does not emit "llm.response" on error', async () => {
      // arrange
      mockCreate.mockRejectedValue(new Error('AuthenticationError'))
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Claude('claude-3-5-haiku-20241022')
      adapter.bindObserver(mockObserver)

      // act
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).rejects.toThrow('AuthenticationError')

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledTimes(1)
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
    })

    it('resolves normally and does not throw when no observer is bound', async () => {
      // arrange
      const adapter = new Claude('claude-3-5-haiku-20241022')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(result).toBeDefined()
      expect(result.stopReason).toBe('end')
    })

    it('two consecutive invocations produce request/response/request/response sequence', async () => {
      // arrange
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Claude('claude-3-5-haiku-20241022')
      adapter.bindObserver(mockObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'first' }])
      await adapter.invoke([{ role: 'user', content: 'second' }])

      // assert
      expect(mockObserver.onEvent).toHaveBeenCalledTimes(4)
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
      expect(mockObserver.onEvent.mock.calls[1]?.[1]).toBe('llm.response')
      expect(mockObserver.onEvent.mock.calls[2]?.[1]).toBe('llm.request')
      expect(mockObserver.onEvent.mock.calls[3]?.[1]).toBe('llm.response')
    })

    it('both "llm.request" and "llm.response" carry the StepContext set via setStepContext', async () => {
      // arrange
      const mockObserver = { onEvent: vi.fn() }
      const adapter = new Claude('claude-3-5-haiku-20241022')
      adapter.bindObserver(mockObserver)
      const ctx = { agentId: 'agent-5', sessionId: 'sess-99', stepName: 'call-model' }
      adapter.setStepContext(ctx)

      // act
      await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(mockObserver.onEvent.mock.calls[0]?.[0]).toEqual(ctx)
      expect(mockObserver.onEvent.mock.calls[1]?.[0]).toEqual(ctx)
      expect(mockObserver.onEvent.mock.calls[0]?.[1]).toBe('llm.request')
      expect(mockObserver.onEvent.mock.calls[1]?.[1]).toBe('llm.response')
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
      expect(secondObserver.onEvent).toHaveBeenCalled()
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
