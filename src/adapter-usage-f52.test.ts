import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Message } from '@noetaris/harness-types'

// mockRetrieve and mockCreate are declared in outer scope so the mock factory can close over them
const mockRetrieve = vi.fn()
const mockCreate = vi.fn()

vi.mock('@anthropic-ai/sdk', () => {
  function MockAnthropic() {
    return {
      models: { retrieve: mockRetrieve },
      messages: { create: mockCreate },
    }
  }
  return { default: MockAnthropic }
})

import { Claude } from './claude.js'
import { MockClaude } from './mock-claude.js'

const messages: Message[] = [{ role: 'user', content: 'Hello' }]

const minimalCreateResponse = {
  content: [{ type: 'text', text: 'Hello' }],
  usage: { input_tokens: 10, output_tokens: 5 },
  stop_reason: 'end_turn',
}

beforeEach(() => {
  vi.clearAllMocks()
  mockRetrieve.mockResolvedValue({ max_input_tokens: 200000, id: 'claude-opus-4-5' })
  mockCreate.mockResolvedValue(minimalCreateResponse)
})

describe('Claude — AdapterUsageF52', () => {

  describe('Group 1: Context Window Fetch, Cache, and Error Suppression', () => {

    it('populates contextWindowSize from max_input_tokens on first invoke call', async () => {
      // arrange
      mockRetrieve.mockResolvedValue({ max_input_tokens: 200000, id: 'claude-opus-4-5' })
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'Hello' }],
        usage: { input_tokens: 10, output_tokens: 5 },
        stop_reason: 'end_turn',
      })
      const claude = new Claude('claude-opus-4-5', { apiKey: 'test-key' })

      // act
      const result = await claude.invoke(messages)

      // assert
      expect(mockRetrieve).toHaveBeenCalledOnce()
      expect(mockRetrieve).toHaveBeenCalledWith('claude-opus-4-5')
      expect(result.usage.contextWindowSize).toBe(200000)
      expect(result.usage.inputTokens).toBe(10)
      expect(result.usage.outputTokens).toBe(5)
    })

    it('does not call models.retrieve on subsequent invoke calls after successful first fetch', async () => {
      // arrange
      mockRetrieve.mockResolvedValue({ max_input_tokens: 200000, id: 'claude-opus-4-5' })
      const claude = new Claude('claude-opus-4-5', { apiKey: 'test-key' })
      await claude.invoke(messages)
      vi.clearAllMocks()

      // act
      const result = await claude.invoke(messages)

      // assert
      expect(mockRetrieve).not.toHaveBeenCalled()
      expect(result.usage.contextWindowSize).toBe(200000)
    })

    it('suppresses models.retrieve error; invoke succeeds with contextWindowSize undefined; emitted event has no contextWindowSize field', async () => {
      // arrange
      mockRetrieve.mockRejectedValue(new Error('Network error'))
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'Hi' }],
        usage: { input_tokens: 8, output_tokens: 3 },
        stop_reason: 'end_turn',
      })
      const observer = { emit: vi.fn() }
      const claude = new Claude('claude-opus-4-5', { apiKey: 'test-key' })
      claude.bindObserver({ onEvent: observer.emit })

      // act
      const result = await claude.invoke(messages)

      // assert
      expect(result.usage.contextWindowSize).toBeUndefined()
      expect(result.usage.inputTokens).toBe(8)
      expect(result.usage.outputTokens).toBe(3)
      expect(result.text).toBe('Hi')
      const event = observer.emit.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event).not.toHaveProperty('contextWindowSize')
    })

    it('does not retry the fetch on subsequent calls after a first-call fetch failure', async () => {
      // arrange
      mockRetrieve.mockRejectedValue(new Error('Timeout'))
      mockCreate.mockResolvedValue(minimalCreateResponse)
      const claude = new Claude('claude-opus-4-5', { apiKey: 'test-key' })
      await claude.invoke(messages)
      vi.clearAllMocks()
      mockCreate.mockResolvedValue(minimalCreateResponse)

      // act
      const result = await claude.invoke(messages)

      // assert
      expect(mockRetrieve).not.toHaveBeenCalled()
      expect(result.usage.contextWindowSize).toBeUndefined()
    })

    it('cache is populated from prefetch even when messages.create subsequently throws; later calls use cache without re-fetching', async () => {
      // arrange
      mockRetrieve.mockResolvedValue({ max_input_tokens: 128000 })
      mockCreate
        .mockRejectedValueOnce(new Error('Provider error'))
        .mockResolvedValue(minimalCreateResponse)
      const claude = new Claude('claude-opus-4-5', { apiKey: 'test-key' })

      // act step 1 — first call throws
      await expect(claude.invoke(messages)).rejects.toThrow('Provider error')
      vi.clearAllMocks()
      mockCreate.mockResolvedValue(minimalCreateResponse)

      // act step 2 — second call succeeds
      const result = await claude.invoke(messages)

      // assert
      expect(mockRetrieve).not.toHaveBeenCalled()
      expect(result.usage.contextWindowSize).toBe(128000)
    })

  })

  describe('Group 2: Token Counts and Event Emission', () => {

    it('emitted LLMUsageEvent has contextWindowSize equal to result.usage.contextWindowSize; token counts match provider response', async () => {
      // arrange
      mockRetrieve.mockResolvedValue({ max_input_tokens: 200000 })
      mockCreate.mockResolvedValue({
        content: [{ type: 'text', text: 'Reply' }],
        usage: { input_tokens: 100, output_tokens: 42 },
        stop_reason: 'end_turn',
      })
      const observer = { onEvent: vi.fn() }
      const claude = new Claude('claude-opus-4-5', { apiKey: 'test-key' })
      claude.bindObserver(observer)

      // act
      const result = await claude.invoke(messages)

      // assert
      expect(result.usage.inputTokens).toBe(100)
      expect(result.usage.outputTokens).toBe(42)
      expect(result.usage.contextWindowSize).toBe(200000)
      const event = observer.onEvent.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event.contextWindowSize).toBe(200000)
      expect(event.tokens.input).toBe(100)
      expect(event.tokens.output).toBe(42)
    })

  })

})

describe('MockClaude — AdapterUsageF52', () => {

  describe('Group 6: Fixed Zero Usage', () => {

    it('invoke returns usage = { inputTokens: 0, outputTokens: 0 } with no contextWindowSize; emitted event has no contextWindowSize', async () => {
      // arrange
      const observer = { onEvent: vi.fn() }
      const mockClaude = new MockClaude({
        text: 'Ok',
        toolCalls: [],
        stopReason: 'end',
        usage: { inputTokens: 0, outputTokens: 0 },
      })
      mockClaude.bindObserver(observer)

      // act
      const result = await mockClaude.invoke(messages)

      // assert
      expect(result.usage).toEqual({ inputTokens: 0, outputTokens: 0 })
      expect(result.usage.contextWindowSize).toBeUndefined()
      const event = observer.onEvent.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event).not.toHaveProperty('contextWindowSize')
    })

  })

})
