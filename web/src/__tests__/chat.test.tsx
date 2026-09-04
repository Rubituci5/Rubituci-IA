import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { ChatInterface } from '@/components/ChatInterface'

// Mock WebSocket
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  removeEventListener: jest.fn(),
  readyState: 1, // OPEN
}

global.WebSocket = jest.fn(() => mockWebSocket) as any

// Mock fetch for auth
global.fetch = jest.fn()

describe('ChatInterface', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    ;(global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ access_token: 'test-token' }),
    })
    localStorage.setItem('access_token', 'test-token')
  })

  afterEach(() => {
    localStorage.clear()
  })

  it('renders chat interface', () => {
    render(<ChatInterface />)
    expect(screen.getByPlaceholderText('Digite sua mensagem...')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /enviar/i })).toBeInTheDocument()
  })

  it('shows connection status', () => {
    render(<ChatInterface />)
    expect(screen.getByText(/conectado/i)).toBeInTheDocument()
  })

  it('sends message on enter', async () => {
    render(<ChatInterface />)
    const input = screen.getByPlaceholderText('Digite sua mensagem...')

    fireEvent.change(input, { target: { value: 'Hello' } })
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })

    await waitFor(() => {
      expect(mockWebSocket.send).toHaveBeenCalledWith(
        expect.stringContaining('"type":"chat"')
      )
    })
  })

  it('displays messages in chat history', async () => {
    render(<ChatInterface />)
    const input = screen.getByPlaceholderText('Digite sua mensagem...')

    fireEvent.change(input, { target: { value: 'Test message' } })
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })

    await waitFor(() => {
      expect(screen.getByText('Test message')).toBeInTheDocument()
    })
  })

  it('shows loading state while waiting for response', async () => {
    render(<ChatInterface />)
    const input = screen.getByPlaceholderText('Digite sua mensagem...')

    fireEvent.change(input, { target: { value: 'Hello' } })
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' })

    await waitFor(() => {
      expect(screen.getByText(/pensando/i)).toBeInTheDocument()
    })
  })

  it('handles websocket disconnect', async () => {
    render(<ChatInterface />)

    // Simulate disconnect
    const wsInstance = (global.WebSocket as jest.Mock).mock.results[0].value
    const closeHandler = wsInstance.addEventListener.mock.calls.find(
      (call: any[]) => call[0] === 'close'
    )?.[1]

    if (closeHandler) {
      closeHandler({ code: 1000, reason: 'Normal closure' })
    }

    await waitFor(() => {
      expect(screen.getByText(/desconectado/i)).toBeInTheDocument()
    })
  })
})