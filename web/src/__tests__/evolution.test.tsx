import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import { EvolutionDashboard } from '@/components/EvolutionDashboard'

global.fetch = jest.fn()

describe('EvolutionDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    ;(global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        generations: [
          {
            number: 1,
            status: 'active',
            metrics: { eval_loss: 2.5, perplexity: 12.3 },
            created_at: '2026-01-01T00:00:00Z',
            is_active: true,
          },
          {
            number: 2,
            status: 'training',
            metrics: { eval_loss: 1.8, perplexity: 6.2 },
            created_at: '2026-02-01T00:00:00Z',
            is_active: false,
          },
        ],
        current_generation: 1,
      }),
    })
  })

  it('renders evolution dashboard', () => {
    render(<EvolutionDashboard />)
    expect(screen.getByText(/evolução/i)).toBeInTheDocument()
    expect(screen.getByText(/geração/i)).toBeInTheDocument()
  })

  it('displays generation list', async () => {
    render(<EvolutionDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Geração 000001')).toBeInTheDocument()
      expect(screen.getByText('Geração 000002')).toBeInTheDocument()
    })
  })

  it('shows metrics for each generation', async () => {
    render(<EvolutionDashboard />)

    await waitFor(() => {
      expect(screen.getByText('2.5')).toBeInTheDocument() // eval_loss
      expect(screen.getByText('12.3')).toBeInTheDocument() // perplexity
    })
  })

  it('highlights active generation', async () => {
    render(<EvolutionDashboard />)

    await waitFor(() => {
      const activeGen = screen.getByText('Geração 000001').closest('tr')
      expect(activeGen).toHaveClass('bg-blue-50')
    })
  })

  it('shows training status badges', async () => {
    render(<EvolutionDashboard />)

    await waitFor(() => {
      expect(screen.getByText(/ativo/i)).toBeInTheDocument()
      expect(screen.getByText(/treinando/i)).toBeInTheDocument()
    })
  })

  it('handles loading state', () => {
    ;(global.fetch as jest.Mock).mockImplementation(() => new Promise(() => {})) // Never resolves

    render(<EvolutionDashboard />)
    expect(screen.getByText(/carregando/i)).toBeInTheDocument()
  })

  it('handles error state', async () => {
    ;(global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 500,
    })

    render(<EvolutionDashboard />)

    await waitFor(() => {
      expect(screen.getByText(/erro/i)).toBeInTheDocument()
    })
  })
})