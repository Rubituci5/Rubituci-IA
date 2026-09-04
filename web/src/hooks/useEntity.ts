"use client"

import { create } from "zustand"
import { EntityStatus, EvolutionMetrics, ResearchMetrics, GenerationInfo } from "@/types"
import { useAuthStore } from "./useAuth"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

interface EntityState {
  entityStatus: EntityStatus["status"] | null
  generation: number | null
  evolutionMetrics: EvolutionMetrics | null
  researchMetrics: ResearchMetrics | null
  generations: GenerationInfo[]
  isLoading: boolean

  fetchStatus: () => Promise<void>
  fetchEvolution: () => Promise<void>
  fetchResearch: () => Promise<void>
  fetchGenerations: () => Promise<void>
  triggerReflection: () => Promise<void>
  triggerConsolidation: () => Promise<void>
  triggerResearch: (query: string) => Promise<void>
}

export const useEntityStore = create<EntityState>((set, get) => ({
  entityStatus: null,
  generation: null,
  evolutionMetrics: null,
  researchMetrics: null,
  generations: [],
  isLoading: true,

  fetchStatus: async () => {
    const { accessToken } = useAuthStore.getState()

    if (!accessToken) {
      set({ isLoading: false })
      return
    }

    try {
      const res = await fetch(`${API_URL}/api/entity/status`, {
        headers: { "Authorization": `Bearer ${accessToken}` },
      })

      if (res.ok) {
        const data = await res.json()
        set({
          entityStatus: data.status,
          generation: data.generation,
        })
      }
    } catch (e) {
      console.error("Failed to fetch entity status:", e)
    } finally {
      set({ isLoading: false })
    }
  },

  fetchEvolution: async () => {
    const { accessToken } = useAuthStore.getState()
    if (!accessToken) return

    try {
      const res = await fetch(`${API_URL}/api/evolution/dashboard`, {
        headers: { "Authorization": `Bearer ${accessToken}` },
      })
      if (res.ok) {
        const data = await res.json()
        set({
          evolutionMetrics: {
            generations: get().generations,
            current_generation: data.current_generation || 1,
            total_training_steps: data.current_metrics?.steps_completed || 0,
            total_memories: data.total_experiences || 0,
            total_beliefs: data.total_beliefs || 0,
          },
          generation: data.current_generation || 1,
        })
      }
    } catch (e) {
      console.error("Failed to fetch evolution:", e)
    }
  },

  fetchResearch: async () => {
    const { accessToken } = useAuthStore.getState()
    if (!accessToken) return

    try {
      const res = await fetch(`${API_URL}/api/research/metrics`, {
        headers: { "Authorization": `Bearer ${accessToken}` },
      })
      if (res.ok) {
        const data = await res.json()
        set({ researchMetrics: data })
      }
    } catch (e) {
      console.error("Failed to fetch research:", e)
    }
  },

  fetchGenerations: async () => {
    const { accessToken } = useAuthStore.getState()
    if (!accessToken) return

    try {
      const res = await fetch(`${API_URL}/api/evolution/generations`, {
        headers: { "Authorization": `Bearer ${accessToken}` },
      })
      if (res.ok) {
        const data = await res.json()
        const generations = data.map((item: any) => ({
          number: item.generation,
          status: item.status,
          metrics: item.eval_metrics || {},
          created_at: item.promoted_at || new Date(0).toISOString(),
          activated_at: item.promoted_at || undefined,
          parent_generation: item.parent_generation,
        }))
        set(state => ({ generations, evolutionMetrics: state.evolutionMetrics ? { ...state.evolutionMetrics, generations } : null }))
      }
    } catch (e) {
      console.error("Failed to fetch generations:", e)
    }
  },

  triggerReflection: async () => {
    const { accessToken } = useAuthStore.getState()
    if (!accessToken) return

    try {
      await fetch(`${API_URL}/api/entity/reflect`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${accessToken}` },
      })
      // Refresh status after
      get().fetchStatus()
    } catch (e) {
      console.error("Failed to trigger reflection:", e)
    }
  },

  triggerConsolidation: async () => {
    const { accessToken } = useAuthStore.getState()
    if (!accessToken) return

    try {
      await fetch(`${API_URL}/api/entity/consolidate`, {
        method: "POST",
        headers: { "Authorization": `Bearer ${accessToken}` },
      })
      get().fetchStatus()
    } catch (e) {
      console.error("Failed to trigger consolidation:", e)
    }
  },

  triggerResearch: async (query) => {
    const { accessToken } = useAuthStore.getState()
    if (!accessToken) return

    try {
      await fetch(`${API_URL}/api/research/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ query }),
      })
      get().fetchResearch()
    } catch (e) {
      console.error("Failed to trigger research:", e)
    }
  },
}))

export function useEntity() {
  const store = useEntityStore()

  return {
    entityStatus: store.entityStatus,
    generation: store.generation,
    evolutionMetrics: store.evolutionMetrics,
    researchMetrics: store.researchMetrics,
    generations: store.generations,
    isLoading: store.isLoading,
    fetchStatus: store.fetchStatus,
    fetchEvolution: store.fetchEvolution,
    fetchResearch: store.fetchResearch,
    fetchGenerations: store.fetchGenerations,
    triggerReflection: store.triggerReflection,
    triggerConsolidation: store.triggerConsolidation,
    triggerResearch: store.triggerResearch,
  }
}
