"use client"

import { useEffect } from "react"
import { TrendingUp, TrendingDown, Activity, Database, Zap, Clock, ArrowUp, ArrowDown, Minus } from "lucide-react"
import { useEntity } from "@/hooks/useEntity"
import { GenerationCard } from "@/components/GenerationCard"
import { MetricCard } from "@/components/MetricCard"

export default function EvolutionPage() {
  const {
    evolutionMetrics,
    generations,
    generation,
    isLoading,
    fetchEvolution,
    fetchGenerations,
  } = useEntity()

  useEffect(() => {
    fetchEvolution()
    fetchGenerations()
  }, [fetchEvolution, fetchGenerations])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[60vh]">
        <div className="animate-spin w-8 h-8 border-4 border-entity-500 border-t-transparent rounded-full" />
      </div>
    )
  }

  const currentGen = generations.find((g) => g.number === generation)

  return (
    <div className="h-full flex flex-col p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-surface-900 dark:text-surface-100">Evolution Dashboard</h1>
        <p className="text-surface-500 dark:text-surface-400 mt-1">
          Track generation progress, training metrics, and entity evolution over time
        </p>
      </div>

      {/* Current Generation Status */}
      <div className="card p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 rounded-2xl bg-entity-100 dark:bg-entity-900/30 flex items-center justify-center">
              <Zap className="w-8 h-8 text-entity-600 dark:text-entity-400" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-surface-900 dark:text-surface-100">
                Generation {generation?.toString().padStart(6, "0") || "—"}
              </h2>
              <p className="text-surface-500 dark:text-surface-400">
                {currentGen?.status || "Unknown"} • {currentGen?.activated_at ? `Activated ${formatRelativeTime(currentGen.activated_at)}` : "Not yet activated"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <MetricCard
              label="Eval Loss"
              value={currentGen?.metrics?.eval_loss?.toFixed(4) || "—"}
              trend={currentGen?.metrics?.eval_loss ? "down" : null}
            />
            <MetricCard
              label="Perplexity"
              value={currentGen?.metrics?.perplexity?.toFixed(2) || "—"}
              trend={currentGen?.metrics?.perplexity ? "down" : null}
            />
            <MetricCard
              label="Train Loss"
              value={currentGen?.metrics?.train_loss?.toFixed(4) || "—"}
            />
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Total Generations"
          value={evolutionMetrics?.generations.length || 0}
          icon={Database}
        />
        <MetricCard
          label="Training Steps"
          value={formatNumber(evolutionMetrics?.total_training_steps || 0)}
          icon={Activity}
        />
        <MetricCard
          label="Total Memories"
          value={formatNumber(evolutionMetrics?.total_memories || 0)}
          icon={Database}
        />
        <MetricCard
          label="Total Beliefs"
          value={formatNumber(evolutionMetrics?.total_beliefs || 0)}
          icon={Zap}
        />
      </div>

      {/* Generations Timeline */}
      <div className="card">
        <div className="p-6 border-b border-surface-200 dark:border-surface-700">
          <h2 className="text-xl font-semibold text-surface-900 dark:text-surface-100">Generation History</h2>
        </div>
        <div className="divide-y divide-surface-200 dark:divide-surface-700">
          {generations.length === 0 ? (
            <div className="p-12 text-center text-surface-500">
              <Zap className="w-12 h-12 mx-auto mb-4 text-surface-300" />
              <p>No generations yet. Training will create the first generation.</p>
            </div>
          ) : (
            generations.map((gen) => (
              <GenerationCard key={gen.number} generation={gen} isCurrent={gen.number === generation} />
            ))
          )}
        </div>
      </div>
    </div>
  )
}

function formatNumber(num: number): string {
  if (num >= 1000000) return (num / 1000000).toFixed(1) + "M"
  if (num >= 1000) return (num / 1000).toFixed(1) + "K"
  return num.toString()
}

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffMinutes = Math.floor(diffMs / (1000 * 60))

  if (diffDays > 0) return `${diffDays}d ago`
  if (diffHours > 0) return `${diffHours}h ago`
  if (diffMinutes > 0) return `${diffMinutes}m ago`
  return "Just now"
}