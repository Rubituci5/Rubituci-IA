"use client"

import { GenerationInfo } from "@/types"
import { CheckCircle, Clock, AlertCircle, Archive, TrendingUp, TrendingDown } from "lucide-react"

interface GenerationCardProps {
  generation: GenerationInfo
  isCurrent: boolean
}

export function GenerationCard({ generation, isCurrent }: GenerationCardProps) {
  const statusConfig = {
    active: { label: "Active", color: "bg-green-500", icon: CheckCircle },
    training: { label: "Training", color: "bg-yellow-500 animate-pulse", icon: Clock },
    rolling_out: { label: "Rolling Out", color: "bg-blue-500", icon: TrendingUp },
    archived: { label: "Archived", color: "bg-gray-500", icon: Archive },
    rejected: { label: "Rejected", color: "bg-red-500", icon: AlertCircle },
    rolled_back: { label: "Rolled Back", color: "bg-orange-500", icon: AlertCircle },
  } as const

  const config = statusConfig[generation.status as keyof typeof statusConfig] || { label: generation.status, color: "bg-gray-500", icon: Clock }
  const Icon = config.icon

  const metrics = generation.metrics || {}

  return (
    <div className={`p-4 hover:bg-surface-50 dark:hover:bg-surface-800/50 transition-colors ${isCurrent ? "bg-entity-50 dark:bg-entity-900/20 border-l-4 border-entity-500" : ""}`}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        {/* Generation Info */}
        <div className="flex items-center gap-4">
          <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${config.color} text-white`}>
            <Icon className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono font-semibold text-surface-900 dark:text-surface-100">
                Gen {generation.number.toString().padStart(6, "0")}
              </span>
              {isCurrent && (
                <span className="px-2 py-0.5 text-xs rounded-full bg-entity-100 dark:bg-entity-900 text-entity-700 dark:text-entity-300">
                  Current
                </span>
              )}
            </div>
            <p className="text-sm text-surface-500 dark:text-surface-400">
              {generation.created_at && `Created ${formatRelativeTime(generation.created_at)}`}
              {generation.parent_generation && ` • Child of Gen ${generation.parent_generation.toString().padStart(6, "0")}`}
            </p>
          </div>
        </div>

        {/* Status Badge */}
        <div className="flex items-center gap-3">
          <span className={`px-3 py-1 rounded-full text-xs font-medium text-white ${config.color}`}>
            {config.label}
          </span>
        </div>

        {/* Metrics */}
        <div className="flex flex-wrap items-center gap-6 text-sm">
          {metrics.eval_loss !== undefined && (
            <div className="flex items-center gap-1.5">
              <TrendingDown className="w-4 h-4 text-green-500" />
              <span className="font-mono text-surface-900 dark:text-surface-100">{metrics.eval_loss.toFixed(4)}</span>
              <span className="text-surface-500">eval loss</span>
            </div>
          )}
          {metrics.perplexity !== undefined && (
            <div className="flex items-center gap-1.5">
              <TrendingDown className="w-4 h-4 text-green-500" />
              <span className="font-mono text-surface-900 dark:text-surface-100">{metrics.perplexity.toFixed(2)}</span>
              <span className="text-surface-500">ppl</span>
            </div>
          )}
          {metrics.train_loss !== undefined && (
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-surface-900 dark:text-surface-100">{metrics.train_loss.toFixed(4)}</span>
              <span className="text-surface-500">train loss</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
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