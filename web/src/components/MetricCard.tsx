"use client"

import { LucideIcon } from "lucide-react"

interface MetricCardProps {
  label: string
  value: string | number
  icon?: LucideIcon
  trend?: "up" | "down" | null
}

export function MetricCard({ label, value, icon: Icon, trend }: MetricCardProps) {
  return (
    <div className="card p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-surface-500 dark:text-surface-400">{label}</p>
          <p className="text-2xl font-bold text-surface-900 dark:text-surface-100 mt-1 font-mono">
            {value}
          </p>
        </div>
        {Icon && (
          <div className="w-10 h-10 rounded-xl bg-entity-100 dark:bg-entity-900/30 flex items-center justify-center">
            <Icon className="w-5 h-5 text-entity-600 dark:text-entity-400" />
          </div>
        )}
      </div>
      {trend && (
        <div className="mt-2 flex items-center gap-1 text-sm">
          {trend === "down" ? (
            <>
              <span className="text-green-600 dark:text-green-400">↓</span>
              <span className="text-green-600 dark:text-green-400">Improving</span>
            </>
          ) : (
            <>
              <span className="text-red-600 dark:text-red-400">↑</span>
              <span className="text-red-600 dark:text-red-400">Increasing</span>
            </>
          )}
        </div>
      )}
    </div>
  )
}