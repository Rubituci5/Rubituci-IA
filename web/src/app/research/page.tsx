"use client"

import { useEffect, useState } from "react"
import { Search, Loader2, Globe, FileText, TrendingUp, Clock, ExternalLink } from "lucide-react"
import { useEntity } from "@/hooks/useEntity"
import { useAuthStore } from "@/hooks/useAuth"

const API_URL = process.env.NEXT_PUBLIC_API_URL || ""

export default function ResearchPage() {
  const { researchMetrics, fetchResearch, triggerResearch } = useEntity()
  const { accessToken } = useAuthStore()
  const [query, setQuery] = useState("")
  const [isSearching, setIsSearching] = useState(false)
  const [results, setResults] = useState<any[]>([])
  const [error, setError] = useState("")

  useEffect(() => {
    fetchResearch()
  }, [fetchResearch])

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || isSearching) return

    setIsSearching(true)
    setError("")

    try {
      const res = await fetch(`${API_URL}/api/research/search`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ query, max_results: 10 }),
      })

      if (!res.ok) throw new Error("Search failed")

      const data = await res.json()
      setResults(data.results || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed")
    } finally {
      setIsSearching(false)
    }
  }

  const handleAutonomousResearch = async () => {
    if (!query.trim()) return

    setIsSearching(true)
    setError("")

    try {
      await triggerResearch(query)
      setResults([])
      fetchResearch()
    } catch (e) {
      setError(e instanceof Error ? e.message : "Research failed")
    } finally {
      setIsSearching(false)
    }
  }

  return (
    <div className="h-full flex flex-col p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-surface-900 dark:text-surface-100">Research</h1>
        <p className="text-surface-500 dark:text-surface-400 mt-1">
          Autonomous web search and information gathering
        </p>
      </div>

      {/* Search Interface */}
      <div className="card p-6">
        <form onSubmit={handleSearch} className="space-y-4">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-surface-400" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search the web... (e.g., 'latest AI alignment research')"
                className="input pl-10"
                disabled={isSearching}
              />
            </div>
            <button type="submit" disabled={!query.trim() || isSearching} className="btn-primary px-6">
              {isSearching ? <Loader2 className="w-5 h-5 animate-spin" /> : "Search"}
            </button>
            <button
              type="button"
              onClick={handleAutonomousResearch}
              disabled={!query.trim() || isSearching}
              className="btn-secondary px-6"
            >
              <Globe className="w-5 h-5 mr-2" />
              Autonomous Research
            </button>
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 dark:text-red-400 text-sm">
              {error}
            </div>
          )}
        </form>
      </div>

      {/* Metrics */}
      {researchMetrics && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="card p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-surface-500">Total Queries</p>
                <p className="text-2xl font-bold text-surface-900 dark:text-surface-100">{researchMetrics.total_queries}</p>
              </div>
              <div className="w-10 h-10 rounded-xl bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                <Search className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              </div>
            </div>
          </div>
          <div className="card p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-surface-500">Total Sources</p>
                <p className="text-2xl font-bold text-surface-900 dark:text-surface-100">{researchMetrics.total_sources}</p>
              </div>
              <div className="w-10 h-10 rounded-xl bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                <FileText className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
            </div>
          </div>
          <div className="card p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-surface-500">Avg Credibility</p>
                <p className="text-2xl font-bold text-surface-900 dark:text-surface-100">
                  {(researchMetrics.avg_credibility * 100).toFixed(0)}%
                </p>
              </div>
              <div className="w-10 h-10 rounded-xl bg-yellow-100 dark:bg-yellow-900/30 flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
              </div>
            </div>
          </div>
          <div className="card p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-surface-500">Source Types</p>
                <p className="text-2xl font-bold text-surface-900 dark:text-surface-100">
                  {Object.keys(researchMetrics.sources_by_type).length}
                </p>
              </div>
              <div className="w-10 h-10 rounded-xl bg-purple-100 dark:bg-purple-900/30 flex items-center justify-center">
                <Globe className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Sources by Type */}
      {researchMetrics && Object.keys(researchMetrics.sources_by_type).length > 0 && (
        <div className="card">
          <div className="p-4 border-b border-surface-200 dark:border-surface-700">
            <h2 className="font-semibold text-surface-900 dark:text-surface-100">Sources by Type</h2>
          </div>
          <div className="p-4">
            <div className="flex flex-wrap gap-2">
              {Object.entries(researchMetrics.sources_by_type).map(([type, count]) => (
                <span key={type} className="px-3 py-1 rounded-full bg-surface-100 dark:bg-surface-800 text-sm font-medium">
                  {type}: {count}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Search Results */}
      {results.length > 0 && (
        <div className="card">
          <div className="p-4 border-b border-surface-200 dark:border-surface-700 flex items-center justify-between">
            <h2 className="font-semibold text-surface-900 dark:text-surface-100">Search Results</h2>
            <span className="text-sm text-surface-500">{results.length} results</span>
          </div>
          <div className="divide-y divide-surface-200 dark:divide-surface-700">
            {results.map((result, index) => (
              <div key={index} className="p-4 hover:bg-surface-50 dark:hover:bg-surface-800/50">
                <div className="flex items-start gap-3">
                  <span className="text-surface-400 font-mono mt-1">{index + 1}.</span>
                  <div className="flex-1 min-w-0">
                    <a
                      href={result.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-medium text-entity-600 dark:text-entity-400 hover:underline truncate block"
                    >
                      {result.title}
                    </a>
                    <p className="text-sm text-surface-500 dark:text-surface-400 mt-1 line-clamp-2">
                      {result.snippet}
                    </p>
                    <div className="flex items-center gap-3 mt-2 text-xs text-surface-400">
                      <span>{result.domain}</span>
                      {result.credibility !== undefined && (
                        <span className="flex items-center gap-1">
                          <TrendingUp className="w-3 h-3" />
                          {(result.credibility * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                  <a
                    href={result.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-surface-400 hover:text-entity-500"
                    title="Open source"
                  >
                    <ExternalLink className="w-4 h-4" />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Queries */}
      {researchMetrics && (researchMetrics.recent_queries?.length ?? 0) > 0 && (
        <div className="card">
          <div className="p-4 border-b border-surface-200 dark:border-surface-700">
            <h2 className="font-semibold text-surface-900 dark:text-surface-100">Recent Queries</h2>
          </div>
          <div className="divide-y divide-surface-200 dark:divide-surface-700">
            {(researchMetrics.recent_queries ?? []).slice(0, 10).map((q: any, index: number) => (
              <div key={index} className="p-4 hover:bg-surface-50 dark:hover:bg-surface-800/50 flex items-center justify-between">
                <div>
                  <p className="font-medium text-surface-900 dark:text-surface-100 truncate max-w-md">{q.query}</p>
                  <p className="text-sm text-surface-500">{q.results_count} results • {formatRelativeTime(q.created_at)}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
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
