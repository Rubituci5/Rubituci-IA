export interface User {
  id: string
  email: string
  username: string
  is_admin: boolean
  created_at: string
}

export interface Message {
  id: string
  conversation_id: string
  role: "user" | "assistant" | "system"
  content: string
  tokens?: number
  model_generation?: number
  sources?: Source[]
  feedback?: Feedback
  created_at: string
}

export interface Source {
  id: string
  url: string
  title: string
  snippet?: string
  credibility?: number
}

export interface Feedback {
  id: string
  message_id: string
  user_id: string
  rating: "positive" | "negative"
  comment?: string
  created_at: string
}

export interface Conversation {
  id: string
  user_id: string
  title: string
  message_count: number
  created_at: string
  updated_at: string
}

export interface EntityStatus {
  status: "active" | "training" | "sleeping" | "offline" | "error"
  generation: number
  uptime_seconds?: number
  last_activity?: string
}

export interface GenerationInfo {
  number: number
  status: string
  metrics?: {
    eval_loss?: number
    perplexity?: number
    train_loss?: number
  }
  created_at: string
  activated_at?: string
  parent_generation?: number
}

export interface EvolutionMetrics {
  generations: GenerationInfo[]
  current_generation: number
  total_training_steps: number
  total_memories: number
  total_beliefs: number
}

export interface ResearchMetrics {
  total_queries: number
  total_sources: number
  sources_by_type: Record<string, number>
  avg_credibility: number
  recent_queries: Array<{
    query: string
    results_count: number
    created_at: string
  }>
}

export type ViewType = "chat" | "history" | "evolution" | "settings"

export interface ApiResponse<T> {
  data?: T
  error?: string
  detail?: string
}

export interface PaginatedResponse<T> {
  items: T[]
  total: number
  page: number
  page_size: number
  total_pages: number
}

export interface WebSocketMessage {
  type: "token" | "done" | "error" | "status"
  content?: string
  conversation_id?: string
  message_id?: string
  error?: string
  status?: string
}