"use client"

import { formatDistanceToNow } from "date-fns"
import { Copy, Check, RotateCcw, Trash2, Flag, ThumbsUp, ThumbsDown, Bot, User, Sparkles, AlertTriangle } from "lucide-react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Message as MessageType } from "@/types"

interface MessageProps {
  message: MessageType
  onRetry: () => void
  onDelete: () => void
  onFeedback: (feedback: "positive" | "negative") => void
  userId?: string
}

export function Message({ message, onRetry, onDelete, onFeedback, userId }: MessageProps) {
  const isUser = message.role === "user"
  const [copied, setCopied] = useState(false)
  const [showFeedback, setShowFeedback] = useState(false)

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
          isUser
            ? "bg-surface-200 dark:bg-surface-700"
            : "bg-entity-100 dark:bg-entity-900/30"
        }`}
      >
        {isUser ? (
          <User className="w-4 h-4 text-surface-600 dark:text-surface-400" />
        ) : (
          <Sparkles className="w-4 h-4 text-entity-600 dark:text-entity-400" />
        )}
      </div>

      <div className={`flex-1 max-w-[calc(100%-3rem)] ${isUser ? "text-right" : ""}`}>
        <div className={`inline-block max-w-[85%] ${isUser ? "text-right" : ""}`}>
          <div
            className={`relative rounded-2xl px-4 py-3 ${
              isUser
                ? "bg-entity-600 text-white rounded-tr-sm"
                : "bg-white dark:bg-surface-900 border border-surface-200 dark:border-surface-700 rounded-tl-sm shadow-sm"
            }`}
          >
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              className="markdown-content prose-sm dark:prose-invert max-w-none"
              components={{
                code: ({ children, ...props }) => (
                  <code {...props} className="bg-surface-100 dark:bg-surface-800 px-1.5 py-0.5 rounded text-sm font-mono">
                    {children}
                  </code>
                ),
                pre: ({ children, ...props }) => (
                  <pre {...props} className="bg-surface-900 dark:bg-surface-950 rounded-lg p-4 overflow-x-auto">
                    {children}
                  </pre>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>

            {/* Sources/Citations */}
            {message.sources && message.sources.length > 0 && (
              <details className="mt-3">
                <summary className="text-xs text-surface-500 dark:text-surface-400 cursor-pointer flex items-center gap-1">
                  <Sparkles className="w-3 h-3" />
                  Sources ({message.sources.length})
                </summary>
                <ul className="mt-2 space-y-1 text-xs text-surface-600 dark:text-surface-400">
                  {message.sources.map((source, i) => (
                    <li key={i} className="flex items-start gap-2">
                      <a
                        href={source.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-entity-600 dark:text-entity-400 hover:underline truncate max-w-[300px]"
                      >
                        {source.title || source.url}
                      </a>
                    </li>
                  ))}
                </ul>
              </details>
            )}

            {/* Metadata */}
            <div className="mt-2 flex items-center justify-end gap-2 text-xs opacity-60">
              <time dateTime={message.created_at}>
                {formatDistanceToNow(new Date(message.created_at), { addSuffix: true })}
              </time>
              {message.tokens && (
                <span>{message.tokens} tokens</span>
              )}
              {message.model_generation && (
                <span>Gen {message.model_generation}</span>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="mt-1.5 flex items-center justify-end gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              onClick={() => {
                navigator.clipboard.writeText(message.content)
                setCopied(true)
                setTimeout(() => setCopied(false), 2000)
              }}
              className="p-1.5 rounded hover:bg-surface-100 dark:hover:bg-surface-800 transition-colors"
              title="Copy"
            >
              {copied ? <Check className="w-4 h-4 text-green-500" /> : <Copy className="w-4 h-4" />}
            </button>

            {!isUser && (
              <>
                <button
                  onClick={onRetry}
                  className="p-1.5 rounded hover:bg-surface-100 dark:hover:bg-surface-800 transition-colors"
                  title="Regenerate"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setShowFeedback(true)}
                  className="p-1.5 rounded hover:bg-surface-100 dark:hover:bg-surface-800 transition-colors"
                  title="Feedback"
                >
                  <Flag className="w-4 h-4" />
                </button>
              </>
            )}

            {isUser && (
              <button
                onClick={onDelete}
                className="p-1.5 rounded hover:bg-red-100 dark:hover:bg-red-900/30 transition-colors text-red-500"
                title="Delete"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        {/* Feedback */}
        {!isUser && showFeedback && (
          <MessageFeedback
            messageId={message.id}
            onSubmit={onFeedback}
            onClose={() => setShowFeedback(false)}
          />
        )}
      </div>
    </div>
  )
}

function MessageFeedback({
  messageId,
  onSubmit,
  onClose,
}: {
  messageId: string
  onSubmit: (feedback: "positive" | "negative") => void
  onClose: () => void
}) {
  const [selected, setSelected] = useState<"positive" | "negative" | null>(null)
  const [comment, setComment] = useState("")

  return (
    <div className="mt-2 p-3 rounded-lg bg-surface-100 dark:bg-surface-800 border border-surface-200 dark:border-surface-700 animate-slide-up">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-surface-900 dark:text-surface-100">Feedback</span>
        <button onClick={onClose} className="p-1 hover:bg-surface-200 dark:hover:bg-surface-700 rounded">✕</button>
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={() => setSelected("positive")}
          className={`flex-1 py-2 px-3 rounded-lg border-2 transition-colors ${
            selected === "positive"
              ? "border-green-500 bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400"
              : "border-surface-300 dark:border-surface-600 hover:border-entity-300"
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            <ThumbsUp className="w-4 h-4" />
            <span>Helpful</span>
          </div>
        </button>
        <button
          onClick={() => setSelected("negative")}
          className={`flex-1 py-2 px-3 rounded-lg border-2 transition-colors ${
            selected === "negative"
              ? "border-red-500 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400"
              : "border-surface-300 dark:border-surface-600 hover:border-entity-300"
          }`}
        >
          <div className="flex items-center justify-center gap-2">
            <ThumbsDown className="w-4 h-4" />
            <span>Not Helpful</span>
          </div>
        </button>
      </div>

      <textarea
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="Optional: What could be better?"
        className="input text-sm min-h-[60px] mb-2"
        rows={2}
      />

      <button
        onClick={() => {
          if (selected) {
            onSubmit(selected)
            onClose()
          }
        }}
        disabled={!selected}
        className="btn-primary w-full text-sm disabled:opacity-50"
      >
        Submit Feedback
      </button>
    </div>
  )
}

interface MessageInputProps {
  value: string
  onChange: (value: string) => void
  onSubmit: (e: React.FormEvent) => void
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void
  disabled: boolean
  placeholder?: string
}

export function MessageInput({
  value,
  onChange,
  onSubmit,
  onKeyDown,
  disabled,
  placeholder = "Message Entity...",
}: MessageInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [value])

  return (
    <form onSubmit={onSubmit} className="relative">
      <textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        className="input resize-none min-h-[52px] max-h-[200px] pr-12"
        rows={1}
        aria-label="Message input"
      />
    </form>
  )
}

export function TypingIndicator() {
  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-entity-100 dark:bg-entity-900/30 flex items-center justify-center">
        <Sparkles className="w-4 h-4 text-entity-600 dark:text-entity-400" />
      </div>
      <div className="bg-white dark:bg-surface-900 border border-surface-200 dark:border-surface-700 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
        <div className="flex gap-1 items-end h-6">
          <span className="w-1.5 h-1.5 bg-entity-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
          <span className="w-1.5 h-1.5 bg-entity-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
          <span className="w-1.5 h-1.5 bg-entity-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
        </div>
      </div>
    </div>
  )
}

import { useState, useRef, useEffect } from "react"