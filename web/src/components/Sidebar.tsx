"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { BrainCircuit, Check, History, LogOut, MessageSquarePlus, Pencil, Search, Settings, Sparkles, Trash2, X } from "lucide-react"
import { ViewType } from "@/types"
import { useAuth } from "@/hooks/useAuth"
import { useChat } from "@/hooks/useChat"

interface Props {
  open: boolean
  onClose: () => void
  currentView: ViewType
  onViewChange: (view: ViewType) => void
  generation: number | null
  entityStatus: string | null
}

const items = [
  { id: "chat", label: "Conversar", icon: Sparkles },
  { id: "history", label: "Histórico", icon: History },
  { id: "evolution", label: "Evolução", icon: BrainCircuit },
  { id: "settings", label: "Configurações", icon: Settings },
] as const

export function Sidebar({ open, onClose, currentView, onViewChange, generation, entityStatus }: Props) {
  const { user, logout } = useAuth()
  const { conversations, currentConversationId, clearConversation, loadConversation, loadConversations, renameConversation, deleteConversation } = useChat()
  const [query, setQuery] = useState("")
  const [editingId, setEditingId] = useState<string | null>(null)
  const [editingTitle, setEditingTitle] = useState("")
  const searchInput = useRef<HTMLInputElement>(null)

  useEffect(() => { loadConversations() }, [loadConversations])
  useEffect(() => {
    const focusSearch = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault()
        searchInput.current?.focus()
      }
    }
    window.addEventListener("keydown", focusSearch)
    return () => window.removeEventListener("keydown", focusSearch)
  }, [])

  const filtered = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase("pt-BR")
    if (!normalized) return conversations
    return conversations.filter(conversation =>
      (conversation.title || "Conversa sem título").toLocaleLowerCase("pt-BR").includes(normalized)
    )
  }, [conversations, query])

  const startConversation = () => {
    clearConversation()
    onViewChange("chat")
    onClose()
  }

  const openConversation = async (id: string) => {
    await loadConversation(id)
    onViewChange("chat")
    onClose()
  }

  const saveTitle = async (id: string) => {
    if (!editingTitle.trim()) return
    await renameConversation(id, editingTitle.trim())
    setEditingId(null)
  }

  const removeConversation = async (id: string) => {
    if (window.confirm("Excluir esta conversa e todas as mensagens?")) await deleteConversation(id)
  }

  return <>
    {open && <button className="fixed inset-0 z-40 bg-black/70 lg:hidden" onClick={onClose} aria-label="Fechar menu" />}
    <aside className={`rubituci-sidebar ${open ? "translate-x-0" : "-translate-x-full lg:translate-x-0"}`}>
      <div className="flex items-center justify-between px-5 py-6">
        <div className="flex items-center gap-3"><div className="brand-mark">R</div><div><strong>Rubituci</strong><p className="text-[10px] uppercase tracking-[.18em] text-white/35">IA comunitária</p></div></div>
        <button onClick={onClose} className="lg:hidden" aria-label="Fechar menu"><X size={18}/></button>
      </div>
      <div className="px-4">
        <button className="new-chat" onClick={startConversation}><MessageSquarePlus size={16}/>Nova conversa</button>
        <label className="search-box"><Search size={14}/><input ref={searchInput} value={query} onChange={event=>setQuery(event.target.value)} placeholder="Buscar conversas" aria-label="Buscar conversas"/><kbd>⌘ K</kbd></label>
      </div>
      <nav className="mt-5 space-y-1 px-3">
        {items.map(item => <button key={item.id} className={`side-item ${currentView === item.id ? "active" : ""}`} onClick={() => { onViewChange(item.id); onClose() }}><item.icon size={17}/>{item.label}</button>)}
      </nav>
      <div className="mx-4 mt-7 min-h-0 flex-1 overflow-y-auto border-t border-white/[.07] pt-5">
        <p className="px-2 text-[10px] uppercase tracking-[.16em] text-white/25">Conversas</p>
        <div className="mt-2 space-y-1">
          {filtered.slice(0, 30).map(conversation => <div key={conversation.id} className={`conversation-row ${currentConversationId === conversation.id ? "active" : ""}`}>
            {editingId === conversation.id ? <input className="conversation-title-input" autoFocus value={editingTitle} maxLength={100} onChange={event=>setEditingTitle(event.target.value)} onKeyDown={event=>{if(event.key==="Enter")saveTitle(conversation.id);if(event.key==="Escape")setEditingId(null)}}/> : <button className="conversation-result" onClick={() => openConversation(conversation.id)} title={conversation.title || "Conversa sem título"}>{conversation.title || "Conversa sem título"}</button>}
            <div className="conversation-actions">{editingId === conversation.id ? <button onClick={()=>saveTitle(conversation.id)} title="Salvar nome" aria-label="Salvar nome"><Check size={13}/></button> : <button onClick={()=>{setEditingId(conversation.id);setEditingTitle(conversation.title || "")}} title="Renomear" aria-label="Renomear conversa"><Pencil size={12}/></button>}<button onClick={()=>removeConversation(conversation.id)} title="Excluir" aria-label="Excluir conversa"><Trash2 size={12}/></button></div>
          </div>)}
          {!filtered.length && <p className="px-2 py-3 text-[11px] text-white/25">{query ? "Nenhuma conversa encontrada." : "Suas conversas aparecerão aqui."}</p>}
        </div>
      </div>
      <div className="mt-auto border-t border-white/[.07] p-4">
        <div className="mb-3 flex items-center gap-2 text-xs text-white/45"><span className="status-dot"/>{entityStatus || "conectando"} · geração {generation || 1}</div>
        <div className="flex items-center gap-3"><div className="avatar">{user?.username?.[0]?.toUpperCase() || "R"}</div><div className="min-w-0 flex-1"><p className="truncate text-sm">{user?.username}</p><p className="truncate text-[11px] text-white/35">{user?.email}</p></div><button onClick={logout} title="Sair" aria-label="Sair" className="text-white/35 hover:text-white"><LogOut size={16}/></button></div>
      </div>
    </aside>
  </>
}
