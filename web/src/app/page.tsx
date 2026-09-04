"use client"

import { useEffect, useRef, useState } from "react"
import { useRouter } from "next/navigation"
import { ExternalLink, Loader2, LogOut, Menu, MoreHorizontal, RefreshCw, Users } from "lucide-react"
import { ChatInterface } from "@/components/ChatInterface"
import { Sidebar } from "@/components/Sidebar"
import { useAuth } from "@/hooks/useAuth"
import { useChat } from "@/hooks/useChat"
import { useEntity } from "@/hooks/useEntity"
import { ViewType } from "@/types"

export default function HomePage() {
  const router = useRouter()
  const { user, isLoading, fetchUser, logout } = useAuth()
  const { entityStatus, generation, fetchStatus } = useEntity()
  const { conversations, loadConversations, loadConversation, clearConversation } = useChat()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)
  const [view, setView] = useState<ViewType>("chat")
  const menu = useRef<HTMLDivElement>(null)

  useEffect(() => { fetchUser() }, [fetchUser])
  useEffect(() => { if (!isLoading && !user) router.replace("/auth?mode=login") }, [isLoading, user, router])
  useEffect(() => { if (user) { fetchStatus(); loadConversations() } }, [user, fetchStatus, loadConversations])
  useEffect(() => {
    const close = (event: MouseEvent) => { if (menu.current && !menu.current.contains(event.target as Node)) setMenuOpen(false) }
    window.addEventListener("mousedown", close)
    return () => window.removeEventListener("mousedown", close)
  }, [])

  if (isLoading || !user) return <div className="grid min-h-screen place-items-center bg-[#07090d]"><Loader2 className="h-7 w-7 animate-spin text-violet-300" /></div>

  const changeView = (next: ViewType) => {
    if (next === "evolution") { router.push("/evolution"); return }
    setView(next)
  }

  return <div className="rubituci-shell">
    <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} currentView={view} onViewChange={changeView} generation={generation} entityStatus={entityStatus} />
    <section className="flex min-w-0 flex-1 flex-col">
      <header className="rubituci-topbar">
        <button className="icon-button lg:hidden" onClick={() => setSidebarOpen(true)} aria-label="Abrir menu"><Menu size={18}/></button>
        <div className="model-pill"><span className="status-dot"/> Rubituci <span className="text-white/35">Gen {generation || 1}</span></div>
        <div className="relative" ref={menu}>
          <button className="icon-button" onClick={() => setMenuOpen(value => !value)} aria-label="Mais opções" aria-expanded={menuOpen}><MoreHorizontal size={18}/></button>
          {menuOpen && <div className="action-menu">
            <button onClick={() => { fetchStatus(); setMenuOpen(false) }}><RefreshCw size={15}/>Atualizar status</button>
            {user.is_admin && <button onClick={() => router.push("/admin/users")}><Users size={15}/>Painel de usuários</button>}
            <a href="https://github.com/Rubituci5/Rubituci-IA" target="_blank" rel="noreferrer"><ExternalLink size={15}/>Código no GitHub</a>
            <button onClick={logout}><LogOut size={15}/>Sair</button>
          </div>}
        </div>
      </header>
      <main className="min-h-0 flex-1">
        {view === "chat" && <ChatInterface/>}
        {view === "history" && <HistoryPanel conversations={conversations} openConversation={async id => { await loadConversation(id); setView("chat") }} />}
        {view === "settings" && <SettingsPanel clearConversation={() => { clearConversation(); setView("chat") }} />}
      </main>
    </section>
  </div>
}

function HistoryPanel({ conversations, openConversation }: { conversations: ReturnType<typeof useChat>["conversations"]; openConversation: (id: string) => Promise<void> }) {
  return <div className="mx-auto max-w-3xl p-6 md:p-10"><h1 className="text-3xl font-medium">Histórico</h1><p className="mt-2 text-sm text-white/40">Abra uma conversa salva no banco.</p><div className="mt-7 space-y-2">{conversations.map(conversation => <button key={conversation.id} onClick={() => openConversation(conversation.id)} className="glass-card flex w-full items-center justify-between p-4 text-left"><span>{conversation.title || "Conversa sem título"}</span><span className="text-xs text-white/30">{new Date(conversation.updated_at).toLocaleString("pt-BR")}</span></button>)}{!conversations.length && <div className="glass-card p-8 text-center text-white/35">Você ainda não tem conversas salvas.</div>}</div></div>
}

function SettingsPanel({ clearConversation }: { clearConversation: () => void }) {
  return <div className="mx-auto max-w-3xl p-6 md:p-10"><h1 className="text-3xl font-medium">Configurações</h1><p className="mt-2 text-sm text-white/40">Controles disponíveis nesta versão.</p><div className="glass-card mt-7 p-5"><h2 className="font-medium">Conversa atual</h2><p className="mt-2 text-sm text-white/40">Comece uma conversa vazia sem apagar seu histórico salvo.</p><button className="tool-button mt-4" onClick={clearConversation}>Iniciar nova conversa</button></div><div className="glass-card mt-3 p-5"><h2 className="font-medium">Projeto aberto</h2><p className="mt-2 text-sm text-white/40">Configurações avançadas e preferências de privacidade ainda estão em desenvolvimento.</p><a className="tool-button mt-4" href="https://github.com/Rubituci5/Rubituci-IA/issues" target="_blank" rel="noreferrer">Sugerir uma configuração <ExternalLink size={14}/></a></div></div>
}
