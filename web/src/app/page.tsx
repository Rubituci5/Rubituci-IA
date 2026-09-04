"use client"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Loader2, Menu, MoreHorizontal } from "lucide-react"
import { ChatInterface } from "@/components/ChatInterface"
import { Sidebar } from "@/components/Sidebar"
import { useAuth } from "@/hooks/useAuth"
import { useEntity } from "@/hooks/useEntity"

export default function HomePage() {
  const router = useRouter(); const { user, isLoading, fetchUser } = useAuth(); const { entityStatus, generation, fetchStatus } = useEntity()
  const [sidebarOpen, setSidebarOpen] = useState(false); const [view, setView] = useState<"chat" | "history" | "evolution" | "settings">("chat")
  useEffect(() => { fetchUser() }, [fetchUser])
  useEffect(() => { if (!isLoading && !user) router.replace("/auth?mode=login") }, [isLoading, user, router])
  useEffect(() => { if (user) fetchStatus() }, [user, fetchStatus])
  if (isLoading || !user) return <div className="min-h-screen grid place-items-center bg-[#07090d]"><Loader2 className="h-7 w-7 animate-spin text-violet-300" /></div>
  return <div className="rubituci-shell"><Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} currentView={view} onViewChange={setView} generation={generation} entityStatus={entityStatus} /><section className="min-w-0 flex-1 flex flex-col"><header className="rubituci-topbar"><button className="icon-button lg:hidden" onClick={() => setSidebarOpen(true)} aria-label="Abrir menu"><Menu size={18}/></button><div className="model-pill"><span className="status-dot"/> Rubituci <span className="text-white/35">Gen {generation || 1}</span></div><button className="icon-button" aria-label="Mais opções"><MoreHorizontal size={18}/></button></header><main className="min-h-0 flex-1">{view === "chat" ? <ChatInterface/> : <Placeholder view={view}/>}</main></section></div>
}
function Placeholder({ view }: { view: string }) { const copy: Record<string,[string,string]> = { history:["Histórico","Suas conversas serão organizadas aqui."], evolution:["Evolução","Acompanhe versões, avaliações e fontes de aprendizado."], settings:["Configurações","Controle privacidade, memória e preferências."] }; const [title,text]=copy[view]||copy.history; return <div className="h-full grid place-items-center p-8"><div className="glass-card max-w-lg p-8 text-center"><h1 className="text-3xl font-medium">{title}</h1><p className="mt-3 text-white/45">{text}</p></div></div> }
