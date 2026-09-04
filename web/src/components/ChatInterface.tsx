"use client"
import { useCallback, useEffect, useRef, useState } from "react"
import { ArrowUp, Globe2, Loader2, Paperclip, Sparkles } from "lucide-react"
import { Message, TypingIndicator } from "./ChatComponents"
import { useChat } from "@/hooks/useChat"
import { useAuth } from "@/hooks/useAuth"

const suggestions=["Explique algo que aprendeu hoje","Ajude-me a escrever com clareza","Pesquise e compare fontes confiáveis"]
export function ChatInterface(){
  const {user}=useAuth(); const {messages,isLoading,error,sendMessage,retryMessage,deleteMessage,submitFeedback}=useChat(); const [input,setInput]=useState(""); const end=useRef<HTMLDivElement>(null); const area=useRef<HTMLTextAreaElement>(null)
  useEffect(()=>{end.current?.scrollIntoView({behavior:"smooth"})},[messages,isLoading])
  const submit=useCallback(async(e:React.FormEvent)=>{e.preventDefault();const value=input.trim();if(!value||isLoading)return;setInput("");if(area.current)area.current.style.height="auto";await sendMessage(value)},[input,isLoading,sendMessage])
  return <div className="chat-stage">
    <div className="chat-scroll scrollbar-thin">
      {messages.length===0?<div className="hero-chat"><div className="ai-orb"><span/></div><p className="eyebrow">Conhecimento aberto · evolução responsável</p><h1>Olá, {user?.username}.<br/><span>O que vamos descobrir?</span></h1><p className="hero-copy">Rubituci é uma IA brasileira open source que aprende com contribuições revisadas da comunidade e mantém a origem do conhecimento.</p></div>:<div className="message-column">{messages.map((message,index)=><Message key={`${message.id}-${index}`} message={message} onRetry={()=>retryMessage(message.id)} onDelete={()=>deleteMessage(message.id)} onFeedback={feedback=>submitFeedback(message.id,feedback)} userId={user?.id}/>)}{isLoading&&<TypingIndicator/>}{error&&<p className="text-sm text-rose-300">{error}</p>}<div ref={end}/></div>}
    </div>
    <div className="composer-wrap"><form onSubmit={submit} className="composer"><textarea ref={area} value={input} onChange={e=>{setInput(e.target.value);e.target.style.height="auto";e.target.style.height=`${Math.min(e.target.scrollHeight,150)}px`}} onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();submit(e as unknown as React.FormEvent)}}} placeholder="Converse com a Rubituci..." aria-label="Digite sua mensagem..." rows={2}/><div className="composer-actions"><div><button type="button" className="tool-button" aria-label="Anexar"><Paperclip size={15}/></button><button type="button" className="tool-button"><Globe2 size={15}/>Pesquisar na web</button></div><button type="submit" className="send-button" disabled={!input.trim()||isLoading} aria-label="Enviar">{isLoading?<Loader2 className="animate-spin" size={17}/>:<ArrowUp size={18}/>}</button></div></form>{messages.length===0&&<div className="suggestion-grid">{suggestions.map((text,i)=><button key={text} onClick={()=>sendMessage(text)}><Sparkles size={15}/><div><strong>{["Curiosidade","Escrita","Pesquisa"][i]}</strong><span>{text}</span></div></button>)}</div>}<p className="mt-3 text-center text-[10px] text-white/25">Rubituci pode cometer erros. Verifique informações importantes.</p></div>
  </div>
}
