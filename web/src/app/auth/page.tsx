"use client"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { ArrowRight, Check, Eye, EyeOff, Loader2 } from "lucide-react"
import { useAuth } from "@/hooks/useAuth"

export default function AuthPage(){
 const router=useRouter(),{login,register}=useAuth();const [isRegister,setIsRegister]=useState(false)
 const [email,setEmail]=useState(""),[password,setPassword]=useState(""),[username,setUsername]=useState(""),[show,setShow]=useState(false),[error,setError]=useState(""),[busy,setBusy]=useState(false)
 useEffect(()=>setIsRegister(new URLSearchParams(window.location.search).get("mode")==="register"),[])
 async function submit(e:React.FormEvent){e.preventDefault();setError("");setBusy(true);try{isRegister?await register(email,username,password):await login(email,password);router.replace("/")}catch(err:any){setError(err.response?.data?.detail||err.message||"Não foi possível continuar.")}finally{setBusy(false)}}
 const apiUrl=process.env.NEXT_PUBLIC_API_URL||"http://localhost:8000"
 return <main className="auth-shell">
  <section className="auth-story"><a className="auth-brand" href="/"><span>R</span>Rubituci</a><div><p className="eyebrow">IA brasileira · código aberto</p><h1>Uma inteligência que cresce com a comunidade.</h1><p>Converse, ensine e acompanhe a evolução de um modelo aberto. Cada contribuição passa por revisão e mantém sua origem.</p><ul><li><Check/>Uso gratuito, sem créditos por mensagem</li><li><Check/>Memória e evolução transparentes</li><li><Check/>Conhecimento comunitário com proveniência</li></ul></div><small>Open source significa liberdade para estudar, executar e contribuir.</small></section>
  <section className="auth-panel"><div className="auth-form-wrap"><div className="mb-8"><p className="text-sm text-violet-300">{isRegister?"Comece agora":"Que bom ter você de volta"}</p><h2>{isRegister?"Crie sua conta":"Entre na Rubituci"}</h2><p>{isRegister?"Leva menos de um minuto.":"Continue suas conversas e contribuições."}</p></div>
   <a className="google-button" href={`${apiUrl}/api/auth/google`}><span className="google-g">G</span>Continuar com Google</a><div className="auth-divider"><span>ou use seu e-mail</span></div>
   <form onSubmit={submit} className="space-y-4">{isRegister&&<label>Como devemos chamar você?<input value={username} onChange={e=>setUsername(e.target.value)} minLength={3} maxLength={100} placeholder="Seu nome de usuário" required/></label>}<label>E-mail<input type="email" value={email} onChange={e=>setEmail(e.target.value)} placeholder="voce@exemplo.com" required/></label><label>Senha<div className="relative"><input type={show?"text":"password"} value={password} onChange={e=>setPassword(e.target.value)} minLength={8} maxLength={128} placeholder="Mínimo de 8 caracteres" required/><button type="button" onClick={()=>setShow(!show)} aria-label="Mostrar senha">{show?<EyeOff/>:<Eye/>}</button></div></label>{error&&<p className="auth-error">{error}</p>}<button className="auth-submit" disabled={busy}>{busy?<Loader2 className="animate-spin"/>:<>{isRegister?"Criar conta":"Entrar"}<ArrowRight/></>}</button></form>
   <p className="mt-6 text-center text-sm text-white/40">{isRegister?"Já possui uma conta? ":"Ainda não possui uma conta? "}<a href={`/auth?mode=${isRegister?"login":"register"}`}>{isRegister?"Entrar":"Cadastrar gratuitamente"}</a></p>{isRegister&&<p className="mt-5 text-center text-[11px] leading-relaxed text-white/25">Ao criar sua conta, você concorda em não enviar dados pessoais de terceiros, segredos ou conteúdo sem licença.</p>}
  </div></section>
 </main>
}
