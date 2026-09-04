"use client"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Loader2 } from "lucide-react"
import { useAuth } from "@/hooks/useAuth"

export default function GoogleCallbackPage(){
  const router=useRouter(); const {completeGoogleLogin}=useAuth(); const [error,setError]=useState("")
  useEffect(()=>{(async()=>{const params=new URLSearchParams(window.location.hash.slice(1));const access=params.get("access_token"),refresh=params.get("refresh_token");history.replaceState(null,"",window.location.pathname);if(!access||!refresh){setError("O Google não devolveu uma sessão válida.");return}try{await completeGoogleLogin(access,refresh);router.replace("/")}catch{setError("Não foi possível concluir o login com Google.")}})()},[completeGoogleLogin,router])
  return <main className="min-h-screen grid place-items-center bg-[#07090d] text-white"><div className="text-center">{error?<><div className="brand-mark mx-auto mb-4">R</div><p>{error}</p><a className="mt-4 inline-block text-violet-300" href="/auth?mode=login">Voltar ao login</a></>:<><Loader2 className="mx-auto animate-spin text-violet-300"/><p className="mt-4 text-sm text-white/45">Concluindo seu login...</p></>}</div></main>
}
