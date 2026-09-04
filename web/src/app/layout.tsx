import type { Metadata } from "next"
import { Inter } from "next/font/google"
import "./globals.css"

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" })

export const metadata: Metadata = {
  title: "Rubituci — inteligência comunitária",
  description: "Uma IA brasileira, aberta e construída com conhecimento comunitário revisado.",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="pt-BR" className={`${inter.variable} antialiased`}>
      <body className="min-h-screen bg-[#07090d]">
        {children}
      </body>
    </html>
  )
}
