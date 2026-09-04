"use client"

import { create } from "zustand"
import { persist } from "zustand/middleware"
import { User } from "@/types"
import axios from "axios"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const api = axios.create({
  baseURL: `${API_URL}/api`,
  withCredentials: true,
})

interface AuthState {
  user: User | null
  accessToken: string | null
  refreshToken: string | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<void>
  register: (email: string, username: string, password: string) => Promise<void>
  logout: () => Promise<void>
  refreshAccessToken: () => Promise<void>
  fetchUser: () => Promise<void>
  setTokens: (access: string, refresh: string) => void
  completeGoogleLogin: (access: string, refresh: string) => Promise<void>
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      accessToken: null,
      refreshToken: null,
      isLoading: true,
      isAuthenticated: false,

      setTokens: (access, refresh) => {
        api.defaults.headers.common["Authorization"] = `Bearer ${access}`
        set({ accessToken: access, refreshToken: refresh, isAuthenticated: true })
      },

      completeGoogleLogin: async (access, refresh) => {
        get().setTokens(access, refresh)
        await get().fetchUser()
      },

      login: async (email, password) => {
        const res = await api.post("/auth/login", { email, password })
        const { access_token, refresh_token } = res.data
        get().setTokens(access_token, refresh_token)
        await get().fetchUser()
      },

      register: async (email, username, password) => {
        const res = await api.post("/auth/register", { email, username, password })
        const { access_token, refresh_token } = res.data
        get().setTokens(access_token, refresh_token)
        await get().fetchUser()
      },

      logout: async () => {
        try {
          await api.post("/auth/logout")
        } catch {}
        api.defaults.headers.common["Authorization"] = ""
        set({ user: null, accessToken: null, refreshToken: null, isAuthenticated: false })
      },

      refreshAccessToken: async () => {
        const { refreshToken } = get()
        if (!refreshToken) throw new Error("No refresh token")

        const res = await api.post("/auth/refresh", { refresh_token: refreshToken })
        const { access_token, refresh_token: newRefresh } = res.data
        get().setTokens(access_token, newRefresh)
      },

      fetchUser: async () => {
        const { accessToken } = get()
        if (!accessToken) {
          set({ isLoading: false, isAuthenticated: false })
          return
        }

        try {
          api.defaults.headers.common["Authorization"] = `Bearer ${accessToken}`
          const res = await api.get("/auth/me")
          set({ user: res.data, isLoading: false, isAuthenticated: true })
        } catch (e) {
          // Try refresh
          try {
            await get().refreshAccessToken()
            const res = await api.get("/auth/me")
            set({ user: res.data, isLoading: false, isAuthenticated: true })
          } catch {
            set({ user: null, isLoading: false, isAuthenticated: false })
          }
        }
      },
    }),
    {
      name: "auth-storage",
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
      }),
      onRehydrateStorage: () => (state) => {
        if (state?.accessToken) {
          api.defaults.headers.common["Authorization"] = `Bearer ${state.accessToken}`
        }
      },
    }
  )
)

export function useAuth() {
  const { user, isLoading, isAuthenticated, fetchUser, login, register, logout, completeGoogleLogin } = useAuthStore()

  return {
    user,
    isLoading,
    isAuthenticated,
    login,
    register,
    logout,
    fetchUser,
    completeGoogleLogin,
  }
}
