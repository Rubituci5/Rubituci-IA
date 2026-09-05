'use client';

import { create } from 'zustand';
import { Message, Conversation, WebSocketMessage } from '@/types';
import { useAuthStore } from './useAuth';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

interface ChatState {
  messages: Message[];
  conversations: Conversation[];
  currentConversationId: string | null;
  isLoading: boolean;
  error: string | null;
  ws: WebSocket | null;

  sendMessage: (content: string, options?: { webSearch?: boolean }) => Promise<void>;
  teachKnowledge: (subject: string, content: string, sourceUrl?: string) => Promise<string>;
  uploadImage: (file: File) => Promise<{ url: string; name: string }>;
  retryMessage: (messageId: string) => Promise<void>;
  deleteMessage: (messageId: string) => Promise<void>;
  submitFeedback: (
    messageId: string,
    rating: 'positive' | 'negative',
    comment?: string
  ) => Promise<void>;
  clearConversation: () => void;
  loadConversations: () => Promise<void>;
  loadConversation: (id: string) => Promise<void>;
  renameConversation: (id: string, title: string) => Promise<void>;
  deleteConversation: (id: string) => Promise<void>;
  createConversation: () => Promise<string>;
  connectWebSocket: (conversationId: string) => void;
  disconnectWebSocket: () => void;
  setConversationId: (id: string | null) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  conversations: [],
  currentConversationId: null,
  isLoading: false,
  error: null,
  ws: null,

  sendMessage: async (content, options = {}) => {
    const { accessToken } = useAuthStore.getState();
    const { currentConversationId, messages, connectWebSocket } = get();

    set({ isLoading: true, error: null });

    // Optimistic user message
    const userMessage: Message = {
      id: `temp-${Date.now()}`,
      conversation_id: currentConversationId || '',
      role: 'user',
      content,
      created_at: new Date().toISOString(),
    };
    set({ messages: [...messages, userMessage] });

    try {
      // Create conversation if needed
      let conversationId = currentConversationId;
      if (!conversationId) {
        conversationId = await get().createConversation();
      }

      // Connect WebSocket
      connectWebSocket(conversationId);

      // Aguarda o socket ficar pronto e envia a mensagem.
      await new Promise<void>((resolve, reject) => {
        const startedAt = Date.now();

        const timer = window.setInterval(() => {
          const socket = get().ws;

          if (socket?.readyState === WebSocket.OPEN) {
            window.clearInterval(timer);
            socket.send(JSON.stringify({ content, web_search: Boolean(options.webSearch) }));
            resolve();
            return;
          }

          if (Date.now() - startedAt > 5000) {
            window.clearInterval(timer);
            reject(new Error('WebSocket connection timeout'));
          }
        }, 50);
      });
    } catch (e) {
      set({ error: e instanceof Error ? e.message : 'Failed to send message', isLoading: false });
      // Remove optimistic message
      set({ messages: messages.filter((m) => m.id !== userMessage.id) });
    }
  },

  teachKnowledge: async (subject, content, sourceUrl) => {
    const { accessToken } = useAuthStore.getState();
    const res = await fetch(`${API_URL}/api/learning/teach`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${accessToken}` },
      body: JSON.stringify({ subject, content, source_url: sourceUrl || null }),
    });
    const result = await res.json().catch(() => null);
    if (!res.ok) throw new Error(result?.detail || 'Não consegui guardar esse conhecimento.');
    return result.message;
  },

  uploadImage: async (file) => {
    const { accessToken } = useAuthStore.getState();
    const data = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(new Error('Não consegui ler a imagem.'));
      reader.readAsDataURL(file);
    });
    const res = await fetch(`${API_URL}/api/uploads/image`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${accessToken}` },
      body: JSON.stringify({ filename: file.name, data }),
    });
    if (!res.ok)
      throw new Error((await res.json().catch(() => null))?.detail || 'Falha ao enviar a imagem.');
    const uploaded = await res.json();
    return { ...uploaded, url: `${API_URL}${uploaded.url}` };
  },

  retryMessage: async (messageId) => {
    const { messages } = get();

    const message = messages.find((m) => m.id === messageId);
    if (!message || message.role !== 'assistant') return;

    // Find the user message before this
    const msgIndex = messages.findIndex((m) => m.id === messageId);
    const userMsg = messages[msgIndex - 1];
    if (!userMsg || userMsg.role !== 'user') return;

    // Remove the assistant message and any after it
    set({ messages: messages.slice(0, msgIndex) });
    await get().sendMessage(userMsg.content);
  },

  deleteMessage: async (messageId) => {
    const { accessToken } = useAuthStore.getState();
    const { messages } = get();

    set({ messages: messages.filter((m) => m.id !== messageId) });

    try {
      await fetch(`${API_URL}/api/messages/${messageId}`, {
        method: 'DELETE',
        headers: { Authorization: `Bearer ${accessToken}` },
      });
    } catch (e) {
      // Reload on error
      if (get().currentConversationId) {
        get().loadConversation(get().currentConversationId!);
      }
    }
  },

  submitFeedback: async (messageId, rating, comment) => {
    const { accessToken } = useAuthStore.getState();

    try {
      await fetch(`${API_URL}/api/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${accessToken}`,
        },
        body: JSON.stringify({ message_id: messageId, rating, comment }),
      });
    } catch (e) {
      console.error('Failed to submit feedback:', e);
    }
  },

  clearConversation: () => {
    set({ messages: [], currentConversationId: null });
    get().disconnectWebSocket();
  },

  loadConversations: async () => {
    const { accessToken } = useAuthStore.getState();

    try {
      const res = await fetch(`${API_URL}/api/conversations`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (res.ok) {
        const data = await res.json();
        set({ conversations: data });
      }
    } catch (e) {
      console.error('Failed to load conversations:', e);
    }
  },

  loadConversation: async (id) => {
    const { accessToken } = useAuthStore.getState();
    const { connectWebSocket } = get();

    set({ isLoading: true, currentConversationId: id });

    try {
      const res = await fetch(`${API_URL}/api/conversations/${id}/messages`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (res.ok) {
        const data = await res.json();
        set({ messages: data, isLoading: false });
        connectWebSocket(id);
      } else {
        set({ messages: [], isLoading: false });
      }
    } catch (e) {
      set({ messages: [], isLoading: false, error: 'Failed to load conversation' });
    }
  },

  renameConversation: async (id, title) => {
    const { accessToken } = useAuthStore.getState();
    const res = await fetch(`${API_URL}/api/conversations/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${accessToken}` },
      body: JSON.stringify({ title }),
    });
    if (!res.ok) throw new Error('Não foi possível renomear a conversa.');
    await get().loadConversations();
  },

  deleteConversation: async (id) => {
    const { accessToken } = useAuthStore.getState();
    const res = await fetch(`${API_URL}/api/conversations/${id}`, {
      method: 'DELETE',
      headers: { Authorization: `Bearer ${accessToken}` },
    });
    if (!res.ok) throw new Error('Não foi possível excluir a conversa.');
    if (get().currentConversationId === id) get().clearConversation();
    await get().loadConversations();
  },

  createConversation: async () => {
    const { accessToken } = useAuthStore.getState();

    const res = await fetch(`${API_URL}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify({ title: 'New Conversation' }),
    });

    if (!res.ok) throw new Error('Failed to create conversation');

    const data = await res.json();
    const { conversations } = get();

    set({
      conversations: [data, ...conversations],
      currentConversationId: data.id,
    });

    return data.id;
  },

  connectWebSocket: (conversationId) => {
    const { ws } = get();
    const { accessToken } = useAuthStore.getState();

    if (ws?.readyState === WebSocket.OPEN) return;

    const wsUrl = accessToken
      ? `${WS_URL}/ws/chat/${conversationId}?token=${encodeURIComponent(accessToken)}`
      : `${WS_URL}/ws/chat/${conversationId}`;

    const newWs = new WebSocket(wsUrl);

    newWs.onopen = () => {
      console.log('WebSocket connected');
    };

    newWs.onmessage = (event) => {
      try {
        const msg: WebSocketMessage = JSON.parse(event.data);

        if (msg.type === 'token') {
          // Append token to last assistant message
          set((state) => {
            const messages = [...state.messages];
            const lastIdx = messages.length - 1;
            const lastMessage = messages[lastIdx];

            if (lastMessage?.role === 'assistant') {
              messages[lastIdx] = {
                ...lastMessage,
                content: lastMessage.content + (msg.content || ''),
              };
            } else {
              messages.push({
                id: `ws-${Date.now()}`,
                conversation_id: conversationId,
                role: 'assistant',
                content: msg.content || '',
                created_at: new Date().toISOString(),
              });
            }

            return { messages };
          });
        } else if (msg.type === 'done') {
          set({ isLoading: false });
          get().loadConversations();
        } else if (msg.type === 'error') {
          set({ error: msg.error, isLoading: false });
        }
      } catch (e) {
        console.error('WS message parse error:', e);
      }
    };

    newWs.onerror = (err) => {
      console.error('WebSocket error:', err);
      set({ error: 'Connection error', isLoading: false });
    };

    newWs.onclose = () => {
      console.log('WebSocket closed');
      set({ ws: null });
    };

    set({ ws: newWs });
  },

  disconnectWebSocket: () => {
    const { ws } = get();
    if (ws) {
      ws.close();
      set({ ws: null });
    }
  },

  setConversationId: (id) => {
    set({ currentConversationId: id });
    if (!id) {
      get().disconnectWebSocket();
    }
  },
}));

export function useChat() {
  return useChatStore();
}
