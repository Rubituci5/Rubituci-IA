'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUp, BookOpen, Globe2, Loader2, Paperclip, Sparkles, X } from 'lucide-react';
import { Message, TypingIndicator } from './ChatComponents';
import { useChat } from '@/hooks/useChat';
import { useAuth } from '@/hooks/useAuth';

const suggestions = [
  'Explique algo que aprendeu hoje',
  'Ajude-me a escrever com clareza',
  'Pesquise e compare fontes confiáveis',
];
export function ChatInterface() {
  const { user } = useAuth();
  const {
    messages,
    isLoading,
    error,
    sendMessage,
    teachKnowledge,
    retryMessage,
    deleteMessage,
    submitFeedback,
    uploadImage,
  } = useChat();
  const [input, setInput] = useState('');
  const [webSearch, setWebSearch] = useState(false);
  const [teaching, setTeaching] = useState(false);
  const [teachSubject, setTeachSubject] = useState('');
  const [teachContent, setTeachContent] = useState('');
  const [teachSource, setTeachSource] = useState('');
  const [teachStatus, setTeachStatus] = useState('');
  const [attachment, setAttachment] = useState<{
    name: string;
    content?: string;
    file?: File;
    preview?: string;
  } | null>(null);
  const [toolError, setToolError] = useState('');
  const end = useRef<HTMLDivElement>(null);
  const area = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);
  const submit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      const value = input.trim();
      if ((!value && !attachment) || isLoading) return;
      try {
        let annex = '';
        if (attachment?.file) {
          const uploaded = await uploadImage(attachment.file);
          annex = `\n\n![${uploaded.name}](${uploaded.url})\n[Imagem anexada: ${uploaded.name}. Minha visão própria ainda está em desenvolvimento.]`;
        } else if (attachment) {
          annex = `\n\n[Arquivo anexado: ${attachment.name}]\n${attachment.content}`;
        }
        setInput('');
        setAttachment(null);
        if (area.current) area.current.style.height = 'auto';
        await sendMessage([value, annex].join('').trim(), { webSearch });
        setWebSearch(false);
      } catch (error) {
        setToolError(error instanceof Error ? error.message : 'Falha ao enviar o anexo.');
      }
    },
    [input, attachment, isLoading, sendMessage, uploadImage, webSearch]
  );
  const attachFile = useCallback((file?: File) => {
    if (!file) return;
    setToolError('');
    if (file.type.startsWith('image/')) {
      if (file.size > 2_000_000) {
        setToolError('A imagem precisa ter no máximo 2 MB.');
        return;
      }
      if (!/image\/(png|jpeg|webp|gif)/.test(file.type)) {
        setToolError('Use imagens PNG, JPEG, WebP ou GIF.');
        return;
      }
      setAttachment({ name: file.name, file, preview: URL.createObjectURL(file) });
      return;
    }
    if (file.size > 500_000) {
      setToolError('O arquivo precisa ter no máximo 500 KB.');
      return;
    }
    const allowed = /\.(txt|md|csv|json)$/i.test(file.name) || file.type.startsWith('text/');
    if (!allowed) {
      setToolError('Anexe imagem, TXT, Markdown, CSV ou JSON.');
      return;
    }
    const reader = new FileReader();
    reader.onload = () =>
      setAttachment({ name: file.name, content: String(reader.result || '').slice(0, 20_000) });
    reader.onerror = () => setToolError('Não consegui ler esse arquivo.');
    reader.readAsText(file);
  }, []);
  const submitTeaching = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!teachSubject.trim() || !teachContent.trim()) return;
      setTeachStatus('Analisando e aprendendo...');
      try {
        setTeachStatus(
          await teachKnowledge(
            teachSubject.trim(),
            teachContent.trim(),
            teachSource.trim() || undefined
          )
        );
        setTeachSubject('');
        setTeachContent('');
        setTeachSource('');
      } catch (error) {
        setTeachStatus(error instanceof Error ? error.message : 'Não consegui aprender.');
      }
    },
    [teachSubject, teachContent, teachSource, teachKnowledge]
  );
  return (
    <div className="chat-stage">
      <div className="chat-scroll scrollbar-thin">
        {messages.length === 0 ? (
          <div className="hero-chat">
            <div className="ai-orb">
              <span />
            </div>
            <p className="eyebrow">Conhecimento aberto · evolução responsável</p>
            <h1>
              Olá, {user?.username}.<br />
              <span>O que vamos descobrir?</span>
            </h1>
            <p className="hero-copy">
              Rubituci é uma IA brasileira open source que aprende com contribuições revisadas da
              comunidade e mantém a origem do conhecimento.
            </p>
          </div>
        ) : (
          <div className="message-column">
            {messages.map((message, index) => (
              <Message
                key={`${message.id}-${index}`}
                message={message}
                onRetry={() => retryMessage(message.id)}
                onDelete={() => deleteMessage(message.id)}
                onFeedback={(feedback) => submitFeedback(message.id, feedback)}
                userId={user?.id}
              />
            ))}
            {isLoading && <TypingIndicator />}
            {error && <p className="text-sm text-rose-300">{error}</p>}
            <div ref={end} />
          </div>
        )}
      </div>
      <div className="composer-wrap">
        {teaching && (
          <form className="teach-panel" onSubmit={submitTeaching}>
            <div className="teach-heading">
              <div><strong>Ensinar algo</strong><span>Ela separa assunto, fatos e etapas e depois pesquisa em segundo plano.</span></div>
              <button type="button" onClick={() => setTeaching(false)} aria-label="Fechar ensino"><X size={15} /></button>
            </div>
            <input value={teachSubject} onChange={(event) => setTeachSubject(event.target.value)} placeholder="Assunto — exemplo: miojo" maxLength={500} />
            <textarea value={teachContent} onChange={(event) => setTeachContent(event.target.value)} placeholder="Explique com fatos ou etapas claras..." rows={4} maxLength={20000} />
            <input value={teachSource} onChange={(event) => setTeachSource(event.target.value)} placeholder="Fonte opcional: https://..." type="url" />
            <div className="teach-footer"><span>{teachStatus}</span><button type="submit" disabled={!teachSubject.trim() || !teachContent.trim()}>Absorver conhecimento</button></div>
          </form>
        )}
        <form onSubmit={submit} className="composer">
          {attachment && (
            <div className="attachment-chip">
              {attachment.preview ? (
                <img src={attachment.preview} alt="Prévia do anexo" />
              ) : (
                <Paperclip size={13} />
              )}
              <span>{attachment.name}</span>
              <button type="button" onClick={() => setAttachment(null)} aria-label="Remover anexo">
                <X size={13} />
              </button>
            </div>
          )}
          <textarea
            ref={area}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = `${Math.min(e.target.scrollHeight, 150)}px`;
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submit(e as unknown as React.FormEvent);
              }
            }}
            placeholder="Converse com a Rubituci..."
            aria-label="Digite sua mensagem..."
            rows={2}
          />
          <div className="composer-actions">
            <div>
              <input
                ref={fileInput}
                type="file"
                className="hidden"
                accept="image/png,image/jpeg,image/webp,image/gif,.txt,.md,.csv,.json,text/*"
                onChange={(e) => {
                  attachFile(e.target.files?.[0]);
                  e.currentTarget.value = '';
                }}
              />
              <button
                type="button"
                className="tool-button"
                aria-label="Anexar arquivo"
                onClick={() => fileInput.current?.click()}
              >
                <Paperclip size={15} />
              </button>
              <button
                type="button"
                className={`tool-button ${webSearch ? 'active' : ''}`}
                aria-pressed={webSearch}
                onClick={() => setWebSearch((value) => !value)}
              >
                <Globe2 size={15} />
                {webSearch ? 'Web ativada' : 'Acessar a web'}
              </button>
              <button type="button" className={`tool-button ${teaching ? 'active' : ''}`} aria-pressed={teaching} onClick={() => { setTeaching((value) => !value); setTeachStatus('') }}>
                <BookOpen size={15} /> Ensinar algo
              </button>
            </div>
            <button
              type="submit"
              className="send-button"
              disabled={(!input.trim() && !attachment) || isLoading}
              aria-label="Enviar"
            >
              {isLoading ? <Loader2 className="animate-spin" size={17} /> : <ArrowUp size={18} />}
            </button>
          </div>
          {toolError && <p className="mt-2 text-xs text-rose-300">{toolError}</p>}
        </form>
        {messages.length === 0 && (
          <div className="suggestion-grid">
            {suggestions.map((text, i) => (
              <button key={text} onClick={() => sendMessage(text, { webSearch: i === 2 })}>
                <Sparkles size={15} />
                <div>
                  <strong>{['Curiosidade', 'Escrita', 'Pesquisa'][i]}</strong>
                  <span>{text}</span>
                </div>
              </button>
            ))}
          </div>
        )}
        <p className="mt-3 text-center text-[10px] text-white/25">
          Rubituci pode cometer erros. Verifique informações importantes.
        </p>
      </div>
    </div>
  );
}
