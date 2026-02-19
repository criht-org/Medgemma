import React, { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

// ─── Helper ────────────────────────────────────────────────────────────────────
function getFileType(file) {
  if (!file) return null;
  if (file.name.toLowerCase().endsWith('.dcm') || file.type === 'application/dicom') return 'dicom';
  if (file.name.toLowerCase().endsWith('.pdf') || file.type === 'application/pdf') return 'pdf';
  if (file.type.startsWith('image/')) return 'image';
  return 'unknown';
}

// ─── Sub-Components ────────────────────────────────────────────────────────────
function ImageCard({ src, alt = "Medical Scan" }) {
  const [lightbox, setLightbox] = useState(false);
  return (
    <>
      <div className="image-preview-card" onClick={() => setLightbox(true)} title="Click to expand">
        <img src={src} alt={alt} />
        <span className="expand-hint">🔍 Expand</span>
      </div>
      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(false)}>
          <img src={src} alt={alt} />
        </div>
      )}
    </>
  );
}

function DicomCard({ filename, meta }) {
  return (
    <div className="dicom-card">
      <div className="dicom-badge">🩻 DICOM Scan</div>
      <div style={{ marginBottom: '0.75rem', fontSize: '0.82rem', color: 'var(--text-muted)' }}>
        📄 {filename}
      </div>
      {meta && (
        <div className="dicom-meta-grid">
          <div className="dicom-meta-item">
            <div className="dicom-meta-label">Modality</div>
            <div className="dicom-meta-value">{meta.Modality || 'Unknown'}</div>
          </div>
          <div className="dicom-meta-item">
            <div className="dicom-meta-label">Date</div>
            <div className="dicom-meta-value">{meta.Date || 'N/A'}</div>
          </div>
          <div className="dicom-meta-item">
            <div className="dicom-meta-label">Patient</div>
            <div className="dicom-meta-value">{meta.Patient || 'Anonymous'}</div>
          </div>
          <div className="dicom-meta-item">
            <div className="dicom-meta-label">Study</div>
            <div className="dicom-meta-value">{meta.Study || 'N/A'}</div>
          </div>
        </div>
      )}
    </div>
  );
}

function TypingBubble() {
  return (
    <div className="msg-row assistant">
      <div className="avatar ai">✦</div>
      <div className="bubble-wrap">
        <div className="bubble" style={{ padding: '14px 18px' }}>
          <div className="typing-indicator">
            <div className="typing-dot" />
            <div className="typing-dot" />
            <div className="typing-dot" />
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Thinking Block (collapsible) ─────────────────────────────────────────────
function ThinkingBlock({ thinking }) {
  const [open, setOpen] = useState(false);
  if (!thinking) return null;
  return (
    <div className="thinking-block">
      <button className="thinking-toggle" onClick={() => setOpen(o => !o)}>
        <span className="thinking-icon">🧠</span>
        <span>Model Thinking Process</span>
        <span className="thinking-chevron">{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div className="thinking-content">
          <pre>{thinking}</pre>
        </div>
      )}
    </div>
  );
}

function MessageRow({ msg }) {
  const isUser = msg.role === 'user';
  const isLocalization = msg.isLocalization;
  const isConsultation = msg.isConsultation;
  return (
    <div className={`msg-row ${msg.role}`}>
      <div className={`avatar ${isUser ? 'user' : 'ai'} ${isConsultation ? 'ai-doctor' : ''}`}>
        {isUser ? '👤' : isConsultation ? '👨‍⚕️' : '✦'}
      </div>
      <div className="bubble-wrap">
        {isConsultation && !isUser && (
          <div className="consultation-badge">🩺 Patient Consultation</div>
        )}
        {msg.img && <ImageCard src={msg.img} />}
        {msg.dicom && <DicomCard filename={msg.dicom.filename} meta={msg.dicom.meta} />}
        {msg.pdf && (
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', padding: '8px 0' }}>
            📄 PDF: {msg.pdf}
          </div>
        )}
        {msg.thinking && <ThinkingBlock thinking={msg.thinking} />}
        {msg.content && (
          <div className={`bubble ${isLocalization ? 'bubble-mono' : ''} ${isConsultation && !isUser ? 'bubble-consultation' : ''}`}>
            {msg.content}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Welcome Screen ─────────────────────────────────────────────────────────────
const QUICK_PROMPTS = [
  { icon: '🩻', title: 'DICOM Analysis', desc: 'Upload a .dcm scan for AI analysis', prompt: 'Analyze this medical scan and describe your findings.' },
  { icon: '🔬', title: 'Radiology Report', desc: 'Generate a structured report', prompt: 'Write a radiology report for this scan.' },
  { icon: '📍', title: 'Lesion Localization', desc: 'Identify and localize findings', prompt: 'Identify any lesions and provide their locations.' },
  { icon: '📋', title: 'Clinical Query', desc: 'Ask a medical question', prompt: 'What are the differential diagnoses for the findings in this scan?' },
];

function WelcomeScreen({ onPrompt }) {
  return (
    <div className="welcome-screen">
      <div>
        <div className="welcome-logo">INNOVDOC AI</div>
        <p className="welcome-subtitle" style={{ marginTop: '0.75rem' }}>
          Expert clinical insights powered by MedGemma. Upload images, DICOM files, or ask any medical question.
        </p>
      </div>
      <div className="capability-grid">
        {QUICK_PROMPTS.map((c, i) => (
          <div key={i} className="capability-card" onClick={() => onPrompt(c.prompt)}>
            <div className="icon">{c.icon}</div>
            <h4>{c.title}</h4>
            <p>{c.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [pendingFile, setPendingFile] = useState(null);
  const [isTyping, setIsTyping] = useState(false);
  const [backendUrl, setBackendUrl] = useState(localStorage.getItem('backend_url') || '');
  const [hfToken, setHfToken] = useState(localStorage.getItem('hf_token') || '');
  const [modelName, setModelName] = useState('google/medgemma-1.5-4b-it');
  const [analysisMode, setAnalysisMode] = useState('General Analysis');
  const [backendStatus, setBackendStatus] = useState('unchecked'); // unchecked | online | offline
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);
  const chatEndRef = useRef(null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, isTyping]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) { ta.style.height = 'auto'; ta.style.height = `${Math.min(ta.scrollHeight, 140)}px`; }
  }, [inputValue]);

  // Persist settings
  useEffect(() => { localStorage.setItem('backend_url', backendUrl); }, [backendUrl]);
  useEffect(() => { localStorage.setItem('hf_token', hfToken); }, [hfToken]);

  // Normalize URL — strip trailing slash
  const apiUrl = (path) => `${backendUrl.replace(/\/+$/, '')}${path}`;

  // Headers needed for ngrok free tier (bypasses browser warning interception)
  const ngrokHeaders = { 'ngrok-skip-browser-warning': 'true' };

  // Ping backend
  const pingBackend = useCallback(async (url) => {
    if (!url) return;
    try {
      const base = url.replace(/\/+$/, '');
      const res = await fetch(`${base}/`, {
        signal: AbortSignal.timeout(5000),
        headers: { 'ngrok-skip-browser-warning': 'true' }
      });
      setBackendStatus(res.ok ? 'online' : 'offline');
    } catch { setBackendStatus('offline'); }
  }, []);

  useEffect(() => { if (backendUrl) pingBackend(backendUrl); }, [backendUrl, pingBackend]);

  // ── File Handling ─────────────────────────────────────────────────────────────
  const handleFile = (file) => {
    if (!file) return;
    const type = getFileType(file);
    const preview = (type === 'image') ? URL.createObjectURL(file) : null;
    setPendingFile({ file, type, preview, name: file.name });
  };

  const handleFileChange = (e) => handleFile(e.target.files[0]);

  // Drag-and-Drop
  const onDragOver = (e) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = () => setIsDragging(false);
  const onDrop = (e) => { e.preventDefault(); setIsDragging(false); handleFile(e.dataTransfer.files[0]); };

  // Paste image
  const onPaste = (e) => {
    const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
    if (item) handleFile(item.getAsFile());
  };

  // ── Send Logic ────────────────────────────────────────────────────────────────
  const handleSend = async () => {
    const prompt = inputValue.trim();
    if (!prompt && !pendingFile) return;

    const userMsg = {
      role: 'user',
      content: prompt,
      img: pendingFile?.type === 'image' ? pendingFile.preview : null,
      dicom: pendingFile?.type === 'dicom' ? { filename: pendingFile.name, meta: null } : null,
      pdf: pendingFile?.type === 'pdf' ? pendingFile.name : null,
    };

    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    const fileToSend = pendingFile;
    setPendingFile(null);
    setIsTyping(true);

    if (!backendUrl) {
      setMessages(prev => [...prev, { role: 'assistant', content: '⚠️ Please enter your Colab Backend URL in the sidebar to enable AI analysis.' }]);
      setIsTyping(false);
      return;
    }

    try {
      const formData = new FormData();
      formData.append('prompt', prompt || 'Analyze this medical file.');
      formData.append('model_name', modelName);
      formData.append('hf_token', hfToken);
      formData.append('analysis_mode', analysisMode);
      if (fileToSend?.file) formData.append('file', fileToSend.file);

      // Send conversation history for multi-turn context
      const historyForBackend = messages
        .filter(m => m.content && (m.role === 'user' || m.role === 'assistant'))
        .map(m => ({ role: m.role, content: m.content }));
      formData.append('history', JSON.stringify(historyForBackend));

      // If DICOM, first get metadata, then do chat
      let dicomMeta = null;
      if (fileToSend?.type === 'dicom') {
        const metaForm = new FormData();
        metaForm.append('file', fileToSend.file);
        const metaRes = await fetch(apiUrl('/process-upload'), {
          method: 'POST',
          body: metaForm,
          headers: ngrokHeaders
        });
        if (metaRes.ok) {
          const metaData = await metaRes.json();
          dicomMeta = metaData.metadata || null;
          setMessages(prev => prev.map((m, i) =>
            i === prev.length - 1 && m.role === 'user' && m.dicom
              ? { ...m, dicom: { ...m.dicom, meta: dicomMeta } }
              : m
          ));
        }
      }

      const res = await fetch(apiUrl('/chat'), {
        method: 'POST',
        body: formData,
        headers: ngrokHeaders
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response,
        thinking: data.thinking || '',
        isLocalization: analysisMode === 'Localization',
        isConsultation: analysisMode === 'Patient Consultation'
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Error: ${err.message || 'Could not reach the backend. Check your Colab URL and ensure the server is running.'}`
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const setQuickPrompt = (p) => { setInputValue(p); textareaRef.current?.focus(); };

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <div className="app-container" onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
      {isDragging && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(91,140,248,0.1)', border: '2px dashed var(--accent)', zIndex: 999, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.5rem', color: 'var(--accent)', pointerEvents: 'none' }}>
          Drop file to attach
        </div>
      )}

      {/* ── Sidebar ─────────────────────────────────────────────────────────── */}
      <aside className="sidebar">
        <div className="sidebar-logo">INNOVDOC AI</div>

        <div className="sidebar-section">
          <h3>Backend</h3>
          <input id="backend-url" className="sidebar-input" placeholder="https://xxxx.ngrok.io"
            value={backendUrl} onChange={e => setBackendUrl(e.target.value)}
            onBlur={() => pingBackend(backendUrl)} />
          <div style={{ marginTop: '0.5rem' }}>
            {backendStatus === 'online' && <span className="status-pill"><span className="status-dot" />Online</span>}
            {backendStatus === 'offline' && <span className="status-pill" style={{ color: '#f87171', background: 'rgba(248,113,113,0.1)', borderColor: 'rgba(248,113,113,0.2)' }}>⚠ Offline</span>}
            {backendStatus === 'unchecked' && <span className="status-pill" style={{ color: 'var(--text-subtle)' }}>— Not Connected</span>}
          </div>
        </div>

        <div className="sidebar-section">
          <h3>Authentication</h3>
          <input id="hf-token" className="sidebar-input" type="password" placeholder="hf_..."
            value={hfToken} onChange={e => setHfToken(e.target.value)} />
        </div>

        <div className="sidebar-section">
          <h3>Model</h3>
          <select id="model-select" className="sidebar-select" value={modelName} onChange={e => setModelName(e.target.value)}>
            <option value="google/medgemma-1.5-4b-it">MedGemma 1.5-4B</option>
            <option value="google/medgemma-4b-it">MedGemma 4B</option>
            <option value="google/medgemma-27b-it">MedGemma 27B</option>
          </select>
        </div>

        <div className="sidebar-section">
          <h3>Mode</h3>
          <select id="analysis-mode" className="sidebar-select" value={analysisMode} onChange={e => setAnalysisMode(e.target.value)}>
            <option>General Analysis</option>
            <option>Radiology Report</option>
            <option>Localization</option>
            <option>Patient Consultation</option>
          </select>
          {analysisMode === 'Patient Consultation' && (
            <div style={{ marginTop: '0.5rem', fontSize: '0.72rem', color: 'var(--accent-2)', lineHeight: 1.4 }}>
              👨‍⚕️ Dr. AI will interview the patient using leading questions.
            </div>
          )}
        </div>

        <div className="sidebar-section">
          <h3>Commands</h3>
          <button className="sidebar-input" style={{ cursor: 'pointer', textAlign: 'left', color: 'var(--text-muted)' }}
            onClick={() => setMessages([])}>🗑 Clear Chat</button>
        </div>

        <div className="sidebar-footer">✨ Developed by Dr. R. K. Ramanan</div>
      </aside>

      {/* ── Main Panel ──────────────────────────────────────────────────────── */}
      <main className="main-content">
        <div className="chat-header">
          <div>
            <h1>INNOVDOC AI</h1>
            <p>Expert medical insights • Upload scans, DICOM files, PDFs</p>
          </div>
        </div>

        {messages.length === 0 && !isTyping ? (
          <WelcomeScreen onPrompt={setQuickPrompt} />
        ) : (
          <div className="messages-area">
            {messages.map((msg, i) => <MessageRow key={i} msg={msg} />)}
            {isTyping && <TypingBubble />}
            <div ref={chatEndRef} />
          </div>
        )}

        {/* ── Input Area ──────────────────────────────────────────────────── */}
        <div className="input-area">
          {pendingFile && (
            <div className="attachment-preview-bar">
              {pendingFile.type === 'image'
                ? <img src={pendingFile.preview} className="attachment-thumb" alt="preview" />
                : <div style={{ width: 38, height: 38, borderRadius: 6, background: 'var(--bg-surface)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.2rem' }}>
                  {pendingFile.type === 'dicom' ? '🩻' : '📄'}
                </div>
              }
              <div className="attachment-info">
                <div className="attachment-name">{pendingFile.name}</div>
                <div className="attachment-type">{pendingFile.type.toUpperCase()} • Ready to send</div>
              </div>
              <button className="remove-attachment" onClick={() => setPendingFile(null)}>✕</button>
            </div>
          )}

          <div className="input-box">
            <button id="upload-btn" className="icon-btn" title="Upload file (image, DICOM, PDF)" onClick={() => fileInputRef.current.click()}>📎</button>
            <input ref={fileInputRef} type="file" accept=".png,.jpg,.jpeg,.dcm,.pdf" style={{ display: 'none' }} onChange={handleFileChange} />
            <textarea
              ref={textareaRef}
              id="chat-input"
              className="chat-textarea"
              rows={1}
              placeholder="Ask a clinical question or describe the scan... (Shift+Enter for new line)"
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              onPaste={onPaste}
            />
            <button id="send-btn" className="send-btn" onClick={handleSend} disabled={isTyping && !inputValue && !pendingFile} title="Send (Enter)">
              ➤
            </button>
          </div>
          <div className="input-hint">Drag & drop or paste files • Enter to send • Shift+Enter for new line</div>
        </div>
      </main>
    </div>
  );
}
