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

function parseFindings(text) {
  const findings = [];
  // Resilient regex to handle markdown (**LOCATION**), extra text, and different casing
  const regex = /(?:FINDING|LABEL|NAME):\s*\**?(.*?)\**?\s*(?:LOCATION|BOX|COORD):\s*\**?\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\**?/gi;

  let match;
  while ((match = regex.exec(text)) !== null) {
    let ymin = parseFloat(match[2]);
    let xmin = parseFloat(match[3]);
    let ymax = parseFloat(match[4]);
    let xmax = parseFloat(match[5]);

    // Auto-normalize: If coordinates are in 0-1000 range, convert to 0-1
    if (ymin > 1 || xmin > 1 || ymax > 1 || xmax > 1) {
      ymin /= 1000;
      xmin /= 1000;
      ymax /= 1000;
      xmax /= 1000;
    }

    findings.push({ name: match[1].replace(/\*/g, '').trim(), ymin, xmin, ymax, xmax });
  }
  return findings;
}

// ─── Sub-Components ────────────────────────────────────────────────────────────
function ImageCard({ src, alt = "Medical Scan", findings = [] }) {
  const [lightbox, setLightbox] = useState(false);
  const containerRef = useRef(null);

  const renderOverlays = () => {
    if (!findings || findings.length === 0) return null;
    return (
      <svg className="localization-svg" viewBox="0 0 100 100" preserveAspectRatio="none">
        {findings.map((f, i) => (
          <g key={i}>
            <rect
              x={f.xmin * 100}
              y={f.ymin * 100}
              width={(f.xmax - f.xmin) * 100}
              height={(f.ymax - f.ymin) * 100}
              className="finding-box"
            />
            <text
              x={f.xmin * 100}
              y={f.ymin * 100 - 2}
              className="finding-label"
              fontSize="3.5"
            >
              {f.name}
            </text>
          </g>
        ))}
      </svg>
    );
  };

  return (
    <>
      <div className="image-preview-card" onClick={() => setLightbox(true)} title="Click to expand">
        <div style={{ position: 'relative' }}>
          <img src={src} alt={alt} style={{ display: 'block' }} />
          {renderOverlays()}
        </div>
        <span className="expand-hint">🔍 Expand</span>
        {findings.length > 0 && <div className="findings-indicator">📍 {findings.length} Findings Detected</div>}
      </div>
      {lightbox && (
        <div className="lightbox-overlay" onClick={() => setLightbox(false)}>
          <div style={{ position: 'relative', maxWidth: '90vw', maxHeight: '90vh' }}>
            <img src={src} alt={alt} />
            {renderOverlays()}
          </div>
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

// Thinking Block removed for cleaner clinical view

// ─── Report Modal ─────────────────────────────────────────────────────────────
function ReportModal({ isOpen, onClose, onSubmit, isSubmitting, onDownload }) {
  const [comment, setComment] = useState('');
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Report Errors</h2>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
            Report inaccuracies to the developer and download the case data for manual sharing.
          </p>
        </div>
        <div className="modal-body">
          <textarea
            className="report-textarea"
            placeholder="Describe the error or inaccuracy..."
            value={comment}
            onChange={e => setComment(e.target.value)}
          />
        </div>
        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose} disabled={isSubmitting}>Cancel</button>
          <button className="btn-secondary" style={{ color: 'var(--accent)' }} onClick={() => onDownload(comment)} disabled={isSubmitting}>
            📥 Download Case
          </button>
          <button className="btn-primary" onClick={() => onSubmit(comment)} disabled={isSubmitting}>
            {isSubmitting ? 'Submitting...' : 'Submit Report'}
          </button>
        </div>
      </div>
    </div>
  );
}

function MessageRow({ msg, onReport }) {
  const isUser = msg.role === 'user';
  const isLocalization = msg.isLocalization;
  const isConsultation = msg.isConsultation;
  const isAssistant = msg.role === 'assistant';

  const findings = !isUser ? parseFindings(msg.content) : [];

  return (
    <div className={`msg-row ${msg.role}`}>
      <div className={`avatar ${isUser ? 'user' : 'ai'} ${isConsultation ? 'ai-doctor' : ''}`}>
        {isUser ? '👤' : isConsultation ? '👨‍⚕️' : '✦'}
      </div>
      <div className="bubble-wrap">
        {isConsultation && !isUser && (
          <div className="consultation-badge">🩺 Patient Consultation</div>
        )}
        {msg.img && <ImageCard src={msg.img} findings={findings} />}
        {msg.dicom && <DicomCard filename={msg.dicom.filename} meta={msg.dicom.meta} />}
        {msg.pdf && (
          <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', padding: '8px 0' }}>
            📄 PDF: {msg.pdf}
          </div>
        )}
        {msg.content && (
          <div className={`bubble ${isLocalization ? 'bubble-mono' : ''} ${isConsultation && !isUser ? 'bubble-consultation' : ''}`}>
            {msg.content}
          </div>
        )}
        {isAssistant && !isUser && (
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <button className="report-btn" onClick={() => onReport(msg)}>
              🚩 Report Errors
            </button>
            {msg.reported && <span className="report-success-msg">✓ Reported</span>}
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
  const [activeFile, setActiveFile] = useState(null); // Persist file across turns
  const [isTyping, setIsTyping] = useState(false);
  const [backendUrl, setBackendUrl] = useState(localStorage.getItem('backend_url') || '');
  const [hfToken, setHfToken] = useState(localStorage.getItem('hf_token') || '');
  const [modelName, setModelName] = useState('google/medgemma-1.5-4b-it');
  const [analysisMode, setAnalysisMode] = useState('General Analysis');
  const [backendStatus, setBackendStatus] = useState('unchecked'); // unchecked | online | offline
  const [isDragging, setIsDragging] = useState(false);
  const [reportingMsg, setReportingMsg] = useState(null); // The message object being reported
  const [isSubmittingReport, setIsSubmittingReport] = useState(false);

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
    const fileObj = { file, type, preview, name: file.name };
    setPendingFile(fileObj);
    setActiveFile(fileObj);
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

      // Use pending file if available, otherwise fallback to activeFile for memory
      const fileContext = fileToSend || activeFile;
      if (fileContext?.file) formData.append('file', fileContext.file);

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
        full_output: data.full_output,
        prompt: prompt,
        isLocalization: analysisMode === 'Localization',
        isConsultation: analysisMode === 'Patient Consultation',
        // Ensure image/dicom context persists for overlay rendering
        img: userMsg.img,
        dicom: userMsg.dicom,
        pdf: userMsg.pdf
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

  // ── Reporting Logic ──────────────────────────────────────────────────────────
  const handleReportSubmit = async (comment) => {
    if (!reportingMsg) return;
    setIsSubmittingReport(true);
    try {
      const payload = {
        prompt: reportingMsg.prompt || "No prompt stored",
        response: reportingMsg.content,
        thinking: reportingMsg.thinking,
        user_comment: comment,
        metadata: {
          model: modelName,
          mode: analysisMode,
          timestamp: new Date().toISOString()
        }
      };

      const res = await fetch(apiUrl('/report'), {
        method: 'POST',
        headers: { ...ngrokHeaders, 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) throw new Error(await res.text());

      // Update message status to "reported"
      setMessages(prev => prev.map(m => m === reportingMsg ? { ...m, reported: true } : m));
      setReportingMsg(null);
    } catch (err) {
      alert(`Failed to send report: ${err.message}`);
    } finally {
      setIsSubmittingReport(false);
    }
  };

  const handleDownloadReport = (comment) => {
    if (!reportingMsg) return;
    const payload = {
      prompt: reportingMsg.prompt || "No prompt stored",
      response: reportingMsg.content,
      thinking: reportingMsg.thinking,
      user_comment: comment,
      metadata: {
        model: modelName,
        mode: analysisMode,
        timestamp: new Date().toISOString()
      }
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `medgemma_err_report_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
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
            <option value="google/medgemma-1.5-1b-it">MedGemma 1.5-1B (Low VRAM)</option>
            <option value="google/medgemma-1.5-4b-it">MedGemma 1.5-4B</option>
            <option value="google/medgemma-4b-it">MedGemma 4B</option>
            <option value="unsloth/medgemma-4b-it-GGUF">MedGemma 4B (Unsloth GGUF)</option>
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

        <div className="sidebar-footer">
          <div>✨ Developed by Dr. R. K. Ramanan</div>
          <div style={{ marginTop: '8px' }}>
            <a href="http://www.c-riht.org" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent)', textDecoration: 'none', fontSize: '0.72rem' }}>www.c-riht.org</a>
          </div>
          <div style={{ marginTop: '4px', fontSize: '0.7rem', opacity: 0.8 }}>c.riht.org@gmail.com</div>
        </div>
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
            {messages.map((msg, i) => <MessageRow key={i} msg={msg} onReport={setReportingMsg} />)}
            {isTyping && <TypingBubble />}
            <div ref={chatEndRef} />
          </div>
        )}

        {/* ── Modals ─────────────────────────────────────────────────────── */}
        <ReportModal
          isOpen={!!reportingMsg}
          onClose={() => setReportingMsg(null)}
          onSubmit={handleReportSubmit}
          isSubmitting={isSubmittingReport}
          onDownload={handleDownloadReport}
        />

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
