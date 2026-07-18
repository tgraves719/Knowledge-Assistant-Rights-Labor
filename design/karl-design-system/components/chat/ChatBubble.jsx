import React from 'react';

/** ChatBubble — Karl chat message. `role="assistant"` renders the citation-
 * ready card (rounded-2xl, tail on bottom-left); `role="user"` renders the
 * solid union-blue bubble with tail on bottom-right — exact shapes from
 * `.assistant-message-card` in app.js. */
export function ChatBubble({ role = 'assistant', children }) {
  const isUser = role === 'user';
  return (
    <div style={{ display: 'flex', justifyContent: isUser ? 'flex-end' : 'flex-start', marginBottom: 'var(--space-3)' }}>
      <div
        style={{
          maxWidth: '85%',
          padding: '0.75rem 1rem',
          fontSize: 'var(--text-sm)',
          lineHeight: 'var(--leading-relaxed)',
          borderRadius: isUser
            ? 'var(--radius-xl) var(--radius-xl) var(--radius-xs) var(--radius-xl)'
            : 'var(--radius-xl) var(--radius-xl) var(--radius-xl) var(--radius-xs)',
          background: isUser ? 'var(--union-blue-dark)' : 'var(--surface-card)',
          color: isUser ? '#fff' : 'var(--text-primary)',
          border: isUser ? 'none' : '1px solid var(--border-default)',
          boxShadow: isUser ? 'none' : 'var(--shadow-sm)',
        }}
      >
        {children}
      </div>
    </div>
  );
}
