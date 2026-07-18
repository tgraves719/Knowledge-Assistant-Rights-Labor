import React from 'react';

/** QuickActionCard — translucent action tile shown above the chat composer
 * (`.quick-action` in index.html: bg-white/10, hover→gold/30, backdrop-blur).
 * Intended to sit on the union-blue gradient header, not on a light page. */
export function QuickActionCard({ icon, children, onClick }) {
  const [hover, setHover] = React.useState(false);
  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.6rem',
        textAlign: 'left',
        padding: '0.65rem 0.9rem',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--border-on-accent)',
        background: hover ? 'rgba(212,160,41,0.3)' : 'rgba(255,255,255,0.1)',
        backdropFilter: 'blur(8px)',
        color: '#fff',
        fontFamily: 'var(--font-sans)',
        fontSize: 'var(--text-sm)',
        cursor: 'pointer',
        transition: 'background-color 150ms var(--ease-standard)',
        width: '100%',
      }}
    >
      {icon}
      <span>{children}</span>
    </button>
  );
}
