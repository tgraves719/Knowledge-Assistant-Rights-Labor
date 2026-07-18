import React from 'react';

/** TabBar — bottom (mobile) / top (desktop) app navigation. Active tab gets
 * a gold glow on icon+label and a soft rounded highlight behind it
 * (`.tab-btn.tab-active` in index.html). */
export function TabBar({ tabs, active, onChange }) {
  return (
    <nav style={{ display: 'flex', background: 'var(--surface-card)', borderTop: '1px solid var(--border-default)', boxShadow: '0 -2px 10px rgba(0,0,0,0.1)' }}>
      {tabs.map((tab) => {
        const isActive = tab.id === active;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange && onChange(tab.id)}
            style={{
              position: 'relative',
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '0.2rem',
              padding: '0.6rem 0.25rem',
              border: 'none',
              outline: 'none',
              background: 'transparent',
              cursor: 'pointer',
              fontFamily: 'var(--font-sans)',
            }}
          >
            {isActive && (
              <span style={{ position: 'absolute', inset: 4, background: 'rgba(212,160,41,0.15)', borderRadius: 'var(--radius-sm)', zIndex: -1 }} />
            )}
            <span style={{ fontSize: 20, lineHeight: 1, color: isActive ? 'var(--union-gold)' : 'var(--ink-500)', filter: isActive ? 'var(--glow-gold-sm)' : 'none' }}>{tab.icon}</span>
            <span style={{ fontSize: 'var(--text-2xs)', fontWeight: 600, color: isActive ? 'var(--union-gold)' : 'var(--ink-500)', textShadow: isActive ? 'var(--glow-gold-text)' : 'none' }}>{tab.label}</span>
          </button>
        );
      })}
    </nav>
  );
}
