import React from 'react';

/** Toggle — pill switch used for dark-mode / developer-mode settings.
 * Track turns gold when active (mirrors #dark-mode-toggle.bg-ufcw-blue
 * dark-mode override which recolors to gold). */
export function Toggle({ checked = false, onChange, label }) {
  return (
    <label style={{ display: 'inline-flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: 'var(--text-sm)', color: 'var(--text-primary)' }}>
      <span
        onClick={() => onChange && onChange(!checked)}
        style={{
          width: 40, height: 22, borderRadius: 'var(--radius-full)',
          background: checked ? 'var(--union-gold)' : 'var(--ink-300)',
          position: 'relative', transition: 'background-color 150ms var(--ease-standard)', flexShrink: 0,
        }}
      >
        <span style={{
          position: 'absolute', top: 2, left: checked ? 20 : 2,
          width: 18, height: 18, borderRadius: 'var(--radius-full)', background: '#fff',
          boxShadow: 'var(--shadow-sm)', transition: 'left 150ms var(--ease-standard)',
        }} />
      </span>
      {label}
    </label>
  );
}
