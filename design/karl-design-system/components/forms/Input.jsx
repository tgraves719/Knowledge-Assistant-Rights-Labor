import React from 'react';

/** Input — text input matching the product's slate-bordered, gold-focus-ring
 * form fields (onboarding, settings, chat composer). */
export function Input({ style, ...rest }) {
  return (
    <input
      style={{
        width: '100%',
        fontFamily: 'var(--font-sans)',
        fontSize: 'var(--text-sm)',
        padding: '0.5rem 0.75rem',
        borderRadius: 'var(--radius-md)',
        border: '1px solid var(--border-strong)',
        background: 'var(--surface-card)',
        color: 'var(--text-primary)',
        outline: 'none',
        ...style,
      }}
      onFocus={(e) => { e.target.style.borderColor = 'var(--ink-500)'; e.target.style.boxShadow = '0 0 0 2px rgba(245,158,11,0.35)'; }}
      onBlur={(e) => { e.target.style.borderColor = 'var(--border-strong)'; e.target.style.boxShadow = 'none'; }}
      {...rest}
    />
  );
}
