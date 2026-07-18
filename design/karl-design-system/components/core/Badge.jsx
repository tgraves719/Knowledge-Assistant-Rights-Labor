import React from 'react';

const TONE = {
  neutral: { bg: 'var(--ink-100)', text: 'var(--ink-700)' },
  citation: { bg: 'var(--citation-bg)', text: 'var(--citation-text)' },
  success: { bg: 'var(--success-bg)', text: 'var(--success-text)', border: 'var(--success-border)' },
  warning: { bg: 'var(--warning-bg)', text: 'var(--warning-text)', border: 'var(--warning-border)' },
  danger: { bg: 'var(--danger-bg)', text: 'var(--danger-text)', border: 'var(--danger-border)' },
  info: { bg: 'var(--info-bg)', text: 'var(--info-text)', border: 'var(--info-border)' },
};

/** Badge — small status/provenance pill. Used for citation source tags
 * (MOA / Base / MOA+Base) and admin status labels. */
export function Badge({ tone = 'neutral', children, style, ...rest }) {
  const t = TONE[tone] || TONE.neutral;
  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.25rem',
        fontFamily: 'var(--font-sans)',
        fontSize: 'var(--text-2xs)',
        fontWeight: 700,
        letterSpacing: 'var(--tracking-wide)',
        textTransform: 'uppercase',
        padding: '0.125rem 0.5rem',
        borderRadius: 'var(--radius-full)',
        background: t.bg,
        color: t.text,
        border: t.border ? `1px solid ${t.border}` : 'none',
        ...style,
      }}
      {...rest}
    >
      {children}
    </span>
  );
}
