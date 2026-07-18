import React from 'react';

/** EmploymentOption — the two-up selectable card used in onboarding for
 * Full-time / Part-time (and similar binary classification choices).
 * Mirrors `.mo-employment-btn` from member-onboarding. */
export function EmploymentOption({ title, subtitle, selected = false, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        textAlign: 'left',
        padding: '0.65rem 0.7rem',
        borderRadius: 'var(--radius-md)',
        border: `2px solid ${selected ? 'var(--union-blue-dark)' : 'var(--ink-300)'}`,
        background: selected ? 'var(--ink-200)' : 'var(--ink-50)',
        color: selected ? 'var(--ink-900)' : 'var(--ink-700)',
        cursor: 'pointer',
        fontFamily: 'var(--font-sans)',
        transition: 'border-color 180ms var(--ease-standard), background-color 180ms var(--ease-standard)',
      }}
    >
      <span style={{ display: 'block', fontSize: 'var(--text-sm)', fontWeight: 700, lineHeight: 1.1 }}>{title}</span>
      {subtitle ? <span style={{ display: 'block', marginTop: '0.15rem', fontSize: 'var(--text-xs)', color: 'var(--ink-500)' }}>{subtitle}</span> : null}
    </button>
  );
}
