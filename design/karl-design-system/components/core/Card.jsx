import React from 'react';

/** Card — generic surface container. Matches the product's
 * `bg-white rounded-2xl shadow-sm border border-slate-200` pattern used for
 * assistant message cards, contract-viewer panels and settings sections. */
export function Card({ padded = true, children, style, ...rest }) {
  return (
    <div
      style={{
        background: 'var(--surface-card)',
        border: '1px solid var(--border-default)',
        borderRadius: 'var(--radius-xl)',
        boxShadow: 'var(--shadow-sm)',
        padding: padded ? 'var(--space-4)' : 0,
        ...style,
      }}
      {...rest}
    >
      {children}
    </div>
  );
}
