import React from 'react';

/** CitationLink — inline dotted-underline link that opens the citation
 * popover. Exact color/hover from `.citation-link` in index.html. */
export function CitationLink({ label, onClick }) {
  return (
    <a
      onClick={onClick}
      style={{
        color: 'var(--union-blue-dark)',
        textDecoration: 'underline',
        textDecorationStyle: 'dotted',
        textUnderlineOffset: '3px',
        cursor: 'pointer',
        fontWeight: 500,
      }}
      onMouseEnter={(e) => (e.target.style.color = 'var(--union-gold)')}
      onMouseLeave={(e) => (e.target.style.color = 'var(--union-blue-dark)')}
    >
      {label}
    </a>
  );
}
