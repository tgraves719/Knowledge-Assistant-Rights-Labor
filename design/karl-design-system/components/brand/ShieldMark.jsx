import React from 'react';

/** ShieldMark — Karl's brand mark: a two-half shield (each half tenant-
 * colored via --union-shield-left / --union-shield-right) that splits apart
 * to reveal a scanning-paper motif while "thinking". Recreated from the
 * `.karl-avatar-shell` SVG in index.html (kept minimal — full scanner/paper
 * animation lives in the product; this is the static + thinking-split mark
 * for use as an avatar/brand glyph).
 */
export function ShieldMark({ size = 46, state = 'idle', leftColor, rightColor }) {
  const left = leftColor || 'var(--union-shield-left)';
  const right = rightColor || 'var(--union-shield-right)';
  const thinking = state === 'thinking';
  return (
    <div style={{ width: size, height: size, filter: 'drop-shadow(0 4px 10px rgba(15,23,42,0.35))' }}>
      <svg viewBox="0 0 200 200" width="100%" height="100%" overflow="visible">
        <g style={{ transform: thinking ? 'translateX(-32px)' : 'translateX(0)', transition: 'transform 800ms cubic-bezier(0.2,0,0.2,1)', transformBox: 'fill-box', transformOrigin: 'center' }}>
          <path d="M100 190C50 190,20 150,20 40Q60 40,100 30Z" fill={left} />
        </g>
        <g style={{ transform: thinking ? 'translateX(32px)' : 'translateX(0)', transition: 'transform 800ms cubic-bezier(0.2,0,0.2,1)', transformBox: 'fill-box', transformOrigin: 'center' }}>
          <path d="M100 190C150 190,180 150,180 40Q140 40,100 30Z" fill={right} />
        </g>
      </svg>
    </div>
  );
}
