import React from 'react';

const VARIANT_STYLE = {
  primary: {
    background: 'var(--union-blue-dark)',
    color: '#fff',
    border: '1px solid transparent',
  },
  gold: {
    background: 'var(--union-gold)',
    color: '#fff',
    border: '1px solid transparent',
  },
  secondary: {
    background: 'var(--paper)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-strong)',
  },
  ghost: {
    background: 'transparent',
    color: 'var(--text-primary)',
    border: '1px solid transparent',
  },
  danger: {
    background: 'var(--danger-strong)',
    color: '#fff',
    border: '1px solid transparent',
  },
};

const SIZE_STYLE = {
  sm: { padding: '0.375rem 0.75rem', fontSize: 'var(--text-xs)', borderRadius: 'var(--radius-sm)' },
  md: { padding: '0.5rem 1rem', fontSize: 'var(--text-sm)', borderRadius: 'var(--radius-md)' },
  lg: { padding: '0.75rem 1.5rem', fontSize: 'var(--text-base)', borderRadius: 'var(--radius-lg)' },
};

/** Button — primary interactive control. Mirrors the product's solid
 * ufcw-blue / ufcw-gold action buttons and the ghost/ ufcw-blue/10 style
 * used for secondary actions. */
export function Button({
  variant = 'primary',
  size = 'md',
  icon = null,
  disabled = false,
  children,
  onClick,
  style,
  ...rest
}) {
  const v = VARIANT_STYLE[variant] || VARIANT_STYLE.primary;
  const s = SIZE_STYLE[size] || SIZE_STYLE.md;
  const [hover, setHover] = React.useState(false);

  const hoverBg = {
    primary: 'var(--union-blue-mid)',
    gold: '#c1922a',
    secondary: 'var(--surface-sunken)',
    ghost: 'rgba(13,59,84,0.08)',
    danger: '#b91c1c',
  }[variant];

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '0.5rem',
        fontFamily: 'var(--font-sans)',
        fontWeight: 600,
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.5 : 1,
        transition: 'background-color 150ms var(--ease-standard), opacity 150ms',
        ...v,
        ...s,
        background: hover && !disabled ? hoverBg : v.background,
        ...style,
      }}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
}
