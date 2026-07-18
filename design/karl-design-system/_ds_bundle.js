/* @ds-bundle: {"format":4,"namespace":"KARLDesignSystem_3d3866","components":[{"name":"ShieldMark","sourcePath":"components/brand/ShieldMark.jsx"},{"name":"ChatBubble","sourcePath":"components/chat/ChatBubble.jsx"},{"name":"CitationLink","sourcePath":"components/chat/CitationLink.jsx"},{"name":"QuickActionCard","sourcePath":"components/chat/QuickActionCard.jsx"},{"name":"Badge","sourcePath":"components/core/Badge.jsx"},{"name":"Button","sourcePath":"components/core/Button.jsx"},{"name":"Card","sourcePath":"components/core/Card.jsx"},{"name":"EmploymentOption","sourcePath":"components/forms/EmploymentOption.jsx"},{"name":"Input","sourcePath":"components/forms/Input.jsx"},{"name":"Toggle","sourcePath":"components/forms/Toggle.jsx"},{"name":"TabBar","sourcePath":"components/navigation/TabBar.jsx"}],"sourceHashes":{"components/brand/ShieldMark.jsx":"605fbfcb1ebc","components/chat/ChatBubble.jsx":"cdb4647925a8","components/chat/CitationLink.jsx":"7e29b4fd9b76","components/chat/QuickActionCard.jsx":"e2d671e58342","components/core/Badge.jsx":"ec0397afa27a","components/core/Button.jsx":"abed4b8bc1a2","components/core/Card.jsx":"dee752cd5bcc","components/forms/EmploymentOption.jsx":"9890354f9f0f","components/forms/Input.jsx":"37f10e93d1eb","components/forms/Toggle.jsx":"25406d1dc7fe","components/navigation/TabBar.jsx":"5ba11fc5d97e"},"inlinedExternals":[],"unexposedExports":[]} */

(() => {

const __ds_ns = (window.KARLDesignSystem_3d3866 = window.KARLDesignSystem_3d3866 || {});

const __ds_scope = {};

(__ds_ns.__errors = __ds_ns.__errors || []);

// components/brand/ShieldMark.jsx
try { (() => {
/** ShieldMark — Karl's brand mark: a two-half shield (each half tenant-
 * colored via --union-shield-left / --union-shield-right) that splits apart
 * to reveal a scanning-paper motif while "thinking". Recreated from the
 * `.karl-avatar-shell` SVG in index.html (kept minimal — full scanner/paper
 * animation lives in the product; this is the static + thinking-split mark
 * for use as an avatar/brand glyph).
 */
function ShieldMark({
  size = 46,
  state = 'idle',
  leftColor,
  rightColor
}) {
  const left = leftColor || 'var(--union-shield-left)';
  const right = rightColor || 'var(--union-shield-right)';
  const thinking = state === 'thinking';
  return /*#__PURE__*/React.createElement("div", {
    style: {
      width: size,
      height: size,
      filter: 'drop-shadow(0 4px 10px rgba(15,23,42,0.35))'
    }
  }, /*#__PURE__*/React.createElement("svg", {
    viewBox: "0 0 200 200",
    width: "100%",
    height: "100%",
    overflow: "visible"
  }, /*#__PURE__*/React.createElement("g", {
    style: {
      transform: thinking ? 'translateX(-32px)' : 'translateX(0)',
      transition: 'transform 800ms cubic-bezier(0.2,0,0.2,1)',
      transformBox: 'fill-box',
      transformOrigin: 'center'
    }
  }, /*#__PURE__*/React.createElement("path", {
    d: "M100 190C50 190,20 150,20 40Q60 40,100 30Z",
    fill: left
  })), /*#__PURE__*/React.createElement("g", {
    style: {
      transform: thinking ? 'translateX(32px)' : 'translateX(0)',
      transition: 'transform 800ms cubic-bezier(0.2,0,0.2,1)',
      transformBox: 'fill-box',
      transformOrigin: 'center'
    }
  }, /*#__PURE__*/React.createElement("path", {
    d: "M100 190C150 190,180 150,180 40Q140 40,100 30Z",
    fill: right
  }))));
}
Object.assign(__ds_scope, { ShieldMark });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/brand/ShieldMark.jsx", error: String((e && e.message) || e) }); }

// components/chat/ChatBubble.jsx
try { (() => {
/** ChatBubble — Karl chat message. `role="assistant"` renders the citation-
 * ready card (rounded-2xl, tail on bottom-left); `role="user"` renders the
 * solid union-blue bubble with tail on bottom-right — exact shapes from
 * `.assistant-message-card` in app.js. */
function ChatBubble({
  role = 'assistant',
  children
}) {
  const isUser = role === 'user';
  return /*#__PURE__*/React.createElement("div", {
    style: {
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 'var(--space-3)'
    }
  }, /*#__PURE__*/React.createElement("div", {
    style: {
      maxWidth: '85%',
      padding: '0.75rem 1rem',
      fontSize: 'var(--text-sm)',
      lineHeight: 'var(--leading-relaxed)',
      borderRadius: isUser ? 'var(--radius-xl) var(--radius-xl) var(--radius-xs) var(--radius-xl)' : 'var(--radius-xl) var(--radius-xl) var(--radius-xl) var(--radius-xs)',
      background: isUser ? 'var(--union-blue-dark)' : 'var(--surface-card)',
      color: isUser ? '#fff' : 'var(--text-primary)',
      border: isUser ? 'none' : '1px solid var(--border-default)',
      boxShadow: isUser ? 'none' : 'var(--shadow-sm)'
    }
  }, children));
}
Object.assign(__ds_scope, { ChatBubble });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/chat/ChatBubble.jsx", error: String((e && e.message) || e) }); }

// components/chat/CitationLink.jsx
try { (() => {
/** CitationLink — inline dotted-underline link that opens the citation
 * popover. Exact color/hover from `.citation-link` in index.html. */
function CitationLink({
  label,
  onClick
}) {
  return /*#__PURE__*/React.createElement("a", {
    onClick: onClick,
    style: {
      color: 'var(--union-blue-dark)',
      textDecoration: 'underline',
      textDecorationStyle: 'dotted',
      textUnderlineOffset: '3px',
      cursor: 'pointer',
      fontWeight: 500
    },
    onMouseEnter: e => e.target.style.color = 'var(--union-gold)',
    onMouseLeave: e => e.target.style.color = 'var(--union-blue-dark)'
  }, label);
}
Object.assign(__ds_scope, { CitationLink });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/chat/CitationLink.jsx", error: String((e && e.message) || e) }); }

// components/chat/QuickActionCard.jsx
try { (() => {
/** QuickActionCard — translucent action tile shown above the chat composer
 * (`.quick-action` in index.html: bg-white/10, hover→gold/30, backdrop-blur).
 * Intended to sit on the union-blue gradient header, not on a light page. */
function QuickActionCard({
  icon,
  children,
  onClick
}) {
  const [hover, setHover] = React.useState(false);
  return /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: onClick,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
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
      width: '100%'
    }
  }, icon, /*#__PURE__*/React.createElement("span", null, children));
}
Object.assign(__ds_scope, { QuickActionCard });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/chat/QuickActionCard.jsx", error: String((e && e.message) || e) }); }

// components/core/Badge.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
const TONE = {
  neutral: {
    bg: 'var(--ink-100)',
    text: 'var(--ink-700)'
  },
  citation: {
    bg: 'var(--citation-bg)',
    text: 'var(--citation-text)'
  },
  success: {
    bg: 'var(--success-bg)',
    text: 'var(--success-text)',
    border: 'var(--success-border)'
  },
  warning: {
    bg: 'var(--warning-bg)',
    text: 'var(--warning-text)',
    border: 'var(--warning-border)'
  },
  danger: {
    bg: 'var(--danger-bg)',
    text: 'var(--danger-text)',
    border: 'var(--danger-border)'
  },
  info: {
    bg: 'var(--info-bg)',
    text: 'var(--info-text)',
    border: 'var(--info-border)'
  }
};

/** Badge — small status/provenance pill. Used for citation source tags
 * (MOA / Base / MOA+Base) and admin status labels. */
function Badge({
  tone = 'neutral',
  children,
  style,
  ...rest
}) {
  const t = TONE[tone] || TONE.neutral;
  return /*#__PURE__*/React.createElement("span", _extends({
    style: {
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
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Badge });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Badge.jsx", error: String((e && e.message) || e) }); }

// components/core/Button.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
const VARIANT_STYLE = {
  primary: {
    background: 'var(--union-blue-dark)',
    color: '#fff',
    border: '1px solid transparent'
  },
  gold: {
    background: 'var(--union-gold)',
    color: '#fff',
    border: '1px solid transparent'
  },
  secondary: {
    background: 'var(--paper)',
    color: 'var(--text-primary)',
    border: '1px solid var(--border-strong)'
  },
  ghost: {
    background: 'transparent',
    color: 'var(--text-primary)',
    border: '1px solid transparent'
  },
  danger: {
    background: 'var(--danger-strong)',
    color: '#fff',
    border: '1px solid transparent'
  }
};
const SIZE_STYLE = {
  sm: {
    padding: '0.375rem 0.75rem',
    fontSize: 'var(--text-xs)',
    borderRadius: 'var(--radius-sm)'
  },
  md: {
    padding: '0.5rem 1rem',
    fontSize: 'var(--text-sm)',
    borderRadius: 'var(--radius-md)'
  },
  lg: {
    padding: '0.75rem 1.5rem',
    fontSize: 'var(--text-base)',
    borderRadius: 'var(--radius-lg)'
  }
};

/** Button — primary interactive control. Mirrors the product's solid
 * ufcw-blue / ufcw-gold action buttons and the ghost/ ufcw-blue/10 style
 * used for secondary actions. */
function Button({
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
    danger: '#b91c1c'
  }[variant];
  return /*#__PURE__*/React.createElement("button", _extends({
    type: "button",
    disabled: disabled,
    onClick: onClick,
    onMouseEnter: () => setHover(true),
    onMouseLeave: () => setHover(false),
    style: {
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
      ...style
    }
  }, rest), icon, children);
}
Object.assign(__ds_scope, { Button });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Button.jsx", error: String((e && e.message) || e) }); }

// components/core/Card.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/** Card — generic surface container. Matches the product's
 * `bg-white rounded-2xl shadow-sm border border-slate-200` pattern used for
 * assistant message cards, contract-viewer panels and settings sections. */
function Card({
  padded = true,
  children,
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("div", _extends({
    style: {
      background: 'var(--surface-card)',
      border: '1px solid var(--border-default)',
      borderRadius: 'var(--radius-xl)',
      boxShadow: 'var(--shadow-sm)',
      padding: padded ? 'var(--space-4)' : 0,
      ...style
    }
  }, rest), children);
}
Object.assign(__ds_scope, { Card });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/core/Card.jsx", error: String((e && e.message) || e) }); }

// components/forms/EmploymentOption.jsx
try { (() => {
/** EmploymentOption — the two-up selectable card used in onboarding for
 * Full-time / Part-time (and similar binary classification choices).
 * Mirrors `.mo-employment-btn` from member-onboarding. */
function EmploymentOption({
  title,
  subtitle,
  selected = false,
  onClick
}) {
  return /*#__PURE__*/React.createElement("button", {
    type: "button",
    onClick: onClick,
    style: {
      textAlign: 'left',
      padding: '0.65rem 0.7rem',
      borderRadius: 'var(--radius-md)',
      border: `2px solid ${selected ? 'var(--union-blue-dark)' : 'var(--ink-300)'}`,
      background: selected ? 'var(--ink-200)' : 'var(--ink-50)',
      color: selected ? 'var(--ink-900)' : 'var(--ink-700)',
      cursor: 'pointer',
      fontFamily: 'var(--font-sans)',
      transition: 'border-color 180ms var(--ease-standard), background-color 180ms var(--ease-standard)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'block',
      fontSize: 'var(--text-sm)',
      fontWeight: 700,
      lineHeight: 1.1
    }
  }, title), subtitle ? /*#__PURE__*/React.createElement("span", {
    style: {
      display: 'block',
      marginTop: '0.15rem',
      fontSize: 'var(--text-xs)',
      color: 'var(--ink-500)'
    }
  }, subtitle) : null);
}
Object.assign(__ds_scope, { EmploymentOption });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/EmploymentOption.jsx", error: String((e && e.message) || e) }); }

// components/forms/Input.jsx
try { (() => {
function _extends() { return _extends = Object.assign ? Object.assign.bind() : function (n) { for (var e = 1; e < arguments.length; e++) { var t = arguments[e]; for (var r in t) ({}).hasOwnProperty.call(t, r) && (n[r] = t[r]); } return n; }, _extends.apply(null, arguments); }
/** Input — text input matching the product's slate-bordered, gold-focus-ring
 * form fields (onboarding, settings, chat composer). */
function Input({
  style,
  ...rest
}) {
  return /*#__PURE__*/React.createElement("input", _extends({
    style: {
      width: '100%',
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--text-sm)',
      padding: '0.5rem 0.75rem',
      borderRadius: 'var(--radius-md)',
      border: '1px solid var(--border-strong)',
      background: 'var(--surface-card)',
      color: 'var(--text-primary)',
      outline: 'none',
      ...style
    },
    onFocus: e => {
      e.target.style.borderColor = 'var(--ink-500)';
      e.target.style.boxShadow = '0 0 0 2px rgba(245,158,11,0.35)';
    },
    onBlur: e => {
      e.target.style.borderColor = 'var(--border-strong)';
      e.target.style.boxShadow = 'none';
    }
  }, rest));
}
Object.assign(__ds_scope, { Input });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Input.jsx", error: String((e && e.message) || e) }); }

// components/forms/Toggle.jsx
try { (() => {
/** Toggle — pill switch used for dark-mode / developer-mode settings.
 * Track turns gold when active (mirrors #dark-mode-toggle.bg-ufcw-blue
 * dark-mode override which recolors to gold). */
function Toggle({
  checked = false,
  onChange,
  label
}) {
  return /*#__PURE__*/React.createElement("label", {
    style: {
      display: 'inline-flex',
      alignItems: 'center',
      gap: '0.6rem',
      cursor: 'pointer',
      fontFamily: 'var(--font-sans)',
      fontSize: 'var(--text-sm)',
      color: 'var(--text-primary)'
    }
  }, /*#__PURE__*/React.createElement("span", {
    onClick: () => onChange && onChange(!checked),
    style: {
      width: 40,
      height: 22,
      borderRadius: 'var(--radius-full)',
      background: checked ? 'var(--union-gold)' : 'var(--ink-300)',
      position: 'relative',
      transition: 'background-color 150ms var(--ease-standard)',
      flexShrink: 0
    }
  }, /*#__PURE__*/React.createElement("span", {
    style: {
      position: 'absolute',
      top: 2,
      left: checked ? 20 : 2,
      width: 18,
      height: 18,
      borderRadius: 'var(--radius-full)',
      background: '#fff',
      boxShadow: 'var(--shadow-sm)',
      transition: 'left 150ms var(--ease-standard)'
    }
  })), label);
}
Object.assign(__ds_scope, { Toggle });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/forms/Toggle.jsx", error: String((e && e.message) || e) }); }

// components/navigation/TabBar.jsx
try { (() => {
/** TabBar — bottom (mobile) / top (desktop) app navigation. Active tab gets
 * a gold glow on icon+label and a soft rounded highlight behind it
 * (`.tab-btn.tab-active` in index.html). */
function TabBar({
  tabs,
  active,
  onChange
}) {
  return /*#__PURE__*/React.createElement("nav", {
    style: {
      display: 'flex',
      background: 'var(--surface-card)',
      borderTop: '1px solid var(--border-default)',
      boxShadow: '0 -2px 10px rgba(0,0,0,0.1)'
    }
  }, tabs.map(tab => {
    const isActive = tab.id === active;
    return /*#__PURE__*/React.createElement("button", {
      key: tab.id,
      type: "button",
      onClick: () => onChange && onChange(tab.id),
      style: {
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
        fontFamily: 'var(--font-sans)'
      }
    }, isActive && /*#__PURE__*/React.createElement("span", {
      style: {
        position: 'absolute',
        inset: 4,
        background: 'rgba(212,160,41,0.15)',
        borderRadius: 'var(--radius-sm)',
        zIndex: -1
      }
    }), /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 20,
        lineHeight: 1,
        color: isActive ? 'var(--union-gold)' : 'var(--ink-500)',
        filter: isActive ? 'var(--glow-gold-sm)' : 'none'
      }
    }, tab.icon), /*#__PURE__*/React.createElement("span", {
      style: {
        fontSize: 'var(--text-2xs)',
        fontWeight: 600,
        color: isActive ? 'var(--union-gold)' : 'var(--ink-500)',
        textShadow: isActive ? 'var(--glow-gold-text)' : 'none'
      }
    }, tab.label));
  }));
}
Object.assign(__ds_scope, { TabBar });
})(); } catch (e) { __ds_ns.__errors.push({ path: "components/navigation/TabBar.jsx", error: String((e && e.message) || e) }); }

__ds_ns.ShieldMark = __ds_scope.ShieldMark;

__ds_ns.ChatBubble = __ds_scope.ChatBubble;

__ds_ns.CitationLink = __ds_scope.CitationLink;

__ds_ns.QuickActionCard = __ds_scope.QuickActionCard;

__ds_ns.Badge = __ds_scope.Badge;

__ds_ns.Button = __ds_scope.Button;

__ds_ns.Card = __ds_scope.Card;

__ds_ns.EmploymentOption = __ds_scope.EmploymentOption;

__ds_ns.Input = __ds_scope.Input;

__ds_ns.Toggle = __ds_scope.Toggle;

__ds_ns.TabBar = __ds_scope.TabBar;

})();
