(function () {
    const TEMPLATE = document.createElement('template');
    TEMPLATE.innerHTML = `
        <style>
            :host {
                --karl-widget-primary: #0d3b54;
                --karl-widget-accent: #d4a029;
                --karl-widget-surface: rgba(255, 255, 255, 0.8);
                display: block;
                width: 100%;
                color: #173246;
                font-family: "Inter", system-ui, sans-serif;
            }

            .shell {
                border-radius: 28px;
                overflow: hidden;
                border: 1px solid rgba(23, 50, 70, 0.12);
                background: var(--karl-widget-surface);
                box-shadow: 0 20px 50px rgba(20, 33, 43, 0.08);
                backdrop-filter: blur(16px);
            }

            .header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 12px;
                padding: 14px 18px;
                color: white;
                background:
                    radial-gradient(circle at top left, color-mix(in srgb, var(--karl-widget-accent) 20%, transparent), transparent 24%),
                    linear-gradient(135deg, color-mix(in srgb, var(--karl-widget-primary) 88%, black), var(--karl-widget-primary));
            }

            .eyebrow {
                margin: 0;
                font-size: 11px;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                color: color-mix(in srgb, white 70%, var(--karl-widget-accent));
            }

            .title {
                margin: 2px 0 0;
                font-size: 1rem;
                font-weight: 600;
            }

            .note {
                font-size: 0.83rem;
                color: rgba(255, 255, 255, 0.82);
            }

            iframe {
                display: block;
                width: 100%;
                min-height: 960px;
                height: var(--karl-widget-height, 1100px);
                border: 0;
                background: white;
            }

            .status {
                padding: 16px 18px;
                font-size: 0.92rem;
                color: #475569;
                background: rgba(255, 255, 255, 0.9);
                border-top: 1px solid rgba(23, 50, 70, 0.08);
            }
        </style>
        <section class="shell">
            <header class="header">
                <div>
                    <p class="eyebrow">Karl Member Widget</p>
                    <p class="title">Union member workspace</p>
                </div>
                <div class="note">Hosted by Karl</div>
            </header>
            <iframe part="frame" referrerpolicy="strict-origin-when-cross-origin" loading="eager"></iframe>
            <div class="status" hidden></div>
        </section>
    `;

    function trimTrailingSlash(value) {
        return String(value || '').replace(/\/+$/, '');
    }

    class KarlMemberWidget extends HTMLElement {
        static get observedAttributes() {
            return ['tenant-slug', 'api-base', 'primary-color', 'accent-color', 'surface-tint', 'heading', 'debug', 'height'];
        }

        constructor() {
            super();
            this.attachShadow({ mode: 'open' });
            this.shadowRoot.appendChild(TEMPLATE.content.cloneNode(true));
        }

        connectedCallback() {
            this.render();
        }

        attributeChangedCallback() {
            if (this.isConnected) {
                this.render();
            }
        }

        get tenantSlug() {
            return String(this.getAttribute('tenant-slug') || '').trim();
        }

        get apiBase() {
            const attrValue = trimTrailingSlash(this.getAttribute('api-base'));
            if (attrValue) return attrValue;
            return trimTrailingSlash(window.location.origin);
        }

        render() {
            const frame = this.shadowRoot.querySelector('iframe');
            const status = this.shadowRoot.querySelector('.status');
            const title = this.shadowRoot.querySelector('.title');
            const shell = this.shadowRoot.querySelector('.shell');
            const height = String(this.getAttribute('height') || '1100').trim();
            const primary = String(this.getAttribute('primary-color') || '').trim();
            const accent = String(this.getAttribute('accent-color') || '').trim();
            const surface = String(this.getAttribute('surface-tint') || '').trim();
            const heading = String(this.getAttribute('heading') || '').trim();
            const debug = this.getAttribute('debug') === 'true' || this.getAttribute('debug') === '1';

            if (primary) this.style.setProperty('--karl-widget-primary', primary);
            if (accent) this.style.setProperty('--karl-widget-accent', accent);
            if (surface) this.style.setProperty('--karl-widget-surface', surface);
            this.style.setProperty('--karl-widget-height', /^\d+$/.test(height) ? `${height}px` : height);

            if (!this.tenantSlug) {
                frame.hidden = true;
                status.hidden = false;
                status.textContent = 'Karl member widget requires a tenant-slug attribute.';
                return;
            }

            const url = new URL(`/embed/member-frame/${encodeURIComponent(this.tenantSlug)}`, `${this.apiBase}/`);
            if (this.apiBase) {
                url.searchParams.set('api_base', this.apiBase);
            }
            if (primary) url.searchParams.set('theme_color', primary);
            if (accent) url.searchParams.set('accent_color', accent);
            if (surface) url.searchParams.set('surface_tint', surface);
            if (heading) url.searchParams.set('welcome_heading', heading);
            if (debug) url.searchParams.set('debug', '1');

            frame.hidden = false;
            status.hidden = true;
            frame.src = url.toString();
            title.textContent = heading || `${this.tenantSlug} member workspace`;
            shell.dataset.tenantSlug = this.tenantSlug;
        }
    }

    if (!window.customElements.get('karl-member-widget')) {
        window.customElements.define('karl-member-widget', KarlMemberWidget);
    }

    window.KarlMemberWidget = {
        mount(target, config = {}) {
            const element = target instanceof HTMLElement ? target : document.querySelector(String(target || ''));
            if (!element) return null;
            const widget = document.createElement('karl-member-widget');
            Object.entries(config || {}).forEach(([key, value]) => {
                if (value === undefined || value === null || value === '') return;
                const attr = key.replace(/[A-Z]/g, (match) => `-${match.toLowerCase()}`);
                widget.setAttribute(attr, String(value));
            });
            element.replaceWith(widget);
            return widget;
        },
    };
})();
