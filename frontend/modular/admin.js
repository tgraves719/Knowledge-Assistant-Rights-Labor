const STORAGE_KEY = 'karl_admin_auth_context';
const USER_PAGE_SIZE = 25;
const ALERTS_PAGE_SIZE = 5;
const PROVIDER_PROFILES = {
    openrouter: {
        label: 'OpenRouter',
        help: 'Use your OpenRouter model name here. Karl fills in the OpenRouter base URL and standard headers automatically.',
        config: () => ({
            base_url: 'https://openrouter.ai/api/v1',
            temperature: 0.2,
            http_referer: window.location.origin,
            x_title: 'Karl',
        }),
    },
    openai: {
        label: 'OpenAI',
        help: 'Use the exact OpenAI model name. Karl uses the standard OpenAI API defaults for this provider.',
        config: () => ({
            temperature: 0.2,
        }),
    },
    gemini: {
        label: 'Google',
        help: 'Use the Gemini model name configured for your Google API key.',
        config: () => ({}),
    },
    anthropic: {
        label: 'Anthropic',
        help: 'Anthropic is listed for future support. Save it here if you want the config stored, but live testing may still depend on backend support.',
        config: () => ({}),
    },
};
const ROLE_ORDER = ['user', 'steward_admin', 'union_admin', 'super_admin'];

const routeContext = resolveRouteContext();
document.body.dataset.routeMode = routeContext.mode;

const state = {
    auth: loadAuthContext(),
    selectedUnion: null,
    tenantBootstrap: null,
    platformSummary: null,
    platformOps: null,
    dashboard: null,
    platformDashboard: null,
    globalTrackingPolicy: null,
    unionTrackingPolicy: null,
    reviewQueue: {
        items: [],
        summary: null,
        query: '',
        unionId: '',
        reviewStatus: '',
        status: '',
    },
    reviewDocument: null,
    sessionTimeline: null,
    userDirectory: {
        items: [],
        page: 1,
        pageSize: USER_PAGE_SIZE,
        total: 0,
        unionTotal: 0,
        query: '',
        field: 'all',
        sort: 'name',
        direction: 'asc',
        open: false,
        activeUserId: null,
    },
    feeds: {
        selected: {
            security: { page: 1, total: 0, pageSize: ALERTS_PAGE_SIZE },
            notifications: { page: 1, total: 0, pageSize: ALERTS_PAGE_SIZE },
            telemetry: { page: 1, total: 0, pageSize: 8, category: '', eventType: '', query: '', sessionId: '' },
        },
        global: {
            security: { page: 1, total: 0, pageSize: ALERTS_PAGE_SIZE },
            notifications: { page: 1, total: 0, pageSize: ALERTS_PAGE_SIZE },
            telemetry: { page: 1, total: 0, pageSize: 8, category: '', eventType: '', query: '', sessionId: '' },
        },
    },
};
let documentPollTimer = null;
let platformOpsPollTimer = null;
const REVIEWABLE_SAFETY_STATES = new Set(['needs_review', 'in_review', 'blocked_pending_superadmin']);

function resolveRouteContext() {
    const path = String(window.location.pathname || '').replace(/\/+$/, '') || '/';
    const tenantAdminMatch = path.match(/^\/u\/([^/]+)\/admin(?:\/index\.html)?$/i);
    if (tenantAdminMatch) {
        return { mode: 'tenant_admin', unionSlug: decodeURIComponent(tenantAdminMatch[1]) };
    }
    if (path === '/karl' || path === '/karl/index.html') {
        return { mode: 'superadmin', unionSlug: null };
    }
    return { mode: 'legacy_admin', unionSlug: null };
}

function loadAuthContext() {
    try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
    } catch (_) {
        return {};
    }
}

function documentNeedsSafetyReview(item) {
    const safetyReviewStatus = String(item?.safety_review_status || '').trim().toLowerCase();
    return Boolean(
        item?.prompt_injection_risk
        || item?.sensitive_data_risk
        || REVIEWABLE_SAFETY_STATES.has(safetyReviewStatus)
    );
}

function saveAuthContext(context) {
    state.auth = context || {};
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.auth));
    renderAuthSummary();
    applyRoleVisibility();
    applyWorkspaceVisibility();
    populateRoleSelects();
}

async function parseJsonResponse(response) {
    const text = await response.text();
    try {
        return text ? JSON.parse(text) : {};
    } catch (_) {
        return { raw: text };
    }
}

function togglePasswordVisibility(inputId, buttonId) {
    const input = document.getElementById(inputId);
    const button = document.getElementById(buttonId);
    if (!input || !button) return;
    const showing = input.type === 'text';
    input.type = showing ? 'password' : 'text';
    button.textContent = showing ? 'Show' : 'Hide';
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function slugifyUnionValue(value) {
    return String(value || '')
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, '-')
        .replace(/^-+|-+$/g, '')
        .slice(0, 120);
}

function updateCreateUnionDerivedFields(force = false) {
    const nameInput = document.getElementById('union-name');
    const slugInput = document.getElementById('union-slug');
    const preview = document.getElementById('create-union-preview');
    if (!nameInput || !slugInput || !preview) return;

    const generated = slugifyUnionValue(nameInput.value);
    const slugTouched = slugInput.dataset.touched === 'true';

    if (force || !slugTouched) slugInput.value = generated;

    if (!generated) {
        preview.textContent = 'Enter a union name to preview the public union web address.';
        return;
    }
    preview.textContent = `Preview: public union web address "${slugInput.value}". Karl will generate the secure internal union ID automatically.`;
}

function notifyAdminSessionExpired(message = 'Your session expired. Please sign in again.') {
    clearAuthContext();
    setAuthModalOpen(true);
    window.alert(message);
}

function setAuthModalOpen(isOpen) {
    const modal = document.getElementById('auth-modal');
    if (!modal) return;
    modal.classList.toggle('hidden', !isOpen);
    document.body.classList.toggle('overflow-hidden', Boolean(isOpen));
    if (isOpen) {
        window.setTimeout(() => document.getElementById('auth-username')?.focus(), 0);
    }
}

function clearAuthContext() {
    state.selectedUnion = null;
    state.platformSummary = null;
    state.platformOps = null;
    state.dashboard = null;
    state.platformDashboard = null;
    state.userDirectory = {
        items: [],
        page: 1,
        pageSize: USER_PAGE_SIZE,
        total: 0,
        unionTotal: 0,
        query: '',
        field: 'all',
        sort: 'name',
        direction: 'asc',
        open: false,
        activeUserId: null,
    };
    state.feeds.selected.security.page = 1;
    state.feeds.selected.notifications.page = 1;
    state.feeds.selected.telemetry.page = 1;
    state.feeds.global.security.page = 1;
    state.feeds.global.notifications.page = 1;
    state.feeds.global.telemetry.page = 1;
    saveAuthContext({});
    setText('detail-title', 'Select a union');
    setText('detail-subtitle', 'This workspace is organized around the tasks a union admin handles most often: documents, members, union settings, quotas, and model setup.');
    renderList('union-list', [], () => '', 'Sign in to load unions.');
    renderList('document-list', [], () => '', 'Select a union first.');
    renderList('chat-list', [], () => '', 'Select a union first.');
    renderList('selected-chat-list', [], () => '', 'Select a union first.');
    renderList('security-list', [], () => '', 'Sign in to load security events.');
    renderList('selected-security-list', [], () => '', 'Select a union first.');
    renderList('global-security-list', [], () => '', 'Sign in to load security events.');
    renderList('notification-list', [], () => '', 'Sign in to load notifications.');
    renderList('selected-notification-list', [], () => '', 'Select a union first.');
    renderList('global-notification-list', [], () => '', 'Sign in to load notifications.');
    renderList('selected-telemetry-list', [], () => '', 'Select a union first.');
    renderList('global-telemetry-list', [], () => '', 'Sign in to load telemetry.');
    renderList('telemetry-list', [], () => '', 'Select a union first.');
    renderList('user-directory-list', [], () => '', 'Select a union first.');
    setText('user-summary', 'Select a union to load member access.');
    setText('user-pagination-summary', 'Select a union to load users.');
    setText('quota-usage-summary', 'Load a union to see current usage.');
    setProviderTestStatus(null);
    setProviderKeyStatus(false);
    setProviderHelp();
    renderPlatformSummary(null);
    renderPlatformOps(null);
    renderDashboard(null, 'dashboard');
    renderDashboard(null, 'superadmin-dashboard');
    renderSelectedUnionAdmins(null);
    renderReviewQueue(null);
    document.getElementById('chat-detail')?.classList.add('hidden');
    document.getElementById('user-directory-panel')?.classList.add('hidden');
    const openUserDirectoryButton = document.getElementById('open-user-directory');
    if (openUserDirectoryButton) openUserDirectoryButton.textContent = 'Open User Directory';
    resetUserEditor();
    setJSON('me-output', {});
    setJSON('superadmin-debug-output', {});
    setText('selected-security-summary', 'No security events loaded.');
    setText('selected-notification-summary', 'No notifications loaded.');
    setText('global-security-summary', 'No security events loaded.');
    setText('global-notification-summary', 'No notifications loaded.');
    setText('selected-telemetry-summary', 'No telemetry loaded.');
    setText('global-telemetry-summary', 'No telemetry loaded.');
    setText('telemetry-summary', 'No telemetry loaded.');
    setAuthModalOpen(false);
}

function applyWorkspaceVisibility() {
    const workspace = document.getElementById('workspace-shell');
    const signedOut = document.getElementById('signed-out-state');
    const authenticated = Boolean(state.auth.authenticated);
    workspace?.classList.toggle('hidden', !authenticated);
    signedOut?.classList.toggle('hidden', authenticated);
    applySelectedUnionVisibility();
}

function applySelectedUnionVisibility() {
    const hasSelectedUnion = Boolean(state.selectedUnion?.id);
    const showUnionTools = routeContext.mode !== 'superadmin' || hasSelectedUnion;
    document.querySelectorAll('.selected-union-only').forEach((node) => {
        node.classList.toggle('hidden', !showUnionTools);
    });
    const statusNode = document.getElementById('selected-union-management-status');
    if (statusNode) {
        statusNode.textContent = hasSelectedUnion
            ? `Managing ${state.selectedUnion.name || state.selectedUnion.slug || 'selected union'}. Union-specific tools are now visible below.`
            : 'Select Manage on a union above to open its union-specific tools.';
    }
    const manageButton = document.getElementById('manage-selected-union');
    if (manageButton && routeContext.mode === 'superadmin') {
        manageButton.textContent = hasSelectedUnion ? 'Managing Selected Union' : 'Manage Selected Union';
    }
    refreshSelectedUnionActions();
}

async function logout() {
    try {
        await fetch('/api/auth/session/logout', {
            method: 'POST',
            headers: authHeaders(),
            credentials: 'same-origin',
        });
    } catch (_) {
        // Best-effort logout.
    }
    clearAuthContext();
}

async function hydrateAuthContext() {
    try {
        const data = await api('/api/auth/session/me');
        if (!data.authenticated) {
            clearAuthContext();
            return false;
        }
        saveAuthContext({
            authenticated: true,
            username: state.auth.username || '',
            union: data.union_slug || routeContext.unionSlug || '',
            user: {
                id: data.user_id,
                email: data.email,
                full_name: data.full_name,
                role: data.role,
                union_id: data.union_id,
                union_slug: data.union_slug,
            },
        });
        if (data.union_id && (routeContext.mode === 'tenant_admin' || routeContext.mode === 'superadmin')) {
            state.selectedUnion = {
                id: data.union_id,
                name: data.union_name || data.union_slug || data.union_id,
                slug: data.union_slug || routeContext.unionSlug || '',
            };
        }
        return true;
    } catch (error) {
        if (String(error?.message || '').includes('Session expired')) {
            return false;
        }
        clearAuthContext();
        return false;
    }
}

function authHeaders() {
    const headers = {};
    if (routeContext.unionSlug) {
        headers['X-Tenant-Slug'] = routeContext.unionSlug;
    }
    return headers;
}

async function api(path, options = {}) {
    const headers = { ...(options.headers || {}), ...authHeaders() };
    const response = await fetch(path, { ...options, headers, credentials: 'same-origin' });
    const text = await response.text();
    let data = {};
    try {
        data = text ? JSON.parse(text) : {};
    } catch (_) {
        data = { raw: text };
    }
    if (!response.ok) {
        if (response.status === 401 && state.auth.authenticated) {
            notifyAdminSessionExpired();
            throw new Error('Session expired. Please sign in again.');
        }
        throw new Error(data.detail || data.raw || JSON.stringify(data));
    }
    return data;
}

async function postTelemetryEvent(category, eventType, metadata = {}, options = {}) {
    const payload = {
        category,
        event_type: eventType,
        route: window.location.pathname,
        metadata,
        session_id: state.auth?.user?.session_id || null,
        union_slug: routeContext.unionSlug || state.selectedUnion?.slug || null,
        surface: options.surface || (state.auth?.user?.role === 'super_admin' ? 'superadmin' : 'admin'),
    };
    try {
        await fetch('/api/telemetry/event', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', ...authHeaders() },
            credentials: 'same-origin',
            keepalive: true,
            body: JSON.stringify(payload),
        });
    } catch (_) {
        // Telemetry should never block the admin UI.
    }
}

function setText(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = value;
}

function setJSON(id, value) {
    const node = document.getElementById(id);
    if (node) node.textContent = JSON.stringify(value, null, 2);
}

function setProviderTestStatus(payload, tone = 'neutral') {
    const node = document.getElementById('provider-test-status');
    if (!node) return;
    if (!payload) {
        node.classList.add('hidden');
        node.textContent = '';
        node.className = 'mt-3 hidden rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700';
        return;
    }
    const tones = {
        success: 'mt-3 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900',
        error: 'mt-3 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900',
        neutral: 'mt-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700',
    };
    node.className = tones[tone] || tones.neutral;
    node.classList.remove('hidden');
    node.textContent = payload;
}

function setProviderKeyStatus(hasSavedKey) {
    const node = document.getElementById('provider-key-status');
    if (!node) return;
    node.textContent = hasSavedKey
        ? 'Saved API key: ********. Leave the field blank if you want to keep the current key.'
        : 'No saved API key yet.';
}

function setProviderHelp(providerName = document.getElementById('provider-name')?.value || 'openrouter') {
    const node = document.getElementById('provider-help');
    if (!node) return;
    const profile = PROVIDER_PROFILES[providerName] || PROVIDER_PROFILES.openrouter;
    node.textContent = profile.help;
}

function applyTrackingPolicyToForm(prefix, policy, { overrideEnabled = true, disableOverrideToggle = false } = {}) {
    const normalized = policy || {};
    const setValue = (id, value) => {
        const node = document.getElementById(id);
        if (node) node.value = value || '';
    };
    setValue(`${prefix}-tracking-mode`, normalized.tracking_mode || 'bug_and_journey');
    setValue(`${prefix}-tracking-privacy`, normalized.privacy_mode || 'anonymized');
    setValue(`${prefix}-tracking-member-choice`, normalized.member_choice_mode || 'bug_only_or_full');
    setValue(`${prefix}-tracking-raw-query`, normalized.raw_query_storage_mode || 'disabled');
    setValue(`${prefix}-tracking-default-member`, normalized.default_member_preference || 'bug_only');
    const overrideNode = document.getElementById(`${prefix}-tracking-override-enabled`);
    if (overrideNode) {
        overrideNode.checked = Boolean(overrideEnabled);
        overrideNode.disabled = Boolean(disableOverrideToggle);
    }
}

function trackingPolicyPayloadFromForm(prefix) {
    return {
        tracking_mode: document.getElementById(`${prefix}-tracking-mode`)?.value || 'bug_and_journey',
        privacy_mode: document.getElementById(`${prefix}-tracking-privacy`)?.value || 'anonymized',
        member_choice_mode: document.getElementById(`${prefix}-tracking-member-choice`)?.value || 'bug_only_or_full',
        raw_query_storage_mode: document.getElementById(`${prefix}-tracking-raw-query`)?.value || 'disabled',
        default_member_preference: document.getElementById(`${prefix}-tracking-default-member`)?.value || 'bug_only',
    };
}

function setTrackingPolicyStatus(id, message, tone = 'neutral') {
    const node = document.getElementById(id);
    if (!node) return;
    node.textContent = message;
    node.className = {
        success: 'rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900',
        error: 'rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900',
        neutral: 'rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600',
    }[tone] || 'rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600';
}

function setPasswordResetModalOpen(isOpen) {
    const modal = document.getElementById('password-reset-modal');
    if (!modal) return;
    modal.classList.toggle('hidden', !isOpen);
    document.body.classList.toggle('overflow-hidden', Boolean(isOpen));
    if (isOpen) {
        window.setTimeout(() => document.getElementById('password-reset-new')?.focus(), 0);
    }
}

function setCreateUnionModalOpen(isOpen) {
    const modal = document.getElementById('create-union-modal');
    if (!modal) return;
    modal.classList.toggle('hidden', !isOpen);
    document.body.classList.toggle('overflow-hidden', Boolean(isOpen));
    if (isOpen) {
        window.setTimeout(() => document.getElementById('union-name')?.focus(), 0);
    }
}

function setDocumentReviewModalOpen(isOpen) {
    const modal = document.getElementById('document-review-modal');
    if (!modal) return;
    modal.classList.toggle('hidden', !isOpen);
    document.body.classList.toggle('overflow-hidden', Boolean(isOpen));
    if (!isOpen) {
        state.reviewDocument = null;
    }
}

function setPasswordResetStatus(message, tone = 'neutral') {
    const node = document.getElementById('password-reset-status');
    if (!node) return;
    node.textContent = message;
    node.className = {
        success: 'rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900',
        error: 'rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900',
        neutral: 'rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600',
    }[tone] || 'rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600';
}

function roleLimit() {
    const role = String(state.auth.user?.role || 'user').toLowerCase();
    const idx = ROLE_ORDER.indexOf(role);
    return idx >= 0 ? idx : 0;
}

function allowedRoleOptions() {
    const maxIndex = routeContext.mode === 'superadmin' && state.auth.user?.role === 'super_admin'
        ? ROLE_ORDER.length - 1
        : Math.min(roleLimit(), ROLE_ORDER.indexOf('union_admin'));
    return ROLE_ORDER.slice(0, maxIndex + 1);
}

function labelForRole(role) {
    return String(role || '').replace(/_/g, ' ');
}

function populateRoleSelect(selectId, selectedValue = null) {
    const node = document.getElementById(selectId);
    if (!node) return;
    const options = allowedRoleOptions();
    node.innerHTML = options.map((value) => `
        <option value="${value}">${labelForRole(value)}</option>
    `).join('');
    if (selectedValue && options.includes(selectedValue)) {
        node.value = selectedValue;
    } else if (options.length) {
        node.value = options[0];
    }
}

function populateRoleSelects() {
    populateRoleSelect('user-role', 'user');
    populateRoleSelect('user-edit-role', 'user');
}

function renderAuthSummary() {
    const user = state.auth.user || {};
    const signInButton = document.getElementById('auth-open-modal');
    const signOutButton = document.getElementById('auth-logout');
    const authUnion = document.getElementById('auth-union');
    const authUnionGroup = document.getElementById('auth-union-group');
    setText(
        'auth-summary',
        state.auth.authenticated
            ? `Signed in as ${user.full_name || user.email || user.username || 'user'} • ${labelForRole(user.role || 'unknown')}`
            : 'Not signed in.',
    );
    const usernameField = document.getElementById('auth-username');
    const passwordField = document.getElementById('auth-password');
    if (usernameField) usernameField.value = state.auth.username || '';
    if (passwordField) passwordField.value = '';
    if (authUnion) {
        authUnion.value = routeContext.unionSlug || state.auth.union || '';
        authUnion.readOnly = Boolean(routeContext.unionSlug);
    }
    authUnionGroup?.classList.toggle('hidden', Boolean(routeContext.unionSlug));
    signInButton?.classList.toggle('hidden', Boolean(state.auth.authenticated));
    signOutButton?.classList.toggle('hidden', !state.auth.authenticated);
}

function applyRoleVisibility() {
    const role = state.auth.user?.role || '';
    const isTenantAdminRoute = routeContext.mode === 'tenant_admin';
    const createUnionForm = document.getElementById('create-union-form');
    const unionDirectoryPanel = document.getElementById('union-directory-panel');
    const loadUnionsButton = document.getElementById('load-unions');
    const loadOpsButton = document.getElementById('load-ops');

    createUnionForm?.classList.toggle('hidden', role !== 'super_admin');
    if (unionDirectoryPanel) {
        unionDirectoryPanel.classList.toggle('hidden', isTenantAdminRoute && role !== 'super_admin');
    }
    loadUnionsButton?.classList.toggle('hidden', isTenantAdminRoute && role !== 'super_admin');
    loadOpsButton?.classList.toggle('hidden', isTenantAdminRoute && role !== 'super_admin');
}

function unionCard(union) {
    const summary = Array.isArray(state.platformSummary?.union_summaries)
        ? (state.platformSummary.union_summaries.find((item) => item.id === union.id) || null)
        : null;
    const metadata = union.metadata || {};
    const authPolicy = metadata.auth_policy || {};
    const isActive = summary?.is_active ?? union.is_active !== false;
    const loginRequired = summary?.member_login_required ?? authPolicy.member_login_required !== false;
    const buttonLabel = state.selectedUnion?.id === union.id ? 'Managing' : 'Manage';
    return `
        <article class="card-hover rounded-[24px] border border-slate-200 bg-white/75 p-4 transition">
            <div class="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <div class="text-lg font-semibold text-slate-900">${union.name}</div>
                    <div class="mono mt-1 text-xs text-slate-500">public address=${union.slug}</div>
                </div>
                <div class="flex flex-wrap gap-2">
                    ${routeContext.mode === 'superadmin' && union.slug ? `<a href="/u/${encodeURIComponent(union.slug)}/admin" target="_blank" rel="noopener" class="rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-slate-100">Open admin console</a>` : ''}
                    <button class="select-union rounded-full px-4 py-2 text-sm font-semibold ${state.selectedUnion?.id === union.id ? 'bg-slate-900 text-white' : 'border border-slate-300 bg-white text-slate-900'}" data-union-id="${union.id}" data-union-name="${union.name}" data-union-slug="${union.slug}">
                        ${buttonLabel}
                    </button>
                </div>
            </div>
            <div class="mt-3 flex flex-wrap gap-2 text-xs text-slate-600">
                <span class="rounded-full bg-slate-100 px-3 py-1">${isActive ? 'active' : 'inactive'}</span>
                <span class="rounded-full bg-slate-100 px-3 py-1">login ${loginRequired ? 'required' : 'optional'}</span>
                <span class="rounded-full bg-slate-100 px-3 py-1">retention ${union.message_retention_enabled ? 'on' : 'off'}</span>
                ${summary ? `<span class="rounded-full bg-slate-100 px-3 py-1">users ${summary.user_count}</span>` : ''}
                ${summary ? `<span class="rounded-full bg-slate-100 px-3 py-1">docs ${summary.document_count}</span>` : ''}
                ${summary && summary.pending_review_count ? `<span class="rounded-full bg-amber-100 px-3 py-1 text-amber-900">reviews ${summary.pending_review_count}</span>` : ''}
            </div>
        </article>
    `;
}

function populateUnionPicker(items = []) {
    const node = document.getElementById('union-picker');
    if (!node) return;
    const selectedValue = state.selectedUnion?.id || '';
    node.innerHTML = `
        <option value="">Choose a union to manage</option>
        ${items.map((item) => `<option value="${item.id}">${item.name}</option>`).join('')}
    `;
    node.value = selectedValue;
}

function refreshSelectedUnionActions() {
    const toggleButton = document.getElementById('selected-union-toggle-active');
    const deleteButton = document.getElementById('selected-union-delete');
    if (!toggleButton && !deleteButton) return;
    const isActive = state.selectedUnion?.is_active !== false;
    if (toggleButton) {
        toggleButton.textContent = isActive ? 'Take This Union Offline' : 'Reactivate This Union';
        toggleButton.dataset.nextActive = isActive ? 'false' : 'true';
        toggleButton.disabled = !state.selectedUnion?.id;
    }
    if (deleteButton) {
        deleteButton.disabled = !state.selectedUnion?.id;
    }
}

function renderPlatformSummary(summary) {
    const node = document.getElementById('platform-summary');
    if (!node) return;
    if (!summary) {
        node.innerHTML = `<div class="rounded-2xl border border-dashed border-slate-300 px-4 py-5 text-sm text-slate-500">Refresh the platform view to load totals.</div>`;
        return;
    }
    const totals = summary.totals || {};
    node.innerHTML = `
        <div class="rounded-2xl border border-slate-200 bg-white/90 px-4 py-4 text-base font-medium text-slate-800 shadow-sm">
            Unions - Total: ${totals.unions || 0} Active: ${totals.active_unions || 0} Inactive: ${totals.inactive_unions || 0} Total Users: ${totals.users || 0}
        </div>
        <div class="mt-3 flex flex-wrap gap-3">
            <button class="summary-nav-button rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-900 transition hover:border-slate-400 hover:shadow-sm" data-target="union-directory-panel" type="button">Unions ${totals.unions || 0}</button>
            <button class="summary-nav-button rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-900 transition hover:border-slate-400 hover:shadow-sm" data-target="review-queue-section" type="button">Reviews ${totals.pending_reviews || 0}</button>
            <button class="summary-nav-button rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-900 transition hover:border-slate-400 hover:shadow-sm" data-target="global-alerts-section" type="button">Alerts ${totals.pending_notifications || 0}</button>
            <button class="summary-nav-button rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-900 transition hover:border-slate-400 hover:shadow-sm" data-target="global-telemetry-panel" type="button">Observability ${(summary.dashboard?.summary?.journey_events_7d || 0) + (summary.dashboard?.summary?.usage_events_7d || 0)}</button>
        </div>
    `;
    node.querySelectorAll('.summary-nav-button').forEach((button) => {
        button.addEventListener('click', () => scrollToSection(button.dataset.target));
    });
}

async function loadGlobalTrackingPolicy() {
    if (state.auth.user?.role !== 'super_admin') return;
    const data = await api('/api/admin/tracking-policy/global');
    state.globalTrackingPolicy = data.policy || null;
    applyTrackingPolicyToForm('global', state.globalTrackingPolicy, { disableOverrideToggle: true });
    const allowOverridesNode = document.getElementById('global-tracking-allow-overrides');
    if (allowOverridesNode) {
        allowOverridesNode.checked = Boolean(state.globalTrackingPolicy?.allow_union_override);
    }
    setTrackingPolicyStatus(
        'global-tracking-policy-status',
        `Global default: ${String(state.globalTrackingPolicy?.tracking_mode || 'bug_and_journey').replace(/_/g, ' ')} with ${state.globalTrackingPolicy?.privacy_mode || 'anonymized'} privacy and raw query storage ${String(state.globalTrackingPolicy?.raw_query_storage_mode || 'disabled').replace(/_/g, ' ')}.`,
        'neutral',
    );
}

async function saveGlobalTrackingPolicy(event) {
    event.preventDefault();
    if (state.auth.user?.role !== 'super_admin') throw new Error('Only superadmins can change the global tracking policy.');
    const payload = {
        ...trackingPolicyPayloadFromForm('global'),
        allow_union_override: document.getElementById('global-tracking-allow-overrides')?.checked !== false,
    };
    const data = await api('/api/admin/tracking-policy/global', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    state.globalTrackingPolicy = data.policy || null;
    setTrackingPolicyStatus('global-tracking-policy-status', 'Global tracking policy saved.', 'success');
    await loadPlatformSummary();
}

async function loadUnionTrackingPolicy() {
    if (state.auth.user?.role !== 'super_admin' || !state.selectedUnion?.id) return;
    const data = await api(`/api/admin/unions/${state.selectedUnion.id}/tracking-policy`);
    state.unionTrackingPolicy = data;
    applyTrackingPolicyToForm(
        'union',
        data.policy || data.effective_policy,
        { overrideEnabled: Boolean(data.override_enabled) },
    );
    const globalAllowsOverride = data.global_allow_union_override !== false;
    const clearButton = document.getElementById('clear-union-tracking-policy');
    if (clearButton) clearButton.disabled = !data.override_enabled;
    const overrideToggle = document.getElementById('union-tracking-override-enabled');
    if (overrideToggle) overrideToggle.disabled = !globalAllowsOverride;
    setUnionTrackingFormEnabled(Boolean(data.override_enabled) && globalAllowsOverride);
    setTrackingPolicyStatus(
        'union-tracking-policy-status',
        !globalAllowsOverride
            ? 'Global policy currently disables union-specific tracking overrides.'
            : data.override_enabled
            ? `This union override is active. Effective mode: ${String(data.effective_policy?.tracking_mode || '').replace(/_/g, ' ')}.`
            : 'This union currently follows the platform default until an override is enabled.',
        'neutral',
    );
}

function setUnionTrackingFormEnabled(enabled) {
    ['union-tracking-mode', 'union-tracking-privacy', 'union-tracking-member-choice', 'union-tracking-raw-query', 'union-tracking-default-member']
        .forEach((id) => {
            const node = document.getElementById(id);
            if (node) node.disabled = !enabled;
        });
    const saveButton = document.querySelector('#union-tracking-policy-form button[type="submit"]');
    if (saveButton) saveButton.disabled = !enabled;
}

async function saveUnionTrackingPolicy(event) {
    event.preventDefault();
    if (state.auth.user?.role !== 'super_admin') throw new Error('Only superadmins can change union tracking overrides.');
    const unionId = requireUnion();
    const overrideEnabled = Boolean(document.getElementById('union-tracking-override-enabled')?.checked);
    if (!overrideEnabled) {
        await clearUnionTrackingPolicy();
        return;
    }
    const data = await api(`/api/admin/unions/${unionId}/tracking-policy`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trackingPolicyPayloadFromForm('union')),
    });
    state.unionTrackingPolicy = data;
    setTrackingPolicyStatus('union-tracking-policy-status', 'Union tracking override saved.', 'success');
}

async function clearUnionTrackingPolicy() {
    if (state.auth.user?.role !== 'super_admin') throw new Error('Only superadmins can clear union tracking overrides.');
    const unionId = requireUnion();
    const data = await api(`/api/admin/unions/${unionId}/tracking-policy`, { method: 'DELETE' });
    state.unionTrackingPolicy = data;
    applyTrackingPolicyToForm('union', data.effective_policy, { overrideEnabled: false });
    const toggle = document.getElementById('union-tracking-override-enabled');
    if (toggle) toggle.checked = false;
    setUnionTrackingFormEnabled(false);
    setTrackingPolicyStatus('union-tracking-policy-status', 'Union override cleared. This union now follows the global default.', 'success');
}

function renderPlatformOps(payload) {
    const summaryNode = document.getElementById('superadmin-ops-summary');
    const usageNode = document.getElementById('superadmin-usage-list');
    const providerNode = document.getElementById('superadmin-provider-list');
    const adminNode = document.getElementById('superadmin-admin-list');
    if (!summaryNode || !usageNode || !providerNode || !adminNode) return;
    if (!payload) {
        summaryNode.textContent = 'Platform-wide health updates appear here.';
        renderList('superadmin-usage-list', [], () => '', 'No usage warnings.');
        renderList('superadmin-provider-list', [], () => '', 'No provider issues.');
        renderList('superadmin-admin-list', [], () => '', 'No admin coverage issues.');
        return;
    }
    const summary = payload.summary || {};
    summaryNode.textContent = `${summary.warning_unions || 0} unions near quota limits, ${summary.provider_issues || 0} with provider issues, ${summary.unions_without_admins || 0} without admin coverage. Updated ${new Date(payload.captured_at).toLocaleTimeString()}.`;
    setText('platform-tools-status', `Platform view refreshed at ${new Date(payload.captured_at).toLocaleTimeString()}. Use Manage, Take Offline, Reactivate, or Delete from the union cards above.`);
    const items = Array.isArray(payload.items) ? payload.items : [];
    const usageItems = items.filter((item) => item.usage?.warning_level && item.usage.warning_level !== 'healthy');
    const providerItems = items.filter((item) => item.provider_health?.status !== 'configured');
    const adminItems = items.filter((item) => !item.admins?.length);
    renderList('superadmin-usage-list', usageItems, (item) => `
        <div class="rounded-2xl border border-slate-200 bg-white p-3">
            <div class="flex items-center justify-between gap-3">
                <div class="font-semibold text-slate-900">${item.name}</div>
                <div class="rounded-full ${item.usage.warning_level === 'limit_reached' ? 'bg-rose-100 text-rose-900' : item.usage.warning_level === 'paused' ? 'bg-slate-200 text-slate-800' : 'bg-amber-100 text-amber-900'} px-3 py-1 text-xs font-semibold">${item.usage.warning_level.replace(/_/g, ' ')}</div>
            </div>
            <div class="mt-2 text-xs text-slate-600">24h requests ${item.usage.requests_last_24h}, tokens ${item.usage.tokens_last_24h}, est. cost $${Number(item.usage.estimated_cost_last_24h || 0).toFixed(2)}</div>
        </div>
    `, 'No unions are currently near their quota limits.');
    renderList('superadmin-provider-list', providerItems, (item) => `
        <button class="provider-issue-card w-full rounded-2xl border border-slate-200 bg-white p-3 text-left hover:border-slate-400" data-union-id="${item.union_id}" data-target="provider-section" type="button">
            <div class="flex items-center justify-between gap-3">
                <div class="font-semibold text-slate-900">${item.name}</div>
                <div class="rounded-full bg-rose-100 px-3 py-1 text-xs font-semibold text-rose-900">${item.provider_health.status === 'missing' ? 'setup needed' : item.provider_health.status}</div>
            </div>
            <div class="mt-2 text-xs text-slate-600">${item.provider_health.provider_name || 'No provider configured yet'} ${item.provider_health.model_name || ''}</div>
            <div class="mt-2 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">${item.provider_health.status === 'missing' ? 'Open this union to finish provider setup' : 'Open this union to review provider health'}</div>
        </button>
    `, 'All unions currently have provider configuration saved.');
    renderList('superadmin-admin-list', adminItems, (item) => `
        <div class="rounded-2xl border border-slate-200 bg-white p-3">
            <div class="font-semibold text-slate-900">${item.name}</div>
            <div class="mt-2 text-xs text-slate-600">No current union admins assigned.</div>
        </div>
    `, 'Every union currently has at least one admin assigned.');
    providerNode.querySelectorAll('.provider-issue-card').forEach((button) => {
        button.addEventListener('click', () => run(async () => {
            window.alert('This union still needs model provider setup. Karl will open that union and take you to the Model Provider section so you can save the provider and API key.');
            await openUnionWorkspace(button.dataset.unionId, button.dataset.target);
        }));
    });
}

function scrollToSection(id) {
    if (!id) return;
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderPaginatedFeed({
    listId,
    summaryId,
    prevId,
    nextId,
    items,
    formatter,
    empty,
    page,
    pageSize,
    total,
    singularLabel,
    pluralLabel,
}) {
    renderList(listId, items, formatter, empty);
    const totalPages = Math.max(1, Math.ceil((total || 0) / Math.max(pageSize || 1, 1)));
    const currentPage = Math.min(Math.max(page || 1, 1), totalPages);
    const start = total ? ((currentPage - 1) * pageSize) + 1 : 0;
    const end = total ? Math.min(total, currentPage * pageSize) : 0;
    setText(summaryId, total ? `${start}-${end} of ${total} ${total === 1 ? singularLabel : pluralLabel}` : `No ${pluralLabel} loaded.`);
    const prev = document.getElementById(prevId);
    const next = document.getElementById(nextId);
    if (prev) prev.disabled = currentPage <= 1;
    if (next) next.disabled = currentPage >= totalPages;
}

function notificationFeedFormatter(item, scope = 'global') {
    const canDismiss = item.status !== 'acknowledged';
    return `
        <div class="min-w-0 rounded-2xl border border-slate-200 bg-white p-3">
            <div class="flex min-w-0 items-start justify-between gap-3">
                <div class="min-w-0 flex-1">
                    <div class="break-words font-semibold text-slate-900">${escapeHtml(item.subject)}</div>
                    <div class="mt-1 break-words text-sm leading-6 text-slate-600">${escapeHtml(item.body)}</div>
                </div>
                <div class="flex shrink-0 flex-col items-end gap-2">
                    <div class="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold">${escapeHtml(item.status)}</div>
                    ${canDismiss ? `<button class="notification-dismiss rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-900 hover:bg-slate-100" data-notification-id="${escapeHtml(item.id)}" data-scope="${escapeHtml(scope)}" type="button">Dismiss</button>` : ''}
                </div>
            </div>
            <div class="mono mt-2 break-all text-xs text-slate-500">${new Date(item.created_at).toLocaleString()}</div>
        </div>
    `;
}

function summarizeTelemetryMetadata(metadata = {}) {
    const entries = Object.entries(metadata || {})
        .filter(([, value]) => value !== null && value !== undefined && value !== '')
        .slice(0, 4)
        .map(([key, value]) => `${key.replace(/_/g, ' ')}: ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`);
    return entries.join(' • ');
}

function telemetryFeedFormatter(item) {
    const badgeClass = item.category === 'usage_ux'
        ? 'bg-emerald-100 text-emerald-900'
        : 'bg-slate-100 text-slate-900';
    const metaSummary = summarizeTelemetryMetadata(item.metadata || {});
    const identityLine = item.union_name
        ? [item.union_name, item.session_id || item.anonymized_user_key || item.user_id || ''].filter(Boolean).join(' • ')
        : [item.session_id || item.anonymized_user_key || item.user_id || '', item.route || ''].filter(Boolean).join(' • ');
    return `
        <div class="min-w-0 rounded-2xl border border-slate-200 bg-white p-3">
            <div class="flex min-w-0 items-start justify-between gap-3">
                <div class="min-w-0 flex-1">
                    <div class="break-words font-semibold text-slate-900">${escapeHtml(item.event_type)}</div>
                    <div class="mt-1 break-all text-xs text-slate-500">${escapeHtml(identityLine || 'Telemetry event')}</div>
                </div>
                <div class="flex shrink-0 flex-col items-end gap-2">
                    <div class="rounded-full px-3 py-1 text-xs font-semibold ${badgeClass}">${item.category === 'usage_ux' ? 'usage/ux' : 'bug/journey'}</div>
                    ${item.session_id ? `<button class="open-session-timeline rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-900 hover:bg-slate-100" data-session-id="${escapeHtml(item.session_id)}" data-union-id="${escapeHtml(item.union_id || '')}" type="button">View Session</button>` : ''}
                </div>
            </div>
            ${metaSummary ? `<div class="mt-2 break-words text-sm text-slate-700">${escapeHtml(metaSummary)}</div>` : ''}
            <div class="mono mt-2 break-all text-xs text-slate-500">${new Date(item.created_at).toLocaleString()}</div>
        </div>
    `;
}

function renderSessionTimelineModal(payload) {
    const summaryNode = document.getElementById('session-timeline-summary');
    const listNode = document.getElementById('session-timeline-list');
    const titleNode = document.getElementById('session-timeline-title');
    const subtitleNode = document.getElementById('session-timeline-subtitle');
    if (!summaryNode || !listNode || !titleNode || !subtitleNode) return;
    if (!payload) {
        titleNode.textContent = 'Session Timeline';
        subtitleNode.textContent = 'Load a session from the observability feed.';
        summaryNode.textContent = 'No session loaded.';
        listNode.innerHTML = '';
        return;
    }
    const summary = payload.summary || {};
    titleNode.textContent = `Session ${payload.session_id}`;
    subtitleNode.textContent = payload.union_name
        ? `${payload.union_name} session journey`
        : 'Session journey';
    summaryNode.textContent = `${summary.total_events || 0} events${summary.started_at ? ` • started ${new Date(summary.started_at).toLocaleString()}` : ''}${summary.ended_at ? ` • ended ${new Date(summary.ended_at).toLocaleString()}` : ''}`;
    renderList('session-timeline-list', payload.items || [], (item) => `
        <div class="rounded-2xl border border-slate-200 bg-white p-4">
            <div class="flex min-w-0 items-start justify-between gap-3">
                <div class="min-w-0 flex-1">
                    <div class="break-words font-semibold text-slate-900">${escapeHtml(item.event_type)}</div>
                    <div class="mt-1 break-all text-xs text-slate-500">${escapeHtml(item.route || 'No route recorded')}</div>
                </div>
                <div class="rounded-full px-3 py-1 text-xs font-semibold ${item.category === 'usage_ux' ? 'bg-emerald-100 text-emerald-900' : 'bg-slate-100 text-slate-900'}">${item.category === 'usage_ux' ? 'usage/ux' : 'bug/journey'}</div>
            </div>
            ${summarizeTelemetryMetadata(item.metadata || {}) ? `<div class="mt-2 break-words text-sm text-slate-700">${escapeHtml(summarizeTelemetryMetadata(item.metadata || {}))}</div>` : ''}
            <div class="mono mt-2 break-all text-xs text-slate-500">${new Date(item.created_at).toLocaleString()}</div>
        </div>
    `, 'No telemetry events recorded for this session.');
}

function setSessionTimelineModalOpen(isOpen) {
    const modal = document.getElementById('session-timeline-modal');
    if (!modal) return;
    modal.classList.toggle('hidden', !isOpen);
    document.body.classList.toggle('overflow-hidden', Boolean(isOpen));
}

async function openSessionTimeline(sessionId, unionId = '') {
    if (!sessionId) return;
    const params = new URLSearchParams();
    if (unionId) params.set('union_id', unionId);
    const query = params.toString();
    const payload = await api(`/api/ops/telemetry-events/session/${encodeURIComponent(sessionId)}${query ? `?${query}` : ''}`);
    state.sessionTimeline = payload;
    renderSessionTimelineModal(payload);
    setSessionTimelineModalOpen(true);
}

function sparklineSvg(values, color = '#0d5c80') {
    const numericValues = Array.isArray(values) ? values.map((value) => Number(value || 0)) : [];
    if (!numericValues.length) {
        return `<svg viewBox="0 0 160 44" class="h-12 w-full"><path d="M0 22 L160 22" fill="none" stroke="rgba(23, 50, 70, 0.18)" stroke-width="2" stroke-dasharray="4 4"></path></svg>`;
    }
    const max = Math.max(...numericValues, 1);
    const min = Math.min(...numericValues, 0);
    const spread = Math.max(max - min, 1);
    const points = numericValues.map((value, index) => {
        const x = (index / Math.max(numericValues.length - 1, 1)) * 160;
        const y = 38 - (((value - min) / spread) * 30);
        return `${x.toFixed(2)},${y.toFixed(2)}`;
    }).join(' ');
    return `<svg viewBox="0 0 160 44" class="h-12 w-full" preserveAspectRatio="none">
        <path d="M0 38 L160 38" fill="none" stroke="rgba(23, 50, 70, 0.08)" stroke-width="1"></path>
        <polyline fill="none" stroke="${color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="${points}"></polyline>
    </svg>`;
}

function dashboardCard(label, value, detail, trendValues, color = '#0d5c80', options = {}) {
    const tone = options.tone || 'default';
    const eyebrow = options.eyebrow || 'Snapshot';
    const accentClass = {
        default: 'from-white via-white to-slate-50',
        primary: 'from-sky-50 via-white to-cyan-50',
        caution: 'from-amber-50 via-white to-orange-50',
        danger: 'from-rose-50 via-white to-red-50',
        success: 'from-emerald-50 via-white to-teal-50',
    }[tone] || 'from-white via-white to-slate-50';
    const badgeClass = {
        default: 'bg-slate-100 text-slate-700',
        primary: 'bg-sky-100 text-sky-800',
        caution: 'bg-amber-100 text-amber-900',
        danger: 'bg-rose-100 text-rose-900',
        success: 'bg-emerald-100 text-emerald-900',
    }[tone] || 'bg-slate-100 text-slate-700';
    return `
        <div class="rounded-[28px] border border-slate-300 bg-gradient-to-br ${accentClass} p-4 shadow-sm shadow-slate-300/70">
            <div class="flex items-start justify-between gap-3">
                <div class="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">${eyebrow}</div>
                <div class="rounded-full px-3 py-1 text-[11px] font-semibold ${badgeClass}">${label}</div>
            </div>
            <div class="mt-4 text-3xl font-bold tracking-tight text-slate-950">${value}</div>
            <div class="mt-2 text-sm leading-6 text-slate-600">${detail}</div>
            <div class="mt-4 rounded-2xl border border-white/70 bg-white/70 px-3 py-2">${sparklineSvg(trendValues, color)}</div>
        </div>
    `;
}

function dashboardHeroCard(summary, payload, trends) {
    const issueTotal = Number(summary.pending_alerts || 0) + Number(summary.query_failures_7d || 0) + Number(summary.open_review_items || 0);
    const healthLabel = issueTotal === 0
        ? 'Calm'
        : issueTotal <= 4
            ? 'Watch'
            : 'Active';
    const healthClass = issueTotal === 0
        ? 'bg-emerald-100 text-emerald-900'
        : issueTotal <= 4
            ? 'bg-amber-100 text-amber-900'
            : 'bg-rose-100 text-rose-900';
    return `
        <div class="rounded-[32px] border border-slate-200 bg-[radial-gradient(circle_at_top_left,_rgba(14,116,144,0.16),_transparent_42%),linear-gradient(135deg,#0f172a_0%,#0f3b52_42%,#155e75_100%)] p-5 text-white shadow-lg shadow-slate-300/40 md:col-span-2 xl:col-span-2">
            <div class="flex flex-wrap items-start justify-between gap-4">
                <div>
                    <div class="text-[11px] font-semibold uppercase tracking-[0.22em] text-white/70">Operations Pulse</div>
                    <div class="mt-2 text-3xl font-bold tracking-tight">${payload.scope_label}</div>
                    <div class="mt-2 max-w-2xl text-sm leading-6 text-white/80">Requests, member activity, review pressure, and app friction are all summarized here for a quick read during demos or live operations.</div>
                </div>
                <div class="rounded-full px-4 py-2 text-sm font-semibold ${healthClass}">${healthLabel}</div>
            </div>
            <div class="mt-5 grid gap-3 sm:grid-cols-3">
                <div class="rounded-2xl border border-white/15 bg-white/10 px-4 py-3">
                    <div class="text-[11px] uppercase tracking-[0.18em] text-white/65">Requests today</div>
                    <div class="mt-2 text-2xl font-bold">${summary.requests_last_24h || 0}</div>
                    <div class="mt-1 text-xs text-white/70">${summary.tokens_last_24h || 0} tokens</div>
                </div>
                <div class="rounded-2xl border border-white/15 bg-white/10 px-4 py-3">
                    <div class="text-[11px] uppercase tracking-[0.18em] text-white/65">People this week</div>
                    <div class="mt-2 text-2xl font-bold">${summary.active_users_7d || 0}</div>
                    <div class="mt-1 text-xs text-white/70">${summary.sign_ins_7d || 0} sign-ins • ${summary.member_workspace_loads_7d || 0} workspace loads</div>
                </div>
                <div class="rounded-2xl border border-white/15 bg-white/10 px-4 py-3">
                    <div class="text-[11px] uppercase tracking-[0.18em] text-white/65">Attention needed</div>
                    <div class="mt-2 text-2xl font-bold">${issueTotal}</div>
                    <div class="mt-1 text-xs text-white/70">${summary.pending_alerts || 0} alerts • ${summary.query_failures_7d || 0} failures • ${summary.open_review_items || 0} review items</div>
                </div>
            </div>
            <div class="mt-5 rounded-[24px] border border-white/15 bg-black/10 px-4 py-3">
                <div class="mb-2 flex items-center justify-between gap-3">
                    <div class="text-sm font-semibold text-white">Seven-day service rhythm</div>
                    <div class="text-xs text-white/70">${(trends.labels || []).join(' • ')}</div>
                </div>
                ${sparklineSvg(trends.requests, '#f8fafc')}
            </div>
        </div>
    `;
}

function renderDashboard(payload, prefix = 'dashboard') {
    const summaryNode = document.getElementById(`${prefix}-summary-cards`);
    const trendNode = document.getElementById(`${prefix}-trend-cards`);
    const scopeNode = document.getElementById(`${prefix}-scope-label`);
    if (!summaryNode || !trendNode || !scopeNode) return;
    if (!payload) {
        scopeNode.textContent = prefix === 'superadmin-dashboard'
            ? 'Recent platform-wide usage, sign-ins, and issue trends appear here.'
            : 'Load a union to see usage, sign-ins, open review work, and issue trends.';
        summaryNode.innerHTML = `<div class="rounded-2xl border border-dashed border-slate-300 px-4 py-5 text-sm text-slate-500 md:col-span-2 xl:col-span-5">No dashboard data loaded yet.</div>`;
        trendNode.innerHTML = '';
        return;
    }
    const summary = payload.summary || {};
    const trends = payload.trends || {};
    scopeNode.textContent = `${payload.scope_label} • Updated ${new Date(payload.captured_at).toLocaleString()}`;
    summaryNode.innerHTML = [
        dashboardHeroCard(summary, payload, trends),
        dashboardCard('Member Activity', summary.active_users_7d || 0, `${summary.sign_ins_7d || 0} sign-ins this week and ${summary.member_workspace_loads_7d || 0} workspace loads.`, trends.active_users, '#1b6b8a', { tone: 'primary', eyebrow: 'People' }),
        dashboardCard('Review Pressure', summary.open_review_items || 0, `${summary.documents_needing_attention || 0} documents still need attention from admins.`, trends.notifications, '#d4a029', { tone: 'caution', eyebrow: 'Documents' }),
        dashboardCard('Friction Signals', summary.query_failures_7d || 0, `${summary.pending_alerts || 0} pending alerts and ${summary.security_events_7d || 0} security events this week.`, trends.query_failures || trends.security_events, '#b45309', { tone: 'danger', eyebrow: 'Issues' }),
        dashboardCard('Service Value', `$${Number(summary.estimated_cost_last_24h || 0).toFixed(2)}`, `${summary.ready_documents || 0} ready documents and ${summary.source_opens_7d || 0} source opens this week.`, trends.usage_events || trends.sign_ins, '#0f766e', { tone: 'success', eyebrow: 'Usage' }),
    ].join('');
    trendNode.innerHTML = `
        <div class="rounded-[28px] border border-slate-200 bg-gradient-to-br from-white via-white to-sky-50 p-5 shadow-sm shadow-slate-200/60">
            <div class="flex items-center justify-between gap-3">
                <div>
                    <div class="text-sm font-semibold text-slate-900">Request flow</div>
                    <div class="mt-1 text-xs text-slate-500">${(trends.labels || []).join(' • ')}</div>
                </div>
                <div class="rounded-full bg-sky-100 px-3 py-1 text-xs font-semibold text-sky-800">${summary.requests_last_24h || 0} today</div>
            </div>
            <div class="mt-4 rounded-2xl border border-white bg-white/80 px-3 py-3">${sparklineSvg(trends.requests, '#0d5c80')}</div>
        </div>
        <div class="rounded-[28px] border border-slate-200 bg-gradient-to-br from-white via-white to-cyan-50 p-5 shadow-sm shadow-slate-200/60">
            <div class="flex items-center justify-between gap-3">
                <div>
                    <div class="text-sm font-semibold text-slate-900">Member movement</div>
                    <div class="mt-1 text-xs text-slate-500">Unique active users, sign-ins, and workspace returns over the last 7 days.</div>
                </div>
                <div class="rounded-full bg-cyan-100 px-3 py-1 text-xs font-semibold text-cyan-800">${summary.active_users_7d || 0} active</div>
            </div>
            <div class="mt-4 grid gap-4">
                <div class="rounded-2xl border border-white bg-white/80 px-3 py-3">
                    <div class="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Active users</div>
                    ${sparklineSvg(trends.active_users, '#1b6b8a')}
                </div>
                <div class="rounded-2xl border border-white bg-white/80 px-3 py-3">
                    <div class="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Sign-ins</div>
                    ${sparklineSvg(trends.sign_ins, '#0f766e')}
                </div>
            </div>
        </div>
        <div class="rounded-[28px] border border-slate-200 bg-gradient-to-br from-white via-white to-rose-50 p-5 shadow-sm shadow-slate-200/60">
            <div class="flex items-center justify-between gap-3">
                <div>
                    <div class="text-sm font-semibold text-slate-900">Issues and friction</div>
                    <div class="mt-1 text-xs text-slate-500">Notifications, security events, and query failures over the last 7 days.</div>
                </div>
                <div class="rounded-full bg-rose-100 px-3 py-1 text-xs font-semibold text-rose-800">${summary.pending_alerts || 0} pending</div>
            </div>
            <div class="mt-4 grid gap-4">
                <div class="rounded-2xl border border-white bg-white/80 px-3 py-3">
                    <div class="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Notifications</div>
                    ${sparklineSvg(trends.notifications, '#d4a029')}
                </div>
                <div class="rounded-2xl border border-white bg-white/80 px-3 py-3">
                    <div class="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Security events</div>
                    ${sparklineSvg(trends.security_events, '#b45309')}
                </div>
                <div class="rounded-2xl border border-white bg-white/80 px-3 py-3">
                    <div class="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Query failures</div>
                    ${sparklineSvg(trends.query_failures, '#be123c')}
                </div>
            </div>
        </div>
    `;
}

function renderSelectedUnionAdmins(payload) {
    const node = document.getElementById('selected-union-admins');
    const statusNode = document.getElementById('selected-union-admin-status');
    if (!node || !statusNode) return;
    if (!payload || !Array.isArray(payload.admins)) {
        node.innerHTML = `<div class="rounded-2xl border border-dashed border-slate-300 px-4 py-5 text-sm text-slate-500">Select a union to load admin coverage.</div>`;
        statusNode.textContent = 'Select a union to manage its admin coverage.';
        return;
    }
    if (!payload.admins.length) {
        node.innerHTML = `<div class="rounded-2xl border border-dashed border-slate-300 px-4 py-5 text-sm text-slate-500">No union admins are currently assigned.</div>`;
    } else {
        node.innerHTML = payload.admins.map((admin) => `
            <div class="rounded-2xl border border-slate-200 bg-white p-3">
                <div class="font-semibold text-slate-900">${admin.full_name}</div>
                <div class="mono mt-1 text-xs text-slate-500">${admin.email}</div>
                <div class="mt-2 inline-flex rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700">${labelForRole(admin.role)}</div>
            </div>
        `).join('');
    }
    statusNode.textContent = `${payload.name} currently has ${payload.admins.length} admin${payload.admins.length === 1 ? '' : 's'} assigned.`;
}

function populateReviewQueueUnionFilter() {
    const node = document.getElementById('review-queue-union');
    if (!node) return;
    const unionSummaries = Array.isArray(state.platformSummary?.union_summaries) ? state.platformSummary.union_summaries : [];
    const selectedValue = state.reviewQueue.unionId || '';
    node.innerHTML = `
        <option value="">All unions</option>
        ${unionSummaries.map((item) => `
            <option value="${item.id}">${item.name}</option>
        `).join('')}
    `;
    node.value = selectedValue;
}

function renderReviewQueue(payload) {
    const summaryNode = document.getElementById('review-queue-summary');
    const listNode = document.getElementById('review-queue-list');
    if (!summaryNode || !listNode) return;
    if (!payload) {
        summaryNode.textContent = 'Load the review queue to see unresolved document work.';
        renderList('review-queue-list', [], () => '', 'No unresolved review items.');
        return;
    }
    const summary = payload.summary || {};
    summaryNode.textContent = `${summary.unresolved_documents || 0} unresolved documents, ${summary.pending_notifications || 0} pending alerts, ${summary.acknowledged_notifications || 0} acknowledged review alerts.`;
    renderList('review-queue-list', payload.items || [], (item) => `
        <div class="rounded-[24px] border border-slate-200 bg-white p-4">
            <div class="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <div class="font-semibold text-slate-900">${item.title}</div>
                    <div class="mt-1 text-sm text-slate-600">${item.union_name ? `${item.union_name} • ` : ''}${item.quality_reason || item.recommended_action || 'Needs review attention.'}</div>
                </div>
                <div class="flex flex-wrap gap-2 text-xs">
                    <span class="rounded-full bg-slate-100 px-3 py-1 font-semibold text-slate-700">${item.review_status || 'review'}</span>
                    ${item.safety_review_status ? `<span class="rounded-full bg-rose-100 px-3 py-1 font-semibold text-rose-900">${String(item.safety_review_status).replace(/_/g, ' ')}</span>` : ''}
                    <span class="rounded-full bg-slate-100 px-3 py-1 font-semibold text-slate-700">${item.status || 'status unknown'}</span>
                    ${item.ready_for_query ? '<span class="rounded-full bg-emerald-100 px-3 py-1 font-semibold text-emerald-900">query ready</span>' : '<span class="rounded-full bg-amber-100 px-3 py-1 font-semibold text-amber-900">not query ready</span>'}
                </div>
            </div>
            <div class="mt-3 flex flex-wrap gap-2 text-xs text-slate-600">
                ${item.ocr_status ? `<span class="rounded-full bg-slate-100 px-3 py-1">ocr ${item.ocr_status}</span>` : ''}
                ${item.scan_likelihood ? `<span class="rounded-full bg-slate-100 px-3 py-1">scan ${item.scan_likelihood}</span>` : ''}
                ${item.safety_status && item.safety_status !== 'clear' ? `<span class="rounded-full ${String(item.safety_status) === 'reviewed_safe' ? 'bg-emerald-100 text-emerald-900' : 'bg-rose-100 text-rose-900'} px-3 py-1">safety ${String(item.safety_status).replace(/_/g, ' ')}</span>` : ''}
                ${item.prompt_injection_risk ? '<span class="rounded-full bg-rose-100 px-3 py-1 text-rose-900">prompt injection risk</span>' : ''}
                ${item.sensitive_data_risk ? '<span class="rounded-full bg-amber-100 px-3 py-1 text-amber-900">sensitive data risk</span>' : ''}
                ${item.member_visible === false ? '<span class="rounded-full bg-rose-100 px-3 py-1 text-rose-900">hidden from members</span>' : ''}
                ${item.latest_job?.status ? `<span class="rounded-full bg-slate-100 px-3 py-1">job ${item.latest_job.status}</span>` : ''}
                <span class="rounded-full bg-slate-100 px-3 py-1">updated ${new Date(item.updated_at).toLocaleString()}</span>
            </div>
            ${Array.isArray(item.safety_reasons) && item.safety_reasons.length ? `
                <div class="mt-3 rounded-2xl border border-rose-200 bg-rose-50 px-3 py-3 text-xs text-rose-900">
                    Safety reasons: ${item.safety_reasons.join(', ')}
                </div>
            ` : ''}
            ${Array.isArray(item.notifications) && item.notifications.length ? `
                <div class="mt-3 rounded-2xl border border-slate-200 bg-slate-50 px-3 py-3 text-xs text-slate-600">
                    ${item.notifications.map((notification) => `
                        <div>${notification.subject} • ${notification.status} • ${new Date(notification.created_at).toLocaleString()}</div>
                    `).join('')}
                </div>
            ` : ''}
            <div class="mt-3 flex flex-wrap gap-3">
                ${documentNeedsSafetyReview(item) ? `<button class="open-document-review rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-slate-100" data-union-id="${item.union_id}" data-document-id="${item.document_id}" type="button">Review Document</button>` : ''}
                <button class="open-review-union rounded-full border border-slate-300 bg-white px-4 py-2 text-sm font-semibold text-slate-900 hover:bg-slate-100" data-union-id="${item.union_id}" type="button">Open Union</button>
            </div>
        </div>
    `, 'No unresolved review items matched these filters.');
    document.querySelectorAll('.open-document-review').forEach((button) => {
        button.addEventListener('click', () => run(() => openDocumentReview(button.dataset.documentId, button.dataset.unionId)));
    });
    document.querySelectorAll('.open-review-union').forEach((button) => {
        button.addEventListener('click', () => run(() => openUnionWorkspace(button.dataset.unionId)));
    });
}

function schedulePlatformOpsPolling() {
    if (platformOpsPollTimer) {
        window.clearTimeout(platformOpsPollTimer);
        platformOpsPollTimer = null;
    }
    if (!state.auth.authenticated || state.auth.user?.role !== 'super_admin') return;
    platformOpsPollTimer = window.setTimeout(() => {
        run(async () => {
            await Promise.allSettled([loadPlatformSummary(), loadPlatformOps()]);
            schedulePlatformOpsPolling();
        });
    }, 30000);
}

function renderList(id, items, formatter, empty = 'No items.') {
    const node = document.getElementById(id);
    if (!node) return;
    if (!items || !items.length) {
        node.innerHTML = `<div class="rounded-2xl border border-dashed border-slate-300 px-4 py-5 text-sm text-slate-500">${empty}</div>`;
        return;
    }
    node.innerHTML = items.map(formatter).join('');
}

function scheduleDocumentPolling(shouldPoll) {
    if (documentPollTimer) {
        window.clearTimeout(documentPollTimer);
        documentPollTimer = null;
    }
    if (!shouldPoll) return;
    documentPollTimer = window.setTimeout(() => run(loadDocuments), 5000);
}

function summarizeDocumentStatus(item) {
    const latestJob = item.latest_ingestion_job || {};
    const jobStatus = String(latestJob.status || '').trim().toLowerCase();
    const etaSeconds = Number.isFinite(Number(latestJob.estimated_ready_seconds))
        ? Number(latestJob.estimated_ready_seconds)
        : null;
    if (item.ready_for_query) return 'Ready for member questions';
    if (jobStatus === 'pending') {
        if (etaSeconds && etaSeconds > 0) {
            const etaMinutes = etaSeconds >= 60 ? `${Math.ceil(etaSeconds / 60)} min` : `${etaSeconds}s`;
            return `Queued for processing. Estimated ready in ${etaMinutes}.`;
        }
        return 'Queued for processing';
    }
    if (jobStatus === 'running' || String(item.status || '').toLowerCase() === 'processing') return 'Processing now';
    if (item.quality_status === 'retrying_with_ocr') return 'Retrying with OCR';
    if (item.review_status === 'needs_review' || item.quality_status === 'needs_review') return 'Needs review before members should rely on it';
    if (jobStatus === 'failed' || String(item.status || '').toLowerCase() === 'failed') return 'Processing failed';
    return 'Waiting on ingestion';
}

function requireUnion() {
    if (!state.selectedUnion?.id && state.tenantBootstrap?.union?.id) {
        state.selectedUnion = {
            id: state.tenantBootstrap.union.id,
            name: state.tenantBootstrap.union.name,
            slug: state.tenantBootstrap.union.slug,
            union_local_id: state.tenantBootstrap.union.union_local_id,
        };
    }
    if (!state.selectedUnion?.id) throw new Error('Select a union first.');
    return state.selectedUnion.id;
}

function buildProviderPayload() {
    const providerName = document.getElementById('provider-name').value.trim();
    const profile = PROVIDER_PROFILES[providerName] || PROVIDER_PROFILES.openrouter;
    return {
        provider_name: providerName,
        model_name: document.getElementById('provider-model').value.trim(),
        api_key: document.getElementById('provider-key').value.trim(),
        config: profile.config(),
    };
}

function resetUserEditor() {
    const form = document.getElementById('user-edit-form');
    form?.reset();
    document.getElementById('user-edit-id').value = '';
    populateRoleSelect('user-edit-role', 'user');
    setText('user-edit-status', 'Choose a user from the list to edit their access.');
    document.getElementById('password-reset-form')?.reset();
    setPasswordResetStatus('Choose a user first, then enter the new password here.', 'neutral');
    state.userDirectory.activeUserId = null;
}

function setUserEditStatus(message, tone = 'neutral') {
    const node = document.getElementById('user-edit-status');
    if (!node) return;
    node.textContent = message;
    node.className = {
        success: 'rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900',
        error: 'rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900',
        neutral: 'rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600',
    }[tone] || 'rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600';
}

function openUserDirectory() {
    state.userDirectory.open = true;
    document.getElementById('user-directory-panel')?.classList.remove('hidden');
    const button = document.getElementById('open-user-directory');
    if (button) button.textContent = 'Close User Directory';
}

function closeUserDirectory() {
    state.userDirectory.open = false;
    document.getElementById('user-directory-panel')?.classList.add('hidden');
    const button = document.getElementById('open-user-directory');
    if (button) button.textContent = 'Open User Directory';
}

function selectUserForEdit(userId) {
    const item = state.userDirectory.items.find((entry) => entry.user_id === userId);
    if (!item) return;
    state.userDirectory.activeUserId = userId;
    document.getElementById('user-edit-id').value = item.user_id;
    document.getElementById('user-edit-name').value = item.full_name || '';
    document.getElementById('user-edit-email').value = item.email || '';
    document.getElementById('user-edit-active').checked = Boolean(item.membership_active && item.user_active);
    document.getElementById('user-edit-username').value = item.username || '';
    populateRoleSelect('user-edit-role', item.role);
    setUserEditStatus(`Editing ${item.full_name || item.email}.`, 'neutral');
    setPasswordResetStatus(`Reset password for ${item.full_name || item.email}.`, 'neutral');
    renderUserDirectory();
}

function renderUserDirectory() {
    renderList('user-directory-list', state.userDirectory.items, (item) => `
        <button type="button" class="user-directory-item w-full rounded-[24px] border ${state.userDirectory.activeUserId === item.user_id ? 'border-slate-900 bg-slate-50' : 'border-slate-200 bg-white'} p-4 text-left hover:border-slate-400" data-user-id="${item.user_id}">
            <div class="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <div class="font-semibold text-slate-900">${item.full_name}</div>
                    <div class="mono mt-1 text-xs text-slate-500">${item.email}</div>
                    ${item.username ? `<div class="mono mt-1 text-xs text-slate-500">username=${item.username}</div>` : ''}
                </div>
                <div class="flex flex-wrap justify-end gap-2 text-xs">
                    <span class="rounded-full bg-slate-100 px-3 py-1 font-semibold text-slate-700">${labelForRole(item.role)}</span>
                    ${item.has_local_auth ? '<span class="rounded-full bg-emerald-100 px-3 py-1 font-semibold text-emerald-900">local sign-in</span>' : ''}
                    ${item.is_active ? '<span class="rounded-full bg-sky-100 px-3 py-1 font-semibold text-sky-900">active</span>' : '<span class="rounded-full bg-slate-200 px-3 py-1 font-semibold text-slate-700">inactive</span>'}
                </div>
            </div>
        </button>
    `, 'No users matched this search.');
    document.querySelectorAll('.user-directory-item').forEach((button) => {
        button.addEventListener('click', () => selectUserForEdit(button.dataset.userId));
    });
}

async function login(event) {
    event.preventDefault();
    const username = document.getElementById('auth-username').value.trim();
    const password = document.getElementById('auth-password').value;
    const union = routeContext.unionSlug || document.getElementById('auth-union').value.trim();
    const response = await fetch('/api/auth/session/login', {
        method: 'POST',
        headers: { ...authHeaders(), 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify({ username, password, union_slug: union || null }),
    });
    const payload = await parseJsonResponse(response);
    if (!response.ok) throw new Error(payload.detail || payload.raw || 'Unable to sign in.');
    saveAuthContext({
        authenticated: true,
        username,
        union: payload.user?.union_slug || union,
        user: payload.user || {},
    });
    setAuthModalOpen(false);
    if (payload.user?.union_id) {
        state.selectedUnion = {
            id: payload.user.union_id,
            name: payload.user.union_slug || payload.user.union_id,
            slug: payload.user.union_slug,
        };
    }
    await loadInitialData();
}

async function loadMe() {
    if (!state.auth.authenticated) {
        setJSON('me-output', { authenticated: false });
        return;
    }
    const data = await api('/api/auth/session/me');
    setJSON('me-output', data);
}

async function loadUnions() {
    if (!state.auth.authenticated) {
        renderList('union-list', [], () => '', 'Sign in to load unions.');
        return;
    }
    if (routeContext.mode === 'tenant_admin' && state.selectedUnion) {
        renderList('union-list', [state.selectedUnion], unionCard, 'No unions visible for this account.');
        bindUnionButtons();
        return;
    }
    const data = await api('/api/admin/unions');
    renderList('union-list', data.items, unionCard, 'No unions visible for this account.');
    populateUnionPicker(data.items || []);
    if (routeContext.mode === 'superadmin' && state.selectedUnion?.id) {
        const matchingUnion = (data.items || []).find((item) => item.id === state.selectedUnion.id);
        if (matchingUnion) {
            state.selectedUnion = matchingUnion;
            setText('detail-title', matchingUnion.name);
            setText('detail-subtitle', `Managing ${matchingUnion.name}.`);
        }
        applySelectedUnionVisibility();
    }
    if (routeContext.mode !== 'superadmin' && !state.selectedUnion && data.items.length === 1) {
        state.selectedUnion = data.items[0];
        setText('detail-title', state.selectedUnion.name);
        setText('detail-subtitle', `Managing ${state.selectedUnion.name}.`);
        applySelectedUnionVisibility();
    }
    bindUnionButtons();
}

async function loadTenantBootstrap() {
    if (!routeContext.unionSlug) return null;
    const response = await fetch(`/api/tenant/${encodeURIComponent(routeContext.unionSlug)}/bootstrap?page_mode=admin`, {
        headers: authHeaders(),
        credentials: 'same-origin',
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Unable to load tenant configuration.');
    state.tenantBootstrap = payload;
    state.selectedUnion = {
        id: payload.union.id,
        name: payload.union.name,
        slug: payload.union.slug,
        union_local_id: payload.union.union_local_id,
        message_retention_enabled: payload.features?.retained_chat_enabled,
    };
    document.title = `${payload.union.name} Admin | Karl`;
    setText('detail-title', payload.union.name);
    setText('detail-subtitle', `Managing ${payload.union.name}.`);
    const heading = document.querySelector('header h1');
    if (heading) heading.textContent = `${payload.union.name} administration, security, and model controls.`;
    return payload;
}

async function openUnionWorkspace(unionId, targetSectionId = null) {
    const unions = await api('/api/admin/unions');
    const selected = (unions.items || []).find((item) => item.id === unionId);
    if (!selected) throw new Error('That union is no longer available.');
    state.selectedUnion = selected;
    state.userDirectory.page = 1;
    state.feeds.selected.security.page = 1;
    state.feeds.selected.notifications.page = 1;
    state.feeds.selected.telemetry.page = 1;
    resetUserEditor();
    setText('detail-title', selected.name);
    setText('detail-subtitle', `Managing ${selected.name}.`);
    applySelectedUnionVisibility();
    postTelemetryEvent('usage_ux', 'admin_union_workspace_opened', {
        managed_union_id: selected.id,
        managed_union_slug: selected.slug,
    });
    await Promise.allSettled([
        loadUnionSettings(),
        loadUsers(),
        loadProvider(),
        loadQuota(),
        loadDocuments(),
        loadSelectedAlerts(),
        loadUnionDebugConfig(),
        loadDashboard(),
        loadInvites(),
        loadUnionTrackingPolicy(),
        loadPlatformOps(),
        loadPlatformDashboard(),
        loadGlobalAlerts(),
        loadReviewQueue(),
    ]);
    setText('platform-tools-status', `Managing ${selected.name}. Superadmin-only tools for that union are now open below.`);
    await loadUnions();
    if (targetSectionId) {
        scrollToSection(targetSectionId);
    }
}

function bindUnionButtons() {
    document.querySelectorAll('.select-union').forEach((button) => {
        button.addEventListener('click', () => run(() => openUnionWorkspace(button.dataset.unionId)));
    });
    document.querySelectorAll('.delete-union').forEach((button) => {
        button.addEventListener('click', () => run(() => deleteUnion(button.dataset.unionId, button.dataset.unionName)));
    });
    document.querySelectorAll('.toggle-union-active').forEach((button) => {
        button.addEventListener('click', () => run(() => toggleUnionActive(button.dataset.unionId, button.dataset.unionName, button.dataset.nextActive === 'true')));
    });
}

async function manageSelectedUnionFromPicker() {
    const picker = document.getElementById('union-picker');
    const unionId = picker?.value || '';
    if (!unionId) throw new Error('Choose a union first.');
    await openUnionWorkspace(unionId);
}

async function createUnion(event) {
    event.preventDefault();
    const slug = document.getElementById('union-slug').value.trim();
    const name = document.getElementById('union-name').value.trim();
    const payload = {
        slug,
        name,
    };
    if (!payload.name) return;
    await api('/api/admin/unions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    event.target.reset();
    const slugInput = document.getElementById('union-slug');
    if (slugInput) slugInput.dataset.touched = 'false';
    updateCreateUnionDerivedFields(true);
    setCreateUnionModalOpen(false);
    await loadUnions();
    setText('platform-tools-status', `Created ${payload.name}. It is now available in the union picker and the union list.`);
}

async function deleteUnion(unionId, unionName) {
    if (state.auth.user?.role !== 'super_admin') throw new Error('Only superadmins can delete unions.');
    if (!window.confirm(`Delete ${unionName}? This removes the union workspace, documents, memberships, quota, provider config, and tenant-scoped history.`)) {
        return;
    }
    await api(`/api/admin/unions/${unionId}`, { method: 'DELETE' });
    if (state.selectedUnion?.id === unionId) {
        state.selectedUnion = null;
        resetUserEditor();
        closeUserDirectory();
        setText('detail-title', 'Select a union');
        setText('detail-subtitle', 'This workspace is organized around the tasks a union admin handles most often: documents, members, union settings, quotas, and model setup.');
        renderList('document-list', [], () => '', 'Select a union first.');
        applySelectedUnionVisibility();
    }
    await loadUnions();
    await loadPlatformSummary();
    await loadOps();
    setText('platform-tools-status', `${unionName} was deleted. The platform readout has been refreshed.`);
}

async function toggleUnionActive(unionId, unionName, nextActive) {
    if (state.auth.user?.role !== 'super_admin') throw new Error('Only superadmins can change union active status.');
    const action = nextActive ? 'reactivate' : 'take offline';
    if (!window.confirm(`${action.charAt(0).toUpperCase() + action.slice(1)} ${unionName}? ${nextActive ? 'Members and admins can resume access after this saves.' : 'This immediately marks the union inactive for live use.'}`)) {
        return;
    }
    await api(`/api/admin/unions/${unionId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ is_active: nextActive }),
    });
    await Promise.allSettled([loadUnions(), loadPlatformSummary(), loadPlatformOps(), loadPlatformDashboard()]);
    if (state.selectedUnion?.id === unionId) {
        await Promise.allSettled([loadUnionSettings(), loadDashboard()]);
    }
    setText('platform-tools-status', `${unionName} was ${nextActive ? 'reactivated' : 'taken offline'}.`);
}

async function takeAllUnionsOffline() {
    if (state.auth.user?.role !== 'super_admin') throw new Error('Only superadmins can take all unions offline.');
    if (!window.confirm('Take all unions offline? This marks every union inactive for live use until reactivated.')) {
        return;
    }
    const data = await api('/api/admin/unions/offline-all', { method: 'POST' });
    await Promise.allSettled([loadUnions(), loadPlatformSummary(), loadPlatformOps(), loadPlatformDashboard()]);
    if (state.selectedUnion?.id) {
        await Promise.allSettled([loadUnionSettings(), loadDashboard()]);
    }
    setText('platform-tools-status', `All-union offline action completed. ${data.changed_unions || 0} unions were changed.`);
}

async function clearSelectedUnion() {
    state.selectedUnion = null;
    state.userDirectory.page = 1;
    state.feeds.selected.security.page = 1;
    state.feeds.selected.notifications.page = 1;
    state.feeds.selected.telemetry.page = 1;
    resetUserEditor();
    closeUserDirectory();
    setText('detail-title', 'Select a union');
    setText('detail-subtitle', 'Select Manage in the superadmin tools above to open a union workspace.');
    renderDashboard(null, 'dashboard');
    renderSelectedUnionAdmins(null);
    renderList('document-list', [], () => '', 'Select a union first.');
    renderList('user-directory-list', [], () => '', 'Select a union first.');
    renderList('selected-chat-list', [], () => '', 'Select a union first.');
    renderList('selected-telemetry-list', [], () => '', 'Select a union first.');
    setText('user-summary', 'Select a union to load member access.');
    setText('quota-usage-summary', 'Load a union to see current usage.');
    setText('selected-telemetry-summary', 'No telemetry loaded.');
    setText('platform-tools-status', 'No union is currently being managed. Choose one from the union picker or the list above.');
    applySelectedUnionVisibility();
    setUnionTrackingFormEnabled(false);
    setTrackingPolicyStatus('union-tracking-policy-status', 'This union currently follows the platform default until an override is enabled.', 'neutral');
    await loadUnions();
}

async function toggleSelectedUnionActive() {
    if (!state.selectedUnion?.id) throw new Error('Select a union first.');
    const nextActive = document.getElementById('selected-union-toggle-active')?.dataset.nextActive === 'true';
    await toggleUnionActive(state.selectedUnion.id, state.selectedUnion.name || state.selectedUnion.slug || 'selected union', nextActive);
}

async function deleteSelectedUnion() {
    if (!state.selectedUnion?.id) throw new Error('Select a union first.');
    await deleteUnion(state.selectedUnion.id, state.selectedUnion.name || state.selectedUnion.slug || 'selected union');
}

async function loadUnionDebugConfig() {
    if (state.auth.user?.role !== 'super_admin') {
        setJSON('superadmin-debug-output', {});
        return;
    }
    const unionId = requireUnion();
    const data = await api(`/api/admin/unions/${unionId}/debug-config`);
    setJSON('superadmin-debug-output', data);
}

async function loadDashboard() {
    if (!state.selectedUnion?.id && !state.tenantBootstrap?.union?.id) {
        state.dashboard = null;
        renderDashboard(null, 'dashboard');
        return;
    }
    const unionId = requireUnion();
    const query = state.auth.user?.role === 'super_admin' ? `?union_id=${encodeURIComponent(unionId)}` : '';
    const data = await api(`/api/ops/dashboard${query}`);
    state.dashboard = data;
    renderDashboard(data, 'dashboard');
}

async function loadPlatformDashboard() {
    if (!state.auth.authenticated || state.auth.user?.role !== 'super_admin') {
        state.platformDashboard = null;
        renderDashboard(null, 'superadmin-dashboard');
        return;
    }
    const data = await api('/api/ops/dashboard');
    state.platformDashboard = data;
    renderDashboard(data, 'superadmin-dashboard');
}

async function downloadUnionExport() {
    if (state.auth.user?.role !== 'super_admin') throw new Error('Only superadmins can export tenant data.');
    const unionId = requireUnion();
    const response = await fetch(`/api/admin/unions/${unionId}/export`, {
        headers: authHeaders(),
        credentials: 'same-origin',
    });
    if (!response.ok) {
        const payload = await parseJsonResponse(response);
        throw new Error(payload.detail || payload.raw || 'Unable to export tenant data.');
    }
    const blob = await response.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    const unionSlug = state.selectedUnion?.slug || 'union';
    link.href = downloadUrl;
    link.download = `${unionSlug}-export.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(downloadUrl);
    const statusNode = document.getElementById('selected-union-admin-status');
    if (statusNode) {
        statusNode.textContent = `Downloaded the latest export bundle for ${state.selectedUnion?.name || unionSlug}.`;
    }
}

async function assignSelectedUserAsUnionAdmin() {
    if (state.auth.user?.role !== 'super_admin') throw new Error('Only superadmins can assign union admins.');
    const unionId = requireUnion();
    const userId = document.getElementById('user-edit-id').value;
    if (!userId) throw new Error('Choose a user first.');
    await api(`/api/admin/unions/${unionId}/users/${userId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ role: 'union_admin' }),
    });
    setUserEditStatus('User promoted to union admin.', 'success');
    await Promise.all([loadUsers(), loadPlatformOps()]);
    selectUserForEdit(userId);
}

async function assignSelfTakeover() {
    if (state.auth.user?.role !== 'super_admin') throw new Error('Only superadmins can take over a union.');
    const unionId = requireUnion();
    await api(`/api/admin/unions/${unionId}/admin-takeover`, {
        method: 'POST',
    });
    const statusNode = document.getElementById('selected-union-admin-status');
    if (statusNode) {
        statusNode.textContent = 'Temporary union admin access assigned to your current superadmin account.';
    }
    await loadPlatformOps();
}

async function loadUnionSettings() {
    const unionId = requireUnion();
    const data = await api('/api/admin/unions');
    const union = (data.items || []).find((item) => item.id === unionId);
    if (!union) return;
    state.selectedUnion = union;
    refreshSelectedUnionActions();
    const metadata = union.metadata || {};
    const branding = metadata.branding || {};
    const authPolicy = metadata.auth_policy || {};
    document.getElementById('union-settings-name').value = union.name || '';
    document.getElementById('union-settings-theme-color').value = branding.theme_color || '#0d5c80';
    document.getElementById('union-settings-accent-color').value = branding.accent_color || '#d4a029';
    document.getElementById('union-settings-surface-tint').value = branding.surface_tint || '#edf5f8';
    document.getElementById('union-settings-welcome-heading').value = branding.welcome_heading || '';
    document.getElementById('union-settings-welcome-subcopy').value = branding.welcome_subcopy || '';
    document.getElementById('union-settings-login-required').checked = authPolicy.member_login_required !== false;
    document.getElementById('union-settings-retention').checked = Boolean(union.message_retention_enabled);
    document.getElementById('union-settings-active').checked = union.is_active !== false;
    if (state.auth.user?.role === 'super_admin') {
        await loadUnionTrackingPolicy();
    }
}

async function saveUnionSettings(event) {
    event.preventDefault();
    const unionId = requireUnion();
    const payload = {
        name: document.getElementById('union-settings-name').value.trim(),
        branding: {
            theme_color: document.getElementById('union-settings-theme-color').value,
            accent_color: document.getElementById('union-settings-accent-color').value,
            surface_tint: document.getElementById('union-settings-surface-tint').value,
            welcome_heading: document.getElementById('union-settings-welcome-heading').value.trim(),
            welcome_subcopy: document.getElementById('union-settings-welcome-subcopy').value.trim(),
        },
        member_login_required: document.getElementById('union-settings-login-required').checked,
        message_retention_enabled: document.getElementById('union-settings-retention').checked,
        is_active: document.getElementById('union-settings-active').checked,
    };
    await api(`/api/admin/unions/${unionId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    await loadUnions();
    await loadUnionSettings();
    window.alert('Settings saved ok.');
}

async function loadUsers() {
    const unionId = requireUnion();
    const params = new URLSearchParams({
        page: String(state.userDirectory.page),
        page_size: String(state.userDirectory.pageSize),
        q_field: state.userDirectory.field,
        sort: state.userDirectory.sort,
        direction: state.userDirectory.direction,
    });
    if (state.userDirectory.query) params.set('q', state.userDirectory.query);
    const data = await api(`/api/admin/unions/${unionId}/users?${params.toString()}`);
    state.userDirectory.items = data.items || [];
    state.userDirectory.total = Number(data.total || 0);
    state.userDirectory.unionTotal = Number(data.union_total || 0);
    state.userDirectory.page = Number(data.page || 1);
    const start = state.userDirectory.total ? ((state.userDirectory.page - 1) * state.userDirectory.pageSize) + 1 : 0;
    const end = Math.min(state.userDirectory.page * state.userDirectory.pageSize, state.userDirectory.total);
    setText(
        'user-summary',
        state.userDirectory.unionTotal
            ? `${state.userDirectory.unionTotal} users in this union.`
            : 'No users yet. Use the form below to add the first member or admin.',
    );
    setText(
        'user-pagination-summary',
        state.userDirectory.unionTotal
            ? (state.userDirectory.query
                ? `${state.userDirectory.total} matching users out of ${state.userDirectory.unionTotal} total. Showing ${start}-${end}.`
                : `Showing ${start}-${end} of ${state.userDirectory.unionTotal} total users`)
            : 'No users matched this search.',
    );
    document.getElementById('user-page-prev').disabled = state.userDirectory.page <= 1;
    document.getElementById('user-page-next').disabled = end >= state.userDirectory.total;
    renderUserDirectory();
}

async function createUser(event) {
    event.preventDefault();
    const unionId = requireUnion();
    const payload = {
        email: document.getElementById('user-email').value.trim(),
        full_name: document.getElementById('user-name').value.trim(),
        role: document.getElementById('user-role').value,
        username: document.getElementById('user-username').value.trim() || null,
        password: document.getElementById('user-password').value || null,
    };
    await api(`/api/admin/unions/${unionId}/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    event.target.reset();
    populateRoleSelect('user-role', 'user');
    await loadUsers();
}

async function saveSelectedUser(event) {
    event.preventDefault();
    const unionId = requireUnion();
    const userId = document.getElementById('user-edit-id').value;
    if (!userId) throw new Error('Choose a user first.');
    const payload = {
        full_name: document.getElementById('user-edit-name').value.trim(),
        email: document.getElementById('user-edit-email').value.trim(),
        role: document.getElementById('user-edit-role').value,
        username: document.getElementById('user-edit-username').value.trim() || null,
        is_active: document.getElementById('user-edit-active').checked,
    };
    await api(`/api/admin/unions/${unionId}/users/${userId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    setUserEditStatus('User access updated.', 'success');
    await loadUsers();
    selectUserForEdit(userId);
}

function openPasswordResetModal() {
    const userId = document.getElementById('user-edit-id').value;
    if (!userId) throw new Error('Choose a user first.');
    document.getElementById('password-reset-form')?.reset();
    setPasswordResetStatus('Enter the new password twice to save it.', 'neutral');
    setPasswordResetModalOpen(true);
}

async function resetSelectedUserPassword(event) {
    event?.preventDefault();
    const unionId = requireUnion();
    const userId = document.getElementById('user-edit-id').value;
    const password = document.getElementById('password-reset-new').value;
    const repeat = document.getElementById('password-reset-repeat').value;
    if (!userId) throw new Error('Choose a user first.');
    if (!password.trim()) throw new Error('Enter a new password first.');
    if (password !== repeat) {
        setPasswordResetStatus('The passwords do not match yet.', 'error');
        throw new Error('The passwords do not match.');
    }
    await api(`/api/admin/unions/${unionId}/users/${userId}/password`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
    });
    document.getElementById('password-reset-form')?.reset();
    setPasswordResetStatus('Password reset saved.', 'success');
    setPasswordResetModalOpen(false);
    setUserEditStatus('Password reset saved.', 'success');
}

async function removeSelectedUser() {
    const unionId = requireUnion();
    const userId = document.getElementById('user-edit-id').value;
    if (!userId) throw new Error('Choose a user first.');
    const purgeUser = window.confirm('Permanently delete this user and purge their account data? This removes their account, sessions, retained chats, usage records, notifications, and union access where allowed. Choose Cancel if you only want to remove union access.');
    if (!purgeUser && !window.confirm('Remove this user from the current union only? Their global account record will remain, but they will lose access to this union.')) {
        return;
    }
    await api(`/api/admin/unions/${unionId}/users/${userId}${purgeUser ? '?purge_user=true' : ''}`, { method: 'DELETE' });
    resetUserEditor();
    await loadUsers();
}

async function loadProvider() {
    const unionId = requireUnion();
    const data = await api(`/api/admin/unions/${unionId}/provider`);
    const provider = data.provider || {};
    const providerName = document.getElementById('provider-name');
    const providerModel = document.getElementById('provider-model');
    const providerKey = document.getElementById('provider-key');
    if (providerName) providerName.value = provider.provider_name || 'openrouter';
    if (providerModel) providerModel.value = provider.model_name || '';
    if (providerKey) providerKey.value = '';
    setProviderKeyStatus(Boolean(provider.has_api_key));
    setProviderHelp(provider.provider_name || 'openrouter');
    setProviderTestStatus(null);
}

async function saveProvider(event) {
    event.preventDefault();
    const unionId = requireUnion();
    const payload = buildProviderPayload();
    await api(`/api/admin/unions/${unionId}/provider`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    await loadProvider();
    setProviderTestStatus('Provider settings saved. Run a live test to verify the model responds.', 'neutral');
    window.alert('Model provider settings saved ok.');
    if (state.auth.user?.role === 'super_admin') {
        await Promise.allSettled([loadPlatformOps(), loadPlatformSummary()]);
    }
}

async function testProvider() {
    const unionId = requireUnion();
    setProviderTestStatus('Testing provider connection...', 'neutral');
    const payload = buildProviderPayload();
    payload.api_key = payload.api_key || null;
    const data = await api(`/api/admin/unions/${unionId}/provider/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    const result = data.result || {};
    if (result.ok) {
        postTelemetryEvent('usage_ux', 'provider_test_completed', {
            union_id: unionId,
            ok: true,
            provider_name: result.provider_name || payload.provider_name || null,
            model_name: result.model_name || payload.model_name || null,
            latency_ms: result.latency_ms || null,
        });
        setProviderTestStatus(`Provider responded in ${result.latency_ms} ms using ${result.provider_name}/${result.model_name}. Preview: ${result.preview || 'OK'}`, 'success');
        return;
    }
    postTelemetryEvent('bug_journey', 'provider_test_failed', {
        union_id: unionId,
        ok: false,
        provider_name: result.provider_name || payload.provider_name || null,
        model_name: result.model_name || payload.model_name || null,
        latency_ms: result.latency_ms || null,
        error_type: result.error_type || null,
    });
    setProviderTestStatus(`Provider test failed after ${result.latency_ms} ms. ${result.error_type || 'error'}: ${result.error_message || 'Unknown provider error.'}`, 'error');
}

async function loadQuota() {
    const unionId = requireUnion();
    const data = await api(`/api/admin/unions/${unionId}/quota`);
    const quota = data.quota || {};
    document.getElementById('quota-requests').value = quota.requests_per_day || '';
    document.getElementById('quota-tokens').value = quota.tokens_per_day || '';
    document.getElementById('quota-cost').value = quota.cost_usd_per_day || '';
    document.getElementById('quota-user-rate').value = quota.per_user_requests_per_hour || '';
    document.getElementById('quota-warn').value = quota.warn_threshold_ratio || '';
    document.getElementById('quota-paused').checked = Boolean(quota.is_paused);
    const snapshot = quota.usage_snapshot || {};
    setText(
        'quota-usage-summary',
        `Current usage: ${snapshot.requests_last_24h || 0} requests and ${snapshot.tokens_last_24h || 0} tokens in the last 24 hours, about $${Number(snapshot.estimated_cost_last_24h || 0).toFixed(2)} estimated cost. ${snapshot.requests_last_hour || 0} requests were recorded in the last hour.`,
    );
}

async function saveQuota(event) {
    event.preventDefault();
    const unionId = requireUnion();
    const payload = {
        requests_per_day: Number(document.getElementById('quota-requests').value || 0),
        tokens_per_day: Number(document.getElementById('quota-tokens').value || 0),
        cost_usd_per_day: Number(document.getElementById('quota-cost').value || 0),
        per_user_requests_per_hour: Number(document.getElementById('quota-user-rate').value || 0),
        warn_threshold_ratio: Number(document.getElementById('quota-warn').value || 0.8),
        is_paused: document.getElementById('quota-paused').checked,
    };
    await api(`/api/admin/unions/${unionId}/quota`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    await loadQuota();
}

async function loadDocuments() {
    const unionId = requireUnion();
    const data = await api(`/api/admin/unions/${unionId}/documents`);
    const shouldPoll = (data.items || []).some((item) => {
        const latestStatus = String(item?.latest_ingestion_job?.status || '').toLowerCase();
        return String(item?.status || '').toLowerCase() === 'processing'
            || latestStatus === 'pending'
            || latestStatus === 'running'
            || item?.quality_status === 'retrying_with_ocr';
    });
    scheduleDocumentPolling(shouldPoll);
    renderList('document-list', data.items, (item) => `
        <div class="rounded-[24px] border border-slate-200 bg-white p-4">
            <div class="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <div class="font-semibold text-slate-900">${item.title}</div>
                    <div class="mt-1 text-sm text-slate-600">${summarizeDocumentStatus(item)}</div>
                    <div class="mt-2 flex flex-wrap gap-2 text-xs text-slate-600">
                        <span class="rounded-full bg-slate-100 px-3 py-1">type ${item.document_type || 'unknown'}</span>
                        <span class="rounded-full bg-slate-100 px-3 py-1">query ${item.ready_for_query ? 'ready' : 'not ready'}</span>
                        <span class="rounded-full bg-slate-100 px-3 py-1">review ${item.review_status || 'n/a'}</span>
                        ${item.safety_status && item.safety_status !== 'clear' ? `<span class="rounded-full ${String(item.safety_status) === 'reviewed_safe' ? 'bg-emerald-100 text-emerald-900' : 'bg-rose-100 text-rose-900'} px-3 py-1">safety ${String(item.safety_status).replace(/_/g, ' ')}</span>` : ''}
                        ${item.prompt_injection_risk ? '<span class="rounded-full bg-rose-100 px-3 py-1 text-rose-900">blocked from members</span>' : ''}
                        ${item.sensitive_data_risk ? '<span class="rounded-full bg-amber-100 px-3 py-1 text-amber-900">member redaction active</span>' : ''}
                        <span class="rounded-full bg-slate-100 px-3 py-1">ingestion ${item.latest_ingestion_job?.status || item.status}</span>
                    </div>
                    ${Array.isArray(item.safety_reasons) && item.safety_reasons.length ? `<div class="mt-2 text-xs text-rose-700">Safety review: ${item.safety_reasons.join(', ')}.</div>` : ''}
                    ${item.recommended_action ? `<div class="mt-1 text-xs text-slate-600">${item.recommended_action}</div>` : ''}
                </div>
                <div class="flex flex-wrap gap-2">
                    <button type="button" class="view-document-file rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-900 hover:bg-slate-100" data-document-id="${item.id}" data-document-title="${item.title.replace(/"/g, '&quot;')}">View File</button>
                    ${documentNeedsSafetyReview(item) ? `<button type="button" class="review-document rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-900 hover:bg-slate-100" data-document-id="${item.id}">Review</button>` : ''}
                    <button type="button" class="delete-document rounded-full border border-rose-300 bg-white px-3 py-1 text-xs font-semibold text-rose-700 hover:bg-rose-50" data-document-id="${item.id}" data-document-title="${item.title.replace(/"/g, '&quot;')}">Delete</button>
                </div>
            </div>
        </div>
    `, 'No uploaded documents yet.');
    document.querySelectorAll('.view-document-file').forEach((button) => {
        button.addEventListener('click', () => run(() => openDocumentFile(button.dataset.documentId, button.dataset.documentTitle)));
    });
    document.querySelectorAll('.review-document').forEach((button) => {
        button.addEventListener('click', () => run(() => openDocumentReview(button.dataset.documentId)));
    });
    document.querySelectorAll('.delete-document').forEach((button) => {
        button.addEventListener('click', () => run(() => deleteDocument(button.dataset.documentId, button.dataset.documentTitle)));
    });
}

async function openDocumentFile(documentId, title = 'document') {
    const unionId = requireUnion();
    const response = await fetch(`/api/admin/unions/${unionId}/documents/${documentId}/content`, {
        headers: authHeaders(),
        credentials: 'same-origin',
    });
    if (!response.ok) {
        const payload = await parseJsonResponse(response);
        throw new Error(payload.detail || payload.raw || `Unable to open ${title}.`);
    }
    const blob = await response.blob();
    const openUrl = window.URL.createObjectURL(blob);
    window.open(openUrl, '_blank', 'noopener,noreferrer');
    window.setTimeout(() => window.URL.revokeObjectURL(openUrl), 60000);
    postTelemetryEvent('usage_ux', 'admin_document_file_opened', {
        document_id: documentId,
        union_id: unionId,
        title,
    });
}

function renderDocumentReviewModal(payload) {
    const detail = payload || {};
    const documentPayload = detail.document || {};
    const reviewActions = detail.review_actions || {};
    const preview = detail.review_preview || {};
    const findings = Array.isArray(preview.safety_findings) ? preview.safety_findings : [];
    state.reviewDocument = {
        documentId: documentPayload.id,
        unionId: detail.review_union_id || state.selectedUnion?.id || documentPayload.union_id || '',
        latestJobId: detail.latest_job?.id || documentPayload.latest_ingestion_job?.id || '',
        reviewActions,
    };

    setText('document-review-title', documentPayload.title || 'Review document');
    setText('document-review-subtitle', reviewActions.approval_effect || 'Review the flagged content below and choose the next step.');
    setText(
        'document-review-summary',
        [
            documentPayload.safety_status && documentPayload.safety_status !== 'clear' ? `Safety status: ${String(documentPayload.safety_status).replace(/_/g, ' ')}` : '',
            documentPayload.review_status ? `Review: ${String(documentPayload.review_status).replace(/_/g, ' ')}` : '',
            documentPayload.ready_for_query ? 'Currently query ready.' : 'Currently not query ready.',
        ].filter(Boolean).join(' '),
    );
    const findingsNode = document.getElementById('document-review-findings');
    if (findingsNode) {
        findingsNode.innerHTML = findings.length
            ? findings.map((finding) => `
                <div class="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                    <div class="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">${finding.category.replace(/_/g, ' ')}</div>
                    <div class="mt-1 text-sm font-semibold text-slate-900">${finding.label.replace(/_/g, ' ')}</div>
                    <div class="mt-1 text-sm text-slate-700">${finding.match_preview}</div>
                    <div class="mt-2 text-xs text-slate-500">${finding.context}</div>
                </div>
            `).join('')
            : '<div class="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600">No specific match details were extracted. You can still use the preview below to review the document.</div>';
    }
    setText('document-review-preview-raw', preview.text_excerpt || 'No parsed preview is available yet for this document.');
    setText('document-review-preview-redacted', preview.redacted_excerpt || 'No redacted member preview is available yet.');
    const noteInput = document.getElementById('document-review-note');
    if (noteInput) noteInput.value = '';
    const approveButton = document.getElementById('document-review-approve');
    if (approveButton) {
        approveButton.classList.toggle('hidden', !reviewActions.can_approve_member_access);
        approveButton.disabled = !reviewActions.can_approve_member_access;
        approveButton.textContent = documentPayload.prompt_injection_risk ? 'Release For Members' : 'Approve Full Member Access';
    }
    const needsSuperadmin = document.getElementById('document-review-needs-superadmin');
    if (needsSuperadmin) {
        if (reviewActions.requires_superadmin_override) {
            needsSuperadmin.classList.remove('hidden');
            needsSuperadmin.textContent = 'A superadmin must approve this prompt-injection override. Union admins can still review the issue and delete the document.';
        } else {
            needsSuperadmin.classList.add('hidden');
            needsSuperadmin.textContent = '';
        }
    }
}

async function openDocumentReview(documentId, unionId = null) {
    const effectiveUnionId = unionId || requireUnion();
    const payload = await api(`/api/admin/unions/${effectiveUnionId}/documents/${documentId}/review-detail`);
    renderDocumentReviewModal({ ...payload, review_union_id: effectiveUnionId });
    setDocumentReviewModalOpen(true);
    postTelemetryEvent('usage_ux', 'document_review_opened', {
        document_id: documentId,
        union_id: effectiveUnionId,
    });
}

async function submitDocumentReviewDecision(decision) {
    const reviewState = state.reviewDocument;
    if (!reviewState?.documentId || !reviewState?.unionId) return;
    const note = document.getElementById('document-review-note')?.value.trim() || '';
    if (decision === 'delete_document') {
        const label = document.getElementById('document-review-title')?.textContent || 'this document';
        await deleteDocument(reviewState.documentId, label, reviewState.unionId);
        setDocumentReviewModalOpen(false);
        await Promise.allSettled([
            loadReviewQueue(),
            state.selectedUnion?.id === reviewState.unionId ? loadDocuments() : Promise.resolve(),
            loadPlatformOps(),
        ]);
        return;
    }
    const payload = await api(`/api/admin/unions/${reviewState.unionId}/documents/${reviewState.documentId}/safety-review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, note }),
    });
    renderDocumentReviewModal({
        ...payload,
        latest_job: { id: reviewState.latestJobId },
    });
    window.alert(decision === 'approve_member_access' ? 'Document approved for full member access.' : 'Document marked in review.');
    await Promise.allSettled([
        loadReviewQueue(),
        state.selectedUnion?.id === reviewState.unionId ? loadDocuments() : Promise.resolve(),
        loadPlatformOps(),
    ]);
    if (state.selectedUnion?.id === reviewState.unionId) {
        await Promise.allSettled([loadSelectedAlerts(), loadDashboard()]);
    }
    setDocumentReviewModalOpen(false);
}

async function deleteDocument(documentId, title, unionId = null) {
    const effectiveUnionId = unionId || requireUnion();
    const label = String(title || 'this document');
    if (!window.confirm(`Delete ${label}? This will remove the uploaded file, ingestion jobs, and retrieval chunks.`)) return;
    await api(`/api/admin/unions/${effectiveUnionId}/documents/${documentId}`, { method: 'DELETE' });
    if (state.selectedUnion?.id === effectiveUnionId) {
        await loadDocuments();
    }
}

async function uploadDocument(event) {
    event.preventDefault();
    const unionId = requireUnion();
    const file = document.getElementById('document-file').files[0];
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);
    const result = await api(`/api/admin/unions/${unionId}/documents`, {
        method: 'POST',
        body: formData,
    });
    event.target.reset();
    await loadDocuments();
    if (result.queued_for_background_processing) {
        window.alert('Document uploaded and queued for background processing. The page will refresh status automatically.');
    }
}

async function loadChats() {
    const unionId = requireUnion();
    const data = await api(`/api/admin/unions/${unionId}/chats`);
    const listId = document.getElementById('selected-chat-list') ? 'selected-chat-list' : 'chat-list';
    renderList(listId, data.items, (item) => `
        <button class="chat-item w-full rounded-2xl border border-slate-200 bg-white p-3 text-left hover:border-slate-400" data-chat-id="${item.id}">
            <div class="font-semibold text-slate-900">${item.session_id}</div>
            <div class="mono text-xs text-slate-500">${item.updated_at}</div>
        </button>
    `, 'No retained chats available.');
    document.querySelectorAll('.chat-item').forEach((button) => {
        button.addEventListener('click', async () => {
            const detail = await api(`/api/admin/unions/${unionId}/chats/${button.dataset.chatId}`);
            const node = document.getElementById('chat-detail');
            node.classList.remove('hidden');
            node.textContent = JSON.stringify(detail, null, 2);
        });
    });
}

async function loadSecurityFeed(scope = 'global') {
    const config = state.feeds[scope]?.security;
    if (!config) return;
    const listId = scope === 'selected'
        ? (document.getElementById('selected-security-list') ? 'selected-security-list' : 'security-list')
        : (document.getElementById('global-security-list') ? 'global-security-list' : 'security-list');
    const summaryId = scope === 'selected' ? 'selected-security-summary' : 'global-security-summary';
    const prevId = scope === 'selected' ? 'selected-security-prev' : 'global-security-prev';
    const nextId = scope === 'selected' ? 'selected-security-next' : 'global-security-next';
    if (!state.auth.authenticated) {
        renderPaginatedFeed({
            listId,
            summaryId,
            prevId,
            nextId,
            items: [],
            formatter: () => '',
            empty: 'Sign in to load security events.',
            page: config.page,
            pageSize: config.pageSize,
            total: 0,
            singularLabel: 'security event',
            pluralLabel: 'security events',
        });
        return;
    }
    const params = new URLSearchParams({
        page: String(config.page),
        page_size: String(config.pageSize),
    });
    if (scope === 'selected' && state.selectedUnion?.id && state.auth.user?.role === 'super_admin') {
        params.set('union_id', state.selectedUnion.id);
    }
    const data = await api(`/api/ops/security-events?${params.toString()}`);
    config.total = Number(data.total || 0);
    renderPaginatedFeed({
        listId,
        summaryId,
        prevId,
        nextId,
        items: data.items || [],
        formatter: (item) => `
        <div class="rounded-2xl border border-slate-200 bg-white p-3">
            <div class="flex items-center justify-between gap-3">
                <div class="font-semibold text-slate-900">${item.event_type}</div>
                <div class="rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold">${item.severity}</div>
            </div>
            <div class="mono mt-2 text-xs text-slate-500">${new Date(item.created_at).toLocaleString()}</div>
        </div>
    `,
        empty: 'No security events.',
        page: config.page,
        pageSize: config.pageSize,
        total: config.total,
        singularLabel: 'security event',
        pluralLabel: 'security events',
    });
}

async function loadNotificationFeed(scope = 'global') {
    const config = state.feeds[scope]?.notifications;
    if (!config) return;
    const listId = scope === 'selected'
        ? (document.getElementById('selected-notification-list') ? 'selected-notification-list' : 'notification-list')
        : (document.getElementById('global-notification-list') ? 'global-notification-list' : 'notification-list');
    const summaryId = scope === 'selected' ? 'selected-notification-summary' : 'global-notification-summary';
    const prevId = scope === 'selected' ? 'selected-notification-prev' : 'global-notification-prev';
    const nextId = scope === 'selected' ? 'selected-notification-next' : 'global-notification-next';
    if (!state.auth.authenticated) {
        renderPaginatedFeed({
            listId,
            summaryId,
            prevId,
            nextId,
            items: [],
            formatter: () => '',
            empty: 'Sign in to load notifications.',
            page: config.page,
            pageSize: config.pageSize,
            total: 0,
            singularLabel: 'notification',
            pluralLabel: 'notifications',
        });
        return;
    }
    const params = new URLSearchParams({
        page: String(config.page),
        page_size: String(config.pageSize),
    });
    if (scope === 'selected' && state.selectedUnion?.id && state.auth.user?.role === 'super_admin') {
        params.set('union_id', state.selectedUnion.id);
    }
    const data = await api(`/api/ops/notifications?${params.toString()}`);
    config.total = Number(data.total || 0);
    renderPaginatedFeed({
        listId,
        summaryId,
        prevId,
        nextId,
        items: data.items || [],
        formatter: (item) => notificationFeedFormatter(item, scope),
        empty: 'No notifications.',
        page: config.page,
        pageSize: config.pageSize,
        total: config.total,
        singularLabel: 'notification',
        pluralLabel: 'notifications',
    });
    const listNode = document.getElementById(listId);
    listNode?.querySelectorAll('.notification-dismiss').forEach((button) => {
        button.addEventListener('click', () => run(() => dismissNotification(button.dataset.notificationId, button.dataset.scope || scope)));
    });
}

async function dismissNotification(notificationId, scope = 'global') {
    if (!notificationId) return;
    await api(`/api/ops/notifications/${notificationId}/acknowledge`, { method: 'POST' });
    if (scope === 'selected') {
        await Promise.allSettled([loadNotificationFeed('selected'), loadSecurityFeed('selected'), loadTelemetryFeed('selected')]);
    } else {
        await Promise.allSettled([loadNotificationFeed('global'), loadSecurityFeed('global'), loadTelemetryFeed('global')]);
    }
    await Promise.allSettled([loadPlatformSummary(), loadPlatformDashboard(), loadReviewQueue()]);
}

async function loadTelemetryFeed(scope = 'global') {
    const config = state.feeds[scope]?.telemetry;
    if (!config) return;
    const listId = scope === 'selected'
        ? (document.getElementById('selected-telemetry-list') ? 'selected-telemetry-list' : 'telemetry-list')
        : 'global-telemetry-list';
    const summaryId = scope === 'selected'
        ? (document.getElementById('selected-telemetry-summary') ? 'selected-telemetry-summary' : 'telemetry-summary')
        : 'global-telemetry-summary';
    const prevId = scope === 'selected'
        ? (document.getElementById('selected-telemetry-prev') ? 'selected-telemetry-prev' : 'telemetry-prev')
        : 'global-telemetry-prev';
    const nextId = scope === 'selected'
        ? (document.getElementById('selected-telemetry-next') ? 'selected-telemetry-next' : 'telemetry-next')
        : 'global-telemetry-next';

    if (!state.auth.authenticated) {
        renderPaginatedFeed({
            listId,
            summaryId,
            prevId,
            nextId,
            items: [],
            formatter: () => '',
            empty: 'Sign in to load telemetry.',
            page: config.page,
            pageSize: config.pageSize,
            total: 0,
            singularLabel: 'telemetry event',
            pluralLabel: 'telemetry events',
        });
        return;
    }

    const params = new URLSearchParams({
        page: String(config.page),
        page_size: String(config.pageSize),
    });
    if (config.category) params.set('category', config.category);
    if (config.eventType) params.set('event_type', config.eventType);
    if (config.query) params.set('q', config.query);
    if (config.sessionId) params.set('session_id', config.sessionId);
    if (scope === 'selected' && state.selectedUnion?.id && state.auth.user?.role === 'super_admin') {
        params.set('union_id', state.selectedUnion.id);
    }

    const data = await api(`/api/ops/telemetry-events?${params.toString()}`);
    config.total = Number(data.total || 0);
    renderPaginatedFeed({
        listId,
        summaryId,
        prevId,
        nextId,
        items: data.items || [],
        formatter: telemetryFeedFormatter,
        empty: 'No telemetry events matched those filters.',
        page: config.page,
        pageSize: config.pageSize,
        total: config.total,
        singularLabel: 'telemetry event',
        pluralLabel: 'telemetry events',
    });
    const listNode = document.getElementById(listId);
    listNode?.querySelectorAll('.open-session-timeline').forEach((button) => {
        button.addEventListener('click', () => run(() => openSessionTimeline(button.dataset.sessionId, button.dataset.unionId)));
    });
}

function updateTelemetryFiltersFromForm(scope = 'selected') {
    const config = state.feeds[scope]?.telemetry;
    if (!config) return;
    const prefix = scope === 'selected'
        ? (document.getElementById('selected-telemetry-form') ? 'selected-telemetry' : 'telemetry')
        : 'global-telemetry';
    config.category = document.getElementById(`${prefix}-category`)?.value || '';
    config.eventType = document.getElementById(`${prefix}-event-type`)?.value.trim() || '';
    config.query = document.getElementById(`${prefix}-query`)?.value.trim() || '';
    config.sessionId = document.getElementById(`${prefix}-session-id`)?.value.trim() || '';
    config.page = 1;
}

async function loadPlatformSummary() {
    if (!state.auth.authenticated || state.auth.user?.role !== 'super_admin') {
        state.platformSummary = null;
        renderPlatformSummary(null);
        return;
    }
    const data = await api('/api/admin/platform-summary');
    state.platformSummary = data;
    renderPlatformSummary(data);
    populateReviewQueueUnionFilter();
}

async function loadPlatformOps() {
    if (!state.auth.authenticated || state.auth.user?.role !== 'super_admin') {
        state.platformOps = null;
        renderPlatformOps(null);
        renderSelectedUnionAdmins(null);
        return;
    }
    const data = await api('/api/admin/platform-ops');
    state.platformOps = data;
    renderPlatformOps(data);
    if (state.selectedUnion?.id) {
        const item = (data.items || []).find((entry) => entry.union_id === state.selectedUnion.id) || null;
        renderSelectedUnionAdmins(item);
    } else {
        renderSelectedUnionAdmins(null);
    }
}

async function loadReviewQueue() {
    if (!state.auth.authenticated || state.auth.user?.role !== 'super_admin') {
        renderReviewQueue(null);
        return;
    }
    const params = new URLSearchParams();
    if (state.reviewQueue.query) params.set('q', state.reviewQueue.query);
    if (state.reviewQueue.unionId) params.set('union_id', state.reviewQueue.unionId);
    if (state.reviewQueue.reviewStatus) params.set('review_status', state.reviewQueue.reviewStatus);
    if (state.reviewQueue.status) params.set('status', state.reviewQueue.status);
    const query = params.toString();
    const data = await api(`/api/ops/review-queue${query ? `?${query}` : ''}`);
    state.reviewQueue.items = data.items || [];
    state.reviewQueue.summary = data.summary || null;
    renderReviewQueue(data);
}

async function loadSelectedAlerts() {
    if (!state.selectedUnion?.id && routeContext.mode === 'superadmin') {
        renderList('selected-chat-list', [], () => '', 'Select a union first.');
        renderPaginatedFeed({
            listId: 'selected-security-list',
            summaryId: 'selected-security-summary',
            prevId: 'selected-security-prev',
            nextId: 'selected-security-next',
            items: [],
            formatter: () => '',
            empty: 'Select a union first.',
            page: state.feeds.selected.security.page,
            pageSize: state.feeds.selected.security.pageSize,
            total: 0,
            singularLabel: 'security event',
            pluralLabel: 'security events',
        });
        renderPaginatedFeed({
            listId: 'selected-notification-list',
            summaryId: 'selected-notification-summary',
            prevId: 'selected-notification-prev',
            nextId: 'selected-notification-next',
            items: [],
            formatter: () => '',
            empty: 'Select a union first.',
            page: state.feeds.selected.notifications.page,
            pageSize: state.feeds.selected.notifications.pageSize,
            total: 0,
            singularLabel: 'notification',
            pluralLabel: 'notifications',
        });
        renderPaginatedFeed({
            listId: document.getElementById('selected-telemetry-list') ? 'selected-telemetry-list' : 'telemetry-list',
            summaryId: document.getElementById('selected-telemetry-summary') ? 'selected-telemetry-summary' : 'telemetry-summary',
            prevId: document.getElementById('selected-telemetry-prev') ? 'selected-telemetry-prev' : 'telemetry-prev',
            nextId: document.getElementById('selected-telemetry-next') ? 'selected-telemetry-next' : 'telemetry-next',
            items: [],
            formatter: () => '',
            empty: 'Select a union first.',
            page: state.feeds.selected.telemetry.page,
            pageSize: state.feeds.selected.telemetry.pageSize,
            total: 0,
            singularLabel: 'telemetry event',
            pluralLabel: 'telemetry events',
        });
        return;
    }
    await Promise.allSettled([loadChats(), loadSecurityFeed('selected'), loadNotificationFeed('selected'), loadTelemetryFeed('selected')]);
}

async function loadGlobalAlerts() {
    await Promise.allSettled([loadSecurityFeed('global'), loadNotificationFeed('global'), loadTelemetryFeed('global')]);
}

async function refreshPlatformOverview() {
    await Promise.allSettled([loadPlatformSummary(), loadPlatformOps(), loadPlatformDashboard()]);
}

async function loadOps() {
    await Promise.allSettled([refreshPlatformOverview(), loadGlobalAlerts(), loadReviewQueue()]);
}

async function loadInitialData() {
    if (routeContext.unionSlug && !state.tenantBootstrap) {
        await loadTenantBootstrap();
    }
    await loadMe();
    await loadPlatformSummary();
    await loadPlatformOps();
    await loadGlobalTrackingPolicy();
    await loadUnions();
    if (state.selectedUnion?.id) {
        await Promise.allSettled([
            loadUnionSettings(),
            loadUsers(),
            loadProvider(),
            loadQuota(),
            loadDocuments(),
            loadSelectedAlerts(),
            loadUnionDebugConfig(),
            loadDashboard(),
        ]);
    }
    await loadOps();
    schedulePlatformOpsPolling();
}

function wire() {
    if (routeContext.mode === 'legacy_admin') {
        window.location.replace('/karl/');
        return;
    }
    renderAuthSummary();
    applyRoleVisibility();
    applyWorkspaceVisibility();
    populateRoleSelects();
    setProviderHelp();
    updateCreateUnionDerivedFields(true);
    setUnionTrackingFormEnabled(false);
    document.getElementById('auth-form').addEventListener('submit', (event) => run(() => login(event)));
    document.getElementById('auth-logout').addEventListener('click', () => run(logout));
    document.getElementById('auth-open-modal')?.addEventListener('click', () => setAuthModalOpen(true));
    document.getElementById('auth-modal-close')?.addEventListener('click', () => setAuthModalOpen(false));
    document.getElementById('auth-modal-cancel')?.addEventListener('click', () => setAuthModalOpen(false));
    document.getElementById('auth-password-toggle')?.addEventListener('click', () => togglePasswordVisibility('auth-password', 'auth-password-toggle'));
    document.getElementById('auth-modal')?.addEventListener('click', (event) => {
        if (event.target?.id === 'auth-modal') setAuthModalOpen(false);
    });
    document.getElementById('provider-name')?.addEventListener('change', () => setProviderHelp());
    document.getElementById('refresh-me').addEventListener('click', () => run(loadMe));
    document.getElementById('load-unions').addEventListener('click', () => run(loadUnions));
    document.getElementById('load-ops').addEventListener('click', () => run(refreshPlatformOverview));
    document.getElementById('refresh-global-tracking-policy')?.addEventListener('click', () => run(loadGlobalTrackingPolicy));
    document.getElementById('global-tracking-policy-form')?.addEventListener('submit', (event) => run(() => saveGlobalTrackingPolicy(event)));
    document.getElementById('open-create-union-modal')?.addEventListener('click', () => {
        updateCreateUnionDerivedFields(true);
        setCreateUnionModalOpen(true);
    });
    document.getElementById('create-union-modal-close')?.addEventListener('click', () => setCreateUnionModalOpen(false));
    document.getElementById('create-union-cancel')?.addEventListener('click', () => setCreateUnionModalOpen(false));
    document.getElementById('create-union-modal')?.addEventListener('click', (event) => {
        if (event.target?.id === 'create-union-modal') setCreateUnionModalOpen(false);
    });
    document.getElementById('document-review-close')?.addEventListener('click', () => setDocumentReviewModalOpen(false));
    document.getElementById('document-review-cancel')?.addEventListener('click', () => setDocumentReviewModalOpen(false));
    document.getElementById('document-review-modal')?.addEventListener('click', (event) => {
        if (event.target?.id === 'document-review-modal') setDocumentReviewModalOpen(false);
    });
    document.getElementById('session-timeline-close')?.addEventListener('click', () => setSessionTimelineModalOpen(false));
    document.getElementById('session-timeline-done')?.addEventListener('click', () => setSessionTimelineModalOpen(false));
    document.getElementById('session-timeline-modal')?.addEventListener('click', (event) => {
        if (event.target?.id === 'session-timeline-modal') setSessionTimelineModalOpen(false);
    });
    document.getElementById('document-review-in-review')?.addEventListener('click', () => run(() => submitDocumentReviewDecision('mark_in_review')));
    document.getElementById('document-review-approve')?.addEventListener('click', () => run(() => submitDocumentReviewDecision('approve_member_access')));
    document.getElementById('document-review-delete')?.addEventListener('click', () => run(() => submitDocumentReviewDecision('delete_document')));
    document.getElementById('manage-selected-union')?.addEventListener('click', () => run(manageSelectedUnionFromPicker));
    document.getElementById('take-all-unions-offline')?.addEventListener('click', () => run(takeAllUnionsOffline));
    document.getElementById('create-union-form').addEventListener('submit', (event) => run(() => createUnion(event)));
    document.getElementById('union-name')?.addEventListener('input', () => updateCreateUnionDerivedFields(false));
    document.getElementById('union-slug')?.addEventListener('input', (event) => {
        event.currentTarget.dataset.touched = 'true';
        updateCreateUnionDerivedFields(false);
    });
    document.getElementById('union-settings-form').addEventListener('submit', (event) => run(() => saveUnionSettings(event)));
    document.getElementById('create-user-form').addEventListener('submit', (event) => run(() => createUser(event)));
    document.getElementById('user-edit-form').addEventListener('submit', (event) => run(() => saveSelectedUser(event)));
    document.getElementById('user-password-button').addEventListener('click', () => run(openPasswordResetModal));
    document.getElementById('user-delete-button').addEventListener('click', () => run(removeSelectedUser));
    document.getElementById('provider-form').addEventListener('submit', (event) => run(() => saveProvider(event)));
    document.getElementById('test-provider').addEventListener('click', () => run(testProvider));
    document.getElementById('quota-form').addEventListener('submit', (event) => run(() => saveQuota(event)));
    document.getElementById('document-form').addEventListener('submit', (event) => run(() => uploadDocument(event)));
    document.getElementById('refresh-users').addEventListener('click', () => run(loadUsers));
    document.getElementById('refresh-union-settings').addEventListener('click', () => run(loadUnionSettings));
    document.getElementById('refresh-provider').addEventListener('click', () => run(loadProvider));
    document.getElementById('refresh-quota').addEventListener('click', () => run(loadQuota));
    document.getElementById('refresh-docs').addEventListener('click', () => run(loadDocuments));
    document.getElementById('refresh-selected-alerts')?.addEventListener('click', () => run(loadSelectedAlerts));
    document.getElementById('refresh-global-alerts')?.addEventListener('click', () => run(loadGlobalAlerts));
    document.getElementById('refresh-selected-telemetry')?.addEventListener('click', () => run(() => loadTelemetryFeed('selected')));
    document.getElementById('refresh-global-telemetry')?.addEventListener('click', () => run(() => loadTelemetryFeed('global')));
    document.getElementById('refresh-telemetry')?.addEventListener('click', () => run(() => loadTelemetryFeed('selected')));
    document.getElementById('refresh-dashboard')?.addEventListener('click', () => run(loadDashboard));
    document.getElementById('refresh-platform-overview')?.addEventListener('click', () => run(refreshPlatformOverview));
    document.getElementById('refresh-selected-admins')?.addEventListener('click', () => run(loadPlatformOps));
    document.getElementById('promote-selected-user-admin')?.addEventListener('click', () => run(assignSelectedUserAsUnionAdmin));
    document.getElementById('assign-self-takeover')?.addEventListener('click', () => run(assignSelfTakeover));
    document.getElementById('clear-selected-union')?.addEventListener('click', () => run(clearSelectedUnion));
    document.getElementById('selected-union-toggle-active')?.addEventListener('click', () => run(toggleSelectedUnionActive));
    document.getElementById('selected-union-delete')?.addEventListener('click', () => run(deleteSelectedUnion));
    document.getElementById('refresh-union-tracking-policy')?.addEventListener('click', () => run(loadUnionTrackingPolicy));
    document.getElementById('union-tracking-policy-form')?.addEventListener('submit', (event) => run(() => saveUnionTrackingPolicy(event)));
    document.getElementById('clear-union-tracking-policy')?.addEventListener('click', () => run(clearUnionTrackingPolicy));
    document.getElementById('union-tracking-override-enabled')?.addEventListener('change', (event) => {
        const enabled = Boolean(event.currentTarget?.checked);
        setUnionTrackingFormEnabled(enabled);
    });
    document.getElementById('download-union-export')?.addEventListener('click', () => run(downloadUnionExport));
    document.getElementById('refresh-debug-drawer')?.addEventListener('click', () => run(loadUnionDebugConfig));
    document.getElementById('refresh-review-queue')?.addEventListener('click', () => run(loadReviewQueue));
    document.getElementById('review-queue-form')?.addEventListener('submit', (event) => {
        event.preventDefault();
        state.reviewQueue.query = document.getElementById('review-queue-query')?.value.trim() || '';
        state.reviewQueue.unionId = document.getElementById('review-queue-union')?.value || '';
        state.reviewQueue.reviewStatus = document.getElementById('review-queue-review-status')?.value || '';
        state.reviewQueue.status = document.getElementById('review-queue-status')?.value || '';
        run(loadReviewQueue);
    });
    document.getElementById('selected-telemetry-form')?.addEventListener('submit', (event) => {
        event.preventDefault();
        updateTelemetryFiltersFromForm('selected');
        run(() => loadTelemetryFeed('selected'));
    });
    document.getElementById('global-telemetry-form')?.addEventListener('submit', (event) => {
        event.preventDefault();
        updateTelemetryFiltersFromForm('global');
        run(() => loadTelemetryFeed('global'));
    });
    document.getElementById('telemetry-form')?.addEventListener('submit', (event) => {
        event.preventDefault();
        updateTelemetryFiltersFromForm('selected');
        run(() => loadTelemetryFeed('selected'));
    });
    document.getElementById('open-user-directory').addEventListener('click', () => {
        if (state.userDirectory.open) {
            closeUserDirectory();
            return;
        }
        openUserDirectory();
        run(loadUsers);
    });
    document.getElementById('close-user-directory').addEventListener('click', closeUserDirectory);
    document.getElementById('user-search-form').addEventListener('submit', (event) => {
        event.preventDefault();
        const rawQuery = document.getElementById('user-search-query').value.trim();
        state.userDirectory.query = rawQuery === '*' ? '' : rawQuery;
        document.getElementById('user-search-query').value = state.userDirectory.query;
        state.userDirectory.field = document.getElementById('user-search-field').value;
        state.userDirectory.sort = document.getElementById('user-sort').value;
        state.userDirectory.direction = document.getElementById('user-direction').value;
        state.userDirectory.page = 1;
        run(loadUsers);
    });
    document.getElementById('user-view-all').addEventListener('click', () => {
        state.userDirectory.query = '';
        state.userDirectory.field = 'all';
        state.userDirectory.sort = 'name';
        state.userDirectory.direction = 'asc';
        state.userDirectory.page = 1;
        document.getElementById('user-search-query').value = '';
        document.getElementById('user-search-field').value = 'all';
        document.getElementById('user-sort').value = 'name';
        document.getElementById('user-direction').value = 'asc';
        run(loadUsers);
    });
    document.getElementById('user-page-prev').addEventListener('click', () => {
        if (state.userDirectory.page <= 1) return;
        state.userDirectory.page -= 1;
        run(loadUsers);
    });
    document.getElementById('user-page-next').addEventListener('click', () => {
        const maxPage = Math.max(1, Math.ceil(state.userDirectory.total / state.userDirectory.pageSize));
        if (state.userDirectory.page >= maxPage) return;
        state.userDirectory.page += 1;
        run(loadUsers);
    });
    document.getElementById('password-reset-form').addEventListener('submit', (event) => run(() => resetSelectedUserPassword(event)));
    document.getElementById('password-reset-close').addEventListener('click', () => setPasswordResetModalOpen(false));
    document.getElementById('password-reset-cancel').addEventListener('click', () => setPasswordResetModalOpen(false));
    document.getElementById('password-reset-new-toggle').addEventListener('click', () => togglePasswordVisibility('password-reset-new', 'password-reset-new-toggle'));
    document.getElementById('password-reset-repeat-toggle').addEventListener('click', () => togglePasswordVisibility('password-reset-repeat', 'password-reset-repeat-toggle'));
    document.getElementById('password-reset-modal')?.addEventListener('click', (event) => {
        if (event.target?.id === 'password-reset-modal') setPasswordResetModalOpen(false);
    });
    document.getElementById('selected-security-prev')?.addEventListener('click', () => {
        if (state.feeds.selected.security.page <= 1) return;
        state.feeds.selected.security.page -= 1;
        run(loadSelectedAlerts);
    });
    document.getElementById('selected-security-next')?.addEventListener('click', () => {
        const maxPage = Math.max(1, Math.ceil(state.feeds.selected.security.total / state.feeds.selected.security.pageSize));
        if (state.feeds.selected.security.page >= maxPage) return;
        state.feeds.selected.security.page += 1;
        run(loadSelectedAlerts);
    });
    document.getElementById('selected-notification-prev')?.addEventListener('click', () => {
        if (state.feeds.selected.notifications.page <= 1) return;
        state.feeds.selected.notifications.page -= 1;
        run(loadSelectedAlerts);
    });
    document.getElementById('selected-notification-next')?.addEventListener('click', () => {
        const maxPage = Math.max(1, Math.ceil(state.feeds.selected.notifications.total / state.feeds.selected.notifications.pageSize));
        if (state.feeds.selected.notifications.page >= maxPage) return;
        state.feeds.selected.notifications.page += 1;
        run(loadSelectedAlerts);
    });
    document.getElementById('selected-telemetry-prev')?.addEventListener('click', () => {
        if (state.feeds.selected.telemetry.page <= 1) return;
        state.feeds.selected.telemetry.page -= 1;
        run(() => loadTelemetryFeed('selected'));
    });
    document.getElementById('selected-telemetry-next')?.addEventListener('click', () => {
        const maxPage = Math.max(1, Math.ceil(state.feeds.selected.telemetry.total / state.feeds.selected.telemetry.pageSize));
        if (state.feeds.selected.telemetry.page >= maxPage) return;
        state.feeds.selected.telemetry.page += 1;
        run(() => loadTelemetryFeed('selected'));
    });
    document.getElementById('telemetry-prev')?.addEventListener('click', () => {
        if (state.feeds.selected.telemetry.page <= 1) return;
        state.feeds.selected.telemetry.page -= 1;
        run(() => loadTelemetryFeed('selected'));
    });
    document.getElementById('telemetry-next')?.addEventListener('click', () => {
        const maxPage = Math.max(1, Math.ceil(state.feeds.selected.telemetry.total / state.feeds.selected.telemetry.pageSize));
        if (state.feeds.selected.telemetry.page >= maxPage) return;
        state.feeds.selected.telemetry.page += 1;
        run(() => loadTelemetryFeed('selected'));
    });
    document.getElementById('global-security-prev')?.addEventListener('click', () => {
        if (state.feeds.global.security.page <= 1) return;
        state.feeds.global.security.page -= 1;
        run(loadGlobalAlerts);
    });
    document.getElementById('global-security-next')?.addEventListener('click', () => {
        const maxPage = Math.max(1, Math.ceil(state.feeds.global.security.total / state.feeds.global.security.pageSize));
        if (state.feeds.global.security.page >= maxPage) return;
        state.feeds.global.security.page += 1;
        run(loadGlobalAlerts);
    });
    document.getElementById('global-notification-prev')?.addEventListener('click', () => {
        if (state.feeds.global.notifications.page <= 1) return;
        state.feeds.global.notifications.page -= 1;
        run(loadGlobalAlerts);
    });
    document.getElementById('global-notification-next')?.addEventListener('click', () => {
        const maxPage = Math.max(1, Math.ceil(state.feeds.global.notifications.total / state.feeds.global.notifications.pageSize));
        if (state.feeds.global.notifications.page >= maxPage) return;
        state.feeds.global.notifications.page += 1;
        run(loadGlobalAlerts);
    });
    document.getElementById('global-telemetry-prev')?.addEventListener('click', () => {
        if (state.feeds.global.telemetry.page <= 1) return;
        state.feeds.global.telemetry.page -= 1;
        run(() => loadTelemetryFeed('global'));
    });
    document.getElementById('global-telemetry-next')?.addEventListener('click', () => {
        const maxPage = Math.max(1, Math.ceil(state.feeds.global.telemetry.total / state.feeds.global.telemetry.pageSize));
        if (state.feeds.global.telemetry.page >= maxPage) return;
        state.feeds.global.telemetry.page += 1;
        run(() => loadTelemetryFeed('global'));
    });
    run(async () => {
        const authenticated = await hydrateAuthContext();
        if (authenticated) {
            await loadInitialData();
            postTelemetryEvent('bug_journey', 'admin_workspace_loaded', {
                role: state.auth.user?.role || null,
                tenant_slug: routeContext.unionSlug || null,
            });
        } else {
            setJSON('me-output', { authenticated: false });
        }
    });
}

async function run(fn) {
    try {
        await fn();
    } catch (error) {
        console.error(error);
        window.alert(error.message || String(error));
    }
}

function inviteStatusBadge(item) {
    const styles = {
        active: 'bg-emerald-100 text-emerald-900',
        revoked: 'bg-rose-100 text-rose-900',
        expired: 'bg-amber-100 text-amber-900',
        exhausted: 'bg-slate-200 text-slate-700',
    };
    return `<span class="rounded-full ${styles[item.status] || 'bg-slate-100'} px-3 py-1 text-xs font-semibold">${escapeHtml(item.status)}</span>`;
}

function inviteAudienceBadge(item) {
    const steward = String(item.audience || 'member').toLowerCase() === 'steward';
    const cls = steward ? 'bg-indigo-100 text-indigo-900' : 'bg-sky-100 text-sky-900';
    const text = steward ? 'Steward · all contracts' : `Member${item.contract_id ? ` · ${escapeHtml(item.contract_id)}` : ''}`;
    return `<span class="rounded-full ${cls} px-3 py-1 text-xs font-semibold">${text}</span>`;
}

function shortWhen(value) {
    return value ? escapeHtml(String(value).slice(0, 10)) : '—';
}

function renderInviteCard(item, unionId) {
    return `
        <div class="rounded-[24px] border border-slate-200 bg-white p-4">
            <div class="flex flex-wrap items-start justify-between gap-4">
                <div class="flex items-start gap-4">
                    <div class="shrink-0 rounded-2xl border border-slate-200 bg-white p-2">
                        <img src="/api/admin/unions/${unionId}/invites/${item.id}/qr?format=svg" alt="QR for ${escapeHtml(item.code)}" class="h-24 w-24" loading="lazy">
                    </div>
                    <div>
                        <div class="flex flex-wrap items-center gap-2">
                            <span class="font-mono text-base font-bold tracking-wide text-slate-900">${escapeHtml(item.code)}</span>
                            ${inviteAudienceBadge(item)}
                            ${inviteStatusBadge(item)}
                        </div>
                        <div class="mt-1 text-sm text-slate-600">${escapeHtml(item.label || 'No label')}</div>
                        <div class="mt-2 flex flex-wrap gap-2 text-xs text-slate-600">
                            <span class="rounded-full bg-slate-100 px-3 py-1">joined ${item.use_count}${item.max_uses ? ` / ${item.max_uses}` : ''}</span>
                            <span class="rounded-full bg-indigo-50 px-3 py-1 text-indigo-900" title="Questions asked through this code">${Number(item.total_requests || 0).toLocaleString()} asks</span>
                            <span class="rounded-full bg-indigo-50 px-3 py-1 text-indigo-900" title="Tokens consumed through this code">${Number(item.total_tokens || 0).toLocaleString()} tokens</span>
                            ${Number(item.total_cost_usd || 0) > 0 ? `<span class="rounded-full bg-indigo-50 px-3 py-1 text-indigo-900" title="Estimated cost">$${Number(item.total_cost_usd).toFixed(4)}</span>` : ''}
                            <span class="rounded-full bg-slate-100 px-3 py-1">first ${shortWhen(item.first_used_at)}</span>
                            <span class="rounded-full bg-slate-100 px-3 py-1">last ${shortWhen(item.last_used_at)}</span>
                            ${item.expires_at ? `<span class="rounded-full bg-slate-100 px-3 py-1">expires ${shortWhen(item.expires_at)}</span>` : ''}
                            <span class="rounded-full bg-slate-100 px-3 py-1">${escapeHtml(item.join_path)}</span>
                        </div>
                    </div>
                </div>
                <div class="flex flex-wrap gap-2">
                    <button type="button" class="copy-invite-link rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-900 hover:bg-slate-100" data-join-path="${escapeHtml(item.join_path)}">Copy Link</button>
                    <a href="/api/admin/unions/${unionId}/invites/${item.id}/qr?format=png" download class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-900 hover:bg-slate-100">Download PNG</a>
                    <a href="/api/admin/unions/${unionId}/invites/${item.id}/card" target="_blank" rel="noopener" class="rounded-full border border-slate-300 bg-white px-3 py-1 text-xs font-semibold text-slate-900 hover:bg-slate-100">Printable Card</a>
                    ${item.status === 'active' ? `<button type="button" class="revoke-invite rounded-full border border-rose-300 bg-white px-3 py-1 text-xs font-semibold text-rose-700 hover:bg-rose-50" data-invite-id="${item.id}" data-invite-code="${escapeHtml(item.code)}" data-disconnect="false">Revoke</button>` : ''}
                    ${item.status === 'active' || item.use_count > 0 ? `<button type="button" class="revoke-invite rounded-full border border-rose-400 bg-rose-600 px-3 py-1 text-xs font-semibold text-white hover:bg-rose-700" data-invite-id="${item.id}" data-invite-code="${escapeHtml(item.code)}" data-disconnect="true">Revoke + Disconnect</button>` : ''}
                </div>
            </div>
        </div>
    `;
}

async function loadInvites() {
    const unionId = requireUnion();
    await populateInviteContracts(unionId);
    const data = await api(`/api/admin/unions/${unionId}/invites`);
    const items = data.items || [];
    const members = items.filter((it) => String(it.audience || 'member').toLowerCase() !== 'steward');
    const stewards = items.filter((it) => String(it.audience || 'member').toLowerCase() === 'steward');
    const usageTotal = (list) => list.reduce((sum, it) => sum + (Number(it.use_count) || 0), 0);
    const tokenTotal = (list) => list.reduce((sum, it) => sum + (Number(it.total_tokens) || 0), 0);
    const container = document.getElementById('invite-list');
    if (container) {
        if (!items.length) {
            container.innerHTML = '<p class="text-sm text-slate-500">No QR codes yet. Create one per placement.</p>';
        } else {
            const section = (title, list) => list.length ? `
                <div>
                    <div class="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                        <span>${title}</span>
                        <span class="rounded-full bg-slate-100 px-2 py-0.5 normal-case text-slate-600">${list.length} code${list.length === 1 ? '' : 's'} · ${usageTotal(list)} joined · ${tokenTotal(list).toLocaleString()} tokens</span>
                    </div>
                    <div class="grid gap-3">${list.map((it) => renderInviteCard(it, unionId)).join('')}</div>
                </div>` : '';
            container.innerHTML = `<div class="grid gap-6">${section('Member codes', members)}${section('Steward codes', stewards)}</div>`;
        }
    }
    document.querySelectorAll('.copy-invite-link').forEach((button) => {
        button.addEventListener('click', () => {
            const url = `${window.location.origin}${button.dataset.joinPath}`;
            navigator.clipboard?.writeText(url).then(
                () => { button.textContent = 'Copied!'; window.setTimeout(() => { button.textContent = 'Copy Link'; }, 1500); },
                () => window.prompt('Copy this join link:', url),
            );
        });
    });
    document.querySelectorAll('.revoke-invite').forEach((button) => {
        button.addEventListener('click', () => run(async () => {
            const disconnect = button.dataset.disconnect === 'true';
            const message = disconnect
                ? `Revoke join code ${button.dataset.inviteCode} AND sign out everyone who joined through it? Use this if the QR code is being misused. Members can rejoin through a different active code.`
                : `Revoke join code ${button.dataset.inviteCode}? Members who already joined stay signed in; the QR placement stops accepting new joins.`;
            if (!window.confirm(message)) return;
            const unionIdNow = requireUnion();
            const result = await api(`/api/admin/unions/${unionIdNow}/invites/${button.dataset.inviteId}/revoke`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ disconnect_sessions: disconnect }),
            });
            if (disconnect) {
                window.alert(`Code revoked. ${result.sessions_disconnected || 0} member session(s) disconnected.`);
            }
            await loadInvites();
        }));
    });
}

function selectedInviteAudience() {
    return document.querySelector('input[name="invite-audience"]:checked')?.value || 'member';
}

function syncInviteContractVisibility() {
    const wrap = document.getElementById('invite-contract-wrap');
    if (!wrap) return;
    const steward = selectedInviteAudience() === 'steward';
    wrap.classList.toggle('hidden', steward);
}

async function populateInviteContracts(unionId) {
    const select = document.getElementById('invite-contract');
    if (!select) return;
    let contractIds = [];
    try {
        const data = await api(`/api/admin/unions/${unionId}/documents`);
        contractIds = [...new Set((data.items || [])
            .map((it) => String(it.contract_id || '').trim())
            .filter(Boolean))].sort();
    } catch (error) {
        console.error('Could not load contracts for invite form', error);
    }
    const previous = select.value;
    select.innerHTML = '<option value="">Select a contract…</option>'
        + contractIds.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join('');
    if (previous && contractIds.includes(previous)) select.value = previous;
}

async function createInvite(event) {
    event.preventDefault();
    const unionId = requireUnion();
    const audience = selectedInviteAudience();
    const label = document.getElementById('invite-label').value.trim();
    const maxUsesRaw = document.getElementById('invite-max-uses').value;
    const expiresRaw = document.getElementById('invite-expires').value;
    const contractId = document.getElementById('invite-contract')?.value || '';
    if (audience === 'member' && !contractId) {
        window.alert('Member codes must be pinned to a contract. Pick a contract, or switch the audience to Steward.');
        return;
    }
    const payload = { audience, label };
    if (audience === 'member') payload.contract_id = contractId;
    if (maxUsesRaw) payload.max_uses = Number(maxUsesRaw);
    if (expiresRaw) payload.expires_at = `${expiresRaw}T23:59:59`;
    await api(`/api/admin/unions/${unionId}/invites`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    document.getElementById('invite-create-form').reset();
    syncInviteContractVisibility();
    await loadInvites();
}

document.getElementById('refresh-invites')?.addEventListener('click', () => run(loadInvites));
document.getElementById('invite-create-form')?.addEventListener('submit', (event) => run(() => createInvite(event)));
document.querySelectorAll('input[name="invite-audience"]').forEach((radio) => {
    radio.addEventListener('change', syncInviteContractVisibility);
});
syncInviteContractVisibility();

wire();
