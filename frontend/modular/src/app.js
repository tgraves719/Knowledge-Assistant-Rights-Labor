let createStewardOnboardingController = null;
let createMemberOnboardingController = null;
const EMBED_THEME_OVERRIDES = (() => {
    if (typeof window.__KARL_EMBED_THEME__ === 'object' && window.__KARL_EMBED_THEME__) {
        return window.__KARL_EMBED_THEME__;
    }
    return {};
})();

        function resolveApiBase() {
            const params = new URLSearchParams(window.location.search);
            const queryOverride = (params.get('api_base') || '').trim();
            const localOverride = (localStorage.getItem('karl_api_base') || '').trim();
            const globalOverride = typeof window.__KARL_API_BASE__ === 'string'
                ? window.__KARL_API_BASE__.trim()
                : '';
            const chosenOverride = globalOverride || queryOverride || localOverride;
            if (chosenOverride) {
                return chosenOverride.replace(/\/+$/, '');
            }
            // When served by FastAPI, same-origin is the correct API target.
            // Keep localhost fallback for file:// local testing.
            if (window.location.protocol === 'file:') {
                return 'http://127.0.0.1:8000';
            }
            return window.location.origin.replace(/\/+$/, '');
        }

        const API_BASE = resolveApiBase();
        function resolveRouteContext() {
            if (window.__KARL_ROUTE_CONTEXT__ && typeof window.__KARL_ROUTE_CONTEXT__ === 'object') {
                const injectedMode = String(window.__KARL_ROUTE_CONTEXT__.mode || '').trim();
                const injectedUnionSlug = String(window.__KARL_ROUTE_CONTEXT__.unionSlug || '').trim() || null;
                if (injectedMode) {
                    return { mode: injectedMode, unionSlug: injectedUnionSlug };
                }
            }
            const path = String(window.location.pathname || '').replace(/\/+$/, '') || '/';
            const tenantAdminMatch = path.match(/^\/u\/([^/]+)\/admin(?:\/index\.html)?$/i);
            if (tenantAdminMatch) {
                return { mode: 'tenant_admin', unionSlug: decodeURIComponent(tenantAdminMatch[1]) };
            }
            const tenantMemberMatch = path.match(/^\/u\/([^/]+)(?:\/index\.html)?$/i);
            if (tenantMemberMatch) {
                return { mode: 'tenant_member', unionSlug: decodeURIComponent(tenantMemberMatch[1]) };
            }
            if (path === '/karl' || path === '/karl/index.html') {
                return { mode: 'superadmin', unionSlug: null };
            }
            return { mode: 'legacy_member', unionSlug: null };
        }
        const ACTIVE_CONTRACT_STORAGE_KEY = 'karl_active_contract_id';
        const MEMBER_AUTH_STORAGE_KEY = 'karl_member_auth_context';
        const CONTRACT_VIEW_MODE_STORAGE_KEY = 'karl_contract_view_mode';
        const CONTRACT_PDF_SOURCE_MODE_STORAGE_KEY = 'karl_contract_pdf_source_mode';
        const CONTRACT_TEXT_SOURCE_MODE_STORAGE_KEY = 'karl_contract_text_source_mode';
        const SESSION_ID_STORAGE_KEY = 'karl_session_id';
        const SESSION_META_STORAGE_KEY = 'karl_session_meta';
        const ONBOARDING_FLOW_STORAGE_KEY = 'karl_onboarding_flow';
        const KARL_VERSION_FALLBACK = '0.8.110';
        let isHealthy = false;
        let onboardingFactoriesPromise = null;
        let userProfile = null;
        let availableContracts = [];
        let classificationOptionsByContract = {};
        let activeContract = null;
        let articleCache = {};
        let articleFirstSectionCache = {};
        let articleTitles = {};
        let contractBrowseGroups = [];
        let currentTocSelection = { kind: null, key: null };
        let currentPopover = {
            articleNum: null,
            sectionNum: null,
            partNum: null,
            tableId: null,
            rowIndex: null,
            citationLabel: null,
            sourceRegistryKey: null,
            sourceDocId: null,
            sourceType: null,
            sourcePdf: null,
            sourcePage: null,
            sourceChoices: [],
            selectedSourceChoiceKey: null,
        };
        let currentArticleNum = null;
        let currentPdfBaseUrl = null;
        let currentPdfPage = null;
        let lastPinnedPdfLocation = null;
        let lastPdfNavigationContext = null;
        let lastContractTextContext = null;
        let contractTextCompareOpen = false;
        let pdfNavRequestSeq = 0;
        let pendingPdfFrameSwapTimer = null;
        let pendingEdgePdfSecondPassTimer = null;
        let stewardOnboarding = null;
        let memberOnboarding = null;
        let preferences = JSON.parse(localStorage.getItem('karl_preferences') || '{}');
        let memberAuth = loadMemberAuth();
        let memberTrackingSettings = {
            tracking_preference: 'system_default',
            effective_policy: null,
            member_choice_available: false,
        };
        let tenantBootstrap = null;
        const routeContext = resolveRouteContext();
        let karlInfo = null;
        let currentKarlDocId = null;
        let contractHistoryById = {};
        let activeContractHistory = null;
        let contractViewerMode = String(localStorage.getItem(CONTRACT_VIEW_MODE_STORAGE_KEY) || 'effective_text').trim().toLowerCase() || 'effective_text';
        let contractPdfSourceMode = String(localStorage.getItem(CONTRACT_PDF_SOURCE_MODE_STORAGE_KEY) || 'effective').trim().toLowerCase() || 'effective';
        let contractPdfSourcePdf = null;
        let contractTextSourceMode = String(localStorage.getItem(CONTRACT_TEXT_SOURCE_MODE_STORAGE_KEY) || 'effective').trim().toLowerCase() || 'effective';
        let citationSourceRegistry = {};
        let citationSourceRegistrySeq = 0;
        let sessionMetaStore = loadSessionMetaStore();
        let SESSION_ID = getOrCreateSessionId();
        let shellLayoutSyncRaf = 0;
        const HEADER_SUBTITLE_DEFAULT = 'Union Document Assistant';
        const HEADER_SUBTITLE_THINKING = 'Reviewing your contract...';
        const HEADER_SUBTITLE_SPEAKING = 'Answering with citations';
        const HEADER_SUBTITLE_QUESTION = 'Needs follow-up';
        const HEADER_SUBTITLE_ERROR = 'Something went wrong';
        let karlAvatarSpeakTimer = null;
        let memberAuthMode = 'signin';
        let appliedBrandingSignature = null;

        function createNoopOnboardingController() {
            return {
                bind() {},
                show() {},
                hide() {},
                isVisible() { return false; },
                toggleDatePicker() {},
                changeYear() {},
                selectMonth() {},
                quickSelect() {},
                initDatePicker() {},
            };
        }

        async function ensureOnboardingFactoriesLoaded() {
            if (createStewardOnboardingController && createMemberOnboardingController) {
                return;
            }
            if (!onboardingFactoriesPromise) {
                onboardingFactoriesPromise = Promise.all([
                    import('./modules/steward-onboarding.js'),
                    import('./modules/member-onboarding.js'),
                ]).then(([stewardModule, memberModule]) => {
                    createStewardOnboardingController = stewardModule.createStewardOnboardingController;
                    createMemberOnboardingController = memberModule.createMemberOnboardingController;
                });
            }
            await onboardingFactoriesPromise;
        }

        function generateSessionId() {
            return 'session_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
        }

        function loadMemberAuth() {
            try {
                return JSON.parse(localStorage.getItem(MEMBER_AUTH_STORAGE_KEY) || '{}') || {};
            } catch (_) {
                return {};
            }
        }

        function notifyMemberSessionExpired(message = 'Your session expired. Please sign in again.') {
            clearMemberAuth({ preserveServerSession: true });
            openMemberAuthModal();
            window.alert(message);
        }

        function saveMemberAuth(auth) {
            memberAuth = auth || {};
            localStorage.setItem(MEMBER_AUTH_STORAGE_KEY, JSON.stringify(memberAuth));
            renderMemberAuthSummary();
            applyTenantUploadModeUI();
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

        function updateMemberAuthModeUI() {
            const isRegister = memberAuthMode === 'register';
            const eyebrow = document.getElementById('member-auth-modal-eyebrow');
            const title = document.getElementById('member-auth-modal-title');
            const copy = document.getElementById('member-auth-modal-copy');
            const submit = document.getElementById('member-auth-submit');
            const fullNameGroup = document.getElementById('member-auth-full-name-group');
            const emailGroup = document.getElementById('member-auth-email-group');
            const signInBtn = document.getElementById('member-auth-mode-signin');
            const registerBtn = document.getElementById('member-auth-mode-register');

            if (eyebrow) eyebrow.textContent = isRegister ? 'Create Account' : 'Sign In';
            if (title) title.textContent = isRegister ? 'Create your union account' : 'Access your union workspace';
            if (copy) {
                copy.textContent = isRegister
                    ? 'Create a member account for this union workspace and start with a clean conversation.'
                    : 'Use your union account to chat with uploaded documents and open supporting sources.';
            }
            if (submit) submit.textContent = isRegister ? 'Create Account' : 'Sign In';
            fullNameGroup?.classList.toggle('hidden', !isRegister);
            emailGroup?.classList.toggle('hidden', !isRegister);

            signInBtn?.classList.toggle('bg-white', !isRegister);
            signInBtn?.classList.toggle('text-slate-900', !isRegister);
            signInBtn?.classList.toggle('shadow-sm', !isRegister);
            signInBtn?.classList.toggle('text-slate-600', isRegister);
            registerBtn?.classList.toggle('bg-white', isRegister);
            registerBtn?.classList.toggle('text-slate-900', isRegister);
            registerBtn?.classList.toggle('shadow-sm', isRegister);
            registerBtn?.classList.toggle('text-slate-600', !isRegister);
        }

        function setMemberAuthMode(mode) {
            memberAuthMode = mode === 'register' ? 'register' : 'signin';
            updateMemberAuthModeUI();
        }

        function setMemberAuthModalOpen(isOpen) {
            const modal = document.getElementById('member-auth-modal');
            if (!modal) return;
            modal.classList.toggle('hidden', !isOpen);
            document.body.classList.toggle('overflow-hidden', Boolean(isOpen));
            if (isOpen) {
                window.setTimeout(() => {
                    document.getElementById('member-auth-username')?.focus();
                }, 0);
            }
        }

        function openMemberAuthModal() {
            updateMemberAuthModeUI();
            setMemberAuthModalOpen(true);
            postTelemetryEvent('bug_journey', 'member_auth_modal_opened', {
                mode: memberAuthMode,
                authenticated: Boolean(memberAuth?.authenticated),
            });
        }

        function closeMemberAuthModal() {
            setMemberAuthModalOpen(false);
            postTelemetryEvent('bug_journey', 'member_auth_modal_closed', {
                mode: memberAuthMode,
                authenticated: Boolean(memberAuth?.authenticated),
            });
        }

        async function logoutMemberSession() {
            try {
                await fetch(`${API_BASE}/api/auth/session/logout`, {
                    method: 'POST',
                    headers: authHeaders(),
                    credentials: 'same-origin',
                });
            } catch (_) {
                // Best-effort logout only.
            }
        }

        function resetMemberWorkspaceState() {
            localStorage.removeItem(SESSION_ID_STORAGE_KEY);
            localStorage.removeItem(SESSION_META_STORAGE_KEY);
            localStorage.removeItem(ACTIVE_CONTRACT_STORAGE_KEY);
            SESSION_ID = generateSessionId();
            localStorage.setItem(SESSION_ID_STORAGE_KEY, SESSION_ID);
            sessionMetaStore = {};
            ensureSessionMeta(SESSION_ID);
            activeContract = null;
            availableContracts = [];
            userProfile = null;
            articleCache = {};
            articleFirstSectionCache = {};
            articleTitles = {};
            contractBrowseGroups = [];
            currentTocSelection = { kind: null, key: null };
            citationSourceRegistry = {};
            citationSourceRegistrySeq = 0;
            resetChatMessages();
            populateContractSelects();
            updateProfileDisplay();
            hideOnboarding();
            updateInteractionLock();
        }

        function clearMemberAuth(options = {}) {
            resetMemberWorkspaceState();
            saveMemberAuth({});
            memberTrackingSettings = {
                tracking_preference: 'system_default',
                effective_policy: null,
                member_choice_available: false,
            };
            updateMemberTrackingUI();
            showOnboarding();
            updateInteractionLock();
            closeMemberAuthModal();
        }

        async function handleMemberLogout() {
            await logoutMemberSession();
            clearMemberAuth({ preserveServerSession: true });
            checkHealth();
        }

        function isTenantUploadAuthMode() {
            return Boolean(routeContext.unionSlug && memberAuth?.authenticated && (memberAuth?.user?.union_slug || memberAuth?.union_slug));
        }

        function isTenantRoute() {
            return routeContext.mode === 'tenant_member' || routeContext.mode === 'tenant_admin';
        }

        function authHeaders(extra = {}) {
            const headers = { ...extra };
            if (routeContext.unionSlug) {
                headers['X-Tenant-Slug'] = routeContext.unionSlug;
            }
            return headers;
        }

        async function postTelemetryEvent(category, eventType, metadata = {}, options = {}) {
            const payload = {
                category,
                event_type: eventType,
                route: window.location.pathname,
                metadata,
                session_id: options.sessionId || SESSION_ID || null,
                union_slug: routeContext.unionSlug || memberAuth?.user?.union_slug || memberAuth?.union_slug || null,
                surface: options.surface || 'member',
            };
            try {
                await fetch(`${API_BASE}/api/telemetry/event`, {
                    method: 'POST',
                    headers: authHeaders({ 'Content-Type': 'application/json' }),
                    credentials: 'same-origin',
                    keepalive: true,
                    body: JSON.stringify(payload),
                });
            } catch (_) {
                // Telemetry should never interrupt the main UX.
            }
        }

        async function fetchWithAuth(url, options = {}) {
            const headers = authHeaders(options.headers || {});
            const timeoutMs = Number(options.timeoutMs || 15000);
            const controller = new AbortController();
            const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
            let response;
            try {
                response = await fetch(url, {
                    ...options,
                    headers,
                    credentials: 'same-origin',
                    signal: controller.signal,
                });
            } catch (error) {
                if (error?.name === 'AbortError') {
                    throw new Error('The request took too long. Please try again.');
                }
                throw error;
            } finally {
                window.clearTimeout(timeoutId);
            }
            if (response.status === 401 && memberAuth?.authenticated) {
                notifyMemberSessionExpired();
                throw new Error('Session expired. Please sign in again.');
            }
            return response;
        }

        function getTenantUploadContract() {
            if (!isTenantRoute()) return null;
            const unionSlug = String(routeContext.unionSlug || memberAuth?.user?.union_slug || memberAuth?.union_slug || '').trim();
            if (!unionSlug) return null;
            return {
                contract_id: 'tenant-upload',
                contract_version: 'current',
                union_local_id: unionSlug,
                employer: 'Uploaded documents',
                is_tenant_upload: true,
            };
        }

        function renderMemberAuthSummary() {
            const summary = document.getElementById('member-auth-summary');
            const adminLink = document.getElementById('member-admin-link');
            const headerAccount = document.getElementById('header-account-summary');
            const headerAccountCompact = document.getElementById('header-account-compact');
            const headerAccountCompactText = document.getElementById('header-account-compact-text');
            const headerSignIn = document.getElementById('header-signin-btn');
            const headerSignOut = document.getElementById('header-signout-btn');
            const fullNameInput = document.getElementById('member-auth-full-name');
            const emailInput = document.getElementById('member-auth-email');
            const usernameInput = document.getElementById('member-auth-username');
            const passwordInput = document.getElementById('member-auth-password');
            const unionInput = document.getElementById('member-auth-union');
            const unionGroup = document.getElementById('member-auth-union-group');
            if (fullNameInput) fullNameInput.value = '';
            if (emailInput) emailInput.value = '';
            if (usernameInput) usernameInput.value = memberAuth?.username || '';
            if (passwordInput) passwordInput.value = '';
            if (unionInput) unionInput.value = memberAuth?.user?.union_slug || memberAuth?.union_slug || '';
            if (unionGroup) unionGroup.classList.toggle('hidden', Boolean(routeContext.unionSlug));
            if (adminLink) {
                const role = safeText(memberAuth?.user?.role).toLowerCase();
                const showAdminLink = role === 'union_admin' || role === 'super_admin' || role === 'steward_admin';
                adminLink.classList.toggle('hidden', !showAdminLink);
            }
            if (!summary) return;
            if (!isTenantUploadAuthMode()) {
                summary.textContent = routeContext.unionSlug
                    ? 'Sign in to access your union workspace.'
                    : 'Not signed in.';
                if (headerAccount) headerAccount.textContent = 'Not signed in';
                if (headerAccountCompactText) headerAccountCompactText.textContent = 'Not signed in';
                headerAccountCompact?.classList.add('hidden');
                headerSignIn?.classList.remove('hidden');
                headerSignOut?.classList.add('hidden');
                return;
            }
            const user = memberAuth?.user || {};
            const roleLabel = getMemberRoleLabel();
            const displayName = user.full_name || memberAuth.username || user.email || 'user';
            summary.textContent = `Signed in as ${displayName} • ${roleLabel}`;
            if (headerAccount) headerAccount.textContent = `${displayName} • ${roleLabel}`;
            if (headerAccountCompactText) headerAccountCompactText.textContent = displayName;
            headerAccountCompact?.classList.remove('hidden');
            headerSignIn?.classList.add('hidden');
            headerSignOut?.classList.remove('hidden');
        }

        function humanizeTrackingMode(mode) {
            const value = String(mode || '').trim().toLowerCase();
            if (value === 'none') return 'No optional tracking';
            if (value === 'bug_and_journey') return 'Bug and journey tracking';
            if (value === 'usage_and_ux') return 'Usage and experience tracking';
            if (value === 'both') return 'Bug, journey, and usage tracking';
            return 'Bug and journey tracking';
        }

        function humanizePrivacyMode(mode) {
            return String(mode || '').trim().toLowerCase() === 'identified'
                ? 'identified'
                : 'anonymized';
        }

        function describeMemberChoiceMode(mode) {
            const value = String(mode || '').trim().toLowerCase();
            if (value === 'none') return 'Members cannot change this workspace tracking choice.';
            if (value === 'full_opt_out') return 'Members can choose bug-only, full tracking, or opt out of optional tracking.';
            return 'Members can choose bug-only or full tracking.';
        }

        function setMemberTrackingStatus(message, tone = 'neutral') {
            const status = document.getElementById('member-tracking-status');
            if (!status) return;
            status.textContent = message || '';
            status.classList.remove('text-slate-500', 'text-green-700', 'text-red-600');
            if (tone === 'success') {
                status.classList.add('text-green-700');
            } else if (tone === 'error') {
                status.classList.add('text-red-600');
            } else {
                status.classList.add('text-slate-500');
            }
        }

        function updateMemberTrackingUI() {
            const section = document.getElementById('member-tracking-section');
            const summary = document.getElementById('member-tracking-summary');
            const select = document.getElementById('member-tracking-select');
            const saveButton = document.getElementById('member-tracking-save');
            const note = document.getElementById('member-tracking-note');
            if (!section || !summary || !select || !saveButton || !note) return;

            const bootstrapTracking = tenantBootstrap?.tracking || null;
            const effective = memberTrackingSettings.effective_policy || bootstrapTracking;
            const authenticated = Boolean(memberAuth?.authenticated);
            const choiceAvailable = Boolean(memberTrackingSettings.member_choice_available && authenticated);
            const memberChoiceMode = String(effective?.member_choice_mode || 'none').trim().toLowerCase();
            const privacyMode = humanizePrivacyMode(effective?.privacy_mode);
            const trackingMode = humanizeTrackingMode(effective?.tracking_mode);
            const rawQueryMode = String(effective?.raw_query_storage_mode || 'disabled').trim().toLowerCase();

            section.classList.toggle('hidden', !isTenantRoute());

            const summaryBits = [
                `${trackingMode} is currently enabled for this workspace.`,
                `Data is handled in ${privacyMode} mode by default.`,
                rawQueryMode === 'disabled'
                    ? 'Raw question text is not being stored for analytics.'
                    : `Raw question storage is ${rawQueryMode.replace(/_/g, ' ')}.`,
            ];
            summary.textContent = summaryBits.join(' ');

            const offOption = select.querySelector('option[value="off"]');
            if (offOption) {
                offOption.hidden = memberChoiceMode !== 'full_opt_out';
                offOption.disabled = memberChoiceMode !== 'full_opt_out';
            }

            const fullOption = select.querySelector('option[value="full"]');
            if (fullOption) {
                fullOption.hidden = memberChoiceMode === 'none';
            }

            const currentPreference = String(memberTrackingSettings.tracking_preference || 'system_default').trim().toLowerCase();
            select.value = currentPreference;
            select.disabled = !choiceAvailable;
            saveButton.disabled = !choiceAvailable;
            saveButton.classList.toggle('opacity-60', !choiceAvailable);
            saveButton.classList.toggle('cursor-not-allowed', !choiceAvailable);

            if (!authenticated) {
                note.textContent = 'Sign in to choose how optional product tracking works for your member workspace.';
                setMemberTrackingStatus('', 'neutral');
                return;
            }
            note.textContent = describeMemberChoiceMode(effective?.member_choice_mode);
            if (!choiceAvailable) {
                setMemberTrackingStatus('This workspace is using a fixed tracking policy.', 'neutral');
            }
        }

        async function loadMemberTrackingPreferences() {
            if (!isTenantRoute()) return;
            if (!memberAuth?.authenticated) {
                memberTrackingSettings = {
                    tracking_preference: 'system_default',
                    effective_policy: null,
                    member_choice_available: false,
                };
                updateMemberTrackingUI();
                return;
            }
            try {
                const res = await fetchWithAuth(`${API_BASE}/api/auth/session/preferences`, {
                    headers: authHeaders(),
                    timeoutMs: 10000,
                });
                const payload = await parseJsonResponse(res);
                if (!res.ok) {
                    throw new Error(payload.detail || payload.raw || 'Unable to load tracking settings.');
                }
                memberTrackingSettings = {
                    tracking_preference: payload.tracking_preference || 'system_default',
                    effective_policy: payload.effective_policy || null,
                    member_choice_available: Boolean(payload.member_choice_available),
                };
                updateMemberTrackingUI();
                setMemberTrackingStatus('', 'neutral');
            } catch (error) {
                console.warn('Tracking preference load failed:', error);
                updateMemberTrackingUI();
                setMemberTrackingStatus('Could not load tracking settings right now.', 'error');
            }
        }

        async function saveMemberTrackingPreference() {
            const select = document.getElementById('member-tracking-select');
            if (!select || select.disabled) return;
            setMemberTrackingStatus('Saving...', 'neutral');
            try {
                const res = await fetchWithAuth(`${API_BASE}/api/auth/session/preferences`, {
                    method: 'PUT',
                    headers: authHeaders({ 'Content-Type': 'application/json' }),
                    body: JSON.stringify({ tracking_preference: select.value }),
                    timeoutMs: 10000,
                });
                const payload = await parseJsonResponse(res);
                if (!res.ok) {
                    throw new Error(payload.detail || payload.raw || 'Unable to save tracking choice.');
                }
                memberTrackingSettings.tracking_preference = payload.tracking_preference || select.value;
                await loadMemberTrackingPreferences();
                setMemberTrackingStatus('Tracking choice saved.', 'success');
            } catch (error) {
                console.warn('Tracking preference save failed:', error);
                setMemberTrackingStatus(error.message || 'Unable to save tracking choice.', 'error');
            }
        }

        async function loadTenantBootstrap() {
            if (!routeContext.unionSlug) return null;
            const response = await fetch(`${API_BASE}/api/tenant/${encodeURIComponent(routeContext.unionSlug)}/bootstrap?page_mode=member`, {
                headers: authHeaders(),
                credentials: 'same-origin',
            });
            const payload = await response.json();
            if (!response.ok) {
                throw new Error(payload.detail || 'Unable to load tenant configuration.');
            }
            tenantBootstrap = payload;
            applyMemberBranding({ ...(payload.branding || {}), ...EMBED_THEME_OVERRIDES });
            document.title = `${payload.union?.name || routeContext.unionSlug} | Karl`;
            const subtitle = document.getElementById('header-subtitle');
            if (subtitle) {
                subtitle.textContent = `${payload.union?.name || routeContext.unionSlug} member workspace`;
            }
            const unionInput = document.getElementById('member-auth-union');
            if (unionInput) {
                unionInput.value = payload.union?.slug || routeContext.unionSlug;
                unionInput.readOnly = true;
            }
            const adminLink = document.getElementById('member-admin-link');
            if (adminLink && payload.routes?.admin) {
                adminLink.href = payload.routes.admin;
            }
            updateMemberTrackingUI();
            return payload;
        }

        function normalizeHexColor(value, fallback) {
            const raw = String(value || '').trim();
            if (/^#[0-9a-f]{6}$/i.test(raw)) return raw;
            if (/^#[0-9a-f]{3}$/i.test(raw)) {
                return `#${raw[1]}${raw[1]}${raw[2]}${raw[2]}${raw[3]}${raw[3]}`;
            }
            return fallback;
        }

        function shiftHexColor(hex, amount) {
            const normalized = normalizeHexColor(hex, '#0D3B54').slice(1);
            const next = [0, 2, 4].map((offset) => {
                const channel = parseInt(normalized.slice(offset, offset + 2), 16);
                const shifted = Math.max(0, Math.min(255, channel + amount));
                return shifted.toString(16).padStart(2, '0');
            });
            return `#${next.join('')}`;
        }

        function applyMemberBranding(branding = {}) {
            const primary = normalizeHexColor(branding.theme_color, '#0D3B54');
            const accent = normalizeHexColor(branding.accent_color, '#D4A029');
            const surface = normalizeHexColor(branding.surface_tint, '#F8FAFC');
            const heading = String(branding.welcome_heading || '').trim();
            const signature = JSON.stringify({ primary, accent, surface, heading });
            if (signature === appliedBrandingSignature) return;
            appliedBrandingSignature = signature;

            document.documentElement.style.setProperty('--karl-brand-primary', primary);
            document.documentElement.style.setProperty('--karl-brand-primary-deep', shiftHexColor(primary, -18));
            document.documentElement.style.setProperty('--karl-brand-primary-soft', shiftHexColor(primary, 24));
            document.documentElement.style.setProperty('--karl-brand-accent', accent);
            document.documentElement.style.setProperty('--karl-brand-accent-soft', shiftHexColor(accent, 22));
            document.documentElement.style.setProperty('--karl-brand-surface', surface);

            let styleEl = document.getElementById('karl-runtime-branding');
            if (!styleEl) {
                styleEl = document.createElement('style');
                styleEl.id = 'karl-runtime-branding';
                document.head.appendChild(styleEl);
            }
            styleEl.textContent = `
                #main-header {
                    background:
                        radial-gradient(ellipse 80% 150% at 38% 50%, color-mix(in srgb, var(--karl-brand-primary-soft) 38%, transparent) 0%, transparent 50%),
                        linear-gradient(-45deg, var(--karl-brand-primary-deep), var(--karl-brand-primary), var(--karl-brand-primary-soft), var(--karl-brand-primary), var(--karl-brand-primary-deep)) !important;
                }
                .bg-ufcw-blue,
                #chat-input-area button,
                #settings-profile-save,
                #member-auth-submit {
                    background-color: var(--karl-brand-primary) !important;
                }
                .hover\\:bg-ufcw-blue-mid:hover,
                #chat-input-area button:hover,
                #settings-profile-save:hover,
                #member-auth-submit:hover {
                    background-color: var(--karl-brand-primary-soft) !important;
                }
                .text-ufcw-blue,
                #chat-profile-edit-btn,
                #tab-contract:hover,
                #tab-settings:hover {
                    color: var(--karl-brand-primary) !important;
                }
                .bg-ufcw-gold,
                .pulse-dot {
                    background-color: var(--karl-brand-accent) !important;
                }
                .text-ufcw-gold-light,
                .dark .text-ufcw-gold-light {
                    color: var(--karl-brand-accent-soft) !important;
                }
                #content-settings,
                body {
                    background-color: var(--karl-brand-surface);
                }
                #member-auth-submit:focus,
                #chat-input-area button:focus,
                #user-input:focus {
                    outline-color: var(--karl-brand-accent) !important;
                }
            `;

            const welcomeCopy = document.getElementById('chat-welcome-copy');
            if (welcomeCopy && heading) {
                welcomeCopy.textContent = heading;
            }
        }

        function showTenantPageFailure(message) {
            const subtitle = document.getElementById('header-subtitle');
            if (subtitle) {
                subtitle.textContent = message;
            }
            const headerAccount = document.getElementById('header-account-summary');
            if (headerAccount) {
                headerAccount.textContent = 'Workspace unavailable';
            }
            const queryInput = document.getElementById('query');
            if (queryInput) {
                queryInput.disabled = true;
                queryInput.placeholder = message;
            }
            const submitButton = document.getElementById('sendButton');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.classList.add('opacity-60', 'cursor-not-allowed');
            }
            revealAppShell();
        }

        function getMemberRoleLabel() {
            const rawRole = safeText(memberAuth?.user?.role || userProfile?.role || '');
            if (!rawRole) return 'Union member';
            return rawRole
                .split('_')
                .filter(Boolean)
                .map(part => toTitleCase(part))
                .join(' ');
        }

        function applyTenantUploadModeUI() {
            const tenantMode = isTenantUploadAuthMode();
            const note = document.getElementById('profile-settings-note');
            const contractSelect = document.getElementById('settings-contract');
            const classificationSelect = document.getElementById('settings-classification');
            const hireInput = document.getElementById('settings-hire-date');
            const saveButton = document.getElementById('settings-profile-save');
            const tenantProfileSummary = document.getElementById('tenant-profile-summary');
            const legacyProfileFields = document.getElementById('legacy-profile-fields');
            const chatProfileEditBtn = document.getElementById('chat-profile-edit-btn');
            const profileSettingsSection = document.getElementById('profile-settings-section');
            const employmentInputs = document.querySelectorAll('input[name="settings_employment"]');

            if (note) {
                if (tenantMode) {
                    note.textContent = 'Your union sign-in already determines which documents this workspace uses.';
                    note.classList.remove('hidden');
                } else {
                    note.textContent = '';
                    note.classList.add('hidden');
                }
            }

            if (tenantProfileSummary) {
                tenantProfileSummary.classList.toggle('hidden', !tenantMode);
            }
            if (legacyProfileFields) {
                legacyProfileFields.classList.toggle('hidden', tenantMode);
            }
            if (profileSettingsSection) {
                profileSettingsSection.classList.toggle('hidden', tenantMode);
            }
            if (chatProfileEditBtn) {
                chatProfileEditBtn.innerHTML = tenantMode
                    ? `
                        <svg class="w-3.5 h-3.5 md:w-4 md:h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v12m6-6H6"/>
                        </svg>
                        Workspace
                    `
                    : `
                        <svg class="w-3.5 h-3.5 md:w-4 md:h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"/>
                        </svg>
                        Edit
                    `;
            }

            [contractSelect, classificationSelect, hireInput].forEach(field => {
                if (!field) return;
                field.disabled = tenantMode;
            });
            employmentInputs.forEach(field => {
                field.disabled = tenantMode;
            });

            if (saveButton) {
                saveButton.textContent = tenantMode ? 'Profile managed by union sign-in' : 'Save Profile';
                saveButton.disabled = tenantMode;
                saveButton.classList.toggle('opacity-60', tenantMode);
                saveButton.classList.toggle('cursor-not-allowed', tenantMode);
            }
        }

        async function refreshMemberAuth() {
            try {
                const res = await fetch(`${API_BASE}/api/auth/session/me`, {
                    headers: authHeaders(),
                    credentials: 'same-origin',
                });
                if (res.status === 401) {
                    notifyMemberSessionExpired();
                    return false;
                }
                if (!res.ok) throw new Error(`Auth refresh failed: ${res.status}`);
                const payload = await res.json();
                if (!payload.authenticated) {
                    saveMemberAuth({});
                    return false;
                }
                saveMemberAuth({
                    ...memberAuth,
                    authenticated: true,
                    user: {
                        ...(memberAuth.user || {}),
                        role: payload.role,
                        union_id: payload.union_id,
                        union_slug: payload.union_slug,
                        email: payload.email,
                        full_name: payload.full_name,
                    },
                    union_slug: payload.union_slug,
                });
                if (payload.tracking) {
                    memberTrackingSettings.effective_policy = payload.tracking;
                }
                updateMemberTrackingUI();
                return true;
            } catch (error) {
                console.warn('Member auth refresh failed:', error);
                saveMemberAuth({});
                memberTrackingSettings = {
                    tracking_preference: 'system_default',
                    effective_policy: null,
                    member_choice_available: false,
                };
                updateMemberTrackingUI();
                return false;
            }
        }

        async function loginMemberAuth(event) {
            event?.preventDefault?.();
            const username = document.getElementById('member-auth-username')?.value.trim() || '';
            const password = document.getElementById('member-auth-password')?.value || '';
            const fullName = document.getElementById('member-auth-full-name')?.value.trim() || '';
            const email = document.getElementById('member-auth-email')?.value.trim() || '';
            const unionSlug = routeContext.unionSlug || document.getElementById('member-auth-union')?.value.trim() || '';
            const endpoint = memberAuthMode === 'register'
                ? `${API_BASE}/api/auth/session/register`
                : `${API_BASE}/api/auth/session/login`;
            const payloadBody = memberAuthMode === 'register'
                ? {
                    username,
                    password,
                    full_name: fullName,
                    email,
                    union_slug: unionSlug || null,
                }
                : {
                    username,
                    password,
                    union_slug: unionSlug || null,
                };
            const res = await fetch(endpoint, {
                method: 'POST',
                headers: authHeaders({ 'Content-Type': 'application/json' }),
                credentials: 'same-origin',
                body: JSON.stringify(payloadBody),
            });
            const payload = await parseJsonResponse(res);
            if (!res.ok) {
                throw new Error(payload.detail || payload.raw || 'Unable to sign in.');
            }
            resetMemberWorkspaceState();
            saveMemberAuth({
                authenticated: true,
                username,
                union_slug: payload.user?.union_slug || unionSlug,
                user: payload.user || {},
            });
            if (payload.tracking) {
                memberTrackingSettings.effective_policy = payload.tracking;
            }
            closeMemberAuthModal();
            localStorage.setItem(ACTIVE_CONTRACT_STORAGE_KEY, 'tenant-upload');
            const tenantContract = getTenantUploadContract();
            if (tenantContract) {
                availableContracts = [tenantContract];
                activeContract = tenantContract;
                populateContractSelects();
                updateProfileDisplay();
                syncSettingsForm();
                hideOnboarding();
                updateInteractionLock();
            }
            await loadMemberTrackingPreferences();
        }

        function loadSessionMetaStore() {
            try {
                return JSON.parse(localStorage.getItem(SESSION_META_STORAGE_KEY) || '{}') || {};
            } catch (_) {
                return {};
            }
        }

        function saveSessionMetaStore() {
            localStorage.setItem(SESSION_META_STORAGE_KEY, JSON.stringify(sessionMetaStore));
        }

        function ensureSessionMeta(sessionId) {
            if (!sessionMetaStore[sessionId]) {
                sessionMetaStore[sessionId] = {
                    submitted_count: 0,
                    contract_id: null,
                    classification: null,
                    created_at: new Date().toISOString()
                };
                saveSessionMetaStore();
            }
            return sessionMetaStore[sessionId];
        }

        function getOrCreateSessionId() {
            let sid = localStorage.getItem(SESSION_ID_STORAGE_KEY);
            if (!sid) {
                sid = generateSessionId();
                localStorage.setItem(SESSION_ID_STORAGE_KEY, sid);
            }
            ensureSessionMeta(sid);
            return sid;
        }

        function getOnboardingFlowPreference() {
            const saved = String(localStorage.getItem(ONBOARDING_FLOW_STORAGE_KEY) || '').trim();
            if (saved === 'steward' || saved === 'member') return saved;
            return null;
        }

        function setOnboardingFlowPreference(flow) {
            const normalized = String(flow || '').trim().toLowerCase();
            if (normalized === 'steward' || normalized === 'member') {
                localStorage.setItem(ONBOARDING_FLOW_STORAGE_KEY, normalized);
            }
        }

        function resetChatMessages() {
            const container = document.getElementById('chat-container');
            if (!container) return;
            container.querySelectorAll('.chat-message').forEach(node => node.remove());
            hideLoading();
            clearKarlAvatarSpeakTimer();
            stopThinking();
            setKarlAvatarState('idle');
            setHeaderSubtitle(HEADER_SUBTITLE_DEFAULT);
        }

        function startNewChatSession() {
            SESSION_ID = generateSessionId();
            localStorage.setItem(SESSION_ID_STORAGE_KEY, SESSION_ID);
            ensureSessionMeta(SESSION_ID);
            resetChatMessages();
        }

        function startNewConversation() {
            if (hasSubmittedChatText()) {
                const confirmed = confirm('Start a new conversation and clear the current chat context?');
                if (!confirmed) return;
            }
            startNewChatSession();
            postTelemetryEvent('bug_journey', 'conversation_reset', {
                trigger: 'new_conversation',
            }, { sessionId: SESSION_ID });
            const input = document.getElementById('user-input');
            if (input) {
                input.value = '';
                input.placeholder = isTenantUploadAuthMode()
                    ? 'Ask a new question about your union documents...'
                    : 'Ask about your agreement...';
                input.focus();
            }
        }

        function focusQuestionInput(prefill = '', options = {}) {
            const input = document.getElementById('user-input');
            if (!input) return;
            if (!hasCompleteProfileContext()) {
                if (!memberAuth?.authenticated) {
                    openMemberAuthModal();
                } else {
                    showOnboarding();
                }
                return;
            }
            if (prefill) {
                input.value = prefill;
            }
            input.placeholder = options.followUp
                ? 'Ask a follow-up about the last answer...'
                : (isTenantUploadAuthMode() ? 'Ask a question about your union documents...' : 'Ask about your agreement...');
            input.focus();
        }

        function hasSubmittedChatText() {
            return (ensureSessionMeta(SESSION_ID).submitted_count || 0) > 0;
        }

        function hasCompleteProfileContext() {
            if (isTenantUploadAuthMode()) {
                return Boolean(getTenantUploadContract());
            }
            const contractId = String(userProfile?.contract_id || activeContract?.contract_id || '').trim();
            return Boolean(contractId);
        }

        function getCachedProfileContext() {
            const meta = ensureSessionMeta(SESSION_ID);
            const contractId = String(meta?.contract_id || '').trim();
            const classification = String(meta?.classification || '').trim();
            if (!contractId) return null;
            return { contract_id: contractId, classification: classification };
        }

        async function hydrateProfileFromCache() {
            const cached = getCachedProfileContext();
            if (!cached) return false;

            userProfile = {
                ...(userProfile || {}),
                contract_id: cached.contract_id,
                classification: cached.classification,
                classification_display: userProfile?.classification_display || cached.classification
            };

            if (cached.contract_id) {
                setActiveContract(cached.contract_id, { persist: true, refreshViewer: false, preserveClassification: false });
                await loadClassificationsForContract(cached.contract_id, { preserveSelection: false });
            }

            updateProfileDisplay();
            hideOnboarding();
            return true;
        }

        function updateInteractionLock() {
            const locked = !hasCompleteProfileContext();
            const input = document.getElementById('user-input');
            const sendBtn = document.getElementById('send-btn');
            const requiresSignIn = !memberAuth?.authenticated;
            if (input) {
                input.disabled = locked;
                input.placeholder = locked
                    ? (requiresSignIn
                        ? 'Sign in to start chat...'
                        : (isTenantUploadAuthMode()
                            ? 'Your union documents will appear here after upload and processing...'
                            : 'Select a contract in onboarding to start chat...'))
                    : (isTenantUploadAuthMode() ? 'Ask a question about your union documents...' : 'Ask about your agreement...');
            }
            if (sendBtn) sendBtn.disabled = locked;

            document.querySelectorAll('[data-requires-profile="true"]').forEach(btn => {
                btn.disabled = locked;
            });
            // The contract explorer works on tenant routes now that uploaded
            // contracts keep their article hierarchy and there are member
            // browse endpoints. It stays hidden only when the workspace has
            // no browsable contract (see refreshTenantContractLibrary).
            const contractTab = document.getElementById('tab-contract');
            if (contractTab && isTenantRoute() && !tenantContractLibrary.length) {
                contractTab.classList.add('hidden');
            }
        }

        function markChatSubmitted() {
            const meta = ensureSessionMeta(SESSION_ID);
            meta.submitted_count = (meta.submitted_count || 0) + 1;
            meta.updated_at = new Date().toISOString();
            saveSessionMetaStore();
        }

        function getActiveContract() {
            if (isTenantUploadAuthMode()) {
                return getTenantUploadContract();
            }
            return activeContract;
        }

        let tenantContractLibrary = [];
        let tenantBrowseContractId = null;

        async function refreshTenantContractLibrary() {
            if (!isTenantRoute()) return;
            try {
                const res = await fetchWithAuth(`${API_BASE}/api/member/contracts`);
                if (!res.ok) throw new Error(`contracts ${res.status}`);
                const payload = await res.json();
                tenantContractLibrary = Array.isArray(payload?.contracts) ? payload.contracts : [];
                // An invite pinned to one agreement must not offer the other.
                tenantBrowseContractId = safeText(payload?.pinned_contract_id)
                    || (tenantContractLibrary[0]?.contract_id ?? null);
            } catch (_) {
                tenantContractLibrary = [];
                tenantBrowseContractId = null;
            }
            const contractTab = document.getElementById('tab-contract');
            if (contractTab) {
                contractTab.classList.toggle('hidden', !tenantContractLibrary.length);
            }
        }

        function getBrowseContractId() {
            if (isTenantRoute()) return tenantBrowseContractId;
            return getActiveContractId();
        }

        function getActiveContractId() {
            return getActiveContract()?.contract_id || null;
        }

        function getContractQueryString() {
            const contractId = getActiveContractId();
            if (!contractId) return '';
            return `?contract_id=${encodeURIComponent(contractId)}`;
        }

        function normalizeContractPdfSourceMode(value) {
            const mode = String(value || '').trim().toLowerCase();
            if (mode === 'base' || mode === 'moa') return mode;
            return 'effective';
        }

        function normalizeContractViewerMode(value) {
            return String(value || '').trim().toLowerCase() === 'original_pdf'
                ? 'original_pdf'
                : 'effective_text';
        }

        function isOriginalPdfViewMode() {
            return normalizeContractViewerMode(contractViewerMode) === 'original_pdf';
        }

        function saveContractPdfSourceMode(mode) {
            contractPdfSourceMode = normalizeContractPdfSourceMode(mode);
            localStorage.setItem(CONTRACT_PDF_SOURCE_MODE_STORAGE_KEY, contractPdfSourceMode);
        }

        function saveContractViewerMode(mode) {
            contractViewerMode = normalizeContractViewerMode(mode);
            localStorage.setItem(CONTRACT_VIEW_MODE_STORAGE_KEY, contractViewerMode);
        }

        function normalizeContractTextSourceMode(value) {
            const mode = String(value || '').trim().toLowerCase();
            return mode === 'base' ? 'base' : 'effective';
        }

        function saveContractTextSourceMode(mode) {
            contractTextSourceMode = normalizeContractTextSourceMode(mode);
            localStorage.setItem(CONTRACT_TEXT_SOURCE_MODE_STORAGE_KEY, contractTextSourceMode);
            const select = document.getElementById('contract-text-source-mode-select');
            if (select) select.value = contractTextSourceMode;
            updateContractTextPanelChrome();
        }

        function _sameTextTarget(a, b) {
            const aKind = safeText(a?.kind).toLowerCase();
            const bKind = safeText(b?.kind).toLowerCase();
            const aKey = safeText(a?.key);
            const bKey = safeText(b?.key);
            return !!aKind && !!aKey && aKind === bKind && aKey === bKey;
        }

        function _contractTextTargetLabel(target = null) {
            const kind = safeText(target?.kind).toLowerCase();
            const key = safeText(target?.key);
            if (kind === 'article') {
                const articleNum = toPositiveIntOrNull(target?.articleNum) || (key.match(/^article:(\d+)$/i)?.[1] ? toPositiveIntOrNull(key.match(/^article:(\d+)$/i)[1]) : null);
                return articleNum ? `Article ${articleNum}` : 'Article';
            }
            if (kind === 'appendix') return `Appendix (${key || 'item'})`;
            if (kind === 'lou') return `LOU ${key.replace(/^lou:/i, '')}`;
            if (kind === 'loa') return `LOA ${key.replace(/^loa:/i, '')}`;
            return key || 'Contract Item';
        }

        function _contractTextPayloadSections(payload) {
            const sections = Array.isArray(payload?.sections) ? payload.sections : [];
            return sections.map((section, idx) => ({
                citation: safeText(section?.citation) || `Part ${idx + 1}`,
                content: String(section?.content || ''),
            }));
        }

        function _contractTextPayloadComparableText(payload) {
            return _contractTextPayloadSections(payload)
                .map((section) => `${section.citation}\n${section.content}`.trim())
                .join('\n\n')
                .trim();
        }

        function _normalizeCompareLine(line) {
            return String(line || '').replace(/\s+/g, ' ').trim();
        }

        function _truncateCompareTextForDiff(text, maxChars = 2400) {
            const raw = String(text || '').trim();
            if (raw.length <= maxChars) {
                return { text: raw, truncated: false };
            }
            return {
                text: `${raw.slice(0, maxChars)}\n...[truncated for diff preview]`,
                truncated: true,
            };
        }

        function _tokenizeInlineDiffText(text) {
            const raw = String(text || '');
            const tokens = raw.match(/\w+|\s+|[^\w\s]/g);
            return Array.isArray(tokens) && tokens.length ? tokens : [raw];
        }

        function _inlineDiffTokenLcs(aTokens, bTokens) {
            const a = Array.isArray(aTokens) ? aTokens : [];
            const b = Array.isArray(bTokens) ? bTokens : [];
            const n = a.length;
            const m = b.length;
            const matrix = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0));
            for (let i = n - 1; i >= 0; i -= 1) {
                for (let j = m - 1; j >= 0; j -= 1) {
                    if (a[i] === b[j]) {
                        matrix[i][j] = matrix[i + 1][j + 1] + 1;
                    } else {
                        matrix[i][j] = Math.max(matrix[i + 1][j], matrix[i][j + 1]);
                    }
                }
            }
            const aMarks = Array(n).fill(false);
            const bMarks = Array(m).fill(false);
            let i = 0;
            let j = 0;
            while (i < n && j < m) {
                if (a[i] === b[j]) {
                    aMarks[i] = true;
                    bMarks[j] = true;
                    i += 1;
                    j += 1;
                } else if (matrix[i + 1][j] >= matrix[i][j + 1]) {
                    i += 1;
                } else {
                    j += 1;
                }
            }
            return { aMarks, bMarks };
        }

        function _renderInlineDiffTokenHtml(tokens, keepMarks, removed = false) {
            const out = [];
            for (let i = 0; i < tokens.length; i += 1) {
                const token = String(tokens[i] || '');
                const isKept = !!keepMarks[i];
                const escaped = escapeHtml(token || ' ');
                if (isKept || !token.trim()) {
                    out.push(escaped || ' ');
                    continue;
                }
                const cls = removed
                    ? 'bg-rose-200/80 dark:bg-rose-700/50'
                    : 'bg-emerald-200/80 dark:bg-emerald-700/50';
                out.push(`<span class="${cls} rounded-sm">${escaped}</span>`);
            }
            return out.join('');
        }

        function _renderInlineTokenDiffPair(effectiveLine, baseLine) {
            const maxChars = 700;
            const effText = String(effectiveLine || '');
            const baseText = String(baseLine || '');
            const effTrunc = effText.length > maxChars;
            const baseTrunc = baseText.length > maxChars;
            const eff = effTrunc ? `${effText.slice(0, maxChars)}…` : effText;
            const base = baseTrunc ? `${baseText.slice(0, maxChars)}…` : baseText;
            const effTokens = _tokenizeInlineDiffText(eff);
            const baseTokens = _tokenizeInlineDiffText(base);
            const { aMarks, bMarks } = _inlineDiffTokenLcs(effTokens, baseTokens);
            return {
                effectiveHtml: _renderInlineDiffTokenHtml(effTokens, aMarks, false),
                baseHtml: _renderInlineDiffTokenHtml(baseTokens, bMarks, true),
                truncated: effTrunc || baseTrunc,
            };
        }

        function _buildContractTextLineDiffRows(effectiveText, baseText) {
            const effPreview = _truncateCompareTextForDiff(effectiveText);
            const basePreview = _truncateCompareTextForDiff(baseText);
            const effLinesRaw = effPreview.text.split('\n');
            const baseLinesRaw = basePreview.text.split('\n');
            const maxLines = 160;
            const effLines = effLinesRaw.slice(0, maxLines);
            const baseLines = baseLinesRaw.slice(0, maxLines);
            const effNorm = effLines.map(_normalizeCompareLine);
            const baseNorm = baseLines.map(_normalizeCompareLine);

            const n = effLines.length;
            const m = baseLines.length;
            const matrix = Array.from({ length: n + 1 }, () => Array(m + 1).fill(0));
            for (let i = n - 1; i >= 0; i -= 1) {
                for (let j = m - 1; j >= 0; j -= 1) {
                    if (effNorm[i] === baseNorm[j]) {
                        matrix[i][j] = matrix[i + 1][j + 1] + 1;
                    } else {
                        matrix[i][j] = Math.max(matrix[i + 1][j], matrix[i][j + 1]);
                    }
                }
            }

            const rows = [];
            let i = 0;
            let j = 0;
            let unchangedCount = 0;
            let effectiveOnlyCount = 0;
            let baseOnlyCount = 0;
            while (i < n && j < m) {
                if (effNorm[i] === baseNorm[j]) {
                    rows.push({ type: 'equal', effective: effLines[i], base: baseLines[j] });
                    unchangedCount += 1;
                    i += 1;
                    j += 1;
                    continue;
                }
                if (matrix[i + 1][j] >= matrix[i][j + 1]) {
                    rows.push({ type: 'effective_only', effective: effLines[i], base: '' });
                    effectiveOnlyCount += 1;
                    i += 1;
                } else {
                    rows.push({ type: 'base_only', effective: '', base: baseLines[j] });
                    baseOnlyCount += 1;
                    j += 1;
                }
            }
            while (i < n) {
                rows.push({ type: 'effective_only', effective: effLines[i], base: '' });
                effectiveOnlyCount += 1;
                i += 1;
            }
            while (j < m) {
                rows.push({ type: 'base_only', effective: '', base: baseLines[j] });
                baseOnlyCount += 1;
                j += 1;
            }

            return {
                rows,
                unchangedCount,
                effectiveOnlyCount,
                baseOnlyCount,
                effectiveTruncated: effPreview.truncated || effLinesRaw.length > maxLines,
                baseTruncated: basePreview.truncated || baseLinesRaw.length > maxLines,
            };
        }

        function _renderContractTextLineDiff(diff) {
            const rows = Array.isArray(diff?.rows) ? diff.rows : [];
            if (!rows.length) {
                return '<p class="text-xs text-slate-500 dark:text-slate-400">No text available for diff.</p>';
            }

            const limitedRows = rows.slice(0, 140);
            let sawInlineTruncation = false;
            const rowHtmlParts = [];
            for (let idx = 0; idx < limitedRows.length; idx += 1) {
                const row = limitedRows[idx];
                const type = safeText(row?.type).toLowerCase();
                const next = limitedRows[idx + 1] || null;
                const nextType = safeText(next?.type).toLowerCase();

                const canPair =
                    (type === 'effective_only' && nextType === 'base_only')
                    || (type === 'base_only' && nextType === 'effective_only');

                if (canPair) {
                    const effLine = type === 'effective_only' ? String(row?.effective || '') : String(next?.effective || '');
                    const baseLine = type === 'base_only' ? String(row?.base || '') : String(next?.base || '');
                    const rendered = _renderInlineTokenDiffPair(effLine, baseLine);
                    if (rendered.truncated) sawInlineTruncation = true;
                    rowHtmlParts.push(`
                    <div class="grid grid-cols-[20px_1fr_1fr] gap-1 items-start">
                        <div class="text-[10px] font-bold text-amber-600 dark:text-amber-300 pt-1 text-center">~</div>
                        <div class="m-0 p-1.5 text-[11px] leading-relaxed whitespace-pre-wrap break-words rounded border border-slate-200 dark:border-slate-700 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-800 dark:text-emerald-200">${rendered.effectiveHtml || ' '}</div>
                        <div class="m-0 p-1.5 text-[11px] leading-relaxed whitespace-pre-wrap break-words rounded border border-slate-200 dark:border-slate-700 bg-rose-50 dark:bg-rose-900/20 text-rose-800 dark:text-rose-200">${rendered.baseHtml || ' '}</div>
                    </div>
                    `);
                    idx += 1;
                    continue;
                }

                let effClass = 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200';
                let baseClass = 'bg-white dark:bg-slate-900 text-slate-700 dark:text-slate-200';
                let marker = '=';
                let markerClass = 'text-slate-400';
                if (type === 'effective_only') {
                    effClass = 'bg-emerald-50 dark:bg-emerald-900/20 text-emerald-800 dark:text-emerald-200';
                    marker = '+';
                    markerClass = 'text-emerald-600 dark:text-emerald-300';
                } else if (type === 'base_only') {
                    baseClass = 'bg-rose-50 dark:bg-rose-900/20 text-rose-800 dark:text-rose-200';
                    marker = '-';
                    markerClass = 'text-rose-600 dark:text-rose-300';
                }
                const effText = escapeHtml(String(row?.effective || '') || ' ');
                const baseText = escapeHtml(String(row?.base || '') || ' ');
                rowHtmlParts.push(`
                    <div class="grid grid-cols-[20px_1fr_1fr] gap-1 items-start">
                        <div class="text-[10px] font-bold ${markerClass} pt-1 text-center">${marker}</div>
                        <pre class="m-0 p-1.5 text-[11px] leading-relaxed whitespace-pre-wrap break-words rounded border border-slate-200 dark:border-slate-700 ${effClass}">${effText}</pre>
                        <pre class="m-0 p-1.5 text-[11px] leading-relaxed whitespace-pre-wrap break-words rounded border border-slate-200 dark:border-slate-700 ${baseClass}">${baseText}</pre>
                    </div>
                `);
            }
            const rowHtml = rowHtmlParts.join('');

            const truncatedNote = (rows.length > 140 || diff?.effectiveTruncated || diff?.baseTruncated || sawInlineTruncation)
                ? '<p class="mt-2 text-[10px] text-slate-500 dark:text-slate-400">Diff preview truncated for UI performance (long lines and/or long targets are clipped).</p>'
                : '';

            return `
                <div class="grid grid-cols-[20px_1fr_1fr] gap-1 mb-1">
                    <div></div>
                    <div class="text-[10px] font-semibold text-slate-500 dark:text-slate-400 px-1">Effective</div>
                    <div class="text-[10px] font-semibold text-slate-500 dark:text-slate-400 px-1">Previous/Base</div>
                </div>
                <div class="space-y-1">${rowHtml}</div>
                ${truncatedNote}
            `;
        }

        function _rememberContractTextPayload(target, payload, sourceMode) {
            const normalizedTarget = target && typeof target === 'object' ? { ...target } : null;
            const mode = normalizeContractTextSourceMode(sourceMode);
            if (!normalizedTarget || !payload || typeof payload !== 'object') {
                updateContractTextPanelChrome();
                return;
            }
            if (!_sameTextTarget(lastContractTextContext?.target, normalizedTarget)) {
                lastContractTextContext = {
                    target: normalizedTarget,
                    payloads: { effective: null, base: null },
                };
                contractTextCompareOpen = false;
            }
            if (!lastContractTextContext?.payloads) {
                lastContractTextContext = {
                    target: normalizedTarget,
                    payloads: { effective: null, base: null },
                };
            }
            lastContractTextContext.payloads[mode] = payload;
            updateContractTextPanelChrome();
            renderContractTextComparePanel();
        }

        function _clearContractTextContext() {
            lastContractTextContext = null;
            contractTextCompareOpen = false;
            updateContractTextPanelChrome();
            renderContractTextComparePanel();
        }

        function updateContractTextPanelChrome() {
            const provenanceEl = document.getElementById('contract-text-provenance');
            const compareBtn = document.getElementById('contract-text-compare-btn');
            const compareHint = document.getElementById('contract-text-compare-hint');
            const target = lastContractTextContext?.target || null;
            const history = getActiveContractHistory();
            const sourceMode = normalizeContractTextSourceMode(contractTextSourceMode);
            const sourceLabel = sourceMode === 'base' ? 'Previous/Base Text' : 'Materialized Effective Text';
            const viewerLabel = isOriginalPdfViewMode() ? 'Original PDF reader' : 'Primary reader';

            if (provenanceEl) {
                const bits = [`Viewer: ${viewerLabel}`, `Text Source: ${sourceLabel}`];
                if (target) {
                    bits.push(`Target: ${_contractTextTargetLabel(target)}`);
                }
                if (history?.effective_version_id) {
                    bits.push(`Snapshot: ${safeText(history.effective_version_id)}`);
                }
                if (Number.isFinite(Number(history?.patch_count))) {
                    bits.push(`Amendments: ${Number(history.patch_count)}`);
                }
                provenanceEl.textContent = bits.join(' | ');
            }

            if (compareBtn) {
                const hasTarget = !!target;
                const compareModeLabel = sourceMode === 'base' ? 'Effective' : 'Base';
                compareBtn.textContent = contractTextCompareOpen ? 'Hide Compare' : `Compare to ${compareModeLabel}`;
                compareBtn.disabled = !hasTarget;
                compareBtn.classList.toggle('opacity-50', !hasTarget);
                compareBtn.classList.toggle('cursor-not-allowed', !hasTarget);
            }
            if (compareHint) {
                if (!target) {
                    compareHint.textContent = '';
                } else if (history?.patch_count > 0) {
                    compareHint.textContent = 'Compare current target text across effective and base artifacts';
                } else {
                    compareHint.textContent = 'No amendments in history; compare may be identical';
                }
            }
        }

        function renderContractTextComparePanel() {
            const panel = document.getElementById('contract-text-compare-panel');
            const body = document.getElementById('contract-text-compare-body');
            if (!panel || !body) return;

            if (!contractTextCompareOpen || !lastContractTextContext?.target) {
                panel.classList.add('hidden');
                body.innerHTML = '';
                return;
            }

            const payloads = lastContractTextContext.payloads || {};
            const effectivePayload = payloads.effective || null;
            const basePayload = payloads.base || null;
            if (!effectivePayload || !basePayload) {
                panel.classList.remove('hidden');
                body.innerHTML = '<p class="text-xs text-slate-500 dark:text-slate-400">Loading comparison...</p>';
                return;
            }

            const effectiveText = _contractTextPayloadComparableText(effectivePayload);
            const baseText = _contractTextPayloadComparableText(basePayload);
            const norm = (s) => String(s || '').replace(/\s+/g, ' ').trim();
            const isSame = norm(effectiveText) === norm(baseText);
            const diff = _buildContractTextLineDiffRows(effectiveText, baseText);

            panel.classList.remove('hidden');
            body.innerHTML = `
                <div class="mb-2 text-xs ${isSame ? 'text-emerald-700 dark:text-emerald-300' : 'text-amber-700 dark:text-amber-300'}">
                    ${isSame ? 'No textual difference detected in available chunk artifacts for this target.' : 'Difference detected between effective and base text artifacts for this target.'}
                </div>
                <div class="mb-2 flex flex-wrap gap-1.5">
                    <span class="inline-flex items-center rounded border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] text-emerald-800 dark:bg-emerald-900/20 dark:text-emerald-200">Effective-only lines: ${Number(diff.effectiveOnlyCount || 0)}</span>
                    <span class="inline-flex items-center rounded border border-rose-200 bg-rose-50 px-2 py-0.5 text-[10px] text-rose-800 dark:bg-rose-900/20 dark:text-rose-200">Base-only lines: ${Number(diff.baseOnlyCount || 0)}</span>
                    <span class="inline-flex items-center rounded border border-slate-200 bg-white px-2 py-0.5 text-[10px] text-slate-700 dark:bg-slate-900 dark:text-slate-200">Unchanged lines: ${Number(diff.unchangedCount || 0)}</span>
                </div>
                <div class="rounded border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-2">
                    ${_renderContractTextLineDiff(diff)}
                </div>
            `;
        }

        function isMoaPdfName(name) {
            return String(name || '').toLowerCase().includes('moa');
        }

        function safeText(value) {
            return String(value || '').trim();
        }

        function registerCitationSourceRecord(source) {
            const row = source && typeof source === 'object' ? source : null;
            if (!row) return null;
            citationSourceRegistrySeq += 1;
            const key = `src_${citationSourceRegistrySeq}`;
            citationSourceRegistry[key] = JSON.parse(JSON.stringify(row));
            return key;
        }

        function getCitationSourceRecord(key) {
            const normalized = safeText(key);
            if (!normalized) return null;
            return citationSourceRegistry[normalized] || null;
        }

        function escapeJsSingleQuoted(value) {
            return String(value || '')
                .replace(/\\/g, "\\\\")
                .replace(/'/g, "\\'")
                .replace(/\n/g, ' ');
        }

        function normalizeRoleClarificationPayload(payload) {
            if (!payload || typeof payload !== 'object' || payload.needs_clarification !== true) {
                return null;
            }
            const options = Array.isArray(payload.options)
                ? payload.options
                    .map((opt) => ({
                        value: safeText(opt?.value),
                        label: safeText(opt?.label) || safeText(opt?.value),
                        role_family: safeText(opt?.role_family),
                    }))
                    .filter((opt) => opt.value && opt.label)
                : [];
            return {
                needs_clarification: true,
                role_family: safeText(payload.role_family),
                matched_label: safeText(payload.matched_label),
                message: safeText(payload.message),
                options,
            };
        }

        function buildRoleClarificationChoicesHtml(payload, options = {}) {
            const normalized = normalizeRoleClarificationPayload(payload);
            if (!normalized || !normalized.options.length) return '';
            const retryQuestion = safeText(options?.retryQuestion);
            const compact = options?.compact === true;
            return `
                <div class="flex flex-wrap gap-2 mt-2">
                    ${normalized.options.map((option) => {
                        const label = escapeHtml(option.label);
                        const value = escapeJsSingleQuoted(option.value);
                        const retryArg = retryQuestion ? `'${escapeJsSingleQuoted(retryQuestion)}'` : 'null';
                        const classes = compact
                            ? 'inline-flex items-center rounded-full border border-amber-300 bg-white px-2.5 py-1 text-[11px] font-semibold text-amber-900 hover:bg-amber-100 transition-colors'
                            : 'inline-flex items-center rounded-full border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-900 hover:bg-amber-100 transition-colors';
                        return `<button type="button" class="${classes}" onclick="handleRoleClarificationChoice('${value}', ${retryArg})">${label}</button>`;
                    }).join('')}
                </div>
            `;
        }

        function renderRoleClarificationPanel(targetId, payload, options = {}) {
            const node = document.getElementById(targetId);
            if (!node) return;

            const normalized = normalizeRoleClarificationPayload(payload);
            if (!normalized) {
                node.innerHTML = '';
                node.classList.add('hidden');
                return;
            }

            const title = options?.title || 'Choose your exact classification';
            node.innerHTML = `
                <div class="flex items-start gap-2">
                    <div class="mt-0.5 h-2.5 w-2.5 rounded-full bg-amber-500 shrink-0"></div>
                    <div class="min-w-0">
                        <p class="font-semibold">${escapeHtml(title)}</p>
                        <p class="mt-1">${escapeHtml(normalized.message || 'I need your exact classification before I can apply the right contract rules.')}</p>
                        ${buildRoleClarificationChoicesHtml(normalized, { compact: targetId === 'mo-role-clarification' })}
                    </div>
                </div>
            `;
            node.classList.remove('hidden');
        }

        function syncRoleClarificationPanels() {
            const clarification = normalizeRoleClarificationPayload(userProfile?.role_clarification);
            renderRoleClarificationPanel('onboard-role-clarification', clarification);
            renderRoleClarificationPanel('settings-role-clarification', clarification);
            renderRoleClarificationPanel('mo-role-clarification', clarification, { title: 'Pick the exact role Karl should use' });
        }

        async function handleRoleClarificationChoice(classificationValue, retryQuestion = null) {
            const classification = safeText(classificationValue);
            if (!classification) return false;

            const payload = { classification };
            const contractId = safeText(userProfile?.contract_id) || safeText(activeContract?.contract_id);
            if (contractId) payload.contract_id = contractId;

            const saved = await saveProfile(payload, {
                allowIncomplete: true,
                source: 'clarification',
                suppressSuccessAlert: true,
            });
            if (saved && safeText(retryQuestion)) {
                await sendQuery(safeText(retryQuestion));
            }
            return saved;
        }

        function choosePreferredProvenanceRef(provenance, preferredMode = null) {
            const refs = Array.isArray(provenance) ? provenance.filter((row) => row && typeof row === 'object') : [];
            if (!refs.length) return null;

            const normalizedMode = normalizeContractPdfSourceMode(preferredMode);
            const rankRef = (row) => {
                const sourceType = safeText(row?.source_type).toLowerCase();
                const pdf = safeText(row?.pdf).toLowerCase();
                const isMoa = sourceType.includes('moa') || sourceType.includes('amend') || pdf.includes('moa');
                const isBase = sourceType.includes('base') || sourceType.includes('cba') || sourceType.includes('contract') || (!!pdf && !isMoa);
                const page = Number.isFinite(Number(row?.pdf_page)) ? Number(row.pdf_page) : null;
                const hasPage = page !== null && page > 0;
                if (normalizedMode === 'moa') {
                    if (isMoa && hasPage) return 0;
                    if (hasPage) return 1;
                    if (isMoa) return 2;
                    if (isBase) return 3;
                    return 9;
                }
                if (normalizedMode === 'base') {
                    if (isBase && hasPage) return 0;
                    if (hasPage) return 1;
                    if (isBase) return 2;
                    if (isMoa) return 3;
                    return 9;
                }
                if (normalizedMode === 'effective') {
                    if (isMoa && hasPage) return 0;
                    if (isBase && hasPage) return 1;
                    if (isMoa) return 2;
                    if (isBase) return 3;
                    if (hasPage) return 4;
                    return 9;
                }
                if (hasPage) return 0;
                if (isMoa) return 1;
                if (isBase) return 2;
                return 9;
            };

            return [...refs].sort((a, b) => rankRef(a) - rankRef(b))[0];
        }

        function resolveCitationSourceHint(source, preferredMode = null) {
            const row = source && typeof source === 'object' ? source : {};
            const ref = choosePreferredProvenanceRef(row.provenance, preferredMode);

            let sourceType = safeText(ref?.source_type || row.source_type).toLowerCase();
            let sourcePdf = safeText(ref?.pdf || row.source_pdf || '');
            let sourceDocId = safeText(ref?.source_doc_id || row.source_doc_id || '');
            let sourcePage = Number.isFinite(Number(ref?.pdf_page))
                ? Number(ref.pdf_page)
                : null;

            if (!sourceType && sourcePdf) {
                sourceType = isMoaPdfName(sourcePdf) ? 'moa' : 'base';
            }
            if (!sourceType) sourceType = null;
            if (!sourcePdf) sourcePdf = null;
            if (!sourceDocId) sourceDocId = null;
            if (sourcePage !== null && sourcePage <= 0) sourcePage = null;

            return { sourceType, sourcePdf, sourceDocId, sourcePage };
        }

        function resolvePdfSourceHintFromPayload(payload, preferredMode = null) {
            const row = payload && typeof payload === 'object' ? payload : {};
            const directHint = resolveCitationSourceHint(row, preferredMode);
            if (directHint.sourceType || directHint.sourcePdf || directHint.sourceDocId || directHint.sourcePage) {
                return directHint;
            }

            const sections = Array.isArray(row.sections) ? row.sections : [];
            for (const section of sections) {
                const sectionHint = resolveCitationSourceHint(section, preferredMode);
                if (sectionHint.sourceType || sectionHint.sourcePdf || sectionHint.sourceDocId || sectionHint.sourcePage) {
                    return sectionHint;
                }
            }

            return { sourceType: null, sourcePdf: null, sourceDocId: null, sourcePage: null };
        }

        function summarizeCitationProvenance(source) {
            const row = source && typeof source === 'object' ? source : {};
            const refs = Array.isArray(row.provenance) ? row.provenance : [];
            let hasMoa = false;
            let hasBase = false;

            refs.forEach((ref) => {
                if (!ref || typeof ref !== 'object') return;
                const sourceType = safeText(ref.source_type).toLowerCase();
                const pdf = safeText(ref.pdf).toLowerCase();
                if (sourceType.includes('moa') || sourceType.includes('amend') || pdf.includes('moa')) {
                    hasMoa = true;
                } else if (sourceType.includes('base') || sourceType.includes('cba') || sourceType.includes('contract') || pdf) {
                    hasBase = true;
                }
            });

            if (!hasMoa && !hasBase) {
                const fallbackType = safeText(row.source_type).toLowerCase();
                const fallbackPdf = safeText(row.source_pdf).toLowerCase();
                if (fallbackType.includes('moa') || fallbackPdf.includes('moa')) {
                    hasMoa = true;
                } else if (fallbackType || fallbackPdf) {
                    hasBase = true;
                }
            }

            if (hasMoa && hasBase) return 'MOA + Base PDFs';
            if (hasMoa) return 'MOA PDF';
            if (hasBase) return 'Base PDF';
            return '';
        }

        function describePdfSourceType(sourceType, sourcePdf = null) {
            const normalizedType = safeText(sourceType).toLowerCase();
            const normalizedPdf = safeText(sourcePdf).toLowerCase();
            if (normalizedType === 'moa' || normalizedType.includes('amend') || normalizedPdf.includes('moa')) {
                return 'MOA PDF';
            }
            if (normalizedType === 'base' || normalizedType.includes('cba') || normalizedType.includes('contract')) {
                return 'Base PDF';
            }
            if (normalizedPdf) {
                return isMoaPdfName(normalizedPdf) ? 'MOA PDF' : 'Base PDF';
            }
            return 'Effective Auto';
        }

        function describePdfSourceChoice(choice = null) {
            const row = choice && typeof choice === 'object' ? choice : {};
            const bits = [describePdfSourceType(row.sourceType, row.sourcePdf)];
            const page = Number.isFinite(Number(row.sourcePage)) ? Number(row.sourcePage) : null;
            const pdf = safeText(row.sourcePdf) || null;
            if (page && page > 0) bits.push(`p.${page}`);
            if (pdf) bits.push(pdf);
            return bits.join(' | ');
        }

        function describeTargetSourceOption(choice = null) {
            const row = choice && typeof choice === 'object' ? choice : {};
            const explicitLabel = safeText(row.label);
            if (explicitLabel) return explicitLabel;
            const key = safeText(row.key);
            const sourceType = safeText(row.source_type || row.sourceType).toLowerCase();
            const isPrevious = row.is_previous === true || sourceType === 'base';
            const pdf = safeText(row.source_pdf || row.sourcePdf) || null;
            const page = Number.isFinite(Number(row.page_number ?? row.sourcePage)) ? Number(row.page_number ?? row.sourcePage) : null;
            if (key === 'effective_auto') {
                return 'Latest article version (Auto)';
            }
            const bits = [isPrevious ? 'Original / base PDF' : 'Latest amended PDF'];
            if (page && page > 0) bits.push(`p.${page}`);
            if (pdf) bits.push(pdf);
            return bits.join(' | ');
        }

        function describeResolvedPdfLocation(loc = null, sourceContext = null) {
            const candidates = Array.isArray(loc?.source_candidates) ? loc.source_candidates : [];
            const selectedKey = safeText(loc?.selected_source_key) || null;
            const selected = selectedKey
                ? (candidates.find((row) => safeText(row?.key) === selectedKey) || null)
                : null;
            if (selected) return describePdfSourceChoice({
                sourceType: selected.source_type,
                sourcePdf: selected.source_pdf,
                sourcePage: selected.page_number,
            });
            const sourceType = safeText(sourceContext?.sourceType).toLowerCase() || null;
            const sourcePdf = safeText(sourceContext?.sourcePdf) || null;
            return describePdfSourceChoice({ sourceType, sourcePdf, sourcePage: null });
        }

        function getActiveContractHistory() {
            const contractId = getActiveContractId();
            if (!contractId) return null;
            return contractHistoryById[contractId] || null;
        }

        async function loadContractHistory(contractId, options = {}) {
            const normalizedContractId = safeText(contractId);
            if (!normalizedContractId) return null;
            const refresh = options?.refresh === true;
            if (!refresh && contractHistoryById[normalizedContractId]) {
                activeContractHistory = contractHistoryById[normalizedContractId];
                return activeContractHistory;
            }
            try {
                const params = new URLSearchParams();
                params.set('contract_id', normalizedContractId);
                const res = await fetch(`${API_BASE}/api/contract-history?${params.toString()}`);
                if (!res.ok) throw new Error(`Contract history load failed (${res.status})`);
                const payload = await res.json();
                contractHistoryById[normalizedContractId] = payload;
                activeContractHistory = payload;
                return payload;
            } catch (err) {
                console.warn('Failed to load contract history:', err);
                contractHistoryById[normalizedContractId] = null;
                activeContractHistory = null;
                return null;
            }
        }

        function _resolveDefaultSourcePdfForMode(mode, history) {
            const normalizedMode = normalizeContractPdfSourceMode(mode);
            const row = history && typeof history === 'object' ? history : {};
            const basePdf = safeText(row.base_pdf);
            const amendmentPdfs = Array.isArray(row.amendment_pdfs) ? row.amendment_pdfs.map((v) => safeText(v)).filter(Boolean) : [];

            if (normalizedMode === 'base') return basePdf || null;
            if (normalizedMode === 'moa') return amendmentPdfs[0] || null;
            return null;
        }

        function _buildContractPdfEndpoint(contractId, sourceType = null, sourcePdf = null, sourceDocId = null) {
            const params = new URLSearchParams();
            params.set('contract_id', contractId);
            const normalizedType = safeText(sourceType).toLowerCase();
            const normalizedPdf = safeText(sourcePdf);
            const normalizedSourceDocId = safeText(sourceDocId);
            if (normalizedType) params.set('source_type', normalizedType);
            if (normalizedPdf) params.set('source_pdf', normalizedPdf);
            if (normalizedSourceDocId) params.set('source_doc_id', normalizedSourceDocId);
            return `${API_BASE}/api/contract-pdf?${params.toString()}`;
        }

        function _resolveSourceContextForPdfNavigation(options = {}) {
            const explicitSourceType = safeText(options?.sourceType).toLowerCase() || null;
            const explicitSourcePdf = safeText(options?.sourcePdf) || null;
            const explicitSourceDocId = safeText(options?.sourceDocId) || null;
            const explicitSourcePage = Number.isFinite(Number(options?.sourcePage))
                ? Number(options.sourcePage)
                : null;
            if (explicitSourceType || explicitSourcePdf || explicitSourceDocId || explicitSourcePage) {
                return {
                    sourceType: explicitSourceType,
                    sourcePdf: explicitSourcePdf,
                    sourceDocId: explicitSourceDocId,
                    sourcePage: explicitSourcePage && explicitSourcePage > 0 ? explicitSourcePage : null,
                };
            }

            if (!isDeveloperModeEnabled()) {
                return { sourceType: null, sourcePdf: null, sourceDocId: null, sourcePage: null };
            }

            const mode = normalizeContractPdfSourceMode(contractPdfSourceMode);
            const history = getActiveContractHistory();
            if (mode === 'effective') {
                return { sourceType: null, sourcePdf: null, sourceDocId: null, sourcePage: null };
            }

            let sourcePdf = safeText(contractPdfSourcePdf) || null;
            if (!sourcePdf) {
                sourcePdf = _resolveDefaultSourcePdfForMode(mode, history);
            }
            return {
                sourceType: mode,
                sourcePdf,
                sourceDocId: null,
                sourcePage: null,
            };
        }

        function updateContractViewerModeUI() {
            const content = document.getElementById('content-contract');
            const pdfDevControls = document.getElementById('contract-pdf-dev-controls');
            const textSourceControls = document.getElementById('contract-text-source-controls');
            const mobileModeSelect = document.getElementById('contract-viewer-mode-select-mobile');
            const desktopModeSelect = document.getElementById('contract-viewer-mode-select-desktop');
            const developerOnlyControls = document.querySelectorAll('.contract-dev-only');
            const isDeveloper = isDeveloperModeEnabled();
            const mode = normalizeContractViewerMode(contractViewerMode);
            const originalPdfMode = mode === 'original_pdf';

            if (content) {
                content.classList.toggle('contract-viewer-effective', !originalPdfMode);
                content.classList.toggle('contract-viewer-original', originalPdfMode);
            }
            if (mobileModeSelect) mobileModeSelect.value = mode;
            if (desktopModeSelect) desktopModeSelect.value = mode;
            developerOnlyControls.forEach((el) => {
                el.classList.toggle('hidden', !isDeveloper);
            });
            if (pdfDevControls) {
                pdfDevControls.classList.toggle('hidden', !isDeveloper);
            }
            if (textSourceControls) {
                textSourceControls.classList.toggle('hidden', !isDeveloper);
            }
            const comparePanel = document.getElementById('contract-text-compare-panel');
            if (comparePanel) {
                comparePanel.classList.toggle('hidden', originalPdfMode || !contractTextCompareOpen || !lastContractTextContext?.target);
            }
            renderCurrentTargetSourceControls();
            if (currentPopover && typeof currentPopover === 'object') {
                renderPopoverSourceControls();
            }
        }

        async function ensureContractViewerSelection() {
            const selectionKind = safeText(currentTocSelection?.kind).toLowerCase() || null;
            const selectionKey = safeText(currentTocSelection?.key) || null;
            if (selectionKind && selectionKey) {
                return refreshCurrentContractTextPane();
            }
            const hasManifest = await ensureManifestLoaded();
            if (!hasManifest) return false;
            const firstArticle = getSortedArticleNumbers()[0] ?? null;
            if (firstArticle === null) return false;
            return loadArticle(firstArticle, { openPdf: false });
        }

        async function handleContractViewerModeChange(mode) {
            saveContractViewerMode(mode);
            if (!isDeveloperModeEnabled()) {
                saveContractPdfSourceMode('effective');
                contractPdfSourcePdf = null;
                saveContractTextSourceMode('effective');
            }
            updateContractViewerModeUI();
            renderContractHistoryPanel();
            if (currentActiveTab !== 'contract') return;
            if (isOriginalPdfViewMode()) {
                await openContractPdfFromContractTab();
                return;
            }
            await ensureContractViewerSelection();
        }

        function renderContractHistoryPanel() {
            const panel = document.getElementById('contract-history-banner');
            const versionEl = document.getElementById('contract-history-version');
            const hashEl = document.getElementById('contract-history-hash');
            const amendmentsEl = document.getElementById('contract-history-amendments');
            const coverageEl = document.getElementById('contract-history-coverage');
            const timelineEl = document.getElementById('contract-history-timeline');
            const patchListEl = document.getElementById('contract-history-patches');
            const modeSelect = document.getElementById('contract-source-mode-select');
            const sourceDocSelect = document.getElementById('contract-source-doc-select');
            const history = getActiveContractHistory();
            const isDeveloper = isDeveloperModeEnabled();

            if (!panel || !versionEl || !hashEl || !amendmentsEl || !coverageEl || !timelineEl || !patchListEl || !modeSelect || !sourceDocSelect) {
                return;
            }

            if (!history) {
                panel.classList.add('hidden');
                versionEl.textContent = 'Current Contract: --';
                amendmentsEl.textContent = 'No applied amendments';
                hashEl.textContent = 'Content ID: --';
                coverageEl.textContent = 'Artifact coverage: --';
                timelineEl.innerHTML = '';
                patchListEl.innerHTML = '';
                modeSelect.innerHTML = '<option value="effective">Effective</option>';
                sourceDocSelect.innerHTML = '<option value="">Auto</option>';
                sourceDocSelect.disabled = true;
                updateContractViewerModeUI();
                return;
            }

            if (!isDeveloper) {
                panel.classList.add('hidden');
            } else {
                panel.classList.remove('hidden');
            }

            const effectiveVersion = safeText(history.effective_version_id);
            versionEl.textContent = effectiveVersion
                ? `Current Contract: ${effectiveVersion}`
                : 'Current Contract: Base agreement';
            const hash = safeText(history.effective_content_hash);
            hashEl.textContent = hash ? `Content ID: ${hash.slice(0, 16)}...` : 'Content ID: unavailable';

            const patchIds = Array.isArray(history.applied_patch_ids) ? history.applied_patch_ids : [];
            amendmentsEl.textContent = patchIds.length
                ? `Includes ${patchIds.length} amendment${patchIds.length === 1 ? '' : 's'}`
                : 'No applied amendments';
            const baseChunkTotal = Number.isFinite(Number(history.base_chunk_total))
                ? Number(history.base_chunk_total)
                : 0;
            const effectiveChunkTotal = Number.isFinite(Number(history.effective_chunk_total))
                ? Number(history.effective_chunk_total)
                : 0;
            const baseCounts = history.base_doc_type_counts && typeof history.base_doc_type_counts === 'object'
                ? history.base_doc_type_counts
                : {};
            const effectiveCounts = history.effective_doc_type_counts && typeof history.effective_doc_type_counts === 'object'
                ? history.effective_doc_type_counts
                : {};
            const allTypes = [...new Set([...Object.keys(baseCounts), ...Object.keys(effectiveCounts)])].sort();
            const typeCoverage = allTypes.map((docType) => {
                const baseCount = Number(baseCounts[docType] || 0);
                const effectiveCount = Number(effectiveCounts[docType] || 0);
                return `${docType}:${effectiveCount}/${baseCount}`;
            }).join(', ');
            coverageEl.textContent = `Artifacts current/base: ${effectiveChunkTotal}/${baseChunkTotal}${typeCoverage ? ` | ${typeCoverage}` : ''}`;

            const patches = Array.isArray(history.patches) ? history.patches : [];
            const timelinePills = [
                `<span class="inline-flex items-center rounded border border-slate-300 bg-white px-2 py-0.5 text-[10px] font-medium text-slate-700">Base Agreement</span>`
            ];
            patches.forEach((patch) => {
                const patchId = escapeHtml(safeText(patch.patch_id) || 'patch');
                const date = escapeHtml(safeText(patch.effective_date) || safeText(patch.ratified_date) || 'undated');
                timelinePills.push('<span class="text-slate-400 text-[10px]">-></span>');
                timelinePills.push(`<span class="inline-flex items-center rounded border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-800">MOA ${patchId} (${date})</span>`);
            });
            const effectiveLabel = escapeHtml(safeText(history.effective_version_id) || 'Current Version');
            timelinePills.push('<span class="text-slate-400 text-[10px]">-></span>');
            timelinePills.push(`<span class="inline-flex items-center rounded border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-800">Current Version ${effectiveLabel}</span>`);
            timelineEl.innerHTML = timelinePills.join('');

            patchListEl.innerHTML = patches.length
                ? patches.map((patch) => {
                    const patchId = escapeHtml(safeText(patch.patch_id) || 'patch');
                    const date = escapeHtml(safeText(patch.effective_date) || 'undated');
                    const pdf = escapeHtml(safeText(patch.source_pdf) || '');
                    return `<span class="inline-flex items-center rounded border border-slate-200 bg-white px-2 py-0.5 text-[10px] text-slate-700">${patchId} (${date}${pdf ? `, ${pdf}` : ''})</span>`;
                }).join('')
                : '<span class="text-[10px] text-slate-500">No patch metadata available.</span>';

            const supportedModes = Array.isArray(history.source_modes) && history.source_modes.length
                ? history.source_modes.map((v) => normalizeContractPdfSourceMode(v))
                : ['effective', 'base'];
            if (!supportedModes.includes(contractPdfSourceMode)) {
                saveContractPdfSourceMode(supportedModes[0] || 'effective');
            }

            modeSelect.innerHTML = supportedModes.map((mode) => {
                const label = mode === 'moa' ? 'MOA PDF' : mode === 'base' ? 'Base PDF' : 'Effective (Auto)';
                return `<option value="${mode}">${label}</option>`;
            }).join('');
            modeSelect.value = contractPdfSourceMode;

            const sourceOptions = [];
            if (contractPdfSourceMode === 'base') {
                if (safeText(history.base_pdf)) sourceOptions.push(safeText(history.base_pdf));
            } else if (contractPdfSourceMode === 'moa') {
                const amendmentPdfs = Array.isArray(history.amendment_pdfs) ? history.amendment_pdfs.map((v) => safeText(v)).filter(Boolean) : [];
                sourceOptions.push(...amendmentPdfs);
            }
            const uniqueSourceOptions = [...new Set(sourceOptions)];

            if (contractPdfSourceMode === 'effective') {
                sourceDocSelect.innerHTML = '<option value="">Auto by citation provenance</option>';
                sourceDocSelect.disabled = true;
                contractPdfSourcePdf = null;
                updateContractViewerModeUI();
                return;
            }

            if (!uniqueSourceOptions.length) {
                sourceDocSelect.innerHTML = '<option value="">No source PDF available</option>';
                sourceDocSelect.disabled = true;
                contractPdfSourcePdf = null;
                updateContractViewerModeUI();
                return;
            }

            sourceDocSelect.disabled = false;
            sourceDocSelect.innerHTML = uniqueSourceOptions.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
            if (!contractPdfSourcePdf || !uniqueSourceOptions.includes(contractPdfSourcePdf)) {
                contractPdfSourcePdf = uniqueSourceOptions[0];
            }
            sourceDocSelect.value = contractPdfSourcePdf;
            updateContractViewerModeUI();
        }

        function handleContractSourceModeChange(mode) {
            saveContractPdfSourceMode(mode);
            if (contractPdfSourceMode === 'effective') {
                contractPdfSourcePdf = null;
            } else {
                const history = getActiveContractHistory();
                contractPdfSourcePdf = _resolveDefaultSourcePdfForMode(contractPdfSourceMode, history);
            }
            renderContractHistoryPanel();
            if (currentActiveTab === 'contract') {
                openContractPdfFromContractTab().catch((err) => {
                    console.warn('Unable to refresh contract PDF source mode:', err);
                });
            }
        }

        function handleContractSourceDocChange(sourcePdf) {
            contractPdfSourcePdf = safeText(sourcePdf) || null;
            if (currentActiveTab === 'contract') {
                openContractPdfFromContractTab().catch((err) => {
                    console.warn('Unable to refresh contract PDF source document:', err);
                });
            }
        }

        async function refreshCurrentContractTextPane() {
            const selectionKind = safeText(currentTocSelection?.kind).toLowerCase() || null;
            const selectionKey = safeText(currentTocSelection?.key) || null;
            if (!selectionKind || !selectionKey) return false;

            if (selectionKind === 'article') {
                const match = selectionKey.match(/^article:(\d+)$/i);
                const articleNum = match ? toPositiveIntOrNull(match[1]) : null;
                if (articleNum === null) return false;
                return loadArticle(articleNum, { openPdf: false });
            }
            return loadContractBrowseItem(selectionKind, selectionKey, { openPdf: false });
        }

        function handleContractTextSourceModeChange(mode) {
            saveContractTextSourceMode(mode);
            if (currentActiveTab === 'contract') {
                refreshCurrentContractTextPane().catch((err) => {
                    console.warn('Unable to refresh contract text source mode:', err);
                });
            }
        }

        function toTitleCase(text) {
            const cleaned = (text || '').replace(/\s+/g, ' ').trim();
            if (!cleaned) return '';
            return cleaned
                .toLowerCase()
                .split(' ')
                .map(part => part ? (part[0].toUpperCase() + part.slice(1)) : part)
                .join(' ');
        }

        function inferContractDepartment(contract) {
            const id = (contract?.contract_id || '').toLowerCase();
            if (id.includes('_meat_')) return 'Meat';
            if (id.includes('_clerks_')) return 'Clerks';
            if (id.includes('_pharmacy_')) return 'Pharmacy';
            return '';
        }

        function getContractLabel(contract) {
            if (!contract) return 'Not set';
            if (contract.is_tenant_upload) {
                const unionLabel = safeText(contract.union_local_id) || 'Union';
                return `${unionLabel} uploaded documents`;
            }
            const contractId = (contract.contract_id || '').toLowerCase();
            const parts = contractId.split('_').filter(Boolean);
            const cleanParts = parts[0]?.match(/^local\d+$/i) ? parts.slice(1) : parts.slice();

            let employer = toTitleCase(contract.employer || '');
            if (/king\s*soopers/i.test(employer) || contractId.includes('kingsoopers')) {
                employer = 'King Soopers';
            } else if (/safeway/i.test(employer) || contractId.includes('safeway')) {
                employer = 'Safeway';
            } else if (/albertsons?/i.test(employer) || contractId.includes('albertsons')) {
                employer = 'Albertsons';
            } else {
                // Drop long corporate suffixes for readability.
                employer = employer
                    .replace(/,?\s+a division of.*$/i, '')
                    .replace(/,?\s+inc\.?$/i, '')
                    .trim();
            }

            const dept = inferContractDepartment(contract);
            const deptToken = dept ? dept.toLowerCase() : '';
            let location = '';
            if (cleanParts.length >= 2) {
                const maybe = cleanParts[1];
                if (maybe && maybe !== deptToken && !/^\d{4}$/.test(maybe)) {
                    location = toTitleCase(maybe);
                }
            }

            const base = employer || toTitleCase((contract.contract_id || '').replace(/_/g, ' '));
            const segments = [base];
            if (location) segments.push(location);
            if (dept && !base.toLowerCase().includes(deptToken)) segments.push(dept);
            return segments.join(' - ');
        }

        function populateClassificationSelects(contractId, classifications = [], options = {}) {
            const { preserveSelection = true } = options;
            const selectIds = ['onboard-classification', 'settings-classification'];
            selectIds.forEach(id => {
                const select = document.getElementById(id);
                if (!select) return;

                const previous = preserveSelection ? select.value : '';
                select.innerHTML = '<option value="">Select your role...</option>';
                classifications.forEach(c => {
                    const option = document.createElement('option');
                    option.value = c.value;
                    const hasWage = c.wage_available !== false;
                    const needsClarification = c.requires_role_clarification === true;
                    option.textContent = needsClarification
                        ? `${c.label} (clarify exact role)`
                        : (
                            hasWage
                                ? c.label
                                : `${c.label} (no Appendix A wage row)`
                        );
                    if (!hasWage) {
                        option.dataset.wageAvailable = 'false';
                    }
                    if (needsClarification) {
                        option.dataset.requiresRoleClarification = 'true';
                    }
                    select.appendChild(option);
                });

                if (previous && classifications.some(c => c.value === previous)) {
                    select.value = previous;
                }
            });
        }

        async function loadClassificationsForContract(contractId, options = {}) {
            if (!contractId) return;
            const { preserveSelection = true } = options;

            if (classificationOptionsByContract[contractId]) {
                populateClassificationSelects(contractId, classificationOptionsByContract[contractId], { preserveSelection });
                return;
            }

            try {
                const res = await fetch(`${API_BASE}/api/classifications?contract_id=${encodeURIComponent(contractId)}&include_unmapped=true`);
                if (!res.ok) throw new Error(`Failed to load classifications: ${res.status}`);
                const data = await res.json();
                const classifications = data.classifications || [];
                classificationOptionsByContract[contractId] = classifications;
                populateClassificationSelects(contractId, classifications, { preserveSelection });
            } catch (e) {
                console.error('Failed to load contract classifications:', e);
            }
        }

        function updateContractDisplay() {
            const contract = getActiveContract();
            const subtitle = document.getElementById('header-subtitle');
            const unionName = tenantBootstrap?.union?.name || contract?.union_local_id || 'Union Documents';
            if (subtitle) {
                subtitle.textContent = contract
                    ? `${getContractLabel(contract)} | ${unionName}`
                    : 'Union Document Assistant';
            }
            const displayContract = document.getElementById('display-contract');
            if (displayContract) {
                displayContract.textContent = isTenantUploadAuthMode()
                    ? (tenantBootstrap?.union?.name || contract?.union_local_id || 'Union documents')
                    : getContractLabel(contract);
            }
        }

        function setActiveContract(contractId, options = {}) {
            const { persist = true, refreshViewer = true, preserveClassification = true } = options;
            if (isTenantUploadAuthMode()) {
                const tenantContract = getTenantUploadContract();
                if (!tenantContract) return;
                activeContract = tenantContract;
                if (persist) {
                    localStorage.setItem(ACTIVE_CONTRACT_STORAGE_KEY, tenantContract.contract_id);
                }
                const onboardSelect = document.getElementById('onboard-contract');
                const settingsSelect = document.getElementById('settings-contract');
                if (onboardSelect) onboardSelect.value = tenantContract.contract_id;
                if (settingsSelect) settingsSelect.value = tenantContract.contract_id;
                updateContractDisplay();
                return;
            }
            if (!contractId) return;

            const contract = availableContracts.find(c => c.contract_id === contractId);
            if (!contract) return;

            const previousId = activeContract?.contract_id;
            activeContract = contract;

            if (persist) {
                localStorage.setItem(ACTIVE_CONTRACT_STORAGE_KEY, contract.contract_id);
            }

            const onboardSelect = document.getElementById('onboard-contract');
            const settingsSelect = document.getElementById('settings-contract');
            if (onboardSelect) onboardSelect.value = contract.contract_id;
            if (settingsSelect) settingsSelect.value = contract.contract_id;

            updateContractDisplay();
            checkHealth();
            loadClassificationsForContract(contract.contract_id, { preserveSelection: preserveClassification });

            if (refreshViewer && previousId !== contract.contract_id) {
                articleCache = {};
                articleFirstSectionCache = {};
                articleTitles = {};
                contractBrowseGroups = [];
                currentTocSelection = { kind: null, key: null };
                activeContractHistory = null;
                currentArticleNum = null;
                currentPdfBaseUrl = null;
                currentPdfPage = null;
                lastPinnedPdfLocation = null;
                _clearPdfNavigationContext();
                _clearContractTextContext();
                contractPdfSourcePdf = null;
                if (pendingPdfFrameSwapTimer) {
                    clearTimeout(pendingPdfFrameSwapTimer);
                    pendingPdfFrameSwapTimer = null;
                }
                if (pendingEdgePdfSecondPassTimer) {
                    clearTimeout(pendingEdgePdfSecondPassTimer);
                    pendingEdgePdfSecondPassTimer = null;
                }
                const overlay = document.getElementById('contract-pdf-overlay');
                const frame = document.getElementById('contract-pdf-frame');
                const label = document.getElementById('contract-pdf-location-label');
                overlay?.classList.add('hidden');
                if (frame) frame.src = '';
                _setEdgePdfFallbackVisible(false);
                if (label) label.textContent = 'Loading PDF location...';
                renderContractHistoryPanel();
                if (currentActiveTab === 'contract') {
                    initContractViewer();
                    if (isOriginalPdfViewMode()) {
                        openContractPdfFromContractTab().catch((err) => {
                            console.warn('Unable to reset contract PDF for new contract:', err);
                        });
                    } else {
                        ensureContractViewerSelection().catch((err) => {
                            console.warn('Unable to reset effective contract view for new contract:', err);
                        });
                    }
                }
            }
        }

        function populateContractSelects() {
            const selects = ['onboard-contract', 'settings-contract'];
            selects.forEach(id => {
                const select = document.getElementById(id);
                if (!select) return;

                const currentValue = select.value;
                select.innerHTML = '<option value="">Select your contract...</option>';
                availableContracts.forEach(contract => {
                    const option = document.createElement('option');
                    option.value = contract.contract_id;
                    option.textContent = getContractLabel(contract);
                    select.appendChild(option);
                });

                if (activeContract) {
                    select.value = activeContract.contract_id;
                } else if (currentValue) {
                    select.value = currentValue;
                }
            });
        }

        async function loadContracts() {
            if (isTenantRoute()) {
                const tenantContract = getTenantUploadContract();
                availableContracts = tenantContract ? [tenantContract] : [];
                activeContract = tenantContract;
                populateContractSelects();
                updateContractDisplay();
                applyTenantUploadModeUI();
                return;
            }
            try {
                const res = await fetch(`${API_BASE}/api/contracts`);
                if (!res.ok) throw new Error(`Failed to load contracts: ${res.status}`);
                const data = await res.json();
                availableContracts = data.contracts || [];

                if (!availableContracts.length) {
                    throw new Error('No contracts found in backend manifest catalog');
                }

                populateContractSelects();
                updateContractDisplay();
                applyTenantUploadModeUI();
            } catch (e) {
                console.error('Failed to load contracts:', e);
                availableContracts = [];
                activeContract = null;
                updateContractDisplay();
                applyTenantUploadModeUI();
            }
        }

        // =============================================================================
        // STEWARD ONBOARDING DATE PICKERS
        // =============================================================================

        function toggleDatePicker(pickerId) {
            ensureStewardOnboardingController().toggleDatePicker(pickerId);
        }

        function changeYear(pickerId, delta) {
            ensureStewardOnboardingController().changeYear(pickerId, delta);
        }

        function selectMonth(pickerId, monthIndex) {
            ensureStewardOnboardingController().selectMonth(pickerId, monthIndex);
        }

        function quickSelect(pickerId, yearsAgo) {
            ensureStewardOnboardingController().quickSelect(pickerId, yearsAgo);
        }

        function initDatePicker(pickerId, existingValue) {
            ensureStewardOnboardingController().initDatePicker(pickerId, existingValue);
        }

        // =============================================================================
        // TAB NAVIGATION
        // =============================================================================

        // Start collapsed on mobile, but will auto-open on first visit
        let tocExpanded = false;
        let contractTabVisited = false;

        function toggleTOC() {
            const mobileList = document.getElementById('toc-list-mobile');
            const chevron = document.getElementById('toc-chevron');
            tocExpanded = !tocExpanded;

            if (tocExpanded) {
                mobileList?.classList.remove('hidden');
                chevron?.classList.add('rotate-180');
            } else {
                mobileList?.classList.add('hidden');
                chevron?.classList.remove('rotate-180');
            }
            scheduleShellLayoutMetricsSync();
        }

        // Initialize TOC collapsed state
        function initTOCState() {
            const mobileList = document.getElementById('toc-list-mobile');
            const chevron = document.getElementById('toc-chevron');

            // Mobile starts collapsed
            mobileList?.classList.add('hidden');
            chevron?.classList.remove('rotate-180');
            tocExpanded = false;
            scheduleShellLayoutMetricsSync();
        }

        // Handle window resize for tab bar active state
        let currentActiveTab = 'chat';
        window.addEventListener('resize', () => {
            // Update active tab border direction on resize
            setActiveTab(currentActiveTab);
            scheduleShellLayoutMetricsSync();
        });

        function setActiveTab(tab) {
            currentActiveTab = tab;

            // Update tab buttons
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('tab-active');
            });

            const activeBtn = document.getElementById(`tab-${tab}`) || (tab === 'karl' ? document.getElementById('tab-settings') : null);
            if (activeBtn) {
                activeBtn.classList.add('tab-active');
            }

            // Show/hide tab content
            document.querySelectorAll('[id^="content-"]').forEach(content => {
                content.classList.add('hidden');
            });

            const activeContent = document.getElementById(`content-${tab}`);
            if (activeContent) {
                activeContent.classList.remove('hidden');
            }

            // Initialize contract viewer on first visit
            if (tab === 'contract') {
                if (!hasCompleteProfileContext()) {
                    showOnboarding();
                }
                if (Object.keys(articleTitles).length === 0) {
                    initContractViewer();
                } else {
                    const contractId = getActiveContractId();
                    if (contractId && !getActiveContractHistory()) {
                        loadContractHistory(contractId).then(() => {
                            renderContractHistoryPanel();
                        }).catch(() => {});
                    } else {
                        renderContractHistoryPanel();
                    }
                }
                if (hasCompleteProfileContext()) {
                    if (isOriginalPdfViewMode()) {
                        if (!isContractPdfOverlayOpen()) {
                            openContractPdfFromContractTab().catch((err) => {
                                console.warn('Unable to open default contract PDF view:', err);
                            });
                        }
                    } else {
                        ensureContractViewerSelection().catch((err) => {
                            console.warn('Unable to load effective contract view:', err);
                        });
                    }
                }

                // Auto-open TOC on first visit to Contract tab (mobile only)
                if (!contractTabVisited && window.innerWidth < 768) {
                    contractTabVisited = true;
                    // Small delay to let the tab switch complete
                    setTimeout(() => {
                        if (!tocExpanded) {
                            toggleTOC();
                        }
                    }, 100);
                }
            }

            // Sync settings form when opening settings
            if (tab === 'settings') {
                syncSettingsForm();
            }
        }

        // =============================================================================
        // CONTRACT VIEWER
        // =============================================================================

        function closeContractPdfOverlay() {
            // Contract view is now PDF-first; keep reader mounted.
            return;
        }

        function isContractPdfOverlayOpen() {
            const overlay = document.getElementById('contract-pdf-overlay');
            const frame = document.getElementById('contract-pdf-frame');
            const hasSrc = !!String(frame?.getAttribute('src') || '').trim();
            return !!overlay && !overlay.classList.contains('hidden') && hasSrc;
        }

        function openPdfInNewTab() {
            const frame = document.getElementById('contract-pdf-frame');
            const frameSrc = String(frame?.getAttribute('src') || '').trim();
            const fallbackBase = frameSrc ? frameSrc.split('#')[0] : '';
            const baseUrl = String(currentPdfBaseUrl || fallbackBase || '').trim();
            if (!baseUrl) return;
            const viewerUrl = _buildFullPdfViewerUrl(baseUrl, currentPdfPage);
            window.open(viewerUrl, '_blank', 'noopener,noreferrer');
        }

        function downloadContractPdf() {
            const frame = document.getElementById('contract-pdf-frame');
            const frameSrc = String(frame?.getAttribute('src') || '').trim();
            const fallbackBase = frameSrc ? frameSrc.split('#')[0] : '';
            const baseUrl = String(currentPdfBaseUrl || fallbackBase || '').trim();
            if (!baseUrl) return;

            const contractId = getActiveContractId();
            const suggestedName = contractId ? `${contractId}.pdf` : 'contract.pdf';
            const anchor = document.createElement('a');
            anchor.href = baseUrl;
            anchor.download = suggestedName;
            anchor.target = '_blank';
            anchor.rel = 'noopener';
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
        }

        function _buildContractPdfUrl(baseUrl, pageNumber = null) {
            const cleanBaseUrl = String(baseUrl || '').split('#')[0].trim();
            if (!cleanBaseUrl) return '';
            const params = [];
            if (Number.isFinite(Number(pageNumber)) && Number(pageNumber) > 0) {
                params.push(`page=${Number(pageNumber)}`);
            }
            params.push('view=FitH,0');
            // Hide default PDF viewer UI chrome when embedded.
            params.push('toolbar=0');
            params.push('navpanes=0');
            params.push('scrollbar=0');
            return `${cleanBaseUrl}#${params.join('&')}`;
        }

        function _buildFullPdfViewerUrl(baseUrl, pageNumber = null) {
            const cleanBaseUrl = String(baseUrl || '').split('#')[0].trim();
            if (!cleanBaseUrl) return '';
            const params = [];
            if (Number.isFinite(Number(pageNumber)) && Number(pageNumber) > 0) {
                params.push(`page=${Number(pageNumber)}`);
            }
            return `${cleanBaseUrl}#${params.join('&')}`;
        }

        function _buildEmbeddedPdfNavigationUrl(baseUrl, pageNumber = null) {
            const cleanBaseUrl = String(baseUrl || '').split('#')[0].trim();
            if (!cleanBaseUrl) return '';
            const navUrl = new URL(cleanBaseUrl, window.location.origin);
            navUrl.searchParams.set('_viewer_nav', String(Date.now()));
            return _buildContractPdfUrl(navUrl.toString(), pageNumber);
        }

        function _isEdgeBrowser() {
            const ua = String(window.navigator?.userAgent || '');
            return ua.includes('Edg/');
        }

        function _replaceContractPdfFrame(nextSrc, options = {}) {
            const currentFrame = document.getElementById('contract-pdf-frame');
            if (!currentFrame) return null;
            const replacement = currentFrame.cloneNode(false);
            replacement.removeAttribute('src');
            currentFrame.replaceWith(replacement);
            const requestToken = Number.isFinite(Number(options?.requestToken))
                ? Number(options.requestToken)
                : null;
            const baseUrl = safeText(options?.baseUrl) || null;
            const pageNumber = Number.isFinite(Number(options?.pageNumber))
                ? Number(options.pageNumber)
                : null;
            if (_isEdgeBrowser() && baseUrl) {
                replacement.addEventListener('load', () => {
                    if (pendingEdgePdfSecondPassTimer) {
                        clearTimeout(pendingEdgePdfSecondPassTimer);
                        pendingEdgePdfSecondPassTimer = null;
                    }
                    pendingEdgePdfSecondPassTimer = setTimeout(() => {
                        if (requestToken !== null && requestToken !== pdfNavRequestSeq) return;
                        if (replacement !== document.getElementById('contract-pdf-frame')) return;
                        replacement.src = _buildEmbeddedPdfNavigationUrl(baseUrl, pageNumber);
                    }, 180);
                }, { once: true });
            }
            replacement.src = nextSrc;
            return replacement;
        }

        function _setEdgePdfFallbackVisible(isVisible) {
            const frame = document.getElementById('contract-pdf-frame');
            const fallback = document.getElementById('contract-pdf-edge-fallback');
            if (frame) frame.classList.toggle('hidden', !!isVisible);
            if (fallback) {
                fallback.classList.toggle('hidden', !isVisible);
                fallback.classList.toggle('flex', !!isVisible);
            }
        }

        function _openEdgePdfFallback(baseUrl, pageNumber = null, locationLabel = 'Contract PDF') {
            const overlay = document.getElementById('contract-pdf-overlay');
            const label = document.getElementById('contract-pdf-location-label');
            currentPdfBaseUrl = String(baseUrl || '').trim() || null;
            currentPdfPage = Number.isFinite(Number(pageNumber)) ? Number(pageNumber) : null;
            if (overlay) overlay.classList.remove('hidden');
            if (label) label.textContent = locationLabel;
            _setEdgePdfFallbackVisible(true);
            const viewerUrl = _buildFullPdfViewerUrl(currentPdfBaseUrl, currentPdfPage);
            try {
                window.open(viewerUrl, '_blank', 'noopener,noreferrer');
            } catch (_) {
                // Keep the in-pane fallback visible even if popup open fails.
            }
        }

        function nextPdfNavToken() {
            pdfNavRequestSeq += 1;
            return pdfNavRequestSeq;
        }

        function toPositiveIntOrNull(value) {
            const parsed = Number(value);
            if (!Number.isFinite(parsed)) return null;
            if (parsed <= 0) return null;
            return Math.trunc(parsed);
        }

        function rememberPinnedPdfLocation(baseUrl, pageNumber = null, locationLabel = 'Contract PDF') {
            const contractId = getActiveContractId();
            if (!contractId || !baseUrl) return;
            lastPinnedPdfLocation = {
                contractId,
                baseUrl,
                pageNumber: Number.isFinite(Number(pageNumber)) ? Number(pageNumber) : null,
                locationLabel: String(locationLabel || 'Contract PDF'),
            };
        }

        function _findPreviousPdfSourceCandidate(context = null) {
            const ctx = context && typeof context === 'object' ? context : (lastPdfNavigationContext || {});
            const candidates = Array.isArray(ctx?.sourceCandidates) ? ctx.sourceCandidates : [];
            if (!candidates.length) return null;

            const byKey = candidates.find((c) => safeText(c?.key) === 'previous_base');
            if (byKey) return byKey;

            return candidates.find((c) => {
                const sourceType = safeText(c?.source_type).toLowerCase();
                return sourceType === 'base';
            }) || null;
        }

        function _findPdfSourceCandidateByKey(context = null, choiceKey = null) {
            const ctx = context && typeof context === 'object' ? context : (lastPdfNavigationContext || {});
            const candidates = Array.isArray(ctx?.sourceCandidates) ? ctx.sourceCandidates : [];
            const normalizedChoiceKey = safeText(choiceKey) || null;
            if (!normalizedChoiceKey) return null;
            return candidates.find((c) => safeText(c?.key) === normalizedChoiceKey) || null;
        }

        function renderCurrentTargetSourceControls() {
            const wrap = document.getElementById('contract-target-source-controls');
            const select = document.getElementById('contract-target-source-select');
            if (!wrap || !select) return;
            if (!isOriginalPdfViewMode()) {
                wrap.classList.add('hidden');
                select.innerHTML = '';
                return;
            }

            const ctx = lastPdfNavigationContext;
            const target = ctx?.target || null;
            const rawCandidates = Array.isArray(ctx?.sourceCandidates) ? ctx.sourceCandidates : [];

            const candidateRank = (candidate = null) => {
                const key = safeText(candidate?.key);
                const sourceType = safeText(candidate?.source_type).toLowerCase();
                if (key === 'effective_auto') return 0;
                if (sourceType === 'moa') return 1;
                if (sourceType === 'base') return 2;
                return 3;
            };
            const candidates = [...rawCandidates].sort((a, b) => {
                const rankDiff = candidateRank(a) - candidateRank(b);
                if (rankDiff !== 0) return rankDiff;
                return safeText(a?.label).localeCompare(safeText(b?.label));
            });

            if (!target || candidates.length <= 1) {
                wrap.classList.add('hidden');
                select.innerHTML = '';
                return;
            }

            let selectedKey = safeText(ctx?.selectedSourceKey) || '';
            if (!selectedKey || !_findPdfSourceCandidateByKey(ctx, selectedKey)) {
                selectedKey = safeText(candidates[0]?.key) || 'effective_auto';
            }

            select.innerHTML = candidates.map((candidate) => {
                const key = escapeHtml(String(candidate?.key || ''));
                const label = escapeHtml(describeTargetSourceOption(candidate));
                const selectedAttr = String(candidate?.key || '') === selectedKey ? ' selected' : '';
                return `<option value="${key}"${selectedAttr}>${label}</option>`;
            }).join('');
            wrap.classList.remove('hidden');
        }

        function updatePreviousPdfButtonState() {
            renderCurrentTargetSourceControls();
        }

        function _rememberPdfNavigationContext(target, loc = null) {
            const normalizedTarget = target && typeof target === 'object' ? { ...target } : null;
            const sourceCandidates = Array.isArray(loc?.source_candidates) ? loc.source_candidates.map((row) => ({ ...(row || {}) })) : [];
            const selectedSourceKey = safeText(loc?.selected_source_key) || null;
            lastPdfNavigationContext = normalizedTarget ? {
                target: normalizedTarget,
                sourceCandidates,
                selectedSourceKey,
            } : null;
            updatePreviousPdfButtonState();
        }

        function _clearPdfNavigationContext() {
            lastPdfNavigationContext = null;
            updatePreviousPdfButtonState();
        }

        async function openSourceChoiceForCurrentSelection(choiceKey) {
            const ctx = lastPdfNavigationContext;
            const target = ctx?.target;
            if (!target) return false;

            const normalizedChoiceKey = safeText(choiceKey) || 'effective_auto';
            const candidate = _findPdfSourceCandidateByKey(ctx, normalizedChoiceKey);

            const sourceType = safeText(candidate?.source_type).toLowerCase() || null;
            const sourcePdf = safeText(candidate?.source_pdf) || null;
            const sourceDocId = safeText(candidate?.source_doc_id) || null;
            const sourcePage = Number.isFinite(Number(candidate?.page_number)) ? Number(candidate.page_number) : null;
            const isEffectiveAuto = normalizedChoiceKey === 'effective_auto' || (!candidate && normalizedChoiceKey);

            return openContractInPdf(
                target.articleNum ?? null,
                target.sectionNum ?? null,
                target.partNum ?? null,
                {
                    tableId: target.tableId ?? null,
                    rowIndex: target.rowIndex ?? null,
                    browseKind: target.browseKind ?? null,
                    browseKey: target.browseKey ?? null,
                    sourceType: isEffectiveAuto ? null : sourceType,
                    sourcePdf: isEffectiveAuto ? null : sourcePdf,
                    sourceDocId: isEffectiveAuto ? null : sourceDocId,
                    sourcePage: isEffectiveAuto ? null : sourcePage,
                }
            );
        }

        async function handleCurrentTargetSourceChange(value) {
            await openSourceChoiceForCurrentSelection(value);
        }

        function _showContractPdfOverlay(baseUrl, pageNumber = null, locationLabel = 'Contract PDF', options = {}) {
            const overlay = document.getElementById('contract-pdf-overlay');
            const label = document.getElementById('contract-pdf-location-label');
            if (!overlay || !baseUrl) return;
            const requestToken = Number.isFinite(Number(options?.requestToken))
                ? Number(options.requestToken)
                : null;
            if (requestToken !== null && requestToken !== pdfNavRequestSeq) {
                return;
            }
            const forceReload = options?.forceReload === true;
            const rememberAnchor = options?.rememberAnchor !== false;

            currentPdfBaseUrl = baseUrl;
            currentPdfPage = Number.isFinite(Number(pageNumber)) ? Number(pageNumber) : null;
            overlay.classList.remove('hidden');
            if (label) label.textContent = locationLabel;
            if (rememberAnchor) {
                rememberPinnedPdfLocation(baseUrl, currentPdfPage, locationLabel);
            }

            if (_isEdgeBrowser() && isOriginalPdfViewMode()) {
                _openEdgePdfFallback(currentPdfBaseUrl, currentPdfPage, locationLabel);
                return;
            }

            _setEdgePdfFallbackVisible(false);

            const nextSrc = _buildEmbeddedPdfNavigationUrl(currentPdfBaseUrl, currentPdfPage);
            const frame = document.getElementById('contract-pdf-frame');
            const previousSrc = String(frame?.getAttribute('src') || '').trim();
            if (pendingPdfFrameSwapTimer) {
                clearTimeout(pendingPdfFrameSwapTimer);
                pendingPdfFrameSwapTimer = null;
            }
            if (pendingEdgePdfSecondPassTimer) {
                clearTimeout(pendingEdgePdfSecondPassTimer);
                pendingEdgePdfSecondPassTimer = null;
            }
            if (forceReload) {
                pendingPdfFrameSwapTimer = setTimeout(() => {
                    if (requestToken !== null && requestToken !== pdfNavRequestSeq) return;
                    _replaceContractPdfFrame(nextSrc, {
                        requestToken,
                        baseUrl: currentPdfBaseUrl,
                        pageNumber: currentPdfPage,
                    });
                }, 60);
            } else if (previousSrc && previousSrc !== nextSrc) {
                pendingPdfFrameSwapTimer = setTimeout(() => {
                    if (requestToken !== null && requestToken !== pdfNavRequestSeq) return;
                    _replaceContractPdfFrame(nextSrc, {
                        requestToken,
                        baseUrl: currentPdfBaseUrl,
                        pageNumber: currentPdfPage,
                    });
                }, 60);
            } else {
                _replaceContractPdfFrame(nextSrc, {
                    requestToken,
                    baseUrl: currentPdfBaseUrl,
                    pageNumber: currentPdfPage,
                });
            }
        }

        async function resetContractPdfView() {
            const contractId = getActiveContractId();
            if (!contractId) return false;
            const requestToken = nextPdfNavToken();
            if (
                lastPinnedPdfLocation &&
                lastPinnedPdfLocation.contractId === contractId &&
                lastPinnedPdfLocation.baseUrl
            ) {
                _showContractPdfOverlay(
                    lastPinnedPdfLocation.baseUrl,
                    lastPinnedPdfLocation.pageNumber,
                    lastPinnedPdfLocation.locationLabel,
                    { requestToken, forceReload: true, rememberAnchor: false }
                );
                return true;
            }
            return openContractPdfFromContractTab();
        }

        function getSortedArticleNumbers() {
            return Object.keys(articleTitles || {})
                .map((raw) => toPositiveIntOrNull(raw))
                .filter((num) => num !== null)
                .sort((a, b) => a - b);
        }

        async function ensureManifestLoaded() {
            if (getSortedArticleNumbers().length > 0) return true;
            await initContractViewer();
            return getSortedArticleNumbers().length > 0;
        }

        function setActiveTocSelection(kind, key) {
            const normalizedKind = safeText(kind).toLowerCase() || null;
            const normalizedKey = safeText(key) || null;
            currentTocSelection = { kind: normalizedKind, key: normalizedKey };
            document.querySelectorAll('.toc-item').forEach(item => {
                item.classList.remove('bg-ufcw-blue', 'text-white');
                item.querySelector('span:first-child')?.classList.remove('text-white');
                item.querySelector('span:first-child')?.classList.add('text-ufcw-blue');
                item.querySelector('span:last-child')?.classList.remove('text-white');
                item.querySelector('span:last-child')?.classList.add('text-slate-600');
            });
            if (!normalizedKind || !normalizedKey) return;
            document.querySelectorAll(`.toc-item[data-toc-kind="${normalizedKind}"][data-toc-key="${normalizedKey}"]`).forEach(activeItem => {
                activeItem.classList.add('bg-ufcw-blue', 'text-white');
                activeItem.querySelector('span:first-child')?.classList.remove('text-ufcw-blue');
                activeItem.querySelector('span:first-child')?.classList.add('text-white');
                activeItem.querySelector('span:last-child')?.classList.remove('text-slate-600');
                activeItem.querySelector('span:last-child')?.classList.add('text-white');
            });
        }

        function setActiveArticleInToc(normalizedArticleNum) {
            // Tenant outlines can carry non-numeric article ids ("appendix-a").
            const articleNum = toPositiveIntOrNull(normalizedArticleNum);
            const key = articleNum !== null ? String(articleNum) : safeText(normalizedArticleNum);
            if (!key) return;
            setActiveTocSelection('article', `article:${key}`);
        }

        async function getFirstSectionLocator(articleNum) {
            const normalizedArticleNum = toPositiveIntOrNull(articleNum);
            if (normalizedArticleNum === null) return null;
            if (Object.prototype.hasOwnProperty.call(articleFirstSectionCache, normalizedArticleNum)) {
                return articleFirstSectionCache[normalizedArticleNum];
            }
            try {
                const res = await fetch(`${API_BASE}/api/article/${normalizedArticleNum}${getContractQueryString()}`);
                if (!res.ok) {
                    articleFirstSectionCache[normalizedArticleNum] = null;
                    return null;
                }
                const data = await res.json();
                const sections = Array.isArray(data?.sections) ? data.sections : [];
                const first = sections.find((section) => toPositiveIntOrNull(section?.section_num) !== null);
                if (!first) {
                    articleFirstSectionCache[normalizedArticleNum] = null;
                    return null;
                }
                const locator = {
                    sectionNum: toPositiveIntOrNull(first.section_num),
                    partNum: first.subsection ? String(first.subsection) : null,
                };
                articleFirstSectionCache[normalizedArticleNum] = locator;
                return locator;
            } catch (_) {
                articleFirstSectionCache[normalizedArticleNum] = null;
                return null;
            }
        }

        async function openArticleInPdf(articleNum, options = {}) {
            if (isTenantRoute()) {
                // No legacy section locator on tenant routes; the tenant branch
                // of openContractInPdf resolves the page from the outline.
                return openContractInPdf(articleNum, null, null, {
                    ...options,
                    requestToken: nextPdfNavToken(),
                });
            }
            const normalizedArticleNum = toPositiveIntOrNull(articleNum);
            if (normalizedArticleNum === null) return false;
            const { preferSection = true } = options;
            const requestToken = nextPdfNavToken();

            const openedAtArticleLevel = await openContractInPdf(
                normalizedArticleNum,
                null,
                null,
                { ...options, requestToken }
            );
            if (openedAtArticleLevel) return true;

            if (preferSection) {
                const locator = await getFirstSectionLocator(normalizedArticleNum);
                if (requestToken !== pdfNavRequestSeq) return false;
                const normalizedSection = toPositiveIntOrNull(locator?.sectionNum);
                if (normalizedSection !== null) {
                    const openedBySection = await openContractInPdf(
                        normalizedArticleNum,
                        normalizedSection,
                        locator?.partNum || null,
                        { ...options, requestToken }
                    );
                    if (openedBySection) return true;
                }
            }
            return false;
        }

        async function openContractInPdf(articleNum, sectionNum = null, partNum = null, options = {}) {
            const contractId = getActiveContractId();
            const normalizedArticleNum = toPositiveIntOrNull(articleNum);
            const normalizedSectionNum = toPositiveIntOrNull(sectionNum);
            const tableId = String(options?.tableId || '').trim() || null;
            const browseKind = safeText(options?.browseKind).toLowerCase() || null;
            const browseKey = safeText(options?.browseKey) || null;
            const normalizedRowIndex = Number.isFinite(Number(options?.rowIndex))
                ? Number(options.rowIndex)
                : null;
            const requestToken = Number.isFinite(Number(options?.requestToken))
                ? Number(options.requestToken)
                : nextPdfNavToken();
            const sourceContext = _resolveSourceContextForPdfNavigation(options);
            // Tenant deployments have no on-disk packs behind /api/pdf-location
            // — the printed book is the attached source PDF served by the
            // member API, and the outline carries each section's printed page.
            if (isTenantRoute()) {
                const outline = tenantContractOutline;
                const pdfUrl = safeText(outline?.source_pdf_url);
                if (!pdfUrl) {
                    _clearPdfNavigationContext();
                    return false;
                }
                const articleKey = safeText(articleNum) || null;
                const article = (outline?.articles || []).find(
                    (item) => String(item.article_num) === articleKey
                );
                let page = sourceContext.sourcePage && sourceContext.sourcePage > 0
                    ? sourceContext.sourcePage
                    : null;
                if (page === null && article) {
                    const sections = article.sections || [];
                    const target = safeText(sectionNum)
                        ? sections.find((entry) => String(entry.section_num) === String(sectionNum))
                        : null;
                    const pick = (target && Number(target.page) > 0)
                        ? target
                        : sections.find((entry) => Number(entry.page) > 0);
                    page = pick && Number(pick.page) > 0 ? Number(pick.page) : null;
                }
                const label = article?.article_title
                    ? `${article.article_title}${page ? ` · Page ${page}` : ''}`
                    : `Contract PDF${page ? ` · Page ${page}` : ''}`;
                _showContractPdfOverlay(new URL(pdfUrl, API_BASE).toString(), page, label, { requestToken });
                return true;
            }
            if (!contractId || (normalizedArticleNum === null && !tableId && !(browseKind && browseKey))) {
                _clearPdfNavigationContext();
                return false;
            }
            const pdfTarget = {
                contractId,
                articleNum: normalizedArticleNum,
                sectionNum: normalizedSectionNum,
                partNum: partNum ? String(partNum) : null,
                tableId,
                rowIndex: normalizedRowIndex,
                browseKind,
                browseKey,
            };

            const overlay = document.getElementById('contract-pdf-overlay');
            const labelEl = document.getElementById('contract-pdf-location-label');
            if (overlay) overlay.classList.remove('hidden');
            if (labelEl) {
                if (tableId) {
                    labelEl.textContent = normalizedRowIndex !== null
                        ? `Locating Table ${tableId}, Row ${normalizedRowIndex + 1}...`
                        : `Locating Table ${tableId}...`;
                } else if (browseKind && browseKey) {
                    labelEl.textContent = `Locating ${browseKind.toUpperCase()} ${browseKey}...`;
                } else {
                    labelEl.textContent = normalizedSectionNum !== null
                        ? `Locating Article ${normalizedArticleNum}, Section ${normalizedSectionNum}...`
                        : `Locating Article ${normalizedArticleNum}...`;
                }
            }

            const params = new URLSearchParams();
            params.set('contract_id', contractId);
            if (normalizedArticleNum !== null) {
                params.set('article_num', String(normalizedArticleNum));
            }
            if (normalizedSectionNum !== null) {
                params.set('section_num', String(normalizedSectionNum));
            }
            if (partNum) {
                params.set('subsection', String(partNum));
            }
            if (tableId) {
                params.set('table_id', tableId);
            }
            if (normalizedRowIndex !== null) {
                params.set('row_index', String(normalizedRowIndex));
            }
            if (browseKind) {
                params.set('browse_kind', browseKind);
            }
            if (browseKey) {
                params.set('browse_key', browseKey);
            }
            if (sourceContext.sourceType) {
                params.set('source_type', String(sourceContext.sourceType));
            }
            if (sourceContext.sourcePdf) {
                params.set('source_pdf', String(sourceContext.sourcePdf));
            }
            if (sourceContext.sourceDocId) {
                params.set('source_doc_id', String(sourceContext.sourceDocId));
            }
            if (sourceContext.sourcePage && sourceContext.sourcePage > 0) {
                params.set('source_page', String(sourceContext.sourcePage));
            }

            try {
                const res = await fetch(`${API_BASE}/api/pdf-location?${params.toString()}`);
                if (requestToken !== pdfNavRequestSeq) return false;
                if (!res.ok) {
                    if (res.status === 404) {
                        _rememberPdfNavigationContext(pdfTarget, null);
                        const baseUrl = _buildContractPdfEndpoint(
                            contractId,
                            sourceContext.sourceType,
                            sourceContext.sourcePdf,
                            sourceContext.sourceDocId
                        );
                        const fallbackLabel = tableId
                            ? `Table ${tableId} -> Contract PDF`
                            : (browseKind && browseKey)
                                ? `${browseKind.toUpperCase()} ${browseKey} -> Contract PDF`
                                : `Article ${normalizedArticleNum} -> Contract PDF`;
                        if (_isEdgeBrowser() && isOriginalPdfViewMode()) {
                            _openEdgePdfFallback(baseUrl, null, fallbackLabel);
                            return true;
                        }
                        _showContractPdfOverlay(baseUrl, null, fallbackLabel, { requestToken });
                        return true;
                    }
                    _clearPdfNavigationContext();
                    return false;
                }
                const loc = await res.json();
                if (requestToken !== pdfNavRequestSeq) return false;
                if (!loc?.pdf_url) {
                    _clearPdfNavigationContext();
                    return false;
                }
                _rememberPdfNavigationContext(pdfTarget, loc);

                const locationBits = [];
                if (tableId) {
                    locationBits.push(`Table ${tableId}`);
                    if (normalizedRowIndex !== null) {
                        locationBits.push(`Row ${normalizedRowIndex + 1}`);
                    }
                } else if (browseKind && browseKey) {
                    locationBits.push(`${browseKind.toUpperCase()} ${browseKey}`);
                } else {
                    locationBits.push(`Article ${normalizedArticleNum}`);
                    if (normalizedSectionNum !== null) {
                        locationBits.push(`Section ${normalizedSectionNum}`);
                    }
                }
                const sectionLabel = locationBits.join(', ');
                const matchedByMap = {
                    section: 'section match',
                    article: 'article match',
                    table: 'table match',
                    table_row: 'table row match',
                    browse_item: 'browse item match',
                };
                const matchedBy = matchedByMap[String(loc?.matched_by || '')] || 'best effort';
                const pageNumber = Number.isFinite(Number(loc?.page_number)) ? Number(loc.page_number) : null;
                const pageLabel = pageNumber ? `Page ${pageNumber}` : 'Top of document';
                const sourceLabel = describeResolvedPdfLocation(loc, sourceContext);
                const label = `${sectionLabel} -> ${pageLabel} (${matchedBy}${sourceLabel ? ` | ${sourceLabel}` : ''})`;
                const rawPdfUrl = String(loc.pdf_url || '').trim();
                const resolvedPdfUrl = /^https?:\/\//i.test(rawPdfUrl)
                    ? rawPdfUrl
                    : `${API_BASE}${rawPdfUrl.startsWith('/') ? '' : '/'}${rawPdfUrl}`;
                if (_isEdgeBrowser() && isOriginalPdfViewMode()) {
                    _openEdgePdfFallback(resolvedPdfUrl, pageNumber, label);
                    return true;
                }
                _showContractPdfOverlay(resolvedPdfUrl, pageNumber, label, { requestToken });
                return true;
            } catch (e) {
                console.error('Failed to open contract PDF location:', e);
                _clearPdfNavigationContext();
                return false;
            }
        }

        async function openContractPdfFromContractTab() {
            const contractId = getActiveContractId();
            if (!contractId) return false;
            if (!activeContractHistory || activeContract?.contract_id !== (activeContractHistory?.contract_id || null)) {
                await loadContractHistory(contractId);
                renderContractHistoryPanel();
            }

            const normalizedCurrentArticle = toPositiveIntOrNull(currentArticleNum);
            if (normalizedCurrentArticle !== null) {
                setActiveArticleInToc(normalizedCurrentArticle);
                return openArticleInPdf(normalizedCurrentArticle, { preferSection: true });
            }

            const hasManifest = await ensureManifestLoaded();
            if (hasManifest) {
                const firstArticle = getSortedArticleNumbers()[0] ?? null;
                if (firstArticle !== null) {
                    currentArticleNum = firstArticle;
                    setActiveArticleInToc(firstArticle);
                    return openArticleInPdf(firstArticle, { preferSection: true });
                }
            }

            const sourceContext = _resolveSourceContextForPdfNavigation();
            const baseUrl = _buildContractPdfEndpoint(
                contractId,
                sourceContext.sourceType,
                sourceContext.sourcePdf,
                sourceContext.sourceDocId
            );
            const requestToken = nextPdfNavToken();
            _clearPdfNavigationContext();
            _showContractPdfOverlay(baseUrl, null, 'Contract PDF', { requestToken });
            return true;
        }

        async function initContractViewer() {
            try {
                const contractId = getActiveContractId();
                if (!contractId) {
                    const promptHTML = '<li class="text-slate-500 text-xs py-2">Complete onboarding to load your contract.</li>';
                    const desktopList = document.getElementById('article-list');
                    const mobileList = document.getElementById('article-list-mobile');
                    if (desktopList) desktopList.innerHTML = promptHTML;
                    if (mobileList) mobileList.innerHTML = promptHTML;
                    contractBrowseGroups = [];
                    currentTocSelection = { kind: null, key: null };
                    _clearPdfNavigationContext();
                    _clearContractTextContext();
                    activeContractHistory = null;
                    renderContractHistoryPanel();
                    return;
                }
                await loadContractHistory(contractId);
                if (isTenantRoute()) {
                    // Tenant workspaces browse their own uploaded contract via
                    // the member API; /api/manifest reads on-disk packs that a
                    // tenant deployment does not have.
                    const outline = await loadTenantContractOutline();
                    renderTOC();
                    initTOCState();
                    renderContractHistoryPanel();
                    return outline;
                }
                const res = await fetch(`${API_BASE}/api/manifest${getContractQueryString()}`);
                if (!res.ok) {
                    throw new Error(`Manifest load failed (${res.status})`);
                }
                const manifest = await res.json();
                articleTitles = manifest.article_titles;
                try {
                    const browseRes = await fetch(`${API_BASE}/api/contract-browse${getContractQueryString()}`);
                    if (browseRes.ok) {
                        const browsePayload = await browseRes.json();
                        contractBrowseGroups = Array.isArray(browsePayload?.groups) ? browsePayload.groups : [];
                    } else {
                        contractBrowseGroups = [];
                    }
                } catch (_) {
                    contractBrowseGroups = [];
                }
                renderTOC();
                initTOCState();
                renderContractHistoryPanel();
            } catch (e) {
                console.error('Failed to load manifest:', e);
                const desktopList = document.getElementById('article-list');
                const mobileList = document.getElementById('article-list-mobile');
                const errorHTML = '<li class="text-red-500 text-xs py-2">Failed to load</li>';
                if (desktopList) desktopList.innerHTML = errorHTML;
                if (mobileList) mobileList.innerHTML = errorHTML;
                contractBrowseGroups = [];
                _clearPdfNavigationContext();
                _clearContractTextContext();
                renderContractHistoryPanel();
            }
        }

        let tenantContractOutline = null;

        async function loadTenantContractOutline() {
            const contractId = getBrowseContractId();
            if (!contractId) {
                articleTitles = {};
                contractBrowseGroups = [];
                return null;
            }
            const res = await fetchWithAuth(
                `${API_BASE}/api/member/contracts/${encodeURIComponent(contractId)}/outline`
            );
            if (!res.ok) throw new Error(`Outline load failed (${res.status})`);
            const outline = await res.json();
            tenantContractOutline = outline;

            articleTitles = {};
            (outline.articles || []).forEach((article) => {
                articleTitles[String(article.article_num)] = article.article_title;
            });
            // Map onto the group/item shape the existing TOC renderer expects.
            contractBrowseGroups = [
                {
                    label: outline.document_title || 'Contract',
                    items: (outline.articles || []).map((article) => ({
                        kind: 'article',
                        // Prefixed to match what setActiveArticleInToc highlights.
                        key: `article:${article.article_num}`,
                        title: article.article_title,
                        article_num: article.article_num,
                    })),
                },
            ];
            return outline;
        }

        function renderTOC() {
            const desktopList = document.getElementById('article-list');
            const mobileList = document.getElementById('article-list-mobile');
            const groups = Array.isArray(contractBrowseGroups) ? contractBrowseGroups : [];

            if (groups.length > 0) {
                const tocHTML = groups.map((group) => {
                    const groupLabel = escapeHtml(String(group?.label || 'Section'));
                    const items = Array.isArray(group?.items) ? group.items : [];
                    const itemsHtml = items.map((item) => {
                        const kind = safeText(item?.kind).toLowerCase() || 'article';
                        const key = safeText(item?.key) || '';
                        const title = String(item?.title || '').split('\n')[0].trim();
                        const articleNum = toPositiveIntOrNull(item?.article_num);
                        // Tenant outlines carry non-numeric article ids
                        // ("appendix-a"); route them through loadArticle, not
                        // the legacy on-disk browse endpoint.
                        const stringArticleKey = kind === 'article' && articleNum === null && isTenantRoute()
                            ? safeText(item?.article_num)
                            : '';
                        let prefix = kind === 'article' && articleNum !== null
                            ? `${articleNum}.`
                            : (safeText(item?.label) || kind.toUpperCase());
                        if (kind === 'appendix' || stringArticleKey.toLowerCase().startsWith('appendix')) {
                            prefix = 'Appx';
                        } else if (stringArticleKey.toLowerCase().startsWith('letter')) {
                            prefix = 'LOU';
                        } else if (stringArticleKey) {
                            prefix = '§';
                        }
                        const titleText = title || safeText(item?.label) || key;
                        const safeKind = escapeJsSingleQuoted(kind);
                        const safeKey = escapeJsSingleQuoted(key);
                        const clickHandler = kind === 'article' && articleNum !== null
                            ? `loadArticle(${articleNum})`
                            : stringArticleKey
                                ? `loadArticle('${escapeJsSingleQuoted(stringArticleKey)}')`
                                : `loadContractBrowseItem('${safeKind}', '${safeKey}')`;
                        return `
                            <li>
                                <button
                                    class="toc-item w-full text-left px-2 py-1.5 rounded text-xs hover:bg-ufcw-blue/10 transition-colors flex items-baseline gap-1.5"
                                    onclick="${clickHandler}"
                                    data-toc-kind="${escapeHtml(kind)}"
                                    data-toc-key="${escapeHtml(key)}"
                                    ${articleNum !== null ? `data-article="${articleNum}"` : ''}
                                >
                                    <span class="font-semibold text-ufcw-blue shrink-0 ${kind === 'article' ? 'w-6' : ''}">${escapeHtml(prefix)}</span>
                                    <span class="text-slate-600 truncate">${escapeHtml(titleText)}</span>
                                </button>
                            </li>
                        `;
                    }).join('');
                    return `
                        <li class="mt-2 first:mt-0">
                            <div class="px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-slate-400">${groupLabel}</div>
                            <ul class="space-y-px">${itemsHtml}</ul>
                        </li>
                    `;
                }).join('');

                if (desktopList) desktopList.innerHTML = tocHTML;
                if (mobileList) mobileList.innerHTML = tocHTML;
                if (currentTocSelection?.kind && currentTocSelection?.key) {
                    setActiveTocSelection(currentTocSelection.kind, currentTocSelection.key);
                }
                return;
            }

            const entries = Object.entries(articleTitles)
                .map(([num, title]) => ({ num: parseInt(num), title: title.split('\n')[0].trim() }))
                .sort((a, b) => a.num - b.num);

            // Compact single-line items for easy scanning
            const tocHTML = entries.map(({ num, title }) => `
                <li>
                    <button
                        class="toc-item w-full text-left px-2 py-1.5 rounded text-xs hover:bg-ufcw-blue/10 transition-colors flex items-baseline gap-1.5"
                        onclick="loadArticle(${num})"
                        data-article="${num}"
                        data-toc-kind="article"
                        data-toc-key="article:${num}"
                    >
                        <span class="font-semibold text-ufcw-blue shrink-0 w-6">${num}.</span>
                        <span class="text-slate-600 truncate">${escapeHtml(title)}</span>
                    </button>
                </li>
            `).join('');

            // Populate both mobile and desktop lists
            if (desktopList) desktopList.innerHTML = tocHTML;
            if (mobileList) mobileList.innerHTML = tocHTML;
        }

        async function fetchContractBrowseItemForTextSource(kind, key, sourceMode) {
            const contractId = getActiveContractId();
            const normalizedKind = safeText(kind).toLowerCase() || null;
            const normalizedKey = safeText(key) || null;
            if (!contractId || !normalizedKind || !normalizedKey) {
                throw new Error('Missing contract browse target');
            }
            const params = new URLSearchParams();
            params.set('contract_id', contractId);
            params.set('kind', normalizedKind);
            params.set('key', normalizedKey);
            params.set('source_view', normalizeContractTextSourceMode(sourceMode));
            const res = await fetch(`${API_BASE}/api/contract-browse-item?${params.toString()}`);
            if (!res.ok) {
                throw new Error(`Browse item load failed (${res.status})`);
            }
            return res.json();
        }

        async function fetchArticleForTextSource(articleNum, sourceMode) {
            const normalizedArticleNum = toPositiveIntOrNull(articleNum);
            const contractId = getActiveContractId();
            if (!contractId || normalizedArticleNum === null) {
                throw new Error('Missing article target');
            }
            const params = new URLSearchParams();
            params.set('contract_id', contractId);
            params.set('source_view', normalizeContractTextSourceMode(sourceMode));
            const res = await fetch(`${API_BASE}/api/article/${normalizedArticleNum}?${params.toString()}`);
            if (!res.ok) {
                throw new Error(`Article load failed (${res.status})`);
            }
            return res.json();
        }

        async function ensureContractTextPayloadForMode(sourceMode) {
            const mode = normalizeContractTextSourceMode(sourceMode);
            const target = lastContractTextContext?.target || null;
            if (!target) return null;
            const existing = lastContractTextContext?.payloads?.[mode] || null;
            if (existing) return existing;

            if (safeText(target.kind).toLowerCase() === 'article') {
                const articleNum = toPositiveIntOrNull(target.articleNum) || toPositiveIntOrNull(String(target.key || '').split(':')[1]);
                if (articleNum === null) return null;
                const payload = await fetchArticleForTextSource(articleNum, mode);
                _rememberContractTextPayload(target, payload, mode);
                return payload;
            }

            const payload = await fetchContractBrowseItemForTextSource(target.kind, target.key, mode);
            _rememberContractTextPayload(target, payload, mode);
            return payload;
        }

        async function toggleContractTextCompare() {
            const hasTarget = !!lastContractTextContext?.target;
            if (!hasTarget) return;
            contractTextCompareOpen = !contractTextCompareOpen;
            if (contractTextCompareOpen) {
                const opposite = normalizeContractTextSourceMode(contractTextSourceMode) === 'base' ? 'effective' : 'base';
                try {
                    await ensureContractTextPayloadForMode(opposite);
                } catch (err) {
                    console.warn('Unable to load comparison text source:', err);
                }
            }
            updateContractTextPanelChrome();
            renderContractTextComparePanel();
        }

        async function loadContractBrowseItem(kind, key, options = {}) {
            const { openPdf = isOriginalPdfViewMode() } = options;
            const contractId = getActiveContractId();
            const normalizedKind = safeText(kind).toLowerCase() || null;
            const normalizedKey = safeText(key) || null;
            if (!contractId || !normalizedKind || !normalizedKey) return false;

            currentArticleNum = null;
            setActiveTocSelection(normalizedKind, normalizedKey);

            if (window.innerWidth < 768 && tocExpanded) {
                toggleTOC();
            }

            let data = null;
            try {
                const textMode = normalizeContractTextSourceMode(contractTextSourceMode);
                data = await fetchContractBrowseItemForTextSource(normalizedKind, normalizedKey, textMode);
                renderNonArticleContent(data);
                _rememberContractTextPayload(
                    { kind: normalizedKind, key: normalizedKey },
                    data,
                    textMode,
                );
            } catch (err) {
                console.error('Failed to load contract browse item:', err);
                renderNonArticleContent(null, { error: 'Failed to load item content.' });
                return false;
            }

            if (!openPdf) return true;
            return openContractInPdf(
                null,
                null,
                null,
                {
                    browseKind: normalizedKind,
                    browseKey: normalizedKey,
                    ...resolvePdfSourceHintFromPayload(data, normalizeContractTextSourceMode(contractTextSourceMode)),
                }
            );
        }

        async function loadTenantArticle(articleNum) {
            const contractId = getBrowseContractId();
            if (!contractId) return null;
            const outline = tenantContractOutline;
            const article = (outline?.articles || []).find(
                (item) => String(item.article_num) === String(articleNum)
            );
            const sections = [];
            // Fetch each section's text so the reading pane shows the article
            // the way the contract reads, not one merged block.
            for (const entry of article?.sections || []) {
                const params = new URLSearchParams({ article_num: String(articleNum), section_num: String(entry.section_num) });
                const res = await fetchWithAuth(
                    `${API_BASE}/api/member/contracts/${encodeURIComponent(contractId)}/section?${params}`
                );
                if (!res.ok) continue;
                const payload = await res.json();
                sections.push({
                    section_num: entry.section_num,
                    section_title: payload.section_label || entry.section_label || '',
                    content: payload.content || '',
                    page_start: payload.page ?? entry.page ?? null,
                    anchor_id: payload.anchor_id || entry.anchor_id || null,
                });
            }
            return {
                article_num: articleNum,
                article_title: article?.article_title || `Article ${articleNum}`,
                sections,
                source_pdf_url: outline?.source_pdf_url || null,
            };
        }

        async function loadArticleTextForCurrentMode(articleNum) {
            // Tenant outlines can carry non-numeric article ids ("appendix-a");
            // the legacy pack endpoints only ever take numbers.
            const numericArticleNum = toPositiveIntOrNull(articleNum);
            const normalizedArticleNum = numericArticleNum !== null
                ? numericArticleNum
                : (isTenantRoute() ? (safeText(articleNum) || null) : null);
            const contractId = isTenantRoute() ? getBrowseContractId() : getActiveContractId();
            if (!contractId || normalizedArticleNum === null) return false;
            if (isTenantRoute()) {
                try {
                    const data = await loadTenantArticle(normalizedArticleNum);
                    if (!data) return null;
                    renderArticleContent(data);
                    _rememberContractTextPayload(
                        { kind: 'article', key: `article:${normalizedArticleNum}`, articleNum: normalizedArticleNum },
                        data,
                        'effective',
                    );
                    return data;
                } catch (err) {
                    console.error('Failed to load tenant article text:', err);
                    renderNonArticleContent(null, { error: 'Failed to load article text.' });
                    return null;
                }
            }
            try {
                const textMode = normalizeContractTextSourceMode(contractTextSourceMode);
                const data = await fetchArticleForTextSource(normalizedArticleNum, textMode);
                renderArticleContent(data);
                _rememberContractTextPayload(
                    { kind: 'article', key: `article:${normalizedArticleNum}`, articleNum: normalizedArticleNum },
                    data,
                    textMode,
                );
                return data;
            } catch (err) {
                console.error('Failed to load article text:', err);
                renderNonArticleContent(null, { error: 'Failed to load article text.' });
                return null;
            }
        }

        async function loadArticle(articleNum, options = {}) {
            const { openPdf = isOriginalPdfViewMode() } = options;
            const contractId = isTenantRoute() ? getBrowseContractId() : getActiveContractId();
            const numericArticleNum = toPositiveIntOrNull(articleNum);
            const normalizedArticleNum = numericArticleNum !== null
                ? numericArticleNum
                : (isTenantRoute() ? (safeText(articleNum) || null) : null);
            if (!contractId || normalizedArticleNum === null) return false;
            currentArticleNum = normalizedArticleNum;
            setActiveArticleInToc(normalizedArticleNum);

            // On mobile, collapse TOC after selection for more reading space
            if (window.innerWidth < 768 && tocExpanded) {
                toggleTOC();
            }

            const articleData = await loadArticleTextForCurrentMode(normalizedArticleNum);

            if (!openPdf) return true;
            return openArticleInPdf(normalizedArticleNum, {
                preferSection: true,
                ...resolvePdfSourceHintFromPayload(articleData, normalizeContractTextSourceMode(contractTextSourceMode)),
            });
        }

        function _isMarkdownTableSeparator(line) {
            return /^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$/.test(line || '');
        }

        function _splitMarkdownTableRow(line) {
            let row = String(line || '').trim();
            if (row.startsWith('|')) row = row.slice(1);
            if (row.endsWith('|')) row = row.slice(0, -1);
            return row.split('|').map(cell => cell.trim());
        }

        function _renderMarkdownTable(lines) {
            if (!lines.length) return '';
            const rows = lines.map(_splitMarkdownTableRow).filter(r => r.length > 0);
            if (rows.length < 2) return escapeHtml(lines.join('\n'));

            const header = rows[0];
            const bodyRows = rows.slice(1).filter(r => !r.every(cell => /^:?-{3,}:?$/.test(cell.replace(/\s+/g, ''))));
            const headerHtml = header.map(cell => `<th>${escapeHtml(cell)}</th>`).join('');
            const bodyHtml = bodyRows.map(
                row => `<tr>${row.map(cell => `<td>${escapeHtml(cell)}</td>`).join('')}</tr>`
            ).join('');

            return `
                <div class="contract-table-wrap">
                    <table class="contract-table">
                        <thead><tr>${headerHtml}</tr></thead>
                        <tbody>${bodyHtml}</tbody>
                    </table>
                </div>
            `;
        }

        function renderMarkdown(text) {
            const lines = String(text || '').replace(/\r\n/g, '\n').split('\n');
            const placeholders = [];
            const outLines = [];

            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                const nextLine = lines[i + 1] || '';
                const looksLikeTableHeader = line.includes('|') && _isMarkdownTableSeparator(nextLine);
                if (looksLikeTableHeader) {
                    const tableLines = [line, nextLine];
                    i += 2;
                    while (i < lines.length && lines[i].includes('|')) {
                        tableLines.push(lines[i]);
                        i += 1;
                    }
                    i -= 1;
                    const idx = placeholders.length;
                    placeholders.push(_renderMarkdownTable(tableLines));
                    outLines.push(`%%PH${idx}%%`);
                    continue;
                }
                outLines.push(line);
            }

            let processed = outLines.join('\n');

            // Convert **bold** to placeholder
            processed = processed.replace(/\*\*(.+?)\*\*/g, (match, content) => {
                const idx = placeholders.length;
                placeholders.push(`<strong>${escapeHtml(content)}</strong>`);
                return `%%PH${idx}%%`;
            });

            // Convert ### Headings to placeholder
            processed = processed.replace(/^### (.+)$/gm, (match, content) => {
                const idx = placeholders.length;
                placeholders.push(`<h4 class="text-sm font-semibold text-slate-800 mt-3 mb-2">${escapeHtml(content)}</h4>`);
                return `%%PH${idx}%%`;
            });

            // Convert ## Headings to placeholder
            processed = processed.replace(/^## (.+)$/gm, (match, content) => {
                const idx = placeholders.length;
                placeholders.push(`<h3 class="text-base font-semibold text-slate-800 mt-4 mb-2">${escapeHtml(content)}</h3>`);
                return `%%PH${idx}%%`;
            });

            // Escape the remaining text
            processed = escapeHtml(processed);

            // Restore placeholders
            placeholders.forEach((html, idx) => {
                processed = processed.replace(`%%PH${idx}%%`, html);
            });

            // Convert numbered lists (1. 2. etc at start of line) - after escaping
            processed = processed.replace(/^(\d+)\. (.+)$/gm, '<div class="ml-4 mb-1"><span class="font-medium">$1.</span> $2</div>');

            // Convert bullet points
            processed = processed.replace(/^[-*\u2022] (.+)$/gm, '<div class="ml-4 mb-1">&bull; $1</div>');

            return processed;
        }

        function renderPopoverMarkdown(text) {
            const rendered = renderMarkdown(text || '');
            return parseCitations(rendered);
        }

        function renderInlineMarkdown(text) {
            let processed = escapeHtml(text || '');
            processed = processed.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold">$1</strong>');
            processed = processed.replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-slate-100 text-[11px]">$1</code>');
            return processed;
        }

        function getPreferredResponseTone() {
            return safeText(preferences.responseTone).toLowerCase() || 'calm';
        }

        function getPreferredResponseVerbosity() {
            return safeText(preferences.responseVerbosity).toLowerCase() || 'balanced';
        }

        function updateKarlVersionUI() {
            const version = safeText(karlInfo?.version) || KARL_VERSION_FALLBACK;
            const releaseChannel = safeText(karlInfo?.release_channel);
            const versionLabel = `KARL v${version}`;
            const settingsVersion = document.getElementById('settings-karl-version');
            const metaLine = document.getElementById('karl-meta-line');
            if (settingsVersion) settingsVersion.textContent = versionLabel;
            if (metaLine) {
                metaLine.textContent = releaseChannel ? `${versionLabel} · ${releaseChannel}` : versionLabel;
            }
        }

        function populateKarlDocSelect() {
            const select = document.getElementById('karl-doc-select');
            if (!select) return;
            const docs = Array.isArray(karlInfo?.documents) ? karlInfo.documents : [];
            const previousValue = currentKarlDocId || select.value || '';
            const options = ['<option value="">Select a document...</option>'];
            docs.forEach((doc) => {
                const id = safeText(doc?.id);
                const title = safeText(doc?.title) || id;
                if (!id) return;
                options.push(`<option value="${escapeHtml(id)}">${escapeHtml(title)}</option>`);
            });
            select.innerHTML = options.join('');
            if (previousValue && docs.some((doc) => safeText(doc?.id) === previousValue)) {
                select.value = previousValue;
            }
        }

        async function loadKarlInfo(force = false) {
            if (karlInfo && !force) {
                updateKarlVersionUI();
                populateKarlDocSelect();
                return karlInfo;
            }
            const res = await fetch(`${API_BASE}/api/karl/info`);
            if (!res.ok) {
                throw new Error(`Unable to load KARL info (${res.status})`);
            }
            karlInfo = await res.json();
            updateKarlVersionUI();
            populateKarlDocSelect();
            return karlInfo;
        }

        function renderKarlDoc(doc) {
            const viewer = document.getElementById('karl-doc-viewer');
            if (!viewer) return;
            if (!doc) {
                viewer.innerHTML = '<p class="text-slate-500">Document unavailable.</p>';
                return;
            }
            const title = safeText(doc.title) || 'KARL Document';
            const path = safeText(doc.path);
            viewer.innerHTML = `
                <article class="max-w-none">
                    <header class="mb-6 border-b border-slate-200 pb-4">
                        <p class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">KARL Library</p>
                        <h1 class="mt-1 text-2xl font-semibold text-slate-900">${escapeHtml(title)}</h1>
                        ${path ? `<p class="mt-2 text-xs text-slate-500">${escapeHtml(path)}</p>` : ''}
                    </header>
                    <div class="text-sm leading-relaxed text-slate-700 whitespace-pre-wrap">${renderMarkdown(doc.content || '')}</div>
                </article>
            `;
        }

        async function loadKarlDoc(docId) {
            const normalizedId = safeText(docId);
            if (!normalizedId) return;
            currentKarlDocId = normalizedId;
            const viewer = document.getElementById('karl-doc-viewer');
            if (viewer) {
                viewer.innerHTML = '<p class="text-slate-500">Loading KARL document...</p>';
            }
            const res = await fetch(`${API_BASE}/api/karl/doc/${encodeURIComponent(normalizedId)}`);
            if (!res.ok) {
                throw new Error(`Unable to load KARL document (${res.status})`);
            }
            const doc = await res.json();
            renderKarlDoc(doc);
            const select = document.getElementById('karl-doc-select');
            if (select) select.value = normalizedId;
        }

        async function handleKarlDocChange(docId) {
            if (!safeText(docId)) return;
            try {
                await loadKarlDoc(docId);
            } catch (error) {
                const viewer = document.getElementById('karl-doc-viewer');
                if (viewer) {
                    viewer.innerHTML = `<p class="text-red-600">${escapeHtml(error.message || 'Unable to load KARL document.')}</p>`;
                }
            }
        }

        async function openKarlTab(docId = null) {
            setActiveTab('karl');
            try {
                await loadKarlInfo();
                const fallbackDocId = safeText(docId) || currentKarlDocId || safeText(karlInfo?.documents?.[0]?.id);
                if (fallbackDocId) {
                    await loadKarlDoc(fallbackDocId);
                } else {
                    renderKarlDoc(null);
                }
            } catch (error) {
                const viewer = document.getElementById('karl-doc-viewer');
                updateKarlVersionUI();
                if (viewer) {
                    viewer.innerHTML = `<p class="text-red-600">${escapeHtml(error.message || 'Unable to load KARL materials.')}</p>`;
                }
            }
        }

        function renderNonArticleContent(data, options = {}) {
            const detail = document.getElementById('article-detail');
            const detailMobile = document.getElementById('article-detail-mobile');
            const errorText = safeText(options?.error) || '';

            if (!detail && !detailMobile) return;

            if (!data || errorText) {
                const errorHtml = `
                    <article class="max-w-none">
                        <header class="mb-6 pb-4 border-b border-slate-200">
                            <p class="text-ufcw-blue font-semibold text-sm uppercase tracking-wider">Contract Content</p>
                            <h1 class="text-xl md:text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">Unavailable</h1>
                        </header>
                        <p class="text-sm text-red-600 dark:text-red-300">${escapeHtml(errorText || 'Content not found.')}</p>
                    </article>
                `;
                if (detail) detail.innerHTML = errorHtml;
                if (detailMobile) detailMobile.innerHTML = errorHtml;
                return;
            }

            const kind = safeText(data?.kind).toLowerCase();
            const label = safeText(data?.label) || 'Contract Item';
            const title = safeText(data?.title) || '';
            const docType = safeText(data?.doc_type).toUpperCase() || kind.toUpperCase();
            const sections = Array.isArray(data?.sections) ? data.sections : [];

            const sectionsHtml = sections.map((section, idx) => {
                const citation = safeText(section?.citation) || `${label} Part ${idx + 1}`;
                const body = renderPopoverMarkdown(section?.content || '');
                const summary = safeText(section?.summary)
                    ? `<p class="text-xs text-slate-500 italic mt-1">${renderInlineMarkdown(section.summary)}</p>`
                    : '';
                return `
                    <section class="mb-5 pb-5 border-b border-slate-100 last:border-b-0 last:pb-0">
                        <h3 class="text-sm font-semibold text-slate-800 dark:text-slate-100">${escapeHtml(citation)}</h3>
                        ${summary}
                        <div class="mt-2 text-sm text-slate-700 dark:text-slate-200 leading-relaxed">${body}</div>
                    </section>
                `;
            }).join('');

            const articleHTML = `
                <article class="max-w-none">
                    <header class="mb-6 pb-4 border-b border-slate-200">
                        <p class="text-ufcw-blue font-semibold text-sm uppercase tracking-wider">${escapeHtml(docType)}</p>
                        <h1 class="text-xl md:text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">${escapeHtml(label)}${title ? `: ${escapeHtml(title)}` : ''}</h1>
                        <p class="text-xs text-slate-500 mt-1">${sections.length} chunk${sections.length === 1 ? '' : 's'} grouped</p>
                    </header>
                    <div>${sectionsHtml || '<p class="text-sm text-slate-500">No content available.</p>'}</div>
                </article>
            `;

            if (detail) detail.innerHTML = articleHTML;
            if (detailMobile) detailMobile.innerHTML = articleHTML;
        }

        function renderArticleContent(data) {
            const detail = document.getElementById('article-detail');
            const detailMobile = document.getElementById('article-detail-mobile');
            const textSize = preferences.textSize || 'medium';
            const textClass = textSize === 'small' ? 'text-sm' : textSize === 'large' ? 'text-base' : 'text-sm';

            // Generate section ID that includes subsection/part info
            const getSectionId = (section) => {
                let id = `section-${section.section_num}`;
                if (section.subsection) {
                    id += `-${section.subsection}`;
                }
                return id;
            };

            // Non-numeric ids come from tenant outlines ("appendix-a", whose
            // sections are wage classifications, not numbered sections).
            const numericArticle = toPositiveIntOrNull(data.article_num);
            const articleEyebrow = numericArticle !== null
                ? `Article ${numericArticle}`
                : String(data.article_num || '').replace(/-/g, ' ').toUpperCase();
            const sectionHeading = (section) => {
                const numericSection = toPositiveIntOrNull(section.section_num);
                if (numericSection !== null) {
                    return `Section ${section.section_num}${section.subsection ? ` (${section.subsection})` : ''}`;
                }
                return String(section.section_title || section.section_num || '');
            };
            const articleHTML = `
                <article class="max-w-none">
                    <header class="mb-6 pb-4 border-b border-slate-200">
                        <p class="text-ufcw-blue font-semibold text-sm uppercase tracking-wider">${escapeHtml(articleEyebrow)}</p>
                        <h1 class="text-xl md:text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">${escapeHtml(data.article_title)}</h1>
                    </header>

                    ${data.sections.map(section => `
                        <section id="${getSectionId(section)}" class="mb-6 pb-6 border-b border-slate-100 last:border-0 scroll-mt-4">
                            <h2 class="text-base font-semibold text-slate-800 mb-2">
                                ${escapeHtml(sectionHeading(section))}
                            </h2>
                            ${section.summary ? `<p class="text-xs text-slate-500 italic mb-3">${escapeHtml(section.summary)}</p>` : ''}
                            <div class="${textClass} text-slate-700 leading-relaxed whitespace-pre-wrap">${renderMarkdown(section.content)}</div>
                        </section>
                    `).join('')}
                </article>
            `;

            if (detail) detail.innerHTML = articleHTML;
            if (detailMobile) detailMobile.innerHTML = articleHTML;
        }

        async function navigateToArticle(articleNum, sectionNum = null, partNum = null) {
            setActiveTab('contract');
            const normalizedSectionNum = toPositiveIntOrNull(sectionNum);
            const openedArticle = await loadArticle(articleNum, { openPdf: normalizedSectionNum === null && isOriginalPdfViewMode() });
            if (sectionNum) {
                if (!isOriginalPdfViewMode()) return openedArticle;
                const openedSection = await openContractInPdf(articleNum, sectionNum, partNum);
                return openedSection || openedArticle;
            }
            return openedArticle;
        }

        // =============================================================================
        // CITATION POPOVER
        // =============================================================================

        function parseCitations(text) {
            // Pattern matches citation formats:
            // - Article N, Section M, Part X
            // - Article N, Section M(a)  (parenthetical subsection)
            // With optional bold markers
            const pattern = /(\*{0,2})(Article\s+(\d+))(?:,?\s*Section\s+(\d+)(?:\(([a-z])\))?)?(?:,?\s*Part\s+([\w\-]+))?(\*{0,2})/gi;

            return text.replace(pattern, (match, boldStart, fullArticle, articleNum, sectionNum, parenSub, partNum, boldEnd) => {
                // Subsection can come from either parenthetical format or "Part X" format
                const subsection = parenSub || partNum;

                let display = `Article ${articleNum}`;
                if (sectionNum) {
                    display += `, Section ${sectionNum}`;
                    if (parenSub) display += `(${parenSub})`;
                }
                if (partNum && !parenSub) display += `, Part ${partNum}`;

                const subArg = subsection ? `'${subsection}'` : 'null';
                return `<a href="#" class="citation-link" onclick="handleCitationClick(event, ${articleNum}, ${sectionNum || 'null'}, ${subArg})">${boldStart}${display}${boldEnd}</a>`;
            });
        }

        function handleCitationClick(
            event,
            articleNum,
            sectionNum,
            partNum = null,
            tableId = null,
            rowIndex = null,
            citationLabel = null,
            sourceType = null,
            sourcePdf = null,
            sourcePage = null,
            sourceDocId = null,
            sourceRegistryKey = null
        ) {
            event.preventDefault();
            event.stopPropagation();
            const normalizedArticleNum = toPositiveIntOrNull(articleNum);
            const normalizedSectionNum = toPositiveIntOrNull(sectionNum);
            const normalizedTableId = String(tableId || '').trim() || null;
            const normalizedRowIndex = Number.isFinite(Number(rowIndex)) ? Number(rowIndex) : null;
            const normalizedSourceType = safeText(sourceType).toLowerCase() || null;
            const normalizedSourcePdf = safeText(sourcePdf) || null;
            const normalizedSourcePage = Number.isFinite(Number(sourcePage)) ? Number(sourcePage) : null;
            const normalizedSourceDocId = safeText(sourceDocId) || null;
            const normalizedSourceRegistryKey = safeText(sourceRegistryKey) || null;
            if (normalizedArticleNum === null && !normalizedTableId) return;

            const style = preferences.citationStyle || 'popover';
            if (style === 'navigate') {
                setActiveTab('contract');
                if (normalizedArticleNum !== null) {
                    setActiveArticleInToc(normalizedArticleNum);
                }
                if (normalizedArticleNum !== null) {
                    const shouldOpenArticlePdf =
                        isOriginalPdfViewMode()
                        && normalizedSectionNum === null
                        && normalizedTableId === null;
                    loadArticle(normalizedArticleNum, { openPdf: shouldOpenArticlePdf }).catch(() => {});
                }
                if (isOriginalPdfViewMode() || normalizedArticleNum === null || normalizedTableId) {
                    openContractInPdf(
                        normalizedArticleNum,
                        normalizedSectionNum,
                        partNum,
                        {
                            tableId: normalizedTableId,
                            rowIndex: normalizedRowIndex,
                            sourceType: normalizedSourceType,
                            sourcePdf: normalizedSourcePdf,
                            sourcePage: normalizedSourcePage,
                            sourceDocId: normalizedSourceDocId,
                        }
                    );
                }
            } else {
                showPopover(
                    normalizedArticleNum,
                    normalizedSectionNum,
                    partNum,
                    event.target,
                    {
                        tableId: normalizedTableId,
                        rowIndex: normalizedRowIndex,
                        citationLabel: citationLabel ? String(citationLabel) : null,
                        sourceType: normalizedSourceType,
                        sourcePdf: normalizedSourcePdf,
                        sourcePage: normalizedSourcePage,
                        sourceDocId: normalizedSourceDocId,
                        sourceRegistryKey: normalizedSourceRegistryKey,
                    }
                );
            }
        }

        function buildPopoverSourceChoices(options = {}) {
            const rows = [];
            rows.push({
                key: 'effective_auto',
                label: 'Current Effective (Auto)',
                sourceType: null,
                sourcePdf: null,
                sourcePage: null,
                sourceDocId: null,
                isPrevious: false,
            });

            const explicitType = safeText(options?.sourceType).toLowerCase() || null;
            const explicitPdf = safeText(options?.sourcePdf) || null;
            const explicitDocId = safeText(options?.sourceDocId) || null;
            const explicitPage = Number.isFinite(Number(options?.sourcePage)) ? Number(options.sourcePage) : null;
            const sourceRegistryKey = safeText(options?.sourceRegistryKey) || null;
            const sourceRecord = sourceRegistryKey ? getCitationSourceRecord(sourceRegistryKey) : null;
            const provenance = Array.isArray(sourceRecord?.provenance) ? sourceRecord.provenance : [];

            provenance.forEach((ref, idx) => {
                if (!ref || typeof ref !== 'object') return;
                const refTypeRaw = safeText(ref.source_type).toLowerCase();
                const refPdf = safeText(ref.pdf) || null;
                const refDocId = safeText(ref.source_doc_id) || null;
                const refPage = Number.isFinite(Number(ref.pdf_page)) ? Number(ref.pdf_page) : null;
                let refType = refTypeRaw || null;
                let labelPrefix = 'Source';
                if (refTypeRaw.includes('moa') || refTypeRaw.includes('amend') || (refPdf && isMoaPdfName(refPdf))) {
                    refType = 'moa';
                    labelPrefix = 'MOA';
                } else if (refPdf || refTypeRaw.includes('base') || refTypeRaw.includes('cba') || refTypeRaw.includes('contract')) {
                    refType = 'base';
                    labelPrefix = 'Base';
                }
                const labelBits = [labelPrefix];
                if (refPage && refPage > 0) labelBits.push(`p.${refPage}`);
                if (refPdf) labelBits.push(refPdf);
                rows.push({
                    key: `prov_${idx}_${refType || 'src'}`,
                    label: labelBits.join(' | '),
                    sourceType: refType,
                    sourcePdf: refPdf,
                    sourcePage: refPage,
                    sourceDocId: refDocId,
                    isPrevious: refType === 'base',
                });
            });

            if ((explicitType || explicitPdf || explicitPage || explicitDocId) && provenance.length === 0) {
                const labelBits = [explicitType === 'moa' ? 'MOA' : (explicitType === 'base' ? 'Base' : 'Source')];
                if (explicitPage && explicitPage > 0) labelBits.push(`p.${explicitPage}`);
                if (explicitPdf) labelBits.push(explicitPdf);
                rows.push({
                    key: 'hint_explicit',
                    label: labelBits.join(' | '),
                    sourceType: explicitType,
                    sourcePdf: explicitPdf,
                    sourcePage: explicitPage && explicitPage > 0 ? explicitPage : null,
                    sourceDocId: explicitDocId,
                    isPrevious: explicitType === 'base',
                });
            }

            const deduped = [];
            const seen = new Set();
            rows.forEach((row) => {
                if (!row || typeof row !== 'object') return;
                const sig = [
                    safeText(row.sourceType).toLowerCase(),
                    safeText(row.sourcePdf).toLowerCase(),
                    safeText(row.sourceDocId).toLowerCase(),
                    Number.isFinite(Number(row.sourcePage)) ? Number(row.sourcePage) : 0,
                ].join('|');
                if (seen.has(sig)) return;
                seen.add(sig);
                deduped.push(row);
            });

            const hasBase = deduped.some((row) => safeText(row.sourceType).toLowerCase() === 'base');
            if (hasBase) {
                const baseRow = deduped.find((row) => safeText(row.sourceType).toLowerCase() === 'base');
                if (baseRow) {
                    const prevSig = [
                        safeText(baseRow.sourceType).toLowerCase(),
                        safeText(baseRow.sourcePdf).toLowerCase(),
                        safeText(baseRow.sourceDocId).toLowerCase(),
                        Number.isFinite(Number(baseRow.sourcePage)) ? Number(baseRow.sourcePage) : 0,
                    ].join('|');
                    const aliasSig = `${prevSig}|previous_alias`;
                    if (!seen.has(aliasSig)) {
                        seen.add(aliasSig);
                        deduped.push({
                            ...baseRow,
                            key: 'previous_base',
                            label: 'Previous (Base)',
                            isPrevious: true,
                        });
                    }
                }
            }

            let selectedKey = 'effective_auto';
            const targetType = explicitType;
            const targetPdf = safeText(explicitPdf).toLowerCase();
            const targetDocId = safeText(explicitDocId).toLowerCase();
            const targetPage = Number.isFinite(Number(explicitPage)) ? Number(explicitPage) : null;
            if (targetType || targetPdf || targetDocId || targetPage) {
                const matched = deduped.find((row) => {
                    const rowType = safeText(row.sourceType).toLowerCase();
                    const rowPdf = safeText(row.sourcePdf).toLowerCase();
                    const rowDocId = safeText(row.sourceDocId).toLowerCase();
                    const rowPage = Number.isFinite(Number(row.sourcePage)) ? Number(row.sourcePage) : null;
                    const typeOk = !targetType || rowType === targetType;
                    const pdfOk = !targetPdf || rowPdf === targetPdf;
                    const docOk = !targetDocId || rowDocId === targetDocId;
                    const pageOk = targetPage === null || rowPage === targetPage;
                    return typeOk && pdfOk && docOk && pageOk;
                });
                if (matched?.key) {
                    selectedKey = String(matched.key);
                }
            }

            return {
                choices: deduped,
                selectedKey,
            };
        }

        function renderPopoverSourceControls() {
            const wrap = document.getElementById('popover-source-controls');
            const select = document.getElementById('popover-source-select');
            if (!wrap || !select) return;
            if (!isDeveloperModeEnabled() || !isOriginalPdfViewMode()) {
                wrap.classList.add('hidden');
                select.innerHTML = '';
                return;
            }

            const choices = Array.isArray(currentPopover.sourceChoices) ? currentPopover.sourceChoices : [];
            if (choices.length <= 1) {
                wrap.classList.add('hidden');
                select.innerHTML = '';
                return;
            }

            const selectedKey = safeText(currentPopover.selectedSourceChoiceKey) || safeText(choices[0]?.key) || 'effective_auto';
            select.innerHTML = choices.map((choice) => {
                const key = escapeHtml(String(choice?.key || ''));
                const label = escapeHtml(String(choice?.label || 'Source'));
                const selectedAttr = String(choice?.key || '') === selectedKey ? ' selected' : '';
                return `<option value="${key}"${selectedAttr}>${label}</option>`;
            }).join('');
            wrap.classList.remove('hidden');
        }

        function handlePopoverSourceChoiceChange(event) {
            const nextKey = safeText(event?.target?.value) || null;
            if (!nextKey) return;
            currentPopover.selectedSourceChoiceKey = nextKey;
        }

        async function showPopover(articleNum, sectionNum, partNum, anchorEl, options = {}) {
            const popover = document.getElementById('citation-popover');
            const titleEl = document.getElementById('popover-title');
            const subtitleEl = document.getElementById('popover-subtitle');
            const loadingEl = document.getElementById('popover-loading');
            const textEl = document.getElementById('popover-text');
            const summaryEl = document.getElementById('popover-summary');
            const openBtn = document.getElementById('popover-open-contract-btn');
            const tableId = String(options?.tableId || '').trim() || null;
            const rowIndex = Number.isFinite(Number(options?.rowIndex)) ? Number(options.rowIndex) : null;
            const citationLabel = options?.citationLabel ? String(options.citationLabel) : null;
            const sourceType = safeText(options?.sourceType).toLowerCase() || null;
            const sourcePdf = safeText(options?.sourcePdf) || null;
            const sourcePage = Number.isFinite(Number(options?.sourcePage)) ? Number(options.sourcePage) : null;
            const sourceDocId = safeText(options?.sourceDocId) || null;
            const sourceRegistryKey = safeText(options?.sourceRegistryKey) || null;
            const sourceChoiceState = buildPopoverSourceChoices({
                sourceType,
                sourcePdf,
                sourcePage,
                sourceDocId,
                sourceRegistryKey,
            });

            currentPopover = {
                articleNum,
                sectionNum,
                partNum,
                tableId,
                rowIndex,
                citationLabel,
                sourceRegistryKey,
                sourceDocId,
                sourceType,
                sourcePdf,
                sourcePage,
                sourceChoices: sourceChoiceState.choices,
                selectedSourceChoiceKey: sourceChoiceState.selectedKey,
            };
            renderPopoverSourceControls();
            if (openBtn) {
                openBtn.textContent = isOriginalPdfViewMode() ? 'Open In Original PDF' : 'Open In Contract Viewer';
            }

            // Set title
            let title = citationLabel || '';
            if (!title) {
                if (tableId) {
                    title = `Appendix Table ${tableId}`;
                    if (rowIndex !== null) title += `, Row ${rowIndex + 1}`;
                } else {
                    title = `Article ${articleNum}`;
                    if (sectionNum) title += `, Section ${sectionNum}`;
                    if (partNum) title += `, Part ${partNum}`;
                }
            }
            titleEl.textContent = title;
            const summaryBits = [];
            const selectedSource = sourceChoiceState.choices.find((choice) => safeText(choice?.key) === sourceChoiceState.selectedKey) || null;
            if (selectedSource && safeText(selectedSource.key) !== 'effective_auto') {
                summaryBits.push(describePdfSourceChoice(selectedSource));
            } else {
                summaryBits.push(isOriginalPdfViewMode() ? 'Original PDF follows effective provenance by default.' : 'Effective contract view is the default reader.');
            }
            if (sourceChoiceState.choices.length > 1) {
                summaryBits.push('Use the PDF Source menu in the contract viewer to switch between latest and original versions.');
            }
            subtitleEl.textContent = summaryBits[0] || '';
            if (summaryEl) {
                summaryEl.textContent = summaryBits.join(' ');
            }

            // Show loading
            loadingEl.classList.remove('hidden');
            textEl.classList.add('hidden');

            // Position popover
            const rect = anchorEl.getBoundingClientRect();
            let top = rect.bottom + 8;
            let left = Math.min(rect.left, window.innerWidth - 420);

            // If would go below viewport, show above
            if (top + 320 > window.innerHeight) {
                top = rect.top - 320 - 8;
            }

            popover.style.top = `${Math.max(8, top)}px`;
            popover.style.left = `${Math.max(8, left)}px`;
            popover.classList.remove('hidden');

            // Fetch content
            try {
                let data;
                let usedFallback = false;

                if (articleNum === null && tableId) {
                    textEl.innerHTML = renderPopoverMarkdown(
                        'This citation points to a table reference. Use **Open In Contract Viewer** to jump directly to the table.'
                    );
                    summaryEl.innerHTML = renderInlineMarkdown(
                        rowIndex !== null ? `Table ${tableId}, Row ${rowIndex + 1}` : `Table ${tableId}`
                    );
                    loadingEl.classList.add('hidden');
                    textEl.classList.remove('hidden');
                    return;
                }

                // Tenant workspaces have no on-disk contract packs behind
                // /api/section//api/article — the member browse API serves the
                // same text from the indexed chunks, so read it from there.
                if (isTenantRoute()) {
                    const browseContractId = getBrowseContractId();
                    if (!browseContractId || toPositiveIntOrNull(articleNum) === null) {
                        textEl.innerHTML = renderPopoverMarkdown('Content not found.');
                        summaryEl.textContent = '';
                    } else {
                        const tenantParams = new URLSearchParams({ article_num: String(articleNum) });
                        if (sectionNum) tenantParams.set('section_num', String(sectionNum));
                        const tenantRes = await fetchWithAuth(
                            `${API_BASE}/api/member/contracts/${encodeURIComponent(browseContractId)}/section?${tenantParams}`
                        );
                        if (tenantRes.ok) {
                            const payload = await tenantRes.json();
                            if (payload.article_title) subtitleEl.textContent = payload.article_title;
                            const content = String(payload.content || '');
                            const preview = content.substring(0, 700);
                            textEl.innerHTML = renderPopoverMarkdown(
                                (preview + (content.length > 700 ? '...' : '')) || 'Content not found.'
                            );
                            const bits = [];
                            if (payload.section_label) bits.push(payload.section_label);
                            if (payload.page) bits.push(`Printed page ${payload.page}`);
                            if (bits.length) summaryEl.innerHTML = renderInlineMarkdown(bits.join(' · '));
                        } else {
                            textEl.innerHTML = renderPopoverMarkdown(
                                'This citation was not found in this contract.'
                            );
                            summaryEl.textContent = '';
                        }
                    }
                    loadingEl.classList.add('hidden');
                    textEl.classList.remove('hidden');
                    return;
                }

                // Try to fetch specific section first
                if (sectionNum) {
                    // Build URL with optional subsection query param
                    const sectionParams = new URLSearchParams();
                    const contractId = getActiveContractId();
                    if (contractId) sectionParams.set('contract_id', contractId);
                    if (partNum) {
                        sectionParams.set('subsection', partNum);
                    }
                    let sectionUrl = `${API_BASE}/api/section/${articleNum}/${sectionNum}`;
                    const sectionQuery = sectionParams.toString();
                    if (sectionQuery) sectionUrl += `?${sectionQuery}`;
                    const res = await fetch(sectionUrl);
                    if (res.ok) {
                        data = await res.json();
                    } else {
                        // Section not found - fall back to article
                        console.log(`Section ${sectionNum} not found, falling back to article ${articleNum}`);
                        const articleRes = await fetch(`${API_BASE}/api/article/${articleNum}${getContractQueryString()}`);
                        if (articleRes.ok) {
                            data = await articleRes.json();
                            usedFallback = true;
                        }
                    }
                } else {
                    // Just fetch the article
                    const res = await fetch(`${API_BASE}/api/article/${articleNum}${getContractQueryString()}`);
                    if (res.ok) {
                        data = await res.json();
                    }
                }

                if (!data) {
                    textEl.textContent = 'Article not found in contract.';
                    summaryEl.textContent = '';
                    loadingEl.classList.add('hidden');
                    textEl.classList.remove('hidden');
                    return;
                }

                subtitleEl.textContent = data.article_title || '';

                if (sectionNum && !usedFallback) {
                    textEl.innerHTML = renderPopoverMarkdown(data.content || 'Content not found.');
                    summaryEl.innerHTML = renderInlineMarkdown(data.summary || '');
                } else {
                    // Show article preview (either requested or as fallback)
                    const firstSection = data.sections?.[0];
                    if (firstSection) {
                        const sectionContent = String(firstSection.content || '');
                        const preview = sectionContent.substring(0, 400);
                        textEl.innerHTML = renderPopoverMarkdown(preview + (sectionContent.length > 400 ? '...' : ''));
                        if (usedFallback) {
                            summaryEl.innerHTML = renderInlineMarkdown(
                                `Section ${sectionNum} not found. Showing Article ${articleNum} (${data.sections.length} sections).`
                            );
                        } else {
                            summaryEl.innerHTML = renderInlineMarkdown(`${data.sections.length} sections total`);
                        }
                    } else {
                        textEl.innerHTML = renderPopoverMarkdown('Content not found.');
                        summaryEl.textContent = '';
                    }
                }

                loadingEl.classList.add('hidden');
                textEl.classList.remove('hidden');

            } catch (e) {
                console.error('Failed to load citation:', e);
                textEl.textContent = 'Failed to load content. Please try again.';
                summaryEl.textContent = '';
                loadingEl.classList.add('hidden');
                textEl.classList.remove('hidden');
            }
        }

        function hidePopover() {
            document.getElementById('citation-popover').classList.add('hidden');
        }

        async function expandToContract() {
            hidePopover();
            const articleNum = toPositiveIntOrNull(currentPopover.articleNum);
            const sectionNum = toPositiveIntOrNull(currentPopover.sectionNum);
            const tableId = String(currentPopover.tableId || '').trim() || null;
            const rowIndex = Number.isFinite(Number(currentPopover.rowIndex)) ? Number(currentPopover.rowIndex) : null;
            const sourceType = safeText(currentPopover.sourceType).toLowerCase() || null;
            const sourcePdf = safeText(currentPopover.sourcePdf) || null;
            const sourcePage = Number.isFinite(Number(currentPopover.sourcePage)) ? Number(currentPopover.sourcePage) : null;
            const sourceDocId = safeText(currentPopover.sourceDocId) || null;
            const sourceChoices = Array.isArray(currentPopover.sourceChoices) ? currentPopover.sourceChoices : [];
            const selectedSourceChoiceKey = safeText(currentPopover.selectedSourceChoiceKey) || null;
            const selectedSourceChoice = selectedSourceChoiceKey
                ? (sourceChoices.find((choice) => safeText(choice?.key) === selectedSourceChoiceKey) || null)
                : null;
            if (articleNum === null && !tableId) return;
            const shouldOpenPdf = isOriginalPdfViewMode() || articleNum === null || !!tableId;
            const stayInCurrentTab = _isEdgeBrowser() && isOriginalPdfViewMode() && shouldOpenPdf;

            if (!stayInCurrentTab) {
                setActiveTab('contract');
            }
            if (articleNum !== null) {
                currentArticleNum = articleNum;
                setActiveArticleInToc(articleNum);
                if (!stayInCurrentTab) {
                    await loadArticle(articleNum, { openPdf: false });
                }
            }
            if (!shouldOpenPdf) return true;
            const opened = await openContractInPdf(
                articleNum,
                sectionNum,
                currentPopover.partNum,
                {
                    tableId,
                    rowIndex,
                    sourceType: selectedSourceChoice ? (safeText(selectedSourceChoice.sourceType).toLowerCase() || null) : sourceType,
                    sourcePdf: selectedSourceChoice ? (safeText(selectedSourceChoice.sourcePdf) || null) : sourcePdf,
                    sourcePage: selectedSourceChoice
                        ? (Number.isFinite(Number(selectedSourceChoice.sourcePage)) ? Number(selectedSourceChoice.sourcePage) : null)
                        : sourcePage,
                    sourceDocId: selectedSourceChoice ? (safeText(selectedSourceChoice.sourceDocId) || null) : sourceDocId,
                }
            );
            if (!opened) {
                console.warn('PDF location unavailable.');
            }
        }

        // Close popover when clicking outside
        document.addEventListener('click', (e) => {
            const popover = document.getElementById('citation-popover');
            const isClickInside = popover.contains(e.target) ||
                                  e.target.closest('.citation-link') ||
                                  e.target.closest('.citation-badge');

            if (!isClickInside && !popover.classList.contains('hidden')) {
                hidePopover();
            }
        });

        // =============================================================================
        // PROFILE & SETTINGS
        // =============================================================================

        async function loadOnboardingOptions() {
            if (isTenantRoute()) {
                return;
            }
            try {
                const res = await fetch(`${API_BASE}/api/onboard/options`);
                await res.json();
            } catch (e) {
                console.error('Failed to load options:', e);
            }
        }

        async function loadProfile() {
            if (!memberAuth?.authenticated) {
                userProfile = null;
                syncRoleClarificationPanels();
                updateProfileDisplay();
                updateInteractionLock();
                return;
            }
            if (isTenantRoute()) {
                const tenantContract = getTenantUploadContract();
                userProfile = {
                    role: memberAuth?.user?.role || 'user',
                    union_slug: routeContext.unionSlug || memberAuth?.user?.union_slug || memberAuth?.union_slug || '',
                    contract_id: tenantContract?.contract_id || null,
                    classification_display: getMemberRoleLabel(),
                    months_employed: null,
                };
                if (tenantContract) {
                    activeContract = tenantContract;
                    populateContractSelects();
                }
                syncRoleClarificationPanels();
                updateProfileDisplay();
                syncSettingsForm();
                hideOnboarding();
                updateInteractionLock();
                return;
            }
            try {
                const res = await fetch(`${API_BASE}/api/profile/${SESSION_ID}`);
                const profile = await res.json();
                if (profile?.contract_id) {
                    setActiveContract(profile.contract_id, { persist: true, refreshViewer: true, preserveClassification: false });
                    await loadClassificationsForContract(profile.contract_id, { preserveSelection: false });
                }

                userProfile = profile;
                syncRoleClarificationPanels();
                if (profile.contract_id) {
                    updateProfileDisplay();
                    ensureSessionMeta(SESSION_ID).contract_id = profile.contract_id || null;
                    ensureSessionMeta(SESSION_ID).classification = profile.classification || null;
                    saveSessionMetaStore();
                    if (profile.role_clarification && !profile.classification) {
                        showOnboarding();
                    } else {
                        hideOnboarding();
                    }
                } else {
                    const hydrated = await hydrateProfileFromCache();
                    if (!hydrated) {
                        showOnboarding();
                    }
                }
                updateInteractionLock();
            } catch (e) {
                console.error('Failed to load profile:', e);
                if (!memberAuth?.authenticated) {
                    updateInteractionLock();
                    return;
                }
                const hydrated = await hydrateProfileFromCache();
                if (!hydrated) {
                    showOnboarding();
                }
                updateInteractionLock();
            }
        }

        async function saveProfile(data, options = {}) {
            const allowIncomplete = options?.allowIncomplete === true;
            const source = safeText(options?.source).toLowerCase();
            try {
                const payload = {};
                Object.entries(data || {}).forEach(([key, value]) => {
                    if (value === null || value === undefined) return;
                    if (typeof value === 'string' && !value.trim()) return;
                    payload[key] = value;
                });

                const nextContractId = payload.contract_id || userProfile?.contract_id || activeContract?.contract_id || null;
                const nextClassification = payload.classification ?? userProfile?.classification ?? null;
                const normalizedNextClassification = typeof nextClassification === 'string'
                    ? nextClassification.trim()
                    : '';

                if (!allowIncomplete && !normalizedNextClassification) {
                    alert('Please select your job classification before saving.');
                    syncSettingsForm();
                    return false;
                }
                if (!allowIncomplete && (!nextContractId || !String(nextContractId).trim())) {
                    alert('Please select your contract/store before saving.');
                    syncSettingsForm();
                    return false;
                }

                if (allowIncomplete && Object.keys(payload).length === 0) {
                    hideOnboarding();
                    updateInteractionLock();
                    return true;
                }

                const currentContractId = (userProfile?.contract_id || '').trim().toLowerCase();
                const currentClassification = (userProfile?.classification || '').trim().toLowerCase();
                const currentEmploymentType = String(userProfile?.employment_type || '').trim().toLowerCase();
                const currentHireMonth = String(userProfile?.hire_date || '').trim().slice(0, 7).toLowerCase();
                const contractChanged = !!nextContractId && nextContractId.trim().toLowerCase() !== currentContractId;
                const classificationChanged =
                    typeof nextClassification === 'string' &&
                    nextClassification.trim().toLowerCase() !== currentClassification;
                const nextEmploymentType = typeof payload.employment_type === 'string'
                    ? payload.employment_type.trim().toLowerCase()
                    : currentEmploymentType;
                const nextHireMonth = typeof payload.hire_date === 'string'
                    ? payload.hire_date.trim().slice(0, 7).toLowerCase()
                    : currentHireMonth;
                const employmentTypeChanged =
                    typeof payload.employment_type === 'string' &&
                    nextEmploymentType !== currentEmploymentType;
                const hireMonthChanged =
                    typeof payload.hire_date === 'string' &&
                    nextHireMonth !== currentHireMonth;

                if ((contractChanged || classificationChanged || employmentTypeChanged || hireMonthChanged) && hasSubmittedChatText()) {
                    const changedParts = [];
                    if (contractChanged) changedParts.push('contract/store');
                    if (classificationChanged) changedParts.push('job classification');
                    if (employmentTypeChanged) changedParts.push('employment type');
                    if (hireMonthChanged) changedParts.push('hire date');
                    const confirmed = confirm(
                        `Changing ${changedParts.join(' and ')} will start a new chat and clear current chat context. Continue?`
                    );
                    if (!confirmed) {
                        syncSettingsForm();
                        return false;
                    }
                    startNewChatSession();
                }

                const res = await fetch(`${API_BASE}/api/profile/${SESSION_ID}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                if (!res.ok) {
                    throw new Error(`Profile update failed (${res.status})`);
                }
                userProfile = await res.json();
                if (userProfile?.contract_id) {
                    setActiveContract(userProfile.contract_id, { persist: true, refreshViewer: true, preserveClassification: false });
                    await loadClassificationsForContract(userProfile.contract_id, { preserveSelection: false });
                }
                syncRoleClarificationPanels();
                updateProfileDisplay();
                ensureSessionMeta(SESSION_ID).contract_id = userProfile.contract_id || null;
                ensureSessionMeta(SESSION_ID).classification = userProfile.classification || null;
                saveSessionMetaStore();
                syncSettingsForm();
                const hasRoleClarification = !!normalizeRoleClarificationPayload(userProfile?.role_clarification);
                if (hasRoleClarification && !safeText(userProfile?.classification)) {
                    if (source === 'member' || source === 'steward') {
                        showOnboarding();
                    }
                    updateInteractionLock();
                    return false;
                }
                hideOnboarding();
                updateInteractionLock();
                return true;
            } catch (e) {
                console.error('Failed to save profile:', e);
                alert('Could not save your profile. Please try again.');
                updateInteractionLock();
                return false;
            }
        }

        function updateProfileDisplay() {
            updateContractDisplay();
            if (!userProfile) {
                updateInteractionLock();
                return;
            }

            document.getElementById('display-classification').textContent =
                userProfile.classification_display
                || userProfile.classification
                || (normalizeRoleClarificationPayload(userProfile.role_clarification) ? 'Clarify role' : 'Not set');

            if (userProfile.months_employed) {
                const years = Math.floor(userProfile.months_employed / 12);
                const months = userProfile.months_employed % 12;
                let tenure = '';
                if (years > 0) tenure += `${years}y `;
                if (months > 0 || years === 0) tenure += `${months}m`;
                document.getElementById('display-tenure').textContent = tenure.trim();
            } else {
                document.getElementById('display-tenure').textContent = '--';
            }
            syncRoleClarificationPanels();
            updateInteractionLock();
        }

        function syncSettingsForm() {
            if (activeContract) {
                const contractSelect = document.getElementById('settings-contract');
                if (contractSelect) contractSelect.value = activeContract.contract_id;
            }
            if (!userProfile) return;

            if (userProfile.classification) {
                document.getElementById('settings-classification').value = userProfile.classification;
                document.getElementById('onboard-classification').value = userProfile.classification;
            } else {
                document.getElementById('settings-classification').value = '';
                document.getElementById('onboard-classification').value = '';
            }
            if (userProfile.employment_type) {
                const radio = document.querySelector(`input[name="settings_employment"][value="${userProfile.employment_type}"]`);
                if (radio) radio.checked = true;
            }
            if (userProfile.hire_date) {
                const hireValue = userProfile.hire_date.substring(0, 7);
                document.getElementById('settings-hire-date').value = hireValue;
                initDatePicker('settings', hireValue);
            }

            // Sync preferences
            document.getElementById('pref-text-size').value = preferences.textSize || 'medium';
            document.getElementById('pref-citation-style').value = preferences.citationStyle || 'popover';
            // Answer Tone / Answer Detail controls were removed from Settings.
            // The preference values still exist and still feed the prompt --
            // they just fall back to their defaults instead of being surfaced.
            updateKarlVersionUI();
            updateDeveloperModeUI(isDeveloperModeEnabled());
            syncRoleClarificationPanels();
        }

        async function saveSettingsProfile() {
            if (isTenantUploadAuthMode()) {
                alert('Your union sign-in already sets your access context. Profile settings are not needed for uploaded-document chat.');
                return;
            }
            const contractId = document.getElementById('settings-contract')?.value;
            const classification = document.getElementById('settings-classification').value;
            const employmentType = document.querySelector('input[name="settings_employment"]:checked')?.value;
            const hireMonth = document.getElementById('settings-hire-date').value;

            const data = {};
            if (contractId) data.contract_id = contractId;
            if (classification) data.classification = classification;
            if (employmentType) data.employment_type = employmentType;
            if (hireMonth) data.hire_date = hireMonth + '-01';

            const saved = await saveProfile(data);
            if (saved) {
                alert('Profile saved!');
            }
        }

        function savePreference(key, value) {
            preferences[key] = value;
            localStorage.setItem('karl_preferences', JSON.stringify(preferences));
            if (key === 'developerMode') {
                updateDeveloperModeUI(value === true);
                checkHealth();
            }
        }

        function isDeveloperModeEnabled() {
            return preferences.developerMode === true;
        }

        function toggleDeveloperMode() {
            const isEnabled = !isDeveloperModeEnabled();
            savePreference('developerMode', isEnabled);
        }

        function updateDeveloperModeUI(isEnabled) {
            const toggle = document.getElementById('developer-mode-toggle');
            const dot = document.getElementById('developer-mode-dot');
            // The header connection pill is diagnostics, not member-facing
            // information, so it only appears in developer mode.
            const statusPill = document.getElementById('status');
            if (statusPill) {
                statusPill.classList.toggle('hidden', !isEnabled);
                statusPill.classList.toggle('flex', isEnabled);
            }
            // Settings already has a bottom-nav tab and a welcome tile; this
            // third shortcut is developer-only.
            const profileEditBtn = document.getElementById('chat-profile-edit-btn');
            if (profileEditBtn) {
                profileEditBtn.classList.toggle('hidden', !isEnabled);
                profileEditBtn.classList.toggle('flex', isEnabled);
            }
            const pdfHeaderCopy = document.getElementById('contract-pdf-header-copy');
            const textHeaderCopy = document.getElementById('contract-text-header-copy');
            const historyBanner = document.getElementById('contract-history-banner');
            if (isEnabled) {
                toggle?.classList.remove('bg-slate-200');
                toggle?.classList.add('bg-ufcw-blue');
                dot?.classList.add('translate-x-5');
            } else {
                toggle?.classList.add('bg-slate-200');
                toggle?.classList.remove('bg-ufcw-blue');
                dot?.classList.remove('translate-x-5');
            }
            if (!isEnabled) {
                saveContractPdfSourceMode('effective');
                contractPdfSourcePdf = null;
                saveContractTextSourceMode('effective');
            }
            pdfHeaderCopy?.classList.toggle('hidden', !isEnabled);
            textHeaderCopy?.classList.toggle('hidden', !isEnabled);
            historyBanner?.classList.toggle('hidden', !isEnabled);
            updateContractViewerModeUI();
            renderContractHistoryPanel();
        }

        // Dark mode is permanent per the KARL design system — the app always runs the
        // dark theme (gold accent lightens for contrast). toggleDarkMode is kept as a
        // no-op so any stale onclick references degrade safely.
        function toggleDarkMode() {
            initDarkMode();
        }

        function updateDarkModeUI(_isDark) {}

        function initDarkMode() {
            document.documentElement.classList.add('dark');
            savePreference('darkMode', true);
        }

        function getKarlAvatar() {
            return document.getElementById('karl-avatar');
        }

        function setKarlAvatarState(state) {
            const avatar = getKarlAvatar();
            if (!avatar) return;
            avatar.setAttribute('data-avatar-state', state);
        }

        function getKarlAvatarState() {
            return getKarlAvatar()?.getAttribute('data-avatar-state') || 'idle';
        }

        function setHeaderSubtitle(text) {
            const subtitle = document.getElementById('header-subtitle');
            if (subtitle) subtitle.textContent = text || HEADER_SUBTITLE_DEFAULT;
        }

        function clearKarlAvatarSpeakTimer() {
            if (karlAvatarSpeakTimer) {
                clearTimeout(karlAvatarSpeakTimer);
                karlAvatarSpeakTimer = null;
            }
        }

        function startSpeaking(content = '', outcome = 'confirm') {
            clearKarlAvatarSpeakTimer();
            const avatarOutcome = ['confirm', 'question', 'error'].includes(outcome)
                ? outcome
                : 'confirm';
            setKarlAvatarState(avatarOutcome);
            if (avatarOutcome === 'question') {
                setHeaderSubtitle(HEADER_SUBTITLE_QUESTION);
            } else if (avatarOutcome === 'error') {
                setHeaderSubtitle(HEADER_SUBTITLE_ERROR);
            } else {
                setHeaderSubtitle(HEADER_SUBTITLE_SPEAKING);
            }

            const textLen = typeof content === 'string' ? content.length : 0;
            const speakingDurationMs = Math.max(1200, Math.min(4200, 850 + (textLen * 14)));
            karlAvatarSpeakTimer = setTimeout(() => {
                if (getKarlAvatarState() === avatarOutcome) {
                    setKarlAvatarState('idle');
                    setHeaderSubtitle(HEADER_SUBTITLE_DEFAULT);
                }
            }, speakingDurationMs);
        }

        // Thinking animation - animated gradient on header
        function startThinking() {
            clearKarlAvatarSpeakTimer();
            const header = document.getElementById('main-header');
            header?.classList.add('thinking-gradient');
            setKarlAvatarState('thinking');
            setHeaderSubtitle(HEADER_SUBTITLE_THINKING);
        }

        function stopThinking() {
            const header = document.getElementById('main-header');
            header?.classList.remove('thinking-gradient');
            if (getKarlAvatarState() === 'thinking') {
                setKarlAvatarState('idle');
                setHeaderSubtitle(HEADER_SUBTITLE_DEFAULT);
            }
        }

        // Header gradient setup (no pointer interaction).
        function initHeaderInteractivity() {
            const header = document.getElementById('main-header');
            if (!header) return;
            header.style.setProperty('--header-flow-duration', '14s');
            header.style.setProperty('--header-flow-thinking-duration', '3.2s');
        }

        function syncShellLayoutMetrics() {
            const root = document.documentElement;
            const header = document.getElementById('main-header');
            const tabBar = document.getElementById('tab-bar');
            const tocHeaderMobile = document.getElementById('toc-header-mobile');
            const isDesktop = window.matchMedia('(min-width: 768px)').matches;

            if (header) {
                const headerHeight = Math.ceil(header.getBoundingClientRect().height);
                if (headerHeight > 0) {
                    root.style.setProperty('--mobile-header-offset', `${headerHeight}px`);
                    root.style.setProperty('--desktop-header-offset', `${headerHeight}px`);
                }
            }

            if (tabBar) {
                const tabHeight = Math.ceil(tabBar.getBoundingClientRect().height);
                if (tabHeight > 0) {
                    if (isDesktop) {
                        root.style.setProperty('--desktop-tabbar-height', `${tabHeight + 2}px`);
                    } else {
                        root.style.setProperty('--mobile-tabbar-height', `${tabHeight}px`);
                    }
                }
            }

            if (tocHeaderMobile && !isDesktop) {
                const tocHeight = Math.ceil(tocHeaderMobile.getBoundingClientRect().height);
                if (tocHeight > 0) {
                    root.style.setProperty('--mobile-contract-toc-height', `${tocHeight}px`);
                }
                const firstRow = tocHeaderMobile.firstElementChild;
                const bannerHeight = firstRow ? Math.ceil(firstRow.getBoundingClientRect().height) : 0;
                if (bannerHeight > 0) {
                    root.style.setProperty('--mobile-contract-toc-banner-height', `${bannerHeight}px`);
                }
            }
        }

        function scheduleShellLayoutMetricsSync() {
            if (shellLayoutSyncRaf) cancelAnimationFrame(shellLayoutSyncRaf);
            shellLayoutSyncRaf = requestAnimationFrame(() => {
                shellLayoutSyncRaf = 0;
                syncShellLayoutMetrics();
            });
        }

        function clearSession() {
            if (confirm('This will clear local conversation history and browser preferences for this workspace. Continue?')) {
                postTelemetryEvent('bug_journey', 'workspace_reset', {
                    has_member_auth: Boolean(memberAuth?.authenticated),
                }, { sessionId: SESSION_ID });
                localStorage.removeItem(SESSION_ID_STORAGE_KEY);
                localStorage.removeItem(SESSION_META_STORAGE_KEY);
                localStorage.removeItem('karl_preferences');
                localStorage.removeItem(ACTIVE_CONTRACT_STORAGE_KEY);
                localStorage.removeItem(MEMBER_AUTH_STORAGE_KEY);
                localStorage.removeItem(CONTRACT_PDF_SOURCE_MODE_STORAGE_KEY);
                localStorage.removeItem(ONBOARDING_FLOW_STORAGE_KEY);
                location.reload();
            }
        }

        function setOnboardingModalOpen(isOpen) {
            document.documentElement.classList.toggle('onboarding-open', isOpen);
            document.body.classList.toggle('onboarding-open', isOpen);
        }

        function revealAppShell() {
            const app = document.getElementById('app');
            if (!app) return;
            app.classList.remove('opacity-0', 'pointer-events-none');
        }

        function refreshOnboardingModalOpenState() {
            const isOpen = Boolean(
                (stewardOnboarding && stewardOnboarding.isVisible()) ||
                (memberOnboarding && memberOnboarding.isVisible())
            );
            setOnboardingModalOpen(isOpen);
        }

        function getMemberContractOptions() {
            return availableContracts.map((contract) => {
                const label = getContractLabel(contract);
                const segments = label.split(' - ').map((part) => part.trim()).filter(Boolean);
                const locationFallback = String(contract.region_id || '')
                    .replace(/[_-]+/g, ' ')
                    .replace(/\s+/g, ' ')
                    .trim();

                return {
                    contract_id: contract.contract_id,
                    label,
                    employer: contract.employer || segments[0] || '',
                    location: segments[1] || locationFallback,
                    region_id: contract.region_id || '',
                };
            });
        }

        async function getMemberClassificationOptions(contractId) {
            if (!contractId) return [];
            await loadClassificationsForContract(contractId, { preserveSelection: false });
            return classificationOptionsByContract[contractId] || [];
        }

        function ensureStewardOnboardingController() {
            if (stewardOnboarding) return stewardOnboarding;
            if (!createStewardOnboardingController) {
                stewardOnboarding = createNoopOnboardingController();
                return stewardOnboarding;
            }
            stewardOnboarding = createStewardOnboardingController({
                modalId: 'steward-onboarding-modal',
                formId: 'steward-onboarding-form',
                pickerIds: ['onboard', 'settings'],
                setModalOpenState: () => refreshOnboardingModalOpenState(),
                getActiveContractId: () => getActiveContractId(),
                getUserProfile: () => userProfile,
                onSubmit: async (data) => {
                    setOnboardingFlowPreference('steward');
                    return saveProfile(data, { allowIncomplete: false, source: 'steward' });
                },
                onVisibilityChange: () => {
                    refreshOnboardingModalOpenState();
                    updateInteractionLock();
                },
            });
            stewardOnboarding.bind();
            return stewardOnboarding;
        }

        function ensureMemberOnboardingController() {
            if (memberOnboarding) return memberOnboarding;
            if (!createMemberOnboardingController) {
                memberOnboarding = createNoopOnboardingController();
                return memberOnboarding;
            }
            memberOnboarding = createMemberOnboardingController({
                modalId: 'member-onboarding-modal',
                setModalOpenState: () => refreshOnboardingModalOpenState(),
                getContracts: () => getMemberContractOptions(),
                getClassifications: async (contractId) => getMemberClassificationOptions(contractId),
                getUserProfile: () => userProfile,
                onRequestStewardFlow: async () => {
                    setOnboardingFlowPreference('steward');
                    ensureStewardOnboardingController().show();
                },
                onSubmitMember: async (data) => {
                    setOnboardingFlowPreference('member');
                    return saveProfile(data, { allowIncomplete: true, source: 'member' });
                },
                onVisibilityChange: () => {
                    refreshOnboardingModalOpenState();
                    updateInteractionLock();
                },
            });
            memberOnboarding.bind();
            return memberOnboarding;
        }

        function showOnboarding() {
            if (isTenantRoute()) {
                hideOnboarding();
                return;
            }
            const preferred = getOnboardingFlowPreference();
            if (preferred === 'steward') {
                ensureMemberOnboardingController().hide();
                ensureStewardOnboardingController().show();
            } else {
                ensureStewardOnboardingController().hide();
                ensureMemberOnboardingController().show();
            }
            updateInteractionLock();
        }

        function hideOnboarding() {
            ensureStewardOnboardingController().hide();
            ensureMemberOnboardingController().hide();
            refreshOnboardingModalOpenState();
            updateInteractionLock();
        }

        // =============================================================================
        // CHAT
        // =============================================================================

        async function checkHealth() {
            const statusEl = document.getElementById('status');
            const showDiagnostics = isDeveloperModeEnabled();
            try {
                const contractId = getActiveContractId();
                if (!contractId) {
                    statusEl.innerHTML = `
                        <span class="w-2 h-2 rounded-full bg-slate-300"></span>
                        <span class="text-xs text-blue-100">${isTenantRoute() ? 'Sign in to access union documents' : 'Select contract'}</span>
                    `;
                    return;
                }
                const suffix = contractId ? `?contract_id=${encodeURIComponent(contractId)}` : '';
                const res = await fetchWithAuth(`${API_BASE}/api/health${suffix}`);
                const data = await res.json();
                isHealthy = data.status === 'healthy';
                const sectionCount = data.contract_chunks ?? data.chunks_loaded ?? 0;

                if (isHealthy) {
                    statusEl.innerHTML = `
                        <span class="w-2 h-2 rounded-full bg-green-400"></span>
                        <span class="text-xs text-blue-100">${showDiagnostics ? `${sectionCount} sections` : 'Online'}</span>
                    `;
                } else {
                    statusEl.innerHTML = `
                        <span class="w-2 h-2 rounded-full bg-yellow-400"></span>
                        <span class="text-xs text-blue-100">${showDiagnostics ? (sectionCount > 0 ? `${sectionCount} sections` : 'Limited') : 'Limited'}</span>
                    `;
                }
            } catch (e) {
                statusEl.innerHTML = `
                    <span class="w-2 h-2 rounded-full bg-red-400 pulse-dot"></span>
                    <span class="text-xs text-blue-100">Offline</span>
                `;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function formatResponse(text) {
            let processed = escapeHtml(text || '');

            // Headings
            processed = processed.replace(/^### (.+)$/gm, '<h4 class="text-sm font-semibold text-slate-900 dark:text-slate-100 mt-3 mb-1">$1</h4>');
            processed = processed.replace(/^## (.+)$/gm, '<h3 class="text-base font-semibold text-slate-900 dark:text-slate-100 mt-3 mb-1">$1</h3>');
            processed = processed.replace(/^# (.+)$/gm, '<h2 class="text-lg font-bold text-slate-900 dark:text-slate-100 mt-3 mb-1">$1</h2>');

            // Inline markdown
            processed = processed.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-slate-900 dark:text-slate-100">$1</strong>');
            processed = processed.replace(/`([^`]+)`/g, '<code class="px-1 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-xs">$1</code>');

            // Citation links: "(Source: file.md, Section N, page P)" opens the
            // matching entry in the Supporting Sources panel below the answer.
            processed = processed.replace(/\(Source:\s*([^)]+?)\)/gi, (match, inner) => {
                const pageMatch = /page\s+(\d+)/i.exec(inner);
                const page = pageMatch ? Number(pageMatch[1]) : null;
                return `<button type="button" class="citation-link inline text-left align-baseline text-xs font-medium text-ufcw-blue hover:underline" data-cite-page="${page ?? ''}" onclick="openAnswerCitation(this, ${page ?? 'null'})">(Source: ${inner})</button>`;
            });

            // Lists and quotes
            processed = processed.replace(/^\s*[-*\u2022]\s+(.+)$/gm, '<div class="ml-4 mb-1">&bull; $1</div>');
            processed = processed.replace(/^\s*(\d+)\.\s+(.+)$/gm, '<div class="ml-4 mb-1"><span class="font-medium">$1.</span> $2</div>');
            processed = processed.replace(/^>\s?(.+)$/gm, '<blockquote class="border-l-2 border-slate-300 pl-3 italic text-slate-600 dark:text-slate-300 my-1">$1</blockquote>');

            // Make citations clickable after markdown transforms.
            processed = parseCitations(processed);

            processed = processed.replace(/\n/g, '<br>');
            return processed;
        }

        function buildRoleClarificationCardHtml(payload, options = {}) {
            const normalized = normalizeRoleClarificationPayload(payload);
            if (!normalized) return '';
            const retryQuestion = safeText(options?.retryQuestion);
            return `
                <div class="bg-amber-50 border border-amber-200 rounded-lg p-3 mt-3">
                    <div class="flex items-start gap-3">
                        <div class="w-9 h-9 bg-amber-100 rounded-full flex items-center justify-center shrink-0">
                            <svg class="w-4 h-4 text-amber-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M10.29 3.86l-7.5 13A1 1 0 003.66 18h16.68a1 1 0 00.87-1.5l-7.5-13a1 1 0 00-1.74 0z"/>
                            </svg>
                        </div>
                        <div class="min-w-0">
                            <p class="text-xs font-semibold uppercase tracking-wide text-amber-700">Need exact role</p>
                            <p class="mt-1 text-sm text-amber-950">${escapeHtml(normalized.message || 'I need your exact classification before I can apply the right contract rules.')}</p>
                            ${buildRoleClarificationChoicesHtml(normalized, { retryQuestion })}
                        </div>
                    </div>
                </div>
            `;
        }

        function buildProviderWarningHtml(providerWarning) {
            if (!providerWarning?.message) return '';
            const detail = safeText(providerWarning?.detail);
            return `
                <div class="bg-amber-50 border border-amber-200 rounded-lg p-3 mt-3">
                    <div class="flex items-start gap-3">
                        <div class="w-8 h-8 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                            <svg class="w-4 h-4 text-amber-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-7.938 4h15.876c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L2.34 16c-.77 1.333.192 3 1.732 3z"/>
                            </svg>
                        </div>
                        <div>
                            <p class="text-xs font-semibold uppercase tracking-wide text-amber-700">Model Provider Warning</p>
                            <p class="mt-1 text-sm text-amber-950">${escapeHtml(providerWarning.message)}</p>
                            ${detail ? `<p class="mt-1 text-xs text-amber-800/80">${escapeHtml(detail)}</p>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }

        function isUploadedSourceRecord(source) {
            return safeText(source?.source_type).toLowerCase() === 'upload' && !!safeText(source?.document_id);
        }

        function resolveUploadedSourcePageLabel(source) {
            const page = Number.isFinite(Number(source?.page_number)) ? Number(source.page_number) : null;
            const pageStart = Number.isFinite(Number(source?.page_start)) ? Number(source.page_start) : null;
            const pageEnd = Number.isFinite(Number(source?.page_end)) ? Number(source.page_end) : null;
            const chunkIndex = Number.isFinite(Number(source?.chunk_index)) ? Number(source.chunk_index) + 1 : null;
            const articleNum = safeText(source?.article_num);
            const sectionNum = safeText(source?.section_num);
            const bits = [];
            if (articleNum) bits.push(`Article ${articleNum}`);
            if (sectionNum) bits.push(`Section ${sectionNum}`);
            if (page && page > 0) bits.push(`Page ${page}`);
            else if (pageStart && pageEnd && pageStart > 0 && pageEnd >= pageStart) bits.push(pageStart === pageEnd ? `Page ${pageStart}` : `Pages ${pageStart}-${pageEnd}`);
            else if (pageStart && pageStart > 0) bits.push(`Page ${pageStart}`);
            if (chunkIndex && chunkIndex > 0) bits.push(`Chunk ${chunkIndex}`);
            return bits.join(' • ');
        }

        function buildUploadedSourceLabel(source, idx) {
            const docTitle = safeText(source?.document_title).replace(/\.(md|markdown|txt)$/i, '');
            const articleNum = safeText(source?.article_num);
            const sectionNum = safeText(source?.section_num);
            // section_label is the short form written at ingest; section_title is
            // the full body and must never be used as a label.
            const sectionLabel = safeText(source?.section_label);
            const parts = [];
            if (articleNum) parts.push(`Article ${articleNum}`);
            if (sectionNum) parts.push(`Section ${sectionNum}`);
            const locator = parts.join(' · ');
            const head = [docTitle, locator].filter(Boolean).join(' — ');
            if (sectionLabel) return head ? `${head}: ${sectionLabel}` : sectionLabel;
            return head || safeText(source?.document_title) || `Source ${idx + 1}`;
        }

        function buildUploadedSourceEvidenceHtml(sources) {
            const uploadedSources = Array.isArray(sources)
                ? sources.filter((source) => isUploadedSourceRecord(source))
                : [];
            if (!uploadedSources.length) return '';

            return `
                <div class="uploaded-source-viewer hidden mt-3 rounded-2xl border border-slate-200 bg-slate-50"></div>
                <details class="mt-3 border-t border-slate-200 pt-3 group">
                    <summary class="flex cursor-pointer list-none items-center justify-between gap-3 rounded-xl bg-slate-50 px-3 py-2 text-sm font-medium text-slate-800 hover:bg-slate-100">
                        <span>Supporting Sources (${uploadedSources.length})</span>
                        <span class="text-xs text-slate-500 group-open:hidden">Show</span>
                        <span class="hidden text-xs text-slate-500 group-open:inline">Hide</span>
                    </summary>
                    <div class="mt-3 space-y-3">
                        ${uploadedSources.map((source, idx) => {
                            const sourceRegistryKey = registerCitationSourceRecord(source);
                            // Build the label from structured fields rather than the
                            // server citation string: that string embeds section_title,
                            // which holds the whole section body and rendered as an
                            // unreadable wall of grey text.
                            const label = buildUploadedSourceLabel(source, idx);
                            const subtitle = resolveUploadedSourcePageLabel(source);
                            const excerpt = safeText(source?.excerpt) || 'No preview excerpt available for this source yet.';
                            const pdfUrl = safeText(source?.document_source_pdf_url);
                            const pdfPage = Number.isFinite(Number(source?.source_page)) ? Number(source.source_page) : null;
                            const buttonLabel = pdfUrl
                                ? (pdfPage ? `View PDF · Page ${pdfPage}` : 'View PDF')
                                : 'Open In Document';
                            return `
                                <div class="rounded-xl border border-slate-200 bg-white px-3 py-3 shadow-sm">
                                    <div class="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                                        <div class="min-w-0">
                                            <p class="text-sm font-semibold text-slate-900">${escapeHtml(label)}</p>
                                            ${subtitle ? `<p class="mt-0.5 text-xs text-slate-500">${escapeHtml(subtitle)}</p>` : ''}
                                        </div>
                                        <button
                                            type="button"
                                            class="inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-lg bg-ufcw-blue px-3 py-2 text-xs font-medium text-white hover:bg-ufcw-blue-mid transition-colors"
                                            data-source-registry-key="${escapeHtml(sourceRegistryKey)}"
                                            onclick="openUploadedSourceViewer(this, '${escapeJsSingleQuoted(sourceRegistryKey)}')"
                                        >
                                            ${escapeHtml(buttonLabel)}
                                        </button>
                                    </div>
                                    <p class="mt-3 text-sm leading-relaxed text-slate-700">${escapeHtml(excerpt)}</p>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </details>
            `;
        }

        function buildUploadedSourceViewerHtml(source, objectUrl) {
            const citation = safeText(source?.citation) || safeText(source?.document_title) || 'Uploaded document';
            const excerpt = safeText(source?.selected_excerpt || source?.excerpt);
            const summary = safeText(source?.selected_summary || source?.summary);
            const articleTitle = safeText(source?.selected_article_title || source?.article_title);
            const sectionTitle = safeText(source?.selected_section_title || source?.section_title);
            const page = Number.isFinite(Number(source?.page_number)) ? Number(source.page_number) : null;
            const pageStart = Number.isFinite(Number(source?.selected_page_start ?? source?.page_start)) ? Number(source?.selected_page_start ?? source?.page_start) : null;
            const contentType = safeText(source?.content_type).toLowerCase();
            const effectivePage = page || pageStart || null;
            const fileLocationLabel = resolveUploadedSourcePageLabel({
                ...source,
                page_number: effectivePage,
            }) || 'Selected evidence';

            return `
                <div class="border-b border-slate-200 bg-white px-4 py-3">
                    <div class="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div class="min-w-0">
                            <p class="text-sm font-semibold text-slate-900">${escapeHtml(citation)}</p>
                            <p class="mt-0.5 text-xs text-slate-500">
                                ${escapeHtml(resolveUploadedSourcePageLabel(source) || 'Uploaded source')}
                            ${contentType ? ` • ${escapeHtml(contentType)}` : ''}
                            </p>
                        </div>
                        <div class="flex items-center gap-2">
                            <button
                                type="button"
                                class="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 transition-colors"
                                onclick="openUploadedSourceInNewTab('${escapeJsSingleQuoted(objectUrl)}', ${effectivePage ?? 'null'})"
                            >
                                Open In New Tab
                            </button>
                            <button
                                type="button"
                                class="inline-flex items-center justify-center rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-100 transition-colors"
                                onclick="closeUploadedSourceViewer(this)"
                            >
                                Close
                            </button>
                        </div>
                    </div>
                    ${articleTitle || sectionTitle ? `<p class="mt-2 text-xs font-medium text-slate-600">${escapeHtml([articleTitle, sectionTitle].filter(Boolean).join(' • '))}</p>` : ''}
                    ${summary ? `<p class="mt-2 rounded-xl bg-blue-50 px-3 py-2 text-sm leading-relaxed text-slate-700"><span class="font-semibold text-slate-900">Matched section:</span> ${escapeHtml(summary)}</p>` : ''}
                </div>
                <div class="p-3">
                    <div class="rounded-xl border border-slate-200 bg-white px-4 py-4 text-sm text-slate-700">
                        <p class="text-xs font-semibold uppercase tracking-wide text-slate-500">Selected Evidence</p>
                        <p class="mt-1 text-xs text-slate-500">${escapeHtml(fileLocationLabel)}</p>
                        <p class="mt-3 whitespace-pre-wrap leading-relaxed">${escapeHtml(excerpt || summary || 'The selected evidence could not be loaded.')}</p>
                        <div class="mt-4 rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 text-xs text-slate-600">
                            The exact matched article or section is shown above. Use “Open In New Tab” to inspect the original uploaded file.
                        </div>
                    </div>
                </div>
            `;
        }

        async function openUploadedSourceViewer(trigger, sourceRegistryKey) {
            const source = getCitationSourceRecord(sourceRegistryKey);
            const card = trigger?.closest('.assistant-message-card');
            const viewer = card?.querySelector('.uploaded-source-viewer');
            if (!source || !viewer) return;

            viewer.classList.remove('hidden');
            viewer.innerHTML = '<div class="px-4 py-6 text-sm text-slate-500">Loading source document...</div>';
            viewer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

            const priorObjectUrl = safeText(viewer.dataset.objectUrl);
            if (priorObjectUrl) {
                URL.revokeObjectURL(priorObjectUrl);
                delete viewer.dataset.objectUrl;
            }
            delete viewer.dataset.sourceRegistryKey;

            const sourcePdfUrl = safeText(source?.document_source_pdf_url);
            if (sourcePdfUrl) {
                // The printed contract is the best evidence there is: render it
                // in the in-card panel, opened to the cited page, rather than
                // bouncing the member to a bare browser tab.
                const pdfPage = Number.isFinite(Number(source?.source_page)) ? Number(source.source_page) : null;
                const resolvedPdf = new URL(sourcePdfUrl, API_BASE).toString();
                const framedPdf = pdfPage ? `${resolvedPdf}#page=${pdfPage}` : resolvedPdf;
                const headerLabel = buildUploadedSourceLabel(source, 0);
                viewer.innerHTML = `
                    <div class="flex items-center justify-between gap-3 border-b border-slate-200 px-4 py-2">
                        <p class="min-w-0 truncate text-xs font-medium text-slate-700">
                            ${escapeHtml(headerLabel)}${pdfPage ? ` · Page ${pdfPage}` : ''}
                        </p>
                        <div class="flex shrink-0 items-center gap-2">
                            <button type="button" class="text-xs font-medium text-ufcw-blue hover:underline"
                                onclick="openContractPdfAtPage('${escapeJsSingleQuoted(resolvedPdf)}', ${pdfPage ?? 'null'})">Open in new tab</button>
                            <button type="button" class="text-xs text-slate-500 hover:text-slate-700"
                                onclick="this.closest('.uploaded-source-viewer').classList.add('hidden')">Close</button>
                        </div>
                    </div>
                    <iframe src="${escapeHtml(framedPdf)}" title="Contract PDF" class="w-full rounded-b-2xl border-0 bg-white" style="height: min(70vh, 640px);"></iframe>
                `;
                return;
            }

            const documentContentUrl = safeText(source?.document_content_url);
            const documentUrl = safeText(source?.document_access_url || documentContentUrl);
            if (!documentUrl) {
                viewer.innerHTML = `
                    <div class="px-4 py-6 text-sm text-slate-600">
                        This source does not have a document viewer URL yet.
                    </div>
                `;
                return;
            }

            try {
                let selectedEvidence = null;
                const selectionUrl = safeText(source?.document_selection_url);
                if (selectionUrl) {
                    const selectionResponse = await fetch(new URL(selectionUrl, API_BASE).toString(), { credentials: 'same-origin' });
                    if (selectionResponse.ok) {
                        selectedEvidence = await selectionResponse.json();
                    }
                }
                const resolvedUrl = new URL(documentUrl, API_BASE).toString();
                let response = safeText(source?.document_access_url)
                    ? await fetch(resolvedUrl, { credentials: 'same-origin' })
                    : await fetchWithAuth(resolvedUrl);
                if (!response.ok && documentContentUrl) {
                    const fallbackUrl = new URL(documentContentUrl, API_BASE).toString();
                    response = await fetchWithAuth(fallbackUrl);
                }
                if (!response.ok) {
                    let detail = `Unable to load source document (${response.status}).`;
                    try {
                        const err = await response.json();
                        if (err?.detail) detail = String(err.detail);
                    } catch (_) {
                        // Ignore parse failure.
                    }
                    throw new Error(detail);
                }
                const blob = await response.blob();
                const objectUrl = URL.createObjectURL(blob);
                viewer.dataset.objectUrl = objectUrl;
                viewer.dataset.sourceRegistryKey = safeText(sourceRegistryKey);
                viewer.innerHTML = buildUploadedSourceViewerHtml(
                    {
                        ...source,
                        selected_excerpt: safeText(selectedEvidence?.excerpt),
                        selected_summary: safeText(selectedEvidence?.summary),
                        selected_article_title: safeText(selectedEvidence?.article_title),
                        selected_section_title: safeText(selectedEvidence?.section_title),
                        selected_page_start: selectedEvidence?.page_start,
                        content_type: source?.content_type || blob.type || '',
                    },
                    objectUrl
                );
                const iframe = viewer.querySelector('iframe');
                if (iframe) {
                    iframe.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
                postTelemetryEvent('usage_ux', 'source_opened', {
                    document_id: source?.document_id || null,
                    citation: source?.citation || null,
                    page_number: source?.page_number || source?.page_start || null,
                });
            } catch (error) {
                postTelemetryEvent('bug_journey', 'source_open_failed', {
                    document_id: source?.document_id || null,
                    reason: error?.message || 'unknown_error',
                });
                viewer.innerHTML = `
                    <div class="px-4 py-6 text-sm text-red-700">
                        ${escapeHtml(error?.message || 'Unable to load the source document.')}
                    </div>
                `;
            }
        }

        function closeUploadedSourceViewer(trigger) {
            const viewer = trigger?.closest('.uploaded-source-viewer');
            if (!viewer) return;
            const priorObjectUrl = safeText(viewer.dataset.objectUrl);
            if (priorObjectUrl) {
                URL.revokeObjectURL(priorObjectUrl);
            }
            viewer.innerHTML = '';
            viewer.classList.add('hidden');
            delete viewer.dataset.objectUrl;
            delete viewer.dataset.sourceRegistryKey;
        }

        function openAnswerCitation(trigger, pageNumber = null) {
            const card = trigger?.closest('.assistant-message-card');
            if (!card) return;
            const details = card.querySelector('details');
            if (details) details.open = true;
            const buttons = [...card.querySelectorAll('[data-source-registry-key]')];
            let target = null;
            if (pageNumber) {
                target = buttons.find((btn) => {
                    const record = getCitationSourceRecord(btn.dataset.sourceRegistryKey);
                    return record && Number(record.source_page) === Number(pageNumber);
                }) || null;
            }
            target = target || buttons[0] || null;
            if (target) {
                openUploadedSourceViewer(target, target.dataset.sourceRegistryKey);
            } else if (details) {
                details.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        }

        function openContractPdfAtPage(pdfUrl, pageNumber = null) {
            // #page= is the standard PDF viewer fragment, honoured by Chrome,
            // Firefox and Safari, so the member lands on the exact page the
            // citation refers to rather than page one of a 200-page book.
            const url = pageNumber ? `${pdfUrl}${pdfUrl.includes('#') ? '' : `#page=${pageNumber}`}` : pdfUrl;
            window.open(url, '_blank', 'noopener');
        }

        function openUploadedSourceInNewTab(objectUrl, pageNumber = null) {
            const page = Number.isFinite(Number(pageNumber)) ? Number(pageNumber) : null;
            const targetUrl = page ? `${objectUrl}#page=${page}` : objectUrl;
            window.open(targetUrl, '_blank', 'noopener,noreferrer');
        }

        function addMessage(content, isUser = false, metadata = null) {
            const container = document.getElementById('chat-container');
            const msgDiv = document.createElement('div');
            msgDiv.className = 'fade-up chat-message';

            if (isUser) {
                msgDiv.innerHTML = `
                    <div class="flex justify-end">
                        <div class="bg-ufcw-blue text-white rounded-2xl rounded-br-sm px-4 py-3 max-w-[85%]">
                            <p class="text-sm">${escapeHtml(content)}</p>
                        </div>
                    </div>
                `;
            } else {
                const hasUploadedEvidence = Array.isArray(metadata?.sources) && metadata.sources.some((source) => isUploadedSourceRecord(source));
                let citationsHtml = '';
                if (!hasUploadedEvidence && metadata?.citations?.length > 0) {
                    const sourceByCitation = new Map();
                    if (Array.isArray(metadata?.sources)) {
                        for (const src of metadata.sources) {
                            const key = String(src?.citation || '').trim().toLowerCase();
                            if (!key || sourceByCitation.has(key)) continue;
                            sourceByCitation.set(key, src || {});
                        }
                    }
                    citationsHtml = `
                        <div class="flex flex-wrap gap-1.5 mt-3 pt-3 border-t border-slate-200">
                            ${metadata.citations.map(c => {
                                const source = sourceByCitation.get(String(c || '').trim().toLowerCase()) || {};
                                const sourceRegistryKey = registerCitationSourceRecord(source);
                                const hint = resolveCitationSourceHint(source, isDeveloperModeEnabled() ? contractPdfSourceMode : 'effective');
                                const provenanceLabel = summarizeCitationProvenance(source);
                                const provenanceHtml = provenanceLabel
                                    ? `<span class="text-[9px] font-semibold uppercase tracking-wide text-slate-500">${escapeHtml(provenanceLabel)}</span>`
                                    : '';
                                const safeCitationLabel = escapeJsSingleQuoted(String(c || ''));
                                const sourceTypeArg = hint.sourceType ? `'${escapeJsSingleQuoted(hint.sourceType)}'` : 'null';
                                const sourcePdfArg = hint.sourcePdf ? `'${escapeJsSingleQuoted(hint.sourcePdf)}'` : 'null';
                                const sourceDocIdArg = hint.sourceDocId ? `'${escapeJsSingleQuoted(hint.sourceDocId)}'` : 'null';
                                const sourcePageArg = hint.sourcePage !== null ? hint.sourcePage : 'null';
                                const sourceRegistryKeyArg = sourceRegistryKey ? `'${escapeJsSingleQuoted(sourceRegistryKey)}'` : 'null';

                                // Match Article, Section (with optional parenthetical), and optional Part
                                const match = c.match(/Article\s+(\d+)(?:,?\s*Section\s+(\d+)(?:\(([a-z])\))?)?(?:,?\s*Part\s+([\w\-]+))?/i);
                                if (match) {
                                    const artNum = match[1];
                                    const secNum = match[2] || null;
                                    const parenSub = match[3] || null;
                                    const partNum = match[4] || null;
                                    const subsection = parenSub || partNum;
                                    const subArg = subsection ? `'${subsection}'` : 'null';
                                    const labelArg = `'${safeCitationLabel}'`;
                                    return `
                                    <button class="citation-badge inline-flex items-center gap-1 bg-ufcw-blue/10 text-ufcw-blue text-[11px] font-medium px-2 py-1 rounded-md hover:bg-ufcw-blue/20 transition-colors cursor-pointer"
                                            onclick="handleCitationClick(event, ${artNum}, ${secNum || 'null'}, ${subArg}, null, null, ${labelArg}, ${sourceTypeArg}, ${sourcePdfArg}, ${sourcePageArg}, ${sourceDocIdArg}, ${sourceRegistryKeyArg})">
                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                                        </svg>
                                        ${escapeHtml(c)}
                                        ${provenanceHtml}
                                    </button>
                                `;
                                }

                                const sourceArticleNum = Number.isFinite(Number(source?.article_num))
                                    ? Number(source.article_num)
                                    : null;
                                const sourceSectionNum = Number.isFinite(Number(source?.section_num))
                                    ? Number(source.section_num)
                                    : null;
                                const sourceSub = source?.subsection ? String(source.subsection) : null;
                                const sourceTableId = source?.table_id ? String(source.table_id).trim() : null;
                                const sourceRowIndex = Number.isFinite(Number(source?.row_index))
                                    ? Number(source.row_index)
                                    : null;
                                if (sourceArticleNum !== null || sourceTableId) {
                                    const safeSourceSub = sourceSub
                                        ? escapeJsSingleQuoted(sourceSub)
                                        : null;
                                    const safeSourceTableId = sourceTableId
                                        ? escapeJsSingleQuoted(sourceTableId)
                                        : null;
                                    const sourceArticleArg = sourceArticleNum !== null ? sourceArticleNum : 'null';
                                    const sourceSecArg = sourceSectionNum !== null ? sourceSectionNum : 'null';
                                    const sourceSubArg = safeSourceSub ? `'${safeSourceSub}'` : 'null';
                                    const sourceTableArg = safeSourceTableId ? `'${safeSourceTableId}'` : 'null';
                                    const sourceRowArg = sourceRowIndex !== null ? sourceRowIndex : 'null';
                                    const sourceLabelArg = `'${safeCitationLabel}'`;
                                    return `
                                    <button class="citation-badge inline-flex items-center gap-1 bg-ufcw-blue/10 text-ufcw-blue text-[11px] font-medium px-2 py-1 rounded-md hover:bg-ufcw-blue/20 transition-colors cursor-pointer"
                                            onclick="handleCitationClick(event, ${sourceArticleArg}, ${sourceSecArg}, ${sourceSubArg}, ${sourceTableArg}, ${sourceRowArg}, ${sourceLabelArg}, ${sourceTypeArg}, ${sourcePdfArg}, ${sourcePageArg}, ${sourceDocIdArg}, ${sourceRegistryKeyArg})">
                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                                        </svg>
                                        ${escapeHtml(c)}
                                        ${provenanceHtml}
                                    </button>
                                `;
                                }

                                return `
                                    <span class="citation-badge inline-flex items-center gap-1 bg-slate-100 text-slate-600 text-[11px] font-medium px-2 py-1 rounded-md">
                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                                        </svg>
                                        ${escapeHtml(c)}
                                    </span>
                                `;
                            }).join('')}
                        </div>
                    `;
                }
                const uploadedEvidenceHtml = hasUploadedEvidence
                    ? buildUploadedSourceEvidenceHtml(metadata?.sources)
                    : '';
                const providerWarningHtml = buildProviderWarningHtml(metadata?.provider_warning);

                let wageHtml = '';
                if (metadata?.wage_info) {
                    const tableEvidence = Array.isArray(metadata.wage_info.table_evidence)
                        ? metadata.wage_info.table_evidence.slice(0, 3)
                        : [];
                    const tableEvidenceHtml = tableEvidence.length > 0
                        ? `
                            <div class="mt-3 pt-2 border-t border-green-200">
                                <p class="text-[11px] font-semibold text-green-700 uppercase tracking-wide">Appendix Table Evidence</p>
                                <div class="mt-1.5 space-y-1">
                                    ${tableEvidence.map(row => {
                                        const tableId = row?.table_id ? escapeHtml(String(row.table_id)) : 'table';
                                        const rowLabel = Number.isInteger(row?.row_index) ? `row ${row.row_index + 1}` : 'row';
                                        const step = row?.step_name ? escapeHtml(String(row.step_name)) : 'Rate';
                                        const rate = typeof row?.rate === 'number'
                                            ? `$${row.rate.toFixed(2)}`
                                            : '';
                                        const hint = resolveCitationSourceHint(row, isDeveloperModeEnabled() ? contractPdfSourceMode : 'effective');
                                        const provenanceLabel = summarizeCitationProvenance(row);
                                        const rawTableId = row?.table_id ? escapeJsSingleQuoted(String(row.table_id)) : null;
                                        const rowIndexArg = Number.isInteger(row?.row_index) ? row.row_index : 'null';
                                        const sourceTypeArg = hint.sourceType ? `'${escapeJsSingleQuoted(hint.sourceType)}'` : 'null';
                                        const sourcePdfArg = hint.sourcePdf ? `'${escapeJsSingleQuoted(hint.sourcePdf)}'` : 'null';
                                        const sourceDocIdArg = hint.sourceDocId ? `'${escapeJsSingleQuoted(hint.sourceDocId)}'` : 'null';
                                        const sourcePageArg = hint.sourcePage !== null ? hint.sourcePage : 'null';
                                        const clickHandler = rawTableId
                                            ? `onclick="handleCitationClick(event, null, null, null, '${rawTableId}', ${rowIndexArg}, 'Appendix A', ${sourceTypeArg}, ${sourcePdfArg}, ${sourcePageArg}, ${sourceDocIdArg}, null)"`
                                            : '';
                                        const tagName = rawTableId ? 'button' : 'div';
                                        return `
                                            <${tagName} ${clickHandler} class="bg-white/70 border border-green-200 rounded px-2 py-1 text-[11px] text-green-800 flex flex-wrap items-center gap-x-2 ${rawTableId ? 'w-full text-left hover:bg-green-100 transition-colors cursor-pointer' : ''}">
                                                <span class="font-semibold">${tableId}</span>
                                                <span>${escapeHtml(rowLabel)}</span>
                                                <span>${step}</span>
                                                <span class="font-semibold">${rate}</span>
                                                ${provenanceLabel ? `<span class="ml-auto text-[9px] font-semibold uppercase tracking-wide text-green-700">${escapeHtml(provenanceLabel)}</span>` : ''}
                                            </${tagName}>
                                        `;
                                    }).join('')}
                                </div>
                            </div>
                        `
                        : '';
                    wageHtml = `
                        <div class="bg-green-50 border border-green-200 rounded-lg p-3 mt-3">
                            <div class="flex items-center gap-3">
                                <div class="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                                    <span class="text-green-600 font-bold">$</span>
                                </div>
                                <div>
                                    <p class="text-xs text-green-600 font-medium">Estimated Rate</p>
                                    <p class="text-lg font-bold text-green-700">$${metadata.wage_info.rate.toFixed(2)}/hr</p>
                                </div>
                            </div>
                            <p class="text-xs text-green-600 mt-2 flex items-start gap-1">
                                <svg class="w-4 h-4 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/>
                                </svg>
                                <span>This is an estimate based on your tenure. Verify via pay stub or Company HR Portal.</span>
                            </p>
                            ${tableEvidenceHtml}
                        </div>
                    `;
                }

                let escalationHtml = '';
                if (metadata?.escalation_required) {
                    escalationHtml = `
                        <div class="bg-amber-50 border border-amber-200 rounded-lg p-3 mt-3 flex items-start gap-3">
                            <div class="w-8 h-8 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                <svg class="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/>
                                </svg>
                            </div>
                            <div>
                                <p class="text-sm font-semibold text-amber-800">Talk to Your Steward</p>
                                <p class="text-xs text-amber-700 mt-0.5">This situation may require union representation. Contact your steward before taking any action.</p>
                            </div>
                        </div>
                    `;
                }

                const roleClarificationHtml = buildRoleClarificationCardHtml(
                    metadata?.role_clarification,
                    { retryQuestion: metadata?.retry_question }
                );

                msgDiv.innerHTML = `
                    <div class="assistant-message-card bg-white border border-slate-200 rounded-2xl rounded-bl-sm px-4 py-3 max-w-[95%] shadow-sm">
                        <div class="text-sm text-slate-700 leading-relaxed">${formatResponse(content)}</div>
                        ${wageHtml}
                        ${roleClarificationHtml}
                        ${escalationHtml}
                        ${providerWarningHtml}
                        ${uploadedEvidenceHtml}
                        ${citationsHtml}
                    </div>
                `;
            }

            if (!isUser) {
                const avatarOutcome = metadata?.avatar_outcome
                    || (metadata?.escalation_required ? 'question' : 'confirm');
                startSpeaking(content, avatarOutcome);
            }

            container.appendChild(msgDiv);
            container.scrollTop = container.scrollHeight;
        }

        function showLoading() {
            const container = document.getElementById('chat-container');
            const loadingDiv = document.createElement('div');
            loadingDiv.id = 'loading';
            loadingDiv.className = 'fade-up';
            loadingDiv.innerHTML = `
                <div class="flex items-center gap-2 text-slate-400 py-2">
                    <div class="flex gap-1">
                        <span class="w-2 h-2 bg-ufcw-blue rounded-full animate-bounce" style="animation-delay: 0ms"></span>
                        <span class="w-2 h-2 bg-ufcw-blue rounded-full animate-bounce" style="animation-delay: 100ms"></span>
                        <span class="w-2 h-2 bg-ufcw-blue rounded-full animate-bounce" style="animation-delay: 200ms"></span>
                    </div>
                    <span class="text-xs">${isTenantUploadAuthMode() ? 'Searching union documents...' : 'Searching contract...'}</span>
                </div>
            `;
            container.appendChild(loadingDiv);
            container.scrollTop = container.scrollHeight;
        }

        function hideLoading() {
            document.getElementById('loading')?.remove();
        }

        async function sendQuery(question) {
            const sendBtn = document.getElementById('send-btn');
            if (!hasCompleteProfileContext()) {
                if (!memberAuth?.authenticated) {
                    openMemberAuthModal();
                    addMessage('Sign in to your union account before asking questions.', false, { avatar_outcome: 'question' });
                    return;
                }
                showOnboarding();
                addMessage(
                    isTenantUploadAuthMode()
                        ? 'Upload documents in the admin workspace before asking questions.'
                        : 'Select your agreement and role in onboarding before asking questions.',
                    false,
                    { avatar_outcome: 'question' }
                );
                return;
            }
            const contract = getActiveContract();
            if (!contract) {
                addMessage(
                    isTenantUploadAuthMode()
                        ? 'Sign in and upload documents before asking a question.'
                        : 'Please select your agreement in Settings before asking a question.',
                    false,
                    { avatar_outcome: 'question' }
                );
                return;
            }

            sendBtn.disabled = true;
            addMessage(question, true);
            markChatSubmitted();
            showLoading();
            startThinking();  // Animated gradient
            postTelemetryEvent('bug_journey', 'query_submitted', {
                question_length: String(question || '').length,
                authenticated: Boolean(memberAuth?.authenticated),
                tenant_mode: isTenantUploadAuthMode(),
            });

            try {
                const res = await fetchWithAuth(`${API_BASE}/api/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    timeoutMs: 25000,
                    body: JSON.stringify({
                        question: question,
                        union_local_id: contract.union_local_id,
                        contract_id: contract.contract_id,
                        contract_version: contract.contract_version,
                        session_id: SESSION_ID,
                        response_tone: getPreferredResponseTone(),
                        response_verbosity: getPreferredResponseVerbosity(),
                    })
                });

                hideLoading();
                stopThinking();  // Stop animated gradient

                if (!res.ok) {
                    let detail = `API error: ${res.status}`;
                    try {
                        const err = await res.json();
                        if (err?.detail) detail = String(err.detail);
                    } catch (_) {
                        // Ignore JSON parse failures and use status fallback.
                    }
                    throw new Error(detail);
                }

                const data = await res.json();
                addMessage(data.answer, false, {
                    citations: data.citations,
                    sources: data.sources,
                    provider_warning: data.provider_warning,
                    wage_info: data.wage_info,
                    role_clarification: data.role_clarification,
                    retry_question: question,
                    escalation_required: data.escalation_required,
                    avatar_outcome: data.role_clarification
                        ? 'question'
                        : (data.escalation_required ? 'question' : 'confirm')
                });

            } catch (error) {
                postTelemetryEvent('bug_journey', 'query_failed', {
                    reason: error?.message || 'unknown_error',
                    tenant_mode: isTenantUploadAuthMode(),
                });
                hideLoading();
                stopThinking();  // Stop animated gradient
                addMessage(`Unable to process this request: ${error.message}`, false, { avatar_outcome: 'error' });
                console.error(error);
            } finally {
                sendBtn.disabled = false;
            }
        }

        function askQuestion(q) {
            if (!hasCompleteProfileContext()) {
                if (!memberAuth?.authenticated) {
                    openMemberAuthModal();
                    return;
                }
                showOnboarding();
                return;
            }
            document.getElementById('user-input').value = q;
            document.getElementById('chat-form').dispatchEvent(new Event('submit'));
        }

        document.getElementById('chat-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const input = document.getElementById('user-input');
            const q = input.value.trim();
            if (!q) return;
            input.value = '';
            await sendQuery(q);
        });

        // =============================================================================
        // INIT
        // =============================================================================

        async function init() {
            if (routeContext.mode === 'legacy_member') {
                await ensureOnboardingFactoriesLoaded();
                ensureStewardOnboardingController();
                ensureMemberOnboardingController();
                window.location.replace('/karl/');
                return;
            }
            if (!isTenantRoute()) {
                await ensureOnboardingFactoriesLoaded();
                ensureStewardOnboardingController();
                ensureMemberOnboardingController();
            }
            initDarkMode();  // Apply dark mode if saved
            updateDeveloperModeUI(isDeveloperModeEnabled());
            // Decides whether the Contract tab is offered at all.
            refreshTenantContractLibrary().catch(() => {});
            saveContractTextSourceMode(contractTextSourceMode);
            initHeaderInteractivity();  // Header gradient baseline (no pointer interaction)
            scheduleShellLayoutMetricsSync();
            if (document.fonts?.ready) {
                document.fonts.ready
                    .then(() => scheduleShellLayoutMetricsSync())
                    .catch(() => {});
            }
            setKarlAvatarState('idle');
            setHeaderSubtitle(HEADER_SUBTITLE_DEFAULT);
            refreshOnboardingModalOpenState();
            if (routeContext.unionSlug) {
                try {
                    await loadTenantBootstrap();
                } catch (error) {
                    showTenantPageFailure(error.message || 'Unable to load tenant page.');
                    alert(error.message || 'Unable to load tenant page.');
                    return;
                }
            }
            renderMemberAuthSummary();
            applyTenantUploadModeUI();
            await refreshMemberAuth();
            await loadMemberTrackingPreferences();
            updateInteractionLock();
            await loadContracts();
            await loadOnboardingOptions();
            await loadProfile();
            postTelemetryEvent('bug_journey', 'member_workspace_loaded', {
                route_mode: routeContext.mode,
                signed_in: Boolean(memberAuth?.authenticated),
                tenant_slug: routeContext.unionSlug || null,
            });
            if (routeContext.unionSlug && !memberAuth?.authenticated) {
                hideOnboarding();
            }
            checkHealth();
            setInterval(checkHealth, 30000);

            const onboardContract = document.getElementById('onboard-contract');
            if (onboardContract) {
                onboardContract.addEventListener('change', async (e) => {
                    if (e.target.value) {
                        await loadClassificationsForContract(e.target.value, { preserveSelection: false });
                    }
                });
            }
            const settingsContract = document.getElementById('settings-contract');
            if (settingsContract) {
                settingsContract.addEventListener('change', async (e) => {
                    if (e.target.value) {
                        await loadClassificationsForContract(e.target.value, { preserveSelection: false });
                    }
                });
            }
            const memberAuthForm = document.getElementById('member-auth-form');
            if (memberAuthForm) {
                memberAuthForm.addEventListener('submit', async (e) => {
                    try {
                        await loginMemberAuth(e);
                        await loadProfile();
                        checkHealth();
                    } catch (error) {
                        alert(error.message || 'Unable to sign in.');
                    }
                });
            }
            const memberAuthLogout = document.getElementById('member-auth-logout');
            if (memberAuthLogout) {
                memberAuthLogout.addEventListener('click', () => {
                    handleMemberLogout().catch((error) => {
                        console.warn('Member logout failed:', error);
                        clearMemberAuth({ preserveServerSession: true });
                        checkHealth();
                    });
                });
            }
            document.getElementById('member-auth-modal-close')?.addEventListener('click', () => closeMemberAuthModal());
            document.getElementById('member-auth-modal-cancel')?.addEventListener('click', () => closeMemberAuthModal());
            document.getElementById('member-auth-mode-signin')?.addEventListener('click', () => setMemberAuthMode('signin'));
            document.getElementById('member-auth-mode-register')?.addEventListener('click', () => setMemberAuthMode('register'));
            document.getElementById('member-auth-password-toggle')?.addEventListener('click', () => togglePasswordVisibility('member-auth-password', 'member-auth-password-toggle'));
            document.getElementById('member-tracking-save')?.addEventListener('click', () => {
                saveMemberTrackingPreference().catch((error) => {
                    console.warn('Tracking preference save failed:', error);
                    setMemberTrackingStatus(error.message || 'Unable to save tracking choice.', 'error');
                });
            });
            document.getElementById('member-auth-modal')?.addEventListener('click', (event) => {
                if (event.target?.id === 'member-auth-modal') {
                    closeMemberAuthModal();
                }
            });
            const headerSignIn = document.getElementById('header-signin-btn');
            if (headerSignIn) {
                headerSignIn.addEventListener('click', () => {
                    openMemberAuthModal();
                });
            }
            const headerSignOut = document.getElementById('header-signout-btn');
            if (headerSignOut) {
                headerSignOut.addEventListener('click', () => {
                    handleMemberLogout().catch((error) => {
                        console.warn('Member logout failed:', error);
                        clearMemberAuth({ preserveServerSession: true });
                        checkHealth();
                    });
                });
            }

            // Initialize tab bar with correct active state
            loadKarlInfo().catch(() => {
                updateKarlVersionUI();
            });
            setActiveTab('chat');
            scheduleShellLayoutMetricsSync();
            updateInteractionLock();
            revealAppShell();
            setTimeout(() => scheduleShellLayoutMetricsSync(), 0);
        }
// Expose handlers needed by inline HTML and dynamic citation markup.
Object.assign(window, {
    hidePopover,
    handlePopoverSourceChoiceChange,
    handleRoleClarificationChoice,
    expandToContract,
    toggleDatePicker,
    changeYear,
    quickSelect,
    setActiveTab,
    toggleTOC,
    handleContractViewerModeChange,
    openContractPdfFromContractTab,
    handleContractSourceModeChange,
    handleContractSourceDocChange,
    handleContractTextSourceModeChange,
    handleCurrentTargetSourceChange,
    toggleContractTextCompare,
    resetContractPdfView,
    downloadContractPdf,
    openPdfInNewTab,
    closeContractPdfOverlay,
    askQuestion,
    saveSettingsProfile,
    openKarlTab,
    handleKarlDocChange,
    toggleDarkMode,
    toggleDeveloperMode,
    clearSession,
    startNewConversation,
    focusQuestionInput,
    loadArticle,
    loadContractBrowseItem,
    handleCitationClick,
    openUploadedSourceViewer,
    openUploadedSourceInNewTab,
    openContractPdfAtPage,
    openAnswerCitation,
    selectMonth,
});

init();
