import { createStewardOnboardingController } from './modules/steward-onboarding.js';
import { createMemberOnboardingController } from './modules/member-onboarding.js';

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
        const ACTIVE_CONTRACT_STORAGE_KEY = 'karl_active_contract_id';
        const CONTRACT_PDF_SOURCE_MODE_STORAGE_KEY = 'karl_contract_pdf_source_mode';
        const CONTRACT_TEXT_SOURCE_MODE_STORAGE_KEY = 'karl_contract_text_source_mode';
        const SESSION_ID_STORAGE_KEY = 'karl_session_id';
        const SESSION_META_STORAGE_KEY = 'karl_session_meta';
        const ONBOARDING_FLOW_STORAGE_KEY = 'karl_onboarding_flow';
        let isHealthy = false;
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
        let stewardOnboarding = null;
        let memberOnboarding = null;
        let preferences = JSON.parse(localStorage.getItem('karl_preferences') || '{}');
        let contractHistoryById = {};
        let activeContractHistory = null;
        let contractPdfSourceMode = String(localStorage.getItem(CONTRACT_PDF_SOURCE_MODE_STORAGE_KEY) || 'effective').trim().toLowerCase() || 'effective';
        let contractPdfSourcePdf = null;
        let contractTextSourceMode = String(localStorage.getItem(CONTRACT_TEXT_SOURCE_MODE_STORAGE_KEY) || 'effective').trim().toLowerCase() || 'effective';
        let citationSourceRegistry = {};
        let citationSourceRegistrySeq = 0;
        let sessionMetaStore = loadSessionMetaStore();
        let SESSION_ID = getOrCreateSessionId();
        let shellLayoutSyncRaf = 0;
        const HEADER_SUBTITLE_DEFAULT = 'Union Contract Assistant';
        const HEADER_SUBTITLE_THINKING = 'Reviewing your contract...';
        const HEADER_SUBTITLE_SPEAKING = 'Answering with citations';
        const HEADER_SUBTITLE_QUESTION = 'Needs follow-up';
        const HEADER_SUBTITLE_ERROR = 'Something went wrong';
        let karlAvatarSpeakTimer = null;

        function generateSessionId() {
            return 'session_' + Math.random().toString(36).substring(2, 15) + Date.now().toString(36);
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

        function hasSubmittedChatText() {
            return (ensureSessionMeta(SESSION_ID).submitted_count || 0) > 0;
        }

        function hasCompleteProfileContext() {
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
            if (input) {
                input.disabled = locked;
                input.placeholder = locked
                    ? 'Select a contract in onboarding to start chat...'
                    : 'Ask about your contract...';
            }
            if (sendBtn) sendBtn.disabled = locked;

            document.querySelectorAll('[data-requires-profile="true"]').forEach(btn => {
                btn.disabled = locked;
            });
        }

        function markChatSubmitted() {
            const meta = ensureSessionMeta(SESSION_ID);
            meta.submitted_count = (meta.submitted_count || 0) + 1;
            meta.updated_at = new Date().toISOString();
            saveSessionMetaStore();
        }

        function getActiveContract() {
            return activeContract;
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

        function saveContractPdfSourceMode(mode) {
            contractPdfSourceMode = normalizeContractPdfSourceMode(mode);
            localStorage.setItem(CONTRACT_PDF_SOURCE_MODE_STORAGE_KEY, contractPdfSourceMode);
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

            if (provenanceEl) {
                const bits = [`Text Source: ${sourceLabel}`];
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
            citationSourceRegistry[key] = row;
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

        function choosePreferredProvenanceRef(provenance, preferredMode = null) {
            const refs = Array.isArray(provenance) ? provenance.filter((row) => row && typeof row === 'object') : [];
            if (!refs.length) return null;

            const normalizedMode = normalizeContractPdfSourceMode(preferredMode);
            const moaRefs = refs.filter((row) => {
                const sourceType = safeText(row.source_type).toLowerCase();
                const pdf = safeText(row.pdf).toLowerCase();
                return sourceType.includes('moa') || sourceType.includes('amend') || pdf.includes('moa');
            });
            const baseRefs = refs.filter((row) => {
                const sourceType = safeText(row.source_type).toLowerCase();
                return sourceType.includes('base') || sourceType.includes('cba') || sourceType.includes('contract');
            });

            if (normalizedMode === 'moa' && moaRefs.length) return moaRefs[0];
            if (normalizedMode === 'base' && baseRefs.length) return baseRefs[0];
            if (moaRefs.length && normalizedMode === 'effective') return moaRefs[0];
            if (baseRefs.length) return baseRefs[0];
            return refs[0];
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

            if (hasMoa && hasBase) return 'MOA+Base';
            if (hasMoa) return 'MOA';
            if (hasBase) return 'Base';
            return '';
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

            const mode = normalizeContractPdfSourceMode(contractPdfSourceMode);
            const history = getActiveContractHistory();
            if (mode === 'effective') {
                return { sourceType: null, sourcePdf: null, sourcePage: null };
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

            if (!panel || !versionEl || !hashEl || !amendmentsEl || !coverageEl || !timelineEl || !patchListEl || !modeSelect || !sourceDocSelect) {
                return;
            }

            if (!history) {
                panel.classList.add('hidden');
                coverageEl.textContent = 'Chunk coverage: --';
                timelineEl.innerHTML = '';
                modeSelect.innerHTML = '<option value="effective">Effective</option>';
                sourceDocSelect.innerHTML = '<option value="">Auto</option>';
                sourceDocSelect.disabled = true;
                return;
            }

            panel.classList.remove('hidden');

            const effectiveVersion = safeText(history.effective_version_id);
            versionEl.textContent = effectiveVersion
                ? `Effective Snapshot: ${effectiveVersion}`
                : 'Effective Snapshot: Base Only';
            const hash = safeText(history.effective_content_hash);
            hashEl.textContent = hash ? `Content Hash: ${hash.slice(0, 16)}...` : 'Content Hash: unavailable';

            const patchIds = Array.isArray(history.applied_patch_ids) ? history.applied_patch_ids : [];
            amendmentsEl.textContent = patchIds.length
                ? `${patchIds.length} amendment${patchIds.length === 1 ? '' : 's'} applied`
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
            coverageEl.textContent = `Chunks base/effective: ${baseChunkTotal}/${effectiveChunkTotal}${typeCoverage ? ` | ${typeCoverage}` : ''}`;

            const patches = Array.isArray(history.patches) ? history.patches : [];
            const timelinePills = [
                `<span class="inline-flex items-center rounded border border-slate-300 bg-white px-2 py-0.5 text-[10px] font-medium text-slate-700">Base${safeText(history.base_pdf) ? `: ${escapeHtml(safeText(history.base_pdf))}` : ''}</span>`
            ];
            patches.forEach((patch) => {
                const patchId = escapeHtml(safeText(patch.patch_id) || 'patch');
                const date = escapeHtml(safeText(patch.effective_date) || safeText(patch.ratified_date) || 'undated');
                timelinePills.push('<span class="text-slate-400 text-[10px]">-></span>');
                timelinePills.push(`<span class="inline-flex items-center rounded border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-medium text-amber-800">MOA ${patchId} (${date})</span>`);
            });
            const effectiveLabel = escapeHtml(safeText(history.effective_version_id) || 'base_only');
            timelinePills.push('<span class="text-slate-400 text-[10px]">-></span>');
            timelinePills.push(`<span class="inline-flex items-center rounded border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-medium text-emerald-800">Effective ${effectiveLabel}</span>`);
            timelineEl.innerHTML = timelinePills.join('');

            patchListEl.innerHTML = patches.length
                ? patches.map((patch) => {
                    const patchId = escapeHtml(safeText(patch.patch_id) || 'patch');
                    const date = escapeHtml(safeText(patch.effective_date) || 'undated');
                    const pdf = escapeHtml(safeText(patch.source_pdf) || '');
                    return `<span class="inline-flex items-center rounded border border-slate-200 bg-white px-2 py-0.5 text-[10px] text-slate-700">${patchId} (${date}${pdf ? `, ${pdf}` : ''})</span>`;
                }).join('')
                : '<span class="text-[10px] text-slate-500">No patch chain metadata available.</span>';

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
                return;
            }

            if (!uniqueSourceOptions.length) {
                sourceDocSelect.innerHTML = '<option value="">No source PDF available</option>';
                sourceDocSelect.disabled = true;
                contractPdfSourcePdf = null;
                return;
            }

            sourceDocSelect.disabled = false;
            sourceDocSelect.innerHTML = uniqueSourceOptions.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join('');
            if (!contractPdfSourcePdf || !uniqueSourceOptions.includes(contractPdfSourcePdf)) {
                contractPdfSourcePdf = uniqueSourceOptions[0];
            }
            sourceDocSelect.value = contractPdfSourcePdf;
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
                    option.textContent = hasWage
                        ? c.label
                        : `${c.label} (no Appendix A wage row)`;
                    if (!hasWage) {
                        option.dataset.wageAvailable = 'false';
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
                const res = await fetch(`${API_BASE}/api/classifications?contract_id=${encodeURIComponent(contractId)}`);
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
            if (subtitle) {
                subtitle.textContent = contract
                    ? `${getContractLabel(contract)} | ${contract.union_local_id || 'Union Contract'}`
                    : 'Union Contract Assistant';
            }
            const displayContract = document.getElementById('display-contract');
            if (displayContract) {
                displayContract.textContent = getContractLabel(contract);
            }
        }

        function setActiveContract(contractId, options = {}) {
            const { persist = true, refreshViewer = true, preserveClassification = true } = options;
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
                const overlay = document.getElementById('contract-pdf-overlay');
                const frame = document.getElementById('contract-pdf-frame');
                const label = document.getElementById('contract-pdf-location-label');
                overlay?.classList.add('hidden');
                if (frame) frame.src = '';
                if (label) label.textContent = 'Loading PDF location...';
                renderContractHistoryPanel();
                if (currentActiveTab === 'contract') {
                    initContractViewer();
                    openContractPdfFromContractTab().catch((err) => {
                        console.warn('Unable to reset contract PDF for new contract:', err);
                    });
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
            } catch (e) {
                console.error('Failed to load contracts:', e);
                availableContracts = [];
                activeContract = null;
                updateContractDisplay();
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
        }

        // Initialize TOC collapsed state
        function initTOCState() {
            const mobileList = document.getElementById('toc-list-mobile');
            const chevron = document.getElementById('toc-chevron');

            // Mobile starts collapsed
            mobileList?.classList.add('hidden');
            chevron?.classList.remove('rotate-180');
            tocExpanded = false;
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

            const activeBtn = document.getElementById(`tab-${tab}`);
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
                // PDF-first contract reader: open on tab enter when profile context is available.
                if (hasCompleteProfileContext() && !isContractPdfOverlayOpen()) {
                    openContractPdfFromContractTab().catch((err) => {
                        console.warn('Unable to open default contract PDF view:', err);
                    });
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
            // Hide default PDF viewer UI chrome when embedded.
            params.push('toolbar=0');
            params.push('navpanes=0');
            params.push('scrollbar=0');
            return `${cleanBaseUrl}#${params.join('&')}`;
        }

        function _buildFullPdfViewerUrl(baseUrl, pageNumber = null) {
            const cleanBaseUrl = String(baseUrl || '').split('#')[0].trim();
            if (!cleanBaseUrl) return '';
            if (Number.isFinite(Number(pageNumber)) && Number(pageNumber) > 0) {
                return `${cleanBaseUrl}#page=${Number(pageNumber)}`;
            }
            return cleanBaseUrl;
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

            const ctx = lastPdfNavigationContext;
            const target = ctx?.target || null;
            const candidates = Array.isArray(ctx?.sourceCandidates) ? ctx.sourceCandidates : [];

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
                const label = escapeHtml(String(candidate?.label || 'Source'));
                const selectedAttr = String(candidate?.key || '') === selectedKey ? ' selected' : '';
                return `<option value="${key}"${selectedAttr}>${label}</option>`;
            }).join('');
            wrap.classList.remove('hidden');
        }

        function updatePreviousPdfButtonState() {
            const button = document.getElementById('contract-view-previous-btn');
            const hint = document.getElementById('contract-view-previous-hint');
            renderCurrentTargetSourceControls();
            if (!button) return;

            const ctx = lastPdfNavigationContext;
            const candidate = _findPreviousPdfSourceCandidate(ctx);
            const hasPrevious = !!candidate;
            button.disabled = !hasPrevious;
            button.classList.toggle('opacity-50', !hasPrevious);
            button.classList.toggle('cursor-not-allowed', !hasPrevious);

            if (hint) {
                if (!ctx?.target) {
                    hint.textContent = '';
                } else if (hasPrevious) {
                    const page = Number.isFinite(Number(candidate?.page_number)) ? Number(candidate.page_number) : null;
                    hint.textContent = page ? `Previous available (p.${page})` : 'Previous available';
                } else {
                    hint.textContent = 'No previous source for current selection';
                }
            }
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

        async function openPreviousForCurrentSelection() {
            const ctx = lastPdfNavigationContext;
            const target = ctx?.target;
            const candidate = _findPreviousPdfSourceCandidate(ctx);
            if (!target || !candidate) return false;
            return openSourceChoiceForCurrentSelection(safeText(candidate.key) || 'previous_base');
        }

        function _showContractPdfOverlay(baseUrl, pageNumber = null, locationLabel = 'Contract PDF', options = {}) {
            const overlay = document.getElementById('contract-pdf-overlay');
            const frame = document.getElementById('contract-pdf-frame');
            const label = document.getElementById('contract-pdf-location-label');
            if (!overlay || !frame || !baseUrl) return;
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

            const nextSrc = _buildContractPdfUrl(currentPdfBaseUrl, currentPdfPage);
            const previousSrc = String(frame.getAttribute('src') || '').trim();
            const previousDoc = previousSrc.split('#')[0];
            const nextDoc = nextSrc.split('#')[0];
            if (pendingPdfFrameSwapTimer) {
                clearTimeout(pendingPdfFrameSwapTimer);
                pendingPdfFrameSwapTimer = null;
            }
            if (forceReload) {
                frame.src = 'about:blank';
                pendingPdfFrameSwapTimer = setTimeout(() => {
                    if (requestToken !== null && requestToken !== pdfNavRequestSeq) return;
                    frame.src = nextSrc;
                }, 30);
            } else if (previousSrc && previousDoc === nextDoc && previousSrc !== nextSrc) {
                // Chromium's built-in PDF viewer can ignore hash-only changes in iframe.
                frame.src = 'about:blank';
                pendingPdfFrameSwapTimer = setTimeout(() => {
                    if (requestToken !== null && requestToken !== pdfNavRequestSeq) return;
                    frame.src = nextSrc;
                }, 30);
            } else {
                frame.src = nextSrc;
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
            const articleNum = toPositiveIntOrNull(normalizedArticleNum);
            if (articleNum === null) return;
            setActiveTocSelection('article', `article:${articleNum}`);
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
            const normalizedArticleNum = toPositiveIntOrNull(articleNum);
            if (normalizedArticleNum === null) return false;
            const { preferSection = true } = options;
            const requestToken = nextPdfNavToken();

            if (preferSection) {
                const locator = await getFirstSectionLocator(normalizedArticleNum);
                if (requestToken !== pdfNavRequestSeq) return false;
                const normalizedSection = toPositiveIntOrNull(locator?.sectionNum);
                if (normalizedSection !== null) {
                    const openedBySection = await openContractInPdf(
                        normalizedArticleNum,
                        normalizedSection,
                        locator?.partNum || null,
                        { requestToken }
                    );
                    if (openedBySection) return true;
                }
            }
            return openContractInPdf(normalizedArticleNum, null, null, { requestToken });
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
                const label = `${sectionLabel} -> ${pageLabel} (${matchedBy})`;
                const rawPdfUrl = String(loc.pdf_url || '').trim();
                const resolvedPdfUrl = /^https?:\/\//i.test(rawPdfUrl)
                    ? rawPdfUrl
                    : `${API_BASE}${rawPdfUrl.startsWith('/') ? '' : '/'}${rawPdfUrl}`;
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
                        let prefix = kind === 'article' && articleNum !== null
                            ? `${articleNum}.`
                            : (safeText(item?.label) || kind.toUpperCase());
                        if (kind === 'appendix') {
                            prefix = 'Appx';
                        }
                        const titleText = title || safeText(item?.label) || key;
                        const safeKind = escapeJsSingleQuoted(kind);
                        const safeKey = escapeJsSingleQuoted(key);
                        const clickHandler = kind === 'article' && articleNum !== null
                            ? `loadArticle(${articleNum})`
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
            const { openPdf = true } = options;
            const contractId = getActiveContractId();
            const normalizedKind = safeText(kind).toLowerCase() || null;
            const normalizedKey = safeText(key) || null;
            if (!contractId || !normalizedKind || !normalizedKey) return false;

            currentArticleNum = null;
            setActiveTocSelection(normalizedKind, normalizedKey);

            if (window.innerWidth < 768 && tocExpanded) {
                toggleTOC();
            }

            try {
                const textMode = normalizeContractTextSourceMode(contractTextSourceMode);
                const data = await fetchContractBrowseItemForTextSource(normalizedKind, normalizedKey, textMode);
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
                }
            );
        }

        async function loadArticleTextForCurrentMode(articleNum) {
            const normalizedArticleNum = toPositiveIntOrNull(articleNum);
            const contractId = getActiveContractId();
            if (!contractId || normalizedArticleNum === null) return false;
            try {
                const textMode = normalizeContractTextSourceMode(contractTextSourceMode);
                const data = await fetchArticleForTextSource(normalizedArticleNum, textMode);
                renderArticleContent(data);
                _rememberContractTextPayload(
                    { kind: 'article', key: `article:${normalizedArticleNum}`, articleNum: normalizedArticleNum },
                    data,
                    textMode,
                );
                return true;
            } catch (err) {
                console.error('Failed to load article text:', err);
                renderNonArticleContent(null, { error: 'Failed to load article text.' });
                return false;
            }
        }

        async function loadArticle(articleNum, options = {}) {
            const { openPdf = true } = options;
            const contractId = getActiveContractId();
            const normalizedArticleNum = toPositiveIntOrNull(articleNum);
            if (!contractId || normalizedArticleNum === null) return false;
            currentArticleNum = normalizedArticleNum;
            setActiveArticleInToc(normalizedArticleNum);

            // On mobile, collapse TOC after selection for more reading space
            if (window.innerWidth < 768 && tocExpanded) {
                toggleTOC();
            }

            await loadArticleTextForCurrentMode(normalizedArticleNum);

            if (!openPdf) return true;
            return openArticleInPdf(normalizedArticleNum, { preferSection: true });
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

            const articleHTML = `
                <article class="max-w-none">
                    <header class="mb-6 pb-4 border-b border-slate-200">
                        <p class="text-ufcw-blue font-semibold text-sm uppercase tracking-wider">Article ${data.article_num}</p>
                        <h1 class="text-xl md:text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">${escapeHtml(data.article_title)}</h1>
                    </header>

                    ${data.sections.map(section => `
                        <section id="${getSectionId(section)}" class="mb-6 pb-6 border-b border-slate-100 last:border-0 scroll-mt-4">
                            <h2 class="text-base font-semibold text-slate-800 mb-2">
                                Section ${section.section_num}${section.subsection ? ` (${section.subsection})` : ''}
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
            const openedArticle = await loadArticle(articleNum, { openPdf: normalizedSectionNum === null });
            if (sectionNum) {
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
            subtitleEl.textContent = '';

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
                        'This citation points to a table reference. Use **Open In Contract PDF** to jump directly to the table.'
                    );
                    summaryEl.innerHTML = renderInlineMarkdown(
                        rowIndex !== null ? `Table ${tableId}, Row ${rowIndex + 1}` : `Table ${tableId}`
                    );
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

            setActiveTab('contract');
            if (articleNum !== null) {
                currentArticleNum = articleNum;
                setActiveArticleInToc(articleNum);
                await loadArticle(articleNum, { openPdf: false });
            }
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
            try {
                const res = await fetch(`${API_BASE}/api/onboard/options`);
                await res.json();
            } catch (e) {
                console.error('Failed to load options:', e);
            }
        }

        async function loadProfile() {
            try {
                const res = await fetch(`${API_BASE}/api/profile/${SESSION_ID}`);
                const profile = await res.json();
                if (profile?.contract_id) {
                    setActiveContract(profile.contract_id, { persist: true, refreshViewer: true, preserveClassification: false });
                    await loadClassificationsForContract(profile.contract_id, { preserveSelection: false });
                }

                userProfile = profile;
                if (profile.contract_id) {
                    updateProfileDisplay();
                    ensureSessionMeta(SESSION_ID).contract_id = profile.contract_id || null;
                    ensureSessionMeta(SESSION_ID).classification = profile.classification || null;
                    saveSessionMetaStore();
                    hideOnboarding();
                } else {
                    const hydrated = await hydrateProfileFromCache();
                    if (!hydrated) {
                        showOnboarding();
                    }
                }
                updateInteractionLock();
            } catch (e) {
                console.error('Failed to load profile:', e);
                const hydrated = await hydrateProfileFromCache();
                if (!hydrated) {
                    showOnboarding();
                }
                updateInteractionLock();
            }
        }

        async function saveProfile(data, options = {}) {
            const allowIncomplete = options?.allowIncomplete === true;
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
                const contractChanged = !!nextContractId && nextContractId.trim().toLowerCase() !== currentContractId;
                const classificationChanged =
                    typeof nextClassification === 'string' &&
                    nextClassification.trim().toLowerCase() !== currentClassification;

                if ((contractChanged || classificationChanged) && hasSubmittedChatText()) {
                    const changedParts = [];
                    if (contractChanged) changedParts.push('contract/store');
                    if (classificationChanged) changedParts.push('job classification');
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
                updateProfileDisplay();
                ensureSessionMeta(SESSION_ID).contract_id = userProfile.contract_id || null;
                ensureSessionMeta(SESSION_ID).classification = userProfile.classification || null;
                saveSessionMetaStore();
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
                userProfile.classification_display || userProfile.classification || 'Not set';

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
            updateDeveloperModeUI(isDeveloperModeEnabled());
        }

        async function saveSettingsProfile() {
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
            if (isEnabled) {
                toggle?.classList.remove('bg-slate-200');
                toggle?.classList.add('bg-ufcw-blue');
                dot?.classList.add('translate-x-5');
            } else {
                toggle?.classList.add('bg-slate-200');
                toggle?.classList.remove('bg-ufcw-blue');
                dot?.classList.remove('translate-x-5');
            }
        }

        // Dark mode
        function toggleDarkMode() {
            const isDark = document.documentElement.classList.toggle('dark');
            savePreference('darkMode', isDark);
            updateDarkModeUI(isDark);
        }

        function updateDarkModeUI(isDark) {
            const toggle = document.getElementById('dark-mode-toggle');
            const dot = document.getElementById('dark-mode-dot');
            if (isDark) {
                toggle?.classList.remove('bg-slate-200');
                toggle?.classList.add('bg-ufcw-blue');
                dot?.classList.add('translate-x-5');
            } else {
                toggle?.classList.add('bg-slate-200');
                toggle?.classList.remove('bg-ufcw-blue');
                dot?.classList.remove('translate-x-5');
            }
        }

        function initDarkMode() {
            const isDark = preferences.darkMode || false;
            if (isDark) {
                document.documentElement.classList.add('dark');
            }
            updateDarkModeUI(isDark);
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
        }

        function scheduleShellLayoutMetricsSync() {
            if (shellLayoutSyncRaf) cancelAnimationFrame(shellLayoutSyncRaf);
            shellLayoutSyncRaf = requestAnimationFrame(() => {
                shellLayoutSyncRaf = 0;
                syncShellLayoutMetrics();
            });
        }

        function clearSession() {
            if (confirm('This will clear your profile and conversation history. Continue?')) {
                localStorage.removeItem(SESSION_ID_STORAGE_KEY);
                localStorage.removeItem(SESSION_META_STORAGE_KEY);
                localStorage.removeItem('karl_preferences');
                localStorage.removeItem(ACTIVE_CONTRACT_STORAGE_KEY);
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
                        <span class="text-xs text-blue-100">Select contract</span>
                    `;
                    return;
                }
                const suffix = contractId ? `?contract_id=${encodeURIComponent(contractId)}` : '';
                const res = await fetch(`${API_BASE}/api/health${suffix}`);
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

            // Lists and quotes
            processed = processed.replace(/^\s*[-*\u2022]\s+(.+)$/gm, '<div class="ml-4 mb-1">&bull; $1</div>');
            processed = processed.replace(/^\s*(\d+)\.\s+(.+)$/gm, '<div class="ml-4 mb-1"><span class="font-medium">$1.</span> $2</div>');
            processed = processed.replace(/^>\s?(.+)$/gm, '<blockquote class="border-l-2 border-slate-300 pl-3 italic text-slate-600 dark:text-slate-300 my-1">$1</blockquote>');

            // Make citations clickable after markdown transforms.
            processed = parseCitations(processed);

            processed = processed.replace(/\n/g, '<br>');
            return processed;
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
                let citationsHtml = '';
                if (metadata?.citations?.length > 0) {
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
                                const hint = resolveCitationSourceHint(source, contractPdfSourceMode);
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
                                        return `
                                            <div class="bg-white/70 border border-green-200 rounded px-2 py-1 text-[11px] text-green-800 flex flex-wrap items-center gap-x-2">
                                                <span class="font-semibold">${tableId}</span>
                                                <span>${escapeHtml(rowLabel)}</span>
                                                <span>${step}</span>
                                                <span class="font-semibold">${rate}</span>
                                            </div>
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

                msgDiv.innerHTML = `
                    <div class="bg-white border border-slate-200 rounded-2xl rounded-bl-sm px-4 py-3 max-w-[95%] shadow-sm">
                        <div class="text-sm text-slate-700 leading-relaxed">${formatResponse(content)}</div>
                        ${wageHtml}
                        ${escalationHtml}
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
                    <span class="text-xs">Searching contract...</span>
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
                showOnboarding();
                addMessage('Select your contract and role in onboarding before asking questions.', false, { avatar_outcome: 'question' });
                return;
            }
            const contract = getActiveContract();
            if (!contract) {
                addMessage('Please select your contract in Settings before asking a question.', false, { avatar_outcome: 'question' });
                return;
            }

            sendBtn.disabled = true;
            addMessage(question, true);
            markChatSubmitted();
            showLoading();
            startThinking();  // Animated gradient

            try {
                const res = await fetch(`${API_BASE}/api/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        question: question,
                        union_local_id: contract.union_local_id,
                        contract_id: contract.contract_id,
                        contract_version: contract.contract_version,
                        session_id: SESSION_ID
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
                    wage_info: data.wage_info,
                    escalation_required: data.escalation_required,
                    avatar_outcome: data.escalation_required ? 'question' : 'confirm'
                });

            } catch (error) {
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
            ensureStewardOnboardingController();
            ensureMemberOnboardingController();
            initDarkMode();  // Apply dark mode if saved
            updateDeveloperModeUI(isDeveloperModeEnabled());
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
            updateInteractionLock();
            await loadContracts();
            await loadOnboardingOptions();
            await loadProfile();
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

            // Initialize tab bar with correct active state
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
    expandToContract,
    toggleDatePicker,
    changeYear,
    quickSelect,
    setActiveTab,
    toggleTOC,
    openContractPdfFromContractTab,
    handleContractSourceModeChange,
    handleContractSourceDocChange,
    handleContractTextSourceModeChange,
    handleCurrentTargetSourceChange,
    toggleContractTextCompare,
    resetContractPdfView,
    openPreviousForCurrentSelection,
    downloadContractPdf,
    openPdfInNewTab,
    closeContractPdfOverlay,
    askQuestion,
    saveSettingsProfile,
    toggleDarkMode,
    toggleDeveloperMode,
    clearSession,
    loadArticle,
    loadContractBrowseItem,
    handleCitationClick,
    selectMonth,
});

init();
