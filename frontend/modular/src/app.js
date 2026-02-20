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
        let currentPopover = {
            articleNum: null,
            sectionNum: null,
            partNum: null,
            tableId: null,
            rowIndex: null,
            citationLabel: null,
        };
        let currentArticleNum = null;
        let currentPdfBaseUrl = null;
        let currentPdfPage = null;
        let lastPinnedPdfLocation = null;
        let pdfNavRequestSeq = 0;
        let pendingPdfFrameSwapTimer = null;
        let stewardOnboarding = null;
        let memberOnboarding = null;
        let preferences = JSON.parse(localStorage.getItem('karl_preferences') || '{}');
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
                currentArticleNum = null;
                currentPdfBaseUrl = null;
                currentPdfPage = null;
                lastPinnedPdfLocation = null;
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

        function setActiveArticleInToc(normalizedArticleNum) {
            document.querySelectorAll('.toc-item').forEach(item => {
                item.classList.remove('bg-ufcw-blue', 'text-white');
                item.querySelector('span:first-child')?.classList.remove('text-white');
                item.querySelector('span:first-child')?.classList.add('text-ufcw-blue');
                item.querySelector('span:last-child')?.classList.remove('text-white');
                item.querySelector('span:last-child')?.classList.add('text-slate-600');
            });
            document.querySelectorAll(`.toc-item[data-article="${normalizedArticleNum}"]`).forEach(activeItem => {
                activeItem.classList.add('bg-ufcw-blue', 'text-white');
                activeItem.querySelector('span:first-child')?.classList.remove('text-ufcw-blue');
                activeItem.querySelector('span:first-child')?.classList.add('text-white');
                activeItem.querySelector('span:last-child')?.classList.remove('text-slate-600');
                activeItem.querySelector('span:last-child')?.classList.add('text-white');
            });
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
            const normalizedRowIndex = Number.isFinite(Number(options?.rowIndex))
                ? Number(options.rowIndex)
                : null;
            const requestToken = Number.isFinite(Number(options?.requestToken))
                ? Number(options.requestToken)
                : nextPdfNavToken();
            if (!contractId || (normalizedArticleNum === null && !tableId)) {
                return false;
            }

            const overlay = document.getElementById('contract-pdf-overlay');
            const labelEl = document.getElementById('contract-pdf-location-label');
            if (overlay) overlay.classList.remove('hidden');
            if (labelEl) {
                if (tableId) {
                    labelEl.textContent = normalizedRowIndex !== null
                        ? `Locating Table ${tableId}, Row ${normalizedRowIndex + 1}...`
                        : `Locating Table ${tableId}...`;
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

            try {
                const res = await fetch(`${API_BASE}/api/pdf-location?${params.toString()}`);
                if (requestToken !== pdfNavRequestSeq) return false;
                if (!res.ok) {
                    if (res.status === 404) {
                        const baseUrl = `${API_BASE}/api/contract-pdf?contract_id=${encodeURIComponent(contractId)}`;
                        const fallbackLabel = tableId
                            ? `Table ${tableId} -> Contract PDF`
                            : `Article ${normalizedArticleNum} -> Contract PDF`;
                        _showContractPdfOverlay(baseUrl, null, fallbackLabel, { requestToken });
                        return true;
                    }
                    return false;
                }
                const loc = await res.json();
                if (requestToken !== pdfNavRequestSeq) return false;
                if (!loc?.pdf_url) return false;

                const locationBits = [];
                if (tableId) {
                    locationBits.push(`Table ${tableId}`);
                    if (normalizedRowIndex !== null) {
                        locationBits.push(`Row ${normalizedRowIndex + 1}`);
                    }
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
                return false;
            }
        }

        async function openContractPdfFromContractTab() {
            const contractId = getActiveContractId();
            if (!contractId) return false;

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

            const baseUrl = `${API_BASE}/api/contract-pdf?contract_id=${encodeURIComponent(contractId)}`;
            const requestToken = nextPdfNavToken();
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
                    return;
                }
                const res = await fetch(`${API_BASE}/api/manifest${getContractQueryString()}`);
                if (!res.ok) {
                    throw new Error(`Manifest load failed (${res.status})`);
                }
                const manifest = await res.json();
                articleTitles = manifest.article_titles;
                renderTOC();
                initTOCState();
            } catch (e) {
                console.error('Failed to load manifest:', e);
                const desktopList = document.getElementById('article-list');
                const mobileList = document.getElementById('article-list-mobile');
                const errorHTML = '<li class="text-red-500 text-xs py-2">Failed to load</li>';
                if (desktopList) desktopList.innerHTML = errorHTML;
                if (mobileList) mobileList.innerHTML = errorHTML;
            }
        }

        function renderTOC() {
            const desktopList = document.getElementById('article-list');
            const mobileList = document.getElementById('article-list-mobile');
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
            citationLabel = null
        ) {
            event.preventDefault();
            event.stopPropagation();
            const normalizedArticleNum = toPositiveIntOrNull(articleNum);
            const normalizedSectionNum = toPositiveIntOrNull(sectionNum);
            const normalizedTableId = String(tableId || '').trim() || null;
            const normalizedRowIndex = Number.isFinite(Number(rowIndex)) ? Number(rowIndex) : null;
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
                    { tableId: normalizedTableId, rowIndex: normalizedRowIndex }
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
                    }
                );
            }
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

            currentPopover = {
                articleNum,
                sectionNum,
                partNum,
                tableId,
                rowIndex,
                citationLabel,
            };

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
                { tableId, rowIndex }
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
                                // Match Article, Section (with optional parenthetical), and optional Part
                                const match = c.match(/Article\s+(\d+)(?:,?\s*Section\s+(\d+)(?:\(([a-z])\))?)?(?:,?\s*Part\s+([\w\-]+))?/i);
                                if (match) {
                                    const artNum = match[1];
                                    const secNum = match[2] || null;
                                    const parenSub = match[3] || null;
                                    const partNum = match[4] || null;
                                    const subsection = parenSub || partNum;
                                    const subArg = subsection ? `'${subsection}'` : 'null';
                                    return `
                                    <button class="citation-badge inline-flex items-center gap-1 bg-ufcw-blue/10 text-ufcw-blue text-[11px] font-medium px-2 py-1 rounded-md hover:bg-ufcw-blue/20 transition-colors cursor-pointer"
                                            onclick="handleCitationClick(event, ${artNum}, ${secNum || 'null'}, ${subArg})">
                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                                        </svg>
                                        ${escapeHtml(c)}
                                    </button>
                                `;
                                }

                                const source = sourceByCitation.get(String(c || '').trim().toLowerCase());
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
                                        ? sourceSub.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, ' ')
                                        : null;
                                    const safeSourceTableId = sourceTableId
                                        ? sourceTableId.replace(/\\/g, "\\\\").replace(/'/g, "\\'").replace(/\n/g, ' ')
                                        : null;
                                    const safeCitationLabel = String(c || '')
                                        .replace(/\\/g, "\\\\")
                                        .replace(/'/g, "\\'")
                                        .replace(/\n/g, ' ');
                                    const sourceArticleArg = sourceArticleNum !== null ? sourceArticleNum : 'null';
                                    const sourceSecArg = sourceSectionNum !== null ? sourceSectionNum : 'null';
                                    const sourceSubArg = safeSourceSub ? `'${safeSourceSub}'` : 'null';
                                    const sourceTableArg = safeSourceTableId ? `'${safeSourceTableId}'` : 'null';
                                    const sourceRowArg = sourceRowIndex !== null ? sourceRowIndex : 'null';
                                    const sourceLabelArg = `'${safeCitationLabel}'`;
                                    return `
                                    <button class="citation-badge inline-flex items-center gap-1 bg-ufcw-blue/10 text-ufcw-blue text-[11px] font-medium px-2 py-1 rounded-md hover:bg-ufcw-blue/20 transition-colors cursor-pointer"
                                            onclick="handleCitationClick(event, ${sourceArticleArg}, ${sourceSecArg}, ${sourceSubArg}, ${sourceTableArg}, ${sourceRowArg}, ${sourceLabelArg})">
                                        <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                                        </svg>
                                        ${escapeHtml(c)}
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
    expandToContract,
    toggleDatePicker,
    changeYear,
    quickSelect,
    setActiveTab,
    toggleTOC,
    openContractPdfFromContractTab,
    resetContractPdfView,
    downloadContractPdf,
    openPdfInNewTab,
    closeContractPdfOverlay,
    askQuestion,
    saveSettingsProfile,
    toggleDarkMode,
    toggleDeveloperMode,
    clearSession,
    loadArticle,
    handleCitationClick,
    selectMonth,
});

init();
