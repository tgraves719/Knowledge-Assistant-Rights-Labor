function wait(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

const STEP_CARD_FADE_MS = 180;
const PAGE_TURN_MS = 360;
const FINAL_PAGE_EXIT_MS = 170;
const SHIELD_CLOSE_SETTLE_MS = 260;

const GENERAL_DEPARTMENT_KEY = '__general__';
const FINAL_PUZZLE_PIECE_ID = 'mo-puzzle-piece-6';
const PUZZLE_PIECE_IDS = [
    'mo-puzzle-piece-1',
    'mo-puzzle-piece-2',
    'mo-puzzle-piece-3',
    'mo-puzzle-piece-4',
    'mo-puzzle-piece-5',
    'mo-puzzle-piece-6',
];

export function createMemberOnboardingController(config = {}) {
    const {
        modalId = 'member-onboarding-modal',
        setModalOpenState = () => {},
        getContracts = () => [],
        getClassifications = async () => [],
        getUserProfile = () => null,
        onRequestStewardFlow = async () => {},
        onSubmitMember = async () => false,
        onVisibilityChange = () => {},
    } = config;

    const SCENES = ['init', 'role', 'transition', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6'];
    const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const EARLIEST_HIRE_YEAR = 1970;

    let bound = false;
    let visible = false;
    let currentScene = 'init';
    let narrativeYear = new Date().getFullYear();
    let contractCatalog = [];
    let classRenderToken = 0;
    let isStepping = false;
    let isFinishing = false;
    let timers = [];
    let draft = {
        role: 'member',
        employer: '',
        location: '',
        department_key: '',
        department_label: '',
        contract_id: '',
        classification: '',
        employment_type: '',
        hire_date: '',
    };

    function el(id) {
        return document.getElementById(id);
    }

    function modalEl() {
        return el(modalId);
    }

    function addTimer(cb, ms) {
        const id = setTimeout(cb, ms);
        timers.push(id);
        return id;
    }

    function clearTimers() {
        for (const id of timers) clearTimeout(id);
        timers = [];
    }

    function normalizeText(value) {
        return String(value || '')
            .replace(/[_-]+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }

    function toKey(value) {
        return normalizeText(value).toLowerCase();
    }

    function toTitleCase(value) {
        const normalized = normalizeText(value);
        if (!normalized) return '';
        return normalized
            .split(' ')
            .map((part) => (part ? part[0].toUpperCase() + part.slice(1).toLowerCase() : part))
            .join(' ');
    }

    function toReadableCase(value) {
        const normalized = normalizeText(value);
        if (!normalized) return '';

        const hasUpper = /[A-Z]/.test(normalized);
        const hasLower = /[a-z]/.test(normalized);
        if (!(hasUpper && !hasLower)) return normalized;

        const preserved = new Set(['UFCW', 'LLC', 'USA', 'CBA']);
        return normalized
            .split(' ')
            .map((word) =>
                word
                    .split('-')
                    .map((part) => {
                        const clean = part.replace(/[^A-Z0-9]/g, '');
                        if (!clean) return part;
                        if (preserved.has(clean)) return clean;
                        if (/^[A-Z]{2}$/.test(clean)) return clean;
                        if (/^\d+$/.test(clean)) return part;
                        const lowered = part.toLowerCase();
                        return lowered ? lowered[0].toUpperCase() + lowered.slice(1) : lowered;
                    })
                    .join('-')
            )
            .join(' ');
    }

    function extractLatestYear(text) {
        const matches = String(text || '').match(/\b(?:19|20)\d{2}\b/g) || [];
        if (!matches.length) return 0;
        return matches.reduce((max, raw) => {
            const year = Number(raw);
            return Number.isFinite(year) ? Math.max(max, year) : max;
        }, 0);
    }

    function inferDepartment(contract, labelSegments = []) {
        const explicit = normalizeText(contract?.department || contract?.contract_type || '');
        if (explicit) return toTitleCase(explicit);

        const contractId = String(contract?.contract_id || '').toLowerCase();
        if (contractId.includes('_clerks_')) return 'Clerks';
        if (contractId.includes('_meat_')) return 'Meat';
        if (contractId.includes('_pharmacy_')) return 'Pharmacy';

        const fromLabel = normalizeText(labelSegments[2] || '');
        if (fromLabel && !/^\d{4}$/.test(fromLabel)) return toTitleCase(fromLabel);
        return 'General';
    }

    function buildContractRow(contract) {
        const value = String(contract?.contract_id || contract?.value || '').trim();
        const label = String(contract?.label || contract?.display_name || contract?.name || contract?.contract_id || '').trim();
        if (!value || !label) return null;

        const segments = label.split(' - ').map((part) => part.trim()).filter(Boolean);
        const employer = toReadableCase(contract?.employer || segments[0] || '');
        const location = toReadableCase(contract?.location || contract?.region_id || segments[1] || '');
        const departmentLabel = inferDepartment(contract, segments);
        const departmentKey = toKey(departmentLabel) || GENERAL_DEPARTMENT_KEY;

        return {
            value,
            label,
            employer,
            employerKey: toKey(employer),
            location,
            locationKey: toKey(location),
            departmentLabel,
            departmentKey,
            sortYear: extractLatestYear(`${value} ${label}`),
        };
    }

    function refreshContractCatalog() {
        contractCatalog = (getContracts() || [])
            .map((contract) => buildContractRow(contract))
            .filter(Boolean);
    }

    function getUniqueOptions(rows, keyGetter, labelGetter) {
        const seen = new Map();
        rows.forEach((row) => {
            const key = String(keyGetter(row) || '').trim();
            const label = String(labelGetter(row) || '').trim();
            if (!key || !label) return;
            if (!seen.has(key)) seen.set(key, label);
        });
        return Array.from(seen.entries())
            .map(([value, label]) => ({ value, label }))
            .sort((a, b) => a.label.localeCompare(b.label));
    }

    function getRowsByEmployerLocation() {
        const employerKey = toKey(draft.employer);
        const locationKey = toKey(draft.location);

        return contractCatalog.filter((row) => {
            if (employerKey && row.employerKey !== employerKey) return false;
            if (locationKey && row.locationKey !== locationKey) return false;
            return true;
        });
    }

    function getFilteredContractOptions() {
        const departmentKey = String(draft.department_key || '').trim();
        const rows = getRowsByEmployerLocation();
        if (!departmentKey) return rows;
        return rows.filter((row) => row.departmentKey === departmentKey);
    }

    function pickPreferredContract(rows) {
        if (!rows.length) return null;
        if (draft.contract_id) {
            const existing = rows.find((row) => row.value === draft.contract_id);
            if (existing) return existing;
        }
        return [...rows].sort((a, b) => {
            if (b.sortYear !== a.sortYear) return b.sortYear - a.sortYear;
            return a.label.localeCompare(b.label);
        })[0];
    }

    function updateEmployerLabel(label = '') {
        const node = el('mo-employer-label');
        if (!node) return;
        node.textContent = label || draft.employer || 'Select employer...';
    }

    function updateLocationLabel(label = '') {
        const node = el('mo-location-label');
        if (!node) return;
        node.textContent = label || draft.location || 'Select location...';
    }

    function updateDepartmentLabel(label = '') {
        const node = el('mo-department-label');
        if (!node) return;
        node.textContent = label || draft.department_label || 'Select department...';
    }

    function updateClassificationLabel(label = '') {
        const node = el('mo-class-label');
        if (!node) return;
        node.textContent = label || draft.classification || 'Select role...';
    }

    function setContractId(nextContractId) {
        const next = String(nextContractId || '').trim();
        if (draft.contract_id === next) return;
        draft.contract_id = next;
        draft.classification = '';
        updateClassificationLabel();
    }

    function syncContractFromFilters() {
        const rows = getFilteredContractOptions();
        const preferred = pickPreferredContract(rows);
        setContractId(preferred?.value || '');
    }

    function setEmployerSelection(value = '', label = '') {
        draft.employer = normalizeText(value);
        updateEmployerLabel(label);
        renderLocationOptions();
        renderDepartmentOptions();
        syncContractFromFilters();
    }

    function setLocationSelection(value = '', label = '') {
        draft.location = normalizeText(value);
        updateLocationLabel(label);
        renderDepartmentOptions();
        syncContractFromFilters();
    }

    function setDepartmentSelection(key = '', label = '') {
        draft.department_key = String(key || '').trim();
        draft.department_label = String(label || '').trim();
        updateDepartmentLabel();
        syncContractFromFilters();
    }

    function setClassificationSelection(value, label) {
        draft.classification = String(value || '').trim();
        updateClassificationLabel(label || '');
    }

    function setEmploymentType(value = '') {
        const normalized = String(value || '').trim();
        draft.employment_type = normalized === 'full_time' || normalized === 'part_time'
            ? normalized
            : '';
        const fullBtn = el('mo-employment-full');
        const partBtn = el('mo-employment-part');
        fullBtn?.classList.toggle('selected', draft.employment_type === 'full_time');
        partBtn?.classList.toggle('selected', draft.employment_type === 'part_time');
    }

    function showScene(sceneName) {
        currentScene = sceneName;
        SCENES.forEach((name) => {
            el(`mo-scene-${name}`)?.classList.remove('active');
        });
        const activeScene = el(`mo-scene-${sceneName}`);
        if (activeScene) {
            activeScene.querySelectorAll('.mo-top-zone, .mo-card-zone').forEach((node) => {
                node.style.opacity = '';
                node.style.pointerEvents = '';
            });
            activeScene.classList.add('active');
        }
        setPuzzleProgress(getScenePuzzleProgress(sceneName));
        updateShieldNav();
    }

    function animateTo(element, keyframes, options) {
        if (!element || !element.animate) return Promise.resolve();
        return element.animate(keyframes, { fill: 'forwards', ...options }).finished.catch(() => {});
    }

    function getScenePuzzleProgress(sceneName) {
        if (sceneName === 'm2') return 1;
        if (sceneName === 'm3') return 2;
        if (sceneName === 'm4') return 3;
        if (sceneName === 'm5') return 4;
        if (sceneName === 'm6') return 5;
        return 0;
    }

    function setPuzzleProgress(count) {
        const filledCount = Math.max(0, Math.min(PUZZLE_PIECE_IDS.length, Number(count) || 0));
        PUZZLE_PIECE_IDS.forEach((id, index) => {
            const node = el(id);
            if (!node) return;
            node.classList.toggle('is-filled', index < filledCount);
            node.classList.remove('is-final-ready');
        });
    }

    async function primeFinalPuzzlePiece() {
        const finalPiece = el(FINAL_PUZZLE_PIECE_ID);
        if (!finalPiece) return;
        finalPiece.classList.add('is-filled');
        finalPiece.classList.add('is-final-ready');
        await wait(280);
    }

    function shieldReset() {
        const left = el('mo-shield-left');
        const right = el('mo-shield-right');
        const paper = el('mo-paper-stack');
        const topPage = el('mo-top-page');
        if (left) left.style.transform = 'translateX(0px)';
        if (right) right.style.transform = 'translateX(0px)';
        if (paper) {
            paper.style.transform = 'translate(100px,110px) scale(0.9)';
            paper.style.opacity = '0';
        }
        if (topPage) {
            topPage.style.transform = 'translate(-45px,-5px) rotate(-8deg)';
            topPage.style.opacity = '0';
        }
        setPuzzleProgress(0);
    }

    function shieldEngage() {
        animateTo(el('mo-shield-left'), [{ transform: 'translateX(-65px)' }], { duration: 800, easing: 'cubic-bezier(0.2, 0, 0.2, 1)' });
        animateTo(el('mo-shield-right'), [{ transform: 'translateX(65px)' }], { duration: 800, easing: 'cubic-bezier(0.2, 0, 0.2, 1)' });
        animateTo(el('mo-paper-stack'), [{ transform: 'translate(100px,110px) scale(1)', opacity: 1 }], { duration: 900, easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)' });
    }

    async function shieldAddPage() {
        await animateTo(
            el('mo-top-page'),
            [
                { transform: 'translate(-45px,-5px) rotate(-8deg)', opacity: 0 },
                { transform: 'translate(0,0) rotate(0deg)', opacity: 1 },
            ],
            { duration: PAGE_TURN_MS, easing: 'cubic-bezier(0.34, 1.56, 0.64, 1)' }
        );
    }

    function shieldClose() {
        animateTo(el('mo-shield-left'), [{ transform: 'translateX(0px)' }], { duration: 700, easing: 'cubic-bezier(0.4, 0, 0.2, 1)' });
        animateTo(el('mo-shield-right'), [{ transform: 'translateX(0px)' }], { duration: 700, easing: 'cubic-bezier(0.4, 0, 0.2, 1)' });
        animateTo(el('mo-paper-stack'), [{ transform: 'translate(100px,110px) scale(0.9)', opacity: 0 }], { duration: 600, easing: 'ease-in-out' });
    }

    function setShieldNavButton(id, isVisible, label = '') {
        const button = el(id);
        if (!button) return;
        if (!isVisible) {
            button.classList.add('hidden');
            return;
        }
        if (label) button.textContent = label;
        button.classList.remove('hidden');
    }

    function updateShieldNav() {
        if (!visible) {
            setShieldNavButton('mo-shield-nav-back', false);
            setShieldNavButton('mo-shield-nav-next', false);
            return;
        }

        if (currentScene === 'm1' || currentScene === 'm2' || currentScene === 'm3' || currentScene === 'm4' || currentScene === 'm5' || currentScene === 'm6') {
            setShieldNavButton('mo-shield-nav-back', true, 'Back');
            setShieldNavButton('mo-shield-nav-next', true, 'Not sure');
            return;
        }

        setShieldNavButton('mo-shield-nav-back', false);
        setShieldNavButton('mo-shield-nav-next', false);
    }

    function goBackScene(target) {
        const normalized = String(target || '').trim();
        if (!normalized) return;
        if (normalized === 'role') {
            clearTimers();
            shieldReset();
        }
        showScene(normalized);
    }

    async function handleShieldBack() {
        if (currentScene === 'm1') {
            shieldClose();
            await wait(320);
            goBackScene('role');
            return;
        }
        if (currentScene === 'm2') return goBackScene('m1');
        if (currentScene === 'm3') return goBackScene('m2');
        if (currentScene === 'm4') return goBackScene('m3');
        if (currentScene === 'm5') return goBackScene('m4');
        if (currentScene === 'm6') return goBackScene('m5');
    }

    async function handleShieldNext() {
        if (currentScene === 'm1') return stepMember(1, true);
        if (currentScene === 'm2') return stepMember(2, true);
        if (currentScene === 'm3') return stepMember(3, true);
        if (currentScene === 'm4') return stepMember(4, true);
        if (currentScene === 'm5') return stepMember(5, true);
        if (currentScene === 'm6') return stepMember(6, true);
    }

    function closeAllDropdowns() {
        ['mo-employer-list', 'mo-location-list', 'mo-department-list', 'mo-class-list'].forEach((id) => {
            const node = el(id);
            if (node?.style) node.style.display = 'none';
        });
    }

    function toggleDropdown(targetId) {
        const target = el(targetId);
        if (!target) return;
        const isOpen = target.style.display === 'block';
        closeAllDropdowns();
        target.style.display = isOpen ? 'none' : 'block';
    }

    function renderSelectionList(listId, options, valueAttr, labelAttr, emptyLabel) {
        const list = el(listId);
        if (!list) return;
        list.innerHTML = '';
        if (!options.length) {
            const emptyBtn = document.createElement('button');
            emptyBtn.type = 'button';
            emptyBtn.disabled = true;
            emptyBtn.textContent = emptyLabel;
            list.appendChild(emptyBtn);
            return;
        }
        options.forEach((option) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.textContent = option.label;
            button.setAttribute(valueAttr, option.value);
            button.setAttribute(labelAttr, option.label);
            list.appendChild(button);
        });
    }

    function renderEmployerOptions() {
        const options = getUniqueOptions(contractCatalog, (row) => row.employerKey, (row) => row.employer);
        const stillValid = options.some((opt) => opt.value === toKey(draft.employer));
        if (!stillValid) {
            draft.employer = '';
            updateEmployerLabel();
        }
        renderSelectionList('mo-employer-list', options, 'data-employer-value', 'data-employer-label', 'No employers found');
    }

    function renderLocationOptions() {
        const employerKey = toKey(draft.employer);
        const scoped = employerKey ? contractCatalog.filter((row) => row.employerKey === employerKey) : contractCatalog;
        const options = getUniqueOptions(scoped, (row) => row.locationKey, (row) => row.location);
        const stillValid = options.some((opt) => opt.value === toKey(draft.location));
        if (!stillValid) {
            draft.location = '';
            updateLocationLabel();
        }
        renderSelectionList('mo-location-list', options, 'data-location-value', 'data-location-label', 'No locations found');
    }

    function renderDepartmentOptions() {
        const scoped = getRowsByEmployerLocation();
        const options = getUniqueOptions(scoped, (row) => row.departmentKey, (row) => row.departmentLabel);
        const stillValid = options.some((opt) => opt.value === draft.department_key);
        if (!stillValid) {
            draft.department_key = '';
            draft.department_label = '';
            updateDepartmentLabel();
        }
        renderSelectionList('mo-department-list', options, 'data-department-key', 'data-department-label', 'No departments found');
    }

    async function renderClassificationOptions(contractId) {
        const token = ++classRenderToken;
        if (!contractId) {
            renderSelectionList('mo-class-list', [], 'data-class-value', 'data-class-label', 'Select department first');
            return;
        }

        const optionsRaw = await getClassifications(contractId);
        if (token !== classRenderToken) return;

        const options = (optionsRaw || [])
            .map((o) => ({
                value: String(o.value || '').trim(),
                label: toReadableCase(String(o.label || o.value || '').trim()),
            }))
            .filter((o) => o.value && o.label);

        if (!options.length) {
            renderSelectionList('mo-class-list', [], 'data-class-value', 'data-class-label', 'No classifications found');
            draft.classification = '';
            updateClassificationLabel();
            return;
        }

        renderSelectionList('mo-class-list', options, 'data-class-value', 'data-class-label', 'No classifications found');
        if (draft.classification) {
            const match = options.find((option) => option.value === draft.classification);
            if (match) {
                updateClassificationLabel(match.label);
            } else {
                draft.classification = '';
                updateClassificationLabel();
            }
        }
    }

    function setYear(year) {
        const now = new Date();
        const thisYear = now.getFullYear();
        const parsed = Number(year) || thisYear;
        narrativeYear = Math.max(EARLIEST_HIRE_YEAR, Math.min(thisYear, parsed));
        const yearDisplay = el('mo-year-display');
        if (yearDisplay) yearDisplay.textContent = String(narrativeYear);
        renderMonths();
    }

    function isFutureMonthSelection(year, monthNumber) {
        const now = new Date();
        const thisYear = now.getFullYear();
        const thisMonth = now.getMonth() + 1;
        if (year > thisYear) return true;
        if (year < thisYear) return false;
        return monthNumber > thisMonth;
    }

    function renderMonths() {
        const grid = el('mo-month-grid');
        if (!grid) return;
        const selected = String(draft.hire_date || '').trim();
        const selectedPrefix = `${narrativeYear}-`;
        grid.innerHTML = MONTHS.map((month, i) => {
            const mm = String(i + 1).padStart(2, '0');
            const isSelected = selected.startsWith(selectedPrefix) && selected.includes(`-${mm}-`);
            const isFuture = isFutureMonthSelection(narrativeYear, i + 1);
            return `<button type="button" class="mo-month-btn${isSelected ? ' selected' : ''}${isFuture ? ' disabled' : ''}" data-month="${i + 1}" ${isFuture ? 'disabled' : ''}>${month}</button>`;
        }).join('');
    }

    function setHireDate(monthNumber) {
        if (isFutureMonthSelection(narrativeYear, monthNumber)) return false;
        const mm = String(monthNumber).padStart(2, '0');
        draft.hire_date = `${narrativeYear}-${mm}-01`;
        const input = el('mo-hire-date');
        if (input) input.value = draft.hire_date;
        return true;
    }

    function hydrateFromProfile() {
        const profile = getUserProfile() || {};
        refreshContractCatalog();

        draft = {
            role: 'member',
            employer: '',
            location: '',
            department_key: '',
            department_label: '',
            contract_id: String(profile.contract_id || '').trim(),
            classification: String(profile.classification || '').trim(),
            employment_type: String(profile.employment_type || '').trim(),
            hire_date: '',
        };

        const contractMatch = draft.contract_id
            ? contractCatalog.find((row) => row.value === draft.contract_id)
            : null;
        if (contractMatch) {
            draft.employer = contractMatch.employer || '';
            draft.location = contractMatch.location || '';
            draft.department_key = contractMatch.departmentKey || '';
            draft.department_label = contractMatch.departmentLabel || '';
        } else if (profile.employer) {
            draft.employer = toReadableCase(profile.employer);
        }

        const hireDate = String(profile.hire_date || '').trim();
        if (hireDate) {
            const normalized = hireDate.slice(0, 10);
            const year = Number(normalized.slice(0, 4));
            const month = Number(normalized.slice(5, 7));
            if (Number.isFinite(year)) setYear(year);
            if (Number.isFinite(year) && Number.isFinite(month) && month >= 1 && month <= 12 && !isFutureMonthSelection(year, month)) {
                draft.hire_date = normalized;
            } else {
                draft.hire_date = '';
            }
            const input = el('mo-hire-date');
            if (input) input.value = draft.hire_date || '';
        } else {
            setYear(new Date().getFullYear());
            const input = el('mo-hire-date');
            if (input) input.value = '';
        }

        updateEmployerLabel();
        updateLocationLabel();
        updateDepartmentLabel();
        updateClassificationLabel(profile.classification_display || '');
        setEmploymentType(draft.employment_type);

        renderEmployerOptions();
        renderLocationOptions();
        renderDepartmentOptions();
        syncContractFromFilters();
        renderClassificationOptions(draft.contract_id);
    }

    async function fadeCardOut(step) {
        const card = el(`mo-card-${step}`);
        if (!card) return;
        card.style.opacity = '0';
        card.style.transform = 'scale(0.88) translateY(12px)';
        await wait(STEP_CARD_FADE_MS);
        card.style.opacity = '1';
        card.style.transform = '';
    }

    async function fadeSceneForClose(sceneName) {
        const scene = el(`mo-scene-${sceneName}`);
        if (!scene) return;

        const nodes = [
            scene.querySelector('.mo-top-zone'),
            scene.querySelector('.mo-card-zone'),
        ].filter(Boolean);

        if (!nodes.length) return;

        const topZone = scene.querySelector('.mo-top-zone');
        const cardZone = scene.querySelector('.mo-card-zone');
        const animations = [];
        if (topZone) {
            animations.push(
                animateTo(topZone, [{ opacity: 1 }, { opacity: 0 }], { duration: FINAL_PAGE_EXIT_MS, easing: 'ease-out' })
            );
        }
        if (cardZone) {
            animations.push(
                animateTo(
                    cardZone,
                    [
                        { opacity: 1, transform: 'translateX(-50%) scale(1)' },
                        { opacity: 0, transform: 'translateX(-50%) scale(0.88) translateY(18px)' },
                    ],
                    { duration: FINAL_PAGE_EXIT_MS, easing: 'cubic-bezier(0.4, 0, 0.2, 1)' }
                )
            );
        }
        await Promise.all(animations);

        nodes.forEach((node) => {
            node.style.opacity = '0';
            node.style.pointerEvents = 'none';
        });
    }

    async function stepMember(step, skip = false) {
        if (isStepping) return;
        isStepping = true;
        await fadeCardOut(step);
        try {
            if (step === 1) {
                if (skip) setEmployerSelection('', 'Not sure yet');
                showScene('m2');
                void shieldAddPage();
                return;
            }

            if (step === 2) {
                if (skip) setLocationSelection('', 'Not sure yet');
                showScene('m3');
                void shieldAddPage();
                return;
            }

            if (step === 3) {
                if (skip) setDepartmentSelection('', 'Not sure yet');
                showScene('m4');
                void shieldAddPage();
                void renderClassificationOptions(draft.contract_id);
                return;
            }

            if (step === 4) {
                if (skip) setClassificationSelection('', 'Not sure yet');
                showScene('m5');
                void shieldAddPage();
                return;
            }

            if (step === 5) {
                if (skip) setEmploymentType('');
                showScene('m6');
                void shieldAddPage();
                return;
            }

            await finishMember(skip);
        } finally {
            isStepping = false;
        }
    }

    async function finishMember(skipDate = false) {
        if (isFinishing) return false;
        isFinishing = true;
        try {
            setPuzzleProgress(PUZZLE_PIECE_IDS.length - 1);
            await primeFinalPuzzlePiece();
            if (skipDate) {
                draft.hire_date = '';
                const input = el('mo-hire-date');
                if (input) input.value = '';
            }

            setShieldNavButton('mo-shield-nav-back', false);
            setShieldNavButton('mo-shield-nav-next', false);
            await fadeSceneForClose(currentScene);
            shieldClose();
            await wait(SHIELD_CLOSE_SETTLE_MS);

            const payload = {};
            if (draft.contract_id) payload.contract_id = draft.contract_id;
            if (draft.classification) payload.classification = draft.classification;
            if (draft.employment_type) payload.employment_type = draft.employment_type;
            if (draft.hire_date) payload.hire_date = draft.hire_date;

            const ok = await onSubmitMember(payload);
            if (ok) {
                hide();
                return true;
            }
            return false;
        } finally {
            isFinishing = false;
        }
    }

    function startMemberSequence() {
        showScene('transition');

        const transitionText = el('mo-trans-text');
        if (transitionText) {
            transitionText.style.transition = 'none';
            transitionText.style.top = '50%';
            transitionText.style.transform = 'translate(-50%, -50%)';
            transitionText.style.opacity = '0';
            requestAnimationFrame(() => {
                requestAnimationFrame(() => {
                    transitionText.style.transition = 'opacity 0.7s ease';
                    transitionText.style.opacity = '1';
                });
            });

            addTimer(() => {
                transitionText.style.transition = 'top 1.1s cubic-bezier(0.4, 0, 0.2, 1), transform 1.1s cubic-bezier(0.4, 0, 0.2, 1)';
                transitionText.style.top = '6vh';
                transitionText.style.transform = 'translateX(-50%)';
            }, 2500);
        }

        addTimer(() => shieldEngage(), 3400);
        addTimer(() => showScene('m1'), 4400);
    }

    function bind() {
        if (bound) return;
        bound = true;

        el('mo-begin-btn')?.addEventListener('click', () => showScene('role'));
        el('mo-role-member')?.addEventListener('click', () => {
            draft.role = 'member';
            startMemberSequence();
        });
        el('mo-role-steward')?.addEventListener('click', async () => {
            hide();
            await onRequestStewardFlow();
        });

        el('mo-employer-trigger')?.addEventListener('click', () => toggleDropdown('mo-employer-list'));
        el('mo-location-trigger')?.addEventListener('click', () => toggleDropdown('mo-location-list'));
        el('mo-department-trigger')?.addEventListener('click', () => toggleDropdown('mo-department-list'));
        el('mo-class-trigger')?.addEventListener('click', async () => {
            await renderClassificationOptions(draft.contract_id);
            toggleDropdown('mo-class-list');
        });

        el('mo-employer-list')?.addEventListener('click', async (event) => {
            const btn = event.target.closest('button[data-employer-value]');
            if (!btn) return;
            setEmployerSelection(btn.dataset.employerValue || '', btn.dataset.employerLabel || '');
            closeAllDropdowns();
            await stepMember(1, false);
        });

        el('mo-location-list')?.addEventListener('click', async (event) => {
            const btn = event.target.closest('button[data-location-value]');
            if (!btn) return;
            setLocationSelection(btn.dataset.locationValue || '', btn.dataset.locationLabel || '');
            closeAllDropdowns();
            await stepMember(2, false);
        });

        el('mo-department-list')?.addEventListener('click', async (event) => {
            const btn = event.target.closest('button[data-department-key]');
            if (!btn) return;
            setDepartmentSelection(btn.dataset.departmentKey || '', btn.dataset.departmentLabel || '');
            closeAllDropdowns();
            await stepMember(3, false);
        });

        el('mo-class-list')?.addEventListener('click', async (event) => {
            const btn = event.target.closest('button[data-class-value]');
            if (!btn) return;
            setClassificationSelection(btn.dataset.classValue || '', btn.dataset.classLabel || '');
            closeAllDropdowns();
            await stepMember(4, false);
        });

        el('mo-employment-full')?.addEventListener('click', async () => {
            setEmploymentType('full_time');
            await stepMember(5, false);
        });
        el('mo-employment-part')?.addEventListener('click', async () => {
            setEmploymentType('part_time');
            await stepMember(5, false);
        });

        el('mo-shield-nav-back')?.addEventListener('click', async () => handleShieldBack());
        el('mo-shield-nav-next')?.addEventListener('click', async () => handleShieldNext());

        el('mo-year-prev')?.addEventListener('click', () => setYear(narrativeYear - 1));
        el('mo-year-next')?.addEventListener('click', () => setYear(narrativeYear + 1));
        el('mo-month-grid')?.addEventListener('click', (event) => {
            const btn = event.target.closest('button[data-month]');
            if (!btn) return;
            const month = Number(btn.dataset.month || 0);
            if (month < 1 || month > 12) return;
            if (!setHireDate(month)) return;
            renderMonths();
        });

        el('mo-finish-member')?.addEventListener('click', async () => finishMember(false));

        document.addEventListener('click', (event) => {
            if (!visible) return;
            const modal = modalEl();
            if (!modal || !modal.contains(event.target)) return;
            const target = event.target instanceof Element ? event.target : null;
            if (!target) return;
            if (target.closest('.mo-drop-list') || target.closest('.mo-drop-trigger')) return;
            closeAllDropdowns();
        });
    }

    function show() {
        const modal = modalEl();
        if (!modal) return;
        clearTimers();
        visible = true;
        modal.classList.remove('hidden');
        setModalOpenState(true);
        onVisibilityChange(true);
        shieldReset();
        hydrateFromProfile();
        showScene('init');
    }

    function hide() {
        const modal = modalEl();
        if (!modal) return;
        clearTimers();
        visible = false;
        closeAllDropdowns();
        modal.classList.add('hidden');
        setModalOpenState(false);
        onVisibilityChange(false);
        updateShieldNav();
    }

    function isVisible() {
        return visible;
    }

    return {
        bind,
        show,
        hide,
        isVisible,
    };
}
