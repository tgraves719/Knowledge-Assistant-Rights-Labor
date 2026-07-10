const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const MONTH_FULL = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

export function createStewardOnboardingController(config = {}) {
    const {
        modalId = 'steward-onboarding-modal',
        formId = 'steward-onboarding-form',
        pickerIds = ['onboard', 'settings'],
        setModalOpenState = () => {},
        getActiveContractId = () => null,
        getUserProfile = () => null,
        onSubmit = async () => false,
        onVisibilityChange = () => {},
    } = config;

    const currentYear = new Date().getFullYear();
    const datePickerState = {};
    pickerIds.forEach((pickerId) => {
        datePickerState[pickerId] = { year: currentYear, month: null, isOpen: false };
    });

    let formBound = false;
    let outsideClickBound = false;

    function getElement(id) {
        return document.getElementById(id);
    }

    function getDatePickerState(pickerId) {
        if (!datePickerState[pickerId]) {
            datePickerState[pickerId] = { year: currentYear, month: null, isOpen: false };
        }
        return datePickerState[pickerId];
    }

    function closeDatePicker(pickerId) {
        const picker = getElement(`${pickerId}-date-picker`);
        if (picker) picker.classList.add('hidden');
        getDatePickerState(pickerId).isOpen = false;
    }

    function closeAllDatePickers(exceptPickerId = null) {
        Object.keys(datePickerState).forEach((pickerId) => {
            if (pickerId !== exceptPickerId) {
                closeDatePicker(pickerId);
            }
        });
    }

    function renderMonths(pickerId) {
        const state = getDatePickerState(pickerId);
        const container = getElement(`${pickerId}-months`);
        if (!container) return;

        const thisYear = new Date().getFullYear();
        const thisMonth = new Date().getMonth();
        const hireValue = String(getElement(`${pickerId}-hire-date`)?.value || '');

        container.innerHTML = MONTHS.map((month, index) => {
            const isFuture = state.year === thisYear && index > thisMonth;
            const isSelected = state.month === index && hireValue.startsWith(`${state.year}-${String(index + 1).padStart(2, '0')}`);
            return `<button type="button"
                onclick="selectMonth('${pickerId}', ${index})"
                class="month-btn ${isSelected ? 'selected' : ''} ${isFuture ? 'opacity-30 cursor-not-allowed' : 'hover:bg-ufcw-gold hover:text-white'}"
                ${isFuture ? 'disabled' : ''}>
                ${month}
            </button>`;
        }).join('');
    }

    function toggleDatePicker(pickerId) {
        const picker = getElement(`${pickerId}-date-picker`);
        if (!picker) return;
        const state = getDatePickerState(pickerId);
        if (state.isOpen) {
            closeDatePicker(pickerId);
            return;
        }
        closeAllDatePickers(pickerId);
        picker.classList.remove('hidden');
        state.isOpen = true;
        renderMonths(pickerId);
    }

    function changeYear(pickerId, delta) {
        const state = getDatePickerState(pickerId);
        const nextYear = state.year + Number(delta || 0);
        const thisYear = new Date().getFullYear();
        if (nextYear > thisYear || nextYear < thisYear - 50) return;
        state.year = nextYear;
        const label = getElement(`${pickerId}-year-display`);
        if (label) label.textContent = String(nextYear);
        renderMonths(pickerId);
    }

    function selectMonth(pickerId, monthIndex) {
        const state = getDatePickerState(pickerId);
        state.month = monthIndex;

        const value = `${state.year}-${String(monthIndex + 1).padStart(2, '0')}`;
        const hiddenInput = getElement(`${pickerId}-hire-date`);
        if (hiddenInput) hiddenInput.value = value;

        const displaySpan = getElement(`${pickerId}-date-display`);
        if (displaySpan) {
            displaySpan.textContent = `${MONTH_FULL[monthIndex]} ${state.year}`;
            displaySpan.classList.remove('text-slate-400');
            displaySpan.classList.add('text-slate-700', 'dark:text-slate-200');
        }

        closeDatePicker(pickerId);
    }

    function quickSelect(pickerId, yearsAgo) {
        const state = getDatePickerState(pickerId);
        const thisYear = new Date().getFullYear();
        const thisMonth = new Date().getMonth();
        state.year = thisYear - Number(yearsAgo || 0);
        const label = getElement(`${pickerId}-year-display`);
        if (label) label.textContent = String(state.year);
        const monthToSelect = Number(yearsAgo || 0) === 0 ? thisMonth : 0;
        selectMonth(pickerId, monthToSelect);
    }

    function initDatePicker(pickerId, existingValue) {
        const value = String(existingValue || '').trim();
        if (!value) return;
        const [yearRaw, monthRaw] = value.split('-');
        const year = Number(yearRaw);
        const month = Number(monthRaw);
        if (!Number.isFinite(year) || !Number.isFinite(month) || month < 1 || month > 12) return;

        const state = getDatePickerState(pickerId);
        state.year = year;
        state.month = month - 1;

        const yearLabel = getElement(`${pickerId}-year-display`);
        if (yearLabel) yearLabel.textContent = String(year);

        const displaySpan = getElement(`${pickerId}-date-display`);
        if (displaySpan) {
            displaySpan.textContent = `${MONTH_FULL[month - 1]} ${year}`;
            displaySpan.classList.remove('text-slate-400');
            displaySpan.classList.add('text-slate-700', 'dark:text-slate-200');
        }
    }

    function syncStewardFormFromContext() {
        const profile = getUserProfile() || null;
        const activeContractId = getActiveContractId();

        const contractSelect = getElement('onboard-contract');
        if (contractSelect) {
            contractSelect.value = String(profile?.contract_id || activeContractId || '');
        }

        if (profile?.classification) {
            const select = getElement('onboard-classification');
            if (select) select.value = profile.classification;
        }

        if (profile?.employment_type) {
            const radio = document.querySelector(`input[name="employment_type"][value="${profile.employment_type}"]`);
            if (radio) radio.checked = true;
        }

        const hireValue = String(profile?.hire_date || '').trim();
        if (hireValue) {
            const monthValue = hireValue.slice(0, 7);
            const input = getElement('onboard-hire-date');
            if (input) input.value = monthValue;
            initDatePicker('onboard', monthValue);
        }
    }

    function readStewardFormData() {
        const contractId = String(getElement('onboard-contract')?.value || '').trim();
        const classification = String(getElement('onboard-classification')?.value || '').trim();
        const employmentType = document.querySelector('input[name="employment_type"]:checked')?.value || '';
        const hireMonth = String(getElement('onboard-hire-date')?.value || '').trim();

        return {
            contractId,
            classification,
            employmentType,
            hireMonth,
        };
    }

    async function handleStewardSubmit(event) {
        event.preventDefault();
        const { contractId, classification, employmentType, hireMonth } = readStewardFormData();
        if (!contractId) {
            alert('Please select your contract/store before continuing.');
            return false;
        }
        if (!classification) {
            alert('Please select your job classification before continuing.');
            return false;
        }

        const data = {};
        data.contract_id = contractId;
        data.classification = classification;
        if (employmentType) data.employment_type = employmentType;
        if (hireMonth) data.hire_date = `${hireMonth}-01`;

        return onSubmit(data);
    }

    function show() {
        const modal = getElement(modalId);
        if (!modal) return;
        modal.classList.remove('hidden');
        modal.scrollTop = 0;
        setModalOpenState(true);
        syncStewardFormFromContext();
        onVisibilityChange(true);
    }

    function hide() {
        const modal = getElement(modalId);
        if (!modal) return;
        modal.classList.add('hidden');
        setModalOpenState(false);
        onVisibilityChange(false);
    }

    function isVisible() {
        const modal = getElement(modalId);
        return !!modal && !modal.classList.contains('hidden');
    }

    function bind() {
        if (!formBound) {
            const form = getElement(formId);
            if (form) {
                form.addEventListener('submit', handleStewardSubmit);
                formBound = true;
            }
        }

        if (!outsideClickBound) {
            document.addEventListener('click', (event) => {
                Object.keys(datePickerState).forEach((pickerId) => {
                    const state = datePickerState[pickerId];
                    if (!state?.isOpen) return;
                    const picker = getElement(`${pickerId}-date-picker`);
                    const trigger = getElement(`${pickerId}-date-trigger`);
                    if (!picker || !trigger) return;
                    if (!picker.contains(event.target) && !trigger.contains(event.target)) {
                        closeDatePicker(pickerId);
                    }
                });
            });
            outsideClickBound = true;
        }
    }

    return {
        bind,
        show,
        hide,
        isVisible,
        syncStewardFormFromContext,
        toggleDatePicker,
        changeYear,
        quickSelect,
        selectMonth,
        initDatePicker,
    };
}
