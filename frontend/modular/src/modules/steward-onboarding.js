// Steward onboarding / contract switcher.
//
// As of the KARL 1.0 QR work this collects ONLY the contract a steward is
// working in. Role (job classification) and start date are no longer asked at
// onboarding — the assistant infers what it needs from the conversation, so a
// steward just picks which contract to read and can switch it any time from
// Settings. The date-picker methods are retained as no-ops so any lingering
// inline handlers stay harmless.
export function createStewardOnboardingController(config = {}) {
    const {
        modalId = 'steward-onboarding-modal',
        formId = 'steward-onboarding-form',
        setModalOpenState = () => {},
        getActiveContractId = () => null,
        getUserProfile = () => null,
        onSubmit = async () => false,
        onVisibilityChange = () => {},
    } = config;

    let formBound = false;

    function getElement(id) {
        return document.getElementById(id);
    }

    function syncStewardFormFromContext() {
        const profile = getUserProfile() || null;
        const activeContractId = getActiveContractId();
        const contractSelect = getElement('onboard-contract');
        if (contractSelect) {
            contractSelect.value = String(profile?.contract_id || activeContractId || '');
        }
    }

    async function handleStewardSubmit(event) {
        event.preventDefault();
        const contractId = String(getElement('onboard-contract')?.value || '').trim();
        if (!contractId) {
            alert('Please select your contract/store before continuing.');
            return false;
        }
        return onSubmit({ contract_id: contractId });
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
        if (formBound) return;
        const form = getElement(formId);
        if (form) {
            form.addEventListener('submit', handleStewardSubmit);
            formBound = true;
        }
    }

    const noop = () => {};

    return {
        bind,
        show,
        hide,
        isVisible,
        syncStewardFormFromContext,
        // Retained no-ops for backward-compatible inline handlers.
        toggleDatePicker: noop,
        changeYear: noop,
        quickSelect: noop,
        selectMonth: noop,
        initDatePicker: noop,
    };
}
