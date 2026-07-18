/**
 * @startingPoint section="Components" subtitle="Pill switch — dark mode / developer mode" viewport="700x100"
 */
export interface ToggleProps {
  checked?: boolean;
  onChange?: (next: boolean) => void;
  label?: string;
}

export function Toggle(props: ToggleProps): JSX.Element;
