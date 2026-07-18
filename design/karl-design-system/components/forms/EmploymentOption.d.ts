/**
 * @startingPoint section="Components" subtitle="Selectable classification card (onboarding)" viewport="700x150"
 */
export interface EmploymentOptionProps {
  title: string;
  subtitle?: string;
  selected?: boolean;
  onClick?: () => void;
}

export function EmploymentOption(props: EmploymentOptionProps): JSX.Element;
