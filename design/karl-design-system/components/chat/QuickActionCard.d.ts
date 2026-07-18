import { ReactNode } from 'react';

/**
 * @startingPoint section="Components" subtitle="Translucent suggestion tile on header gradient" viewport="700x140"
 */
export interface QuickActionCardProps {
  icon?: ReactNode;
  children?: ReactNode;
  onClick?: () => void;
}

export function QuickActionCard(props: QuickActionCardProps): JSX.Element;
