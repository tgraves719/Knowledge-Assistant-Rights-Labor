import { ReactNode, CSSProperties } from 'react';

/**
 * @startingPoint section="Components" subtitle="Status & citation-provenance pill" viewport="700x140"
 */
export interface BadgeProps {
  tone?: 'neutral' | 'citation' | 'success' | 'warning' | 'danger' | 'info';
  children?: ReactNode;
  style?: CSSProperties;
}

export function Badge(props: BadgeProps): JSX.Element;
