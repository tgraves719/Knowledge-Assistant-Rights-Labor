import { ReactNode, CSSProperties, MouseEventHandler } from 'react';

/**
 * @startingPoint section="Components" subtitle="Primary action button, five variants" viewport="700x160"
 */
export interface ButtonProps {
  variant?: 'primary' | 'gold' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  icon?: ReactNode;
  disabled?: boolean;
  children?: ReactNode;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  style?: CSSProperties;
}

export function Button(props: ButtonProps): JSX.Element;
