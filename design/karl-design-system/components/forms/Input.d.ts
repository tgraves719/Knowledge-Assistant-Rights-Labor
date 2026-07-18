import { CSSProperties, InputHTMLAttributes } from 'react';

/**
 * @startingPoint section="Components" subtitle="Text input, gold focus ring" viewport="700x120"
 */
export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  style?: CSSProperties;
}

export function Input(props: InputProps): JSX.Element;
