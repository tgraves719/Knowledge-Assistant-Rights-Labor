/**
 * @startingPoint section="Components" subtitle="Karl's split-shield brand mark / avatar" viewport="700x160"
 */
export interface ShieldMarkProps {
  size?: number;
  state?: 'idle' | 'thinking';
  leftColor?: string;
  rightColor?: string;
}

export function ShieldMark(props: ShieldMarkProps): JSX.Element;
