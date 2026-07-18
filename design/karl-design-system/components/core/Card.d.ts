import { ReactNode, CSSProperties } from 'react';

/**
 * @startingPoint section="Components" subtitle="Generic bordered surface container" viewport="700x180"
 */
export interface CardProps {
  padded?: boolean;
  children?: ReactNode;
  style?: CSSProperties;
}

export function Card(props: CardProps): JSX.Element;
