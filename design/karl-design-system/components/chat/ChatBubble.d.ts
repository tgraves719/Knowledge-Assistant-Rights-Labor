import { ReactNode } from 'react';

/**
 * @startingPoint section="Components" subtitle="Assistant / user message bubble" viewport="700x220"
 */
export interface ChatBubbleProps {
  role?: 'assistant' | 'user';
  children?: ReactNode;
}

export function ChatBubble(props: ChatBubbleProps): JSX.Element;
