/**
 * @startingPoint section="Components" subtitle="App tab bar with glowing active state" viewport="700x120"
 */
export interface TabBarTab {
  id: string;
  label: string;
  icon?: any;
}
export interface TabBarProps {
  tabs: TabBarTab[];
  active: string;
  onChange?: (id: string) => void;
}

export function TabBar(props: TabBarProps): JSX.Element;
