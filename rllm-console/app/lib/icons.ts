/**
 * Map panel `icon` strings (lucide names from the backend) to React
 * components. Unknown names fall back to a generic dot.
 */
import {
  Activity,
  Box,
  Circle,
  GraduationCap,
  List,
  Play,
  Settings,
  type LucideIcon,
} from "lucide-react";

const KNOWN: Record<string, LucideIcon> = {
  activity: Activity,
  box: Box,
  circle: Circle,
  list: List,
  play: Play,
  settings: Settings,
  "graduation-cap": GraduationCap,
};

export function iconForPanel(name: string): LucideIcon {
  return KNOWN[name] ?? Circle;
}
