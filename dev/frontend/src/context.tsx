import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { api, useApi } from "./lib/api";
import type { Meta, Role } from "./lib/types";

const ROLE_KEY = "jalvaani.role";
const DISTRICT_KEY = "jalvaani.district";

export const TOURS: Record<"farmer" | "government", string[]> = {
  farmer: ["lang-switch", "district-picker", "card-water-now", "card-forecast", "reliability-badge", "card-rain-soil",
           "advice-list", "chat-button"],
  government: ["freshness-bar", "kpi-row", "map-layers", "map", "glance-panel", "priority-table", "needs-panel", "download-csv", "chat-button"],
};

function read(key: string): string | null {
  try { return localStorage.getItem(key); } catch { return null; }
}
function write(key: string, v: string) {
  try { localStorage.setItem(key, v); } catch { /* ignore */ }
}

interface Ctx {
  meta: Meta | null;
  metaError: string | null;
  role: Role | null;
  setRole: (r: Role) => void;
  savedDistrict: string | null;
  saveDistrict: (slug: string) => void;
  /** guided tour / single highlight: list of element ids and the current step */
  tour: { name: string; steps: string[]; i: number } | null;
  startTour: (name: "farmer" | "government") => void;
  highlight: (element: string) => void;
  stepTour: (delta: number) => void;
  endTour: () => void;
  /** what the user sees now, sent to the chat assistant as context */
  visible: Record<string, unknown>;
  setVisible: (v: Record<string, unknown>) => void;
}

const AppCtx = createContext<Ctx | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const metaState = useApi(api.meta, "meta");
  const [role, setRoleState] = useState<Role | null>(() => {
    const v = read(ROLE_KEY);
    return v === "farmer" || v === "government" ? v : null;
  });
  const [savedDistrict, setSavedDistrict] = useState<string | null>(() => read(DISTRICT_KEY));
  const [tour, setTour] = useState<Ctx["tour"]>(null);
  const [visible, setVisible] = useState<Record<string, unknown>>({});

  const setRole = useCallback((r: Role) => { setRoleState(r); write(ROLE_KEY, r); }, []);
  const saveDistrict = useCallback((s: string) => { setSavedDistrict(s); write(DISTRICT_KEY, s); }, []);
  const present = (ids: string[]) => ids.filter((id) => document.querySelector(`[data-tour="${id}"]`));
  const startTour = useCallback((name: "farmer" | "government") => {
    const steps = present(TOURS[name]);
    if (steps.length) setTour({ name, steps, i: 0 });
  }, []);
  const highlight = useCallback((element: string) => {
    // wait a frame so a navigation triggered by the same chat reply can render first
    setTimeout(() => { if (present([element]).length) setTour({ name: "highlight", steps: [element], i: 0 }); }, 350);
  }, []);
  const stepTour = useCallback((d: number) => setTour((t) => {
    if (!t) return t;
    const i = t.i + d;
    return i < 0 || i >= t.steps.length ? null : { ...t, i };
  }), []);
  const endTour = useCallback(() => setTour(null), []);

  return (
    <AppCtx.Provider value={{ meta: metaState.data, metaError: metaState.error, role, setRole, savedDistrict, saveDistrict,
                              tour, startTour, highlight, stepTour, endTour, visible, setVisible }}>
      {children}
    </AppCtx.Provider>
  );
}

export function useApp() {
  const c = useContext(AppCtx);
  if (!c) throw new Error("useApp outside AppProvider");
  return c;
}
