import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { useApp } from "../context";

/** Guided tour / single highlight. Steps are elements marked with data-tour="<id>"; texts come from tour.<name>.<id>. */
export default function Tour() {
  const { t } = useTranslation();
  const { tour, stepTour, endTour, role } = useApp();
  const [box, setBox] = useState<DOMRect | null>(null);
  const id = tour ? tour.steps[tour.i] : null;

  useEffect(() => {
    if (!id) return;
    const el = document.querySelector<HTMLElement>(`[data-tour="${id}"]`);
    if (!el) { endTour(); return; }
    el.scrollIntoView({ block: "center", behavior: "smooth" });
    el.classList.add("tour-highlight");
    const place = () => setBox(el.getBoundingClientRect());
    const timer = setTimeout(place, 400);
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") endTour();
      if (e.key === "ArrowRight") stepTour(1);
      if (e.key === "ArrowLeft") stepTour(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      clearTimeout(timer);
      el.classList.remove("tour-highlight");
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("keydown", onKey);
    };
  }, [id, endTour, stepTour]);

  if (!tour || !id || !box) return null;
  const name = tour.name === "highlight" ? (role === "government" ? "government" : "farmer") : tour.name;
  const text = t(`tour.${name}.${id}`, { defaultValue: t(`tour.farmer.${id}`, { defaultValue: t(`tour.government.${id}`, { defaultValue: "" }) }) });
  const below = box.bottom + 180 < window.innerHeight;
  const top = below ? box.bottom + 12 : Math.max(12, box.top - 160);
  const left = Math.min(Math.max(12, box.left), window.innerWidth - 332);
  const single = tour.steps.length === 1;
  return (
    <div role="dialog" aria-modal="false" aria-live="polite" aria-label={text}
      className="fixed z-[70] w-[320px] max-w-[calc(100vw-24px)] tile p-4 shadow-xl fade-in" style={{ top, left }}>
      <p className="mb-3">{text}</p>
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm text-muted tabnum">{single ? "" : `${tour.i + 1} / ${tour.steps.length}`}</span>
        <div className="flex gap-2">
          {!single && tour.i > 0 && <button className="btn" onClick={() => stepTour(-1)}>{t("common.back")}</button>}
          {!single && tour.i < tour.steps.length - 1
            ? <button autoFocus className="btn btn-primary" onClick={() => stepTour(1)}>{t("common.next")}</button>
            : <button autoFocus className="btn btn-primary" onClick={endTour}>{single ? t("common.close") : t("common.done")}</button>}
        </div>
      </div>
    </div>
  );
}
