import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { useTranslation } from "react-i18next";
import ChatWidget from "./components/ChatWidget";
import FreshnessBar from "./components/FreshnessBar";
import Header from "./components/Header";
import Tour from "./components/Tour";
import { Loading } from "./components/Bits";
import FarmerPage from "./pages/FarmerPage";
import StartPage from "./pages/StartPage";

// the government pages (map, tables, charts) and help load only when opened: a small first download for farmers
const GovPage = lazy(() => import("./pages/GovPage"));
const GovDistrictPage = lazy(() => import("./pages/GovDistrictPage"));
const HelpPage = lazy(() => import("./pages/HelpPage"));

export default function App() {
  const { t } = useTranslation();
  return (
    <div className="min-h-screen flex flex-col">
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 bg-surface px-3 py-2 rounded">
        {t("common.skip")}
      </a>
      <Header />
      <FreshnessBar />
      <main id="main" className="flex-1 w-full px-4 sm:px-6 lg:px-8 pt-6 pb-28">
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/" element={<StartPage />} />
            <Route path="/farmer" element={<FarmerPage />} />
            <Route path="/farmer/:slug" element={<FarmerPage />} />
            <Route path="/gov" element={<GovPage />} />
            <Route path="/gov/district/:slug" element={<GovDistrictPage />} />
            <Route path="/help" element={<HelpPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </main>
      <ChatWidget />
      <Tour />
    </div>
  );
}
