"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import type { AttackInfo, Corpus, DesignInfo, MethodInfo, MetricInfo } from "@/lib/types";

export interface AsyncState<T> {
  data: T | undefined;
  error: string | null;
  loading: boolean;
  reload: () => void;
}

/** Runs ``load`` on mount and whenever ``deps`` change; keeps the previous data while reloading. */
export function useAsync<T>(load: () => Promise<T>, deps: unknown[] = []): AsyncState<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const loader = useRef(load);
  loader.current = load;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    loader
      .current()
      .then((value) => {
        if (!cancelled) {
          setData(value);
          setError(null);
        }
      })
      .catch((reason: Error) => !cancelled && setError(reason.message))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick, ...deps]);

  const reload = useCallback(() => setTick((value) => value + 1), []);
  return { data, error, loading, reload };
}

/** Re-runs ``callback`` every ``ms`` while ``active``. */
export function useInterval(callback: () => void, ms: number, active = true) {
  const saved = useRef(callback);
  saved.current = callback;
  useEffect(() => {
    if (!active) return;
    const handle = window.setInterval(() => saved.current(), ms);
    return () => window.clearInterval(handle);
  }, [ms, active]);
}

export interface Catalog {
  methods: MethodInfo[];
  metrics: MetricInfo[];
  attacks: AttackInfo[];
  designs: DesignInfo[];
  corpora: Corpus[];
}

// The catalogue does not change while the API runs: fetch it once per page load.
let catalogPromise: Promise<Catalog> | null = null;

function loadCatalog(): Promise<Catalog> {
  if (!catalogPromise) {
    catalogPromise = Promise.all([api.methods(), api.metrics(), api.attacks(), api.designs(), api.corpora()])
      .then(([methods, metrics, attacks, designs, corpora]) => ({ methods, metrics, attacks, designs, corpora }))
      .catch((error) => {
        catalogPromise = null;
        throw error;
      });
  }
  return catalogPromise;
}

export function useCatalog(): AsyncState<Catalog> {
  return useAsync(loadCatalog, []);
}

/**
 * Method label -> the catalogue's short name for figures. Handles registry
 * names, specifications ("QIM_METHOD:step_scale=0.1" -> "QIM:step_scale=0.1")
 * and descriptions; unknown labels are returned unchanged.
 */
export function useMethodAbbreviation(): (label: string) => string {
  const catalog = useCatalog();
  return (label: string) => {
    for (const method of catalog.data?.methods ?? []) {
      if (!method.abbreviation) continue;
      if (label === method.name || label === method.description) return method.abbreviation;
      if (label.startsWith(`${method.name}:`)) return method.abbreviation + label.slice(method.name.length);
    }
    return label;
  };
}
