"use client";

import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { AttackInfo, DatasetInfo, MethodInfo, MetricInfo } from "@/lib/types";

export interface Catalog {
  methods: MethodInfo[];
  metrics: MetricInfo[];
  attacks: AttackInfo[];
  datasets: DatasetInfo[];
}

export function useCatalog() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    Promise.all([api.methods(), api.metrics(), api.attacks(), api.datasets()])
      .then(([methods, metrics, attacks, datasets]) => {
        setCatalog({ methods, metrics, attacks, datasets });
        setError(null);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { catalog, error, refresh };
}
