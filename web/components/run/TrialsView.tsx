"use client";

import Link from "next/link";
import { useState } from "react";
import { AudioLines, ChevronLeft, ChevronRight } from "lucide-react";

import { ErrorNotice, LoadingLine } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/select";
import { api } from "@/lib/api";
import { useAsync } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";

const PAGE = 50;

/** Every trial of a run, filterable, with a link to its inspector. */
export function TrialsView({ runId, refreshKey }: { runId: string; refreshKey: number }) {
  const { t, tOr, number } = useI18n();
  const [method, setMethod] = useState("");
  const [attack, setAttack] = useState<string | null>(null);
  const [status, setStatus] = useState("");
  const [offset, setOffset] = useState(0);
  const facets = useAsync(() => api.facets(runId), [runId, refreshKey]);
  const rows = useAsync(
    () =>
      api.rows(runId, {
        method: method || undefined,
        attack: attack === null ? undefined : attack,
        status: status || undefined,
        offset,
        limit: PAGE,
      }),
    [runId, method, attack, status, offset, refreshKey]
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <Select value={method} onChange={(event) => (setMethod(event.target.value), setOffset(0))} className="w-72">
          <option value="">{t("run.filterMethod")}: {t("common.all")}</option>
          {(facets.data?.method ?? []).map((entry) => (
            <option key={entry} value={entry}>
              {entry}
            </option>
          ))}
        </Select>
        <Select
          value={attack === null ? "__all" : attack}
          onChange={(event) => (setAttack(event.target.value === "__all" ? null : event.target.value), setOffset(0))}
          className="w-64"
        >
          <option value="__all">{t("run.filterAttack")}: {t("common.all")}</option>
          {(facets.data?.attack ?? []).map((entry) => (
            <option key={entry ?? "__baseline"} value={entry ?? ""}>
              {entry ?? t("run.baseline")}
            </option>
          ))}
        </Select>
        <Select value={status} onChange={(event) => (setStatus(event.target.value), setOffset(0))} className="w-44">
          <option value="">{t("run.filterStatus")}: {t("common.all")}</option>
          <option value="ok">ok</option>
          <option value="error">{t("run.failures")}</option>
        </Select>
        <span className="num text-xs text-muted-foreground">
          {rows.data ? `${rows.data.total} ${t("common.rows")}` : ""}
        </span>
      </div>

      {rows.error && <ErrorNotice error={rows.error} onRetry={rows.reload} />}
      {!rows.data ? (
        <LoadingLine />
      ) : (
        <div className={rows.loading ? "opacity-60 transition-opacity" : ""}>
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs text-muted-foreground">
                  <th className="px-3 py-2 font-medium">{t("common.file")}</th>
                  <th className="px-3 py-2 font-medium">{t("run.filterMethod")}</th>
                  <th className="px-3 py-2 text-right font-medium">{t("run.bits")}</th>
                  <th className="px-3 py-2 text-right font-medium">{t("run.repShort")}</th>
                  <th className="px-3 py-2 font-medium">{t("run.filterAttack")}</th>
                  <th className="px-3 py-2 text-right font-medium">BER</th>
                  <th className="px-3 py-2 font-medium">{t("common.status")}</th>
                  <th className="px-3 py-2" />
                </tr>
              </thead>
              <tbody>
                {rows.data.rows.map(({ id, row }) => (
                  <tr key={id} className="border-b last:border-0 hover:bg-accent/40">
                    <td className="max-w-[12rem] truncate px-3 py-1.5 text-xs" title={row.file_name}>
                      {row.file_name}
                    </td>
                    <td className="max-w-[18rem] truncate px-3 py-1.5 text-xs" title={row.method}>
                      {row.method}
                    </td>
                    <td className="num px-3 py-1.5 text-right text-xs">{row.payload_length}</td>
                    <td className="num px-3 py-1.5 text-right text-xs">{row.repetition}</td>
                    <td className="max-w-[14rem] truncate px-3 py-1.5 font-mono text-[11px]" title={row.attack ?? ""}>
                      {row.attack ?? <span className="text-muted-foreground">{t("run.baseline")}</span>}
                    </td>
                    <td className="num px-3 py-1.5 text-right text-xs">{number(row.ber, 3)}</td>
                    <td className="px-3 py-1.5 text-xs">
                      {row.status === "ok" ? (
                        row.decode_success ? (
                          <span style={{ color: "hsl(var(--success))" }}>✓ {t("run.exactShort")}</span>
                        ) : (
                          <span className="text-muted-foreground">{t("run.bitErrors")}</span>
                        )
                      ) : (
                        <span title={row.error ?? ""} style={{ color: row.failure_kind === "over_capacity" ? "var(--status-serious)" : "var(--status-critical)" }}>
                          ✕ {tOr(`failureKinds.${row.failure_kind}`, row.failure_kind ?? "error")}
                        </span>
                      )}
                    </td>
                    <td className="px-3 py-1.5 text-right">
                      {row.status === "ok" && (
                        <Link href={`/runs/${runId}/trials/${id}`} className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
                          <AudioLines className="h-3.5 w-3.5" /> {t("run.inspect")}
                        </Link>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 flex items-center justify-end gap-2">
            <span className="num text-xs text-muted-foreground">
              {rows.data.total ? offset + 1 : 0}–{Math.min(offset + PAGE, rows.data.total)} {t("common.of")} {rows.data.total}
            </span>
            <Button variant="outline" size="icon" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button variant="outline" size="icon" disabled={offset + PAGE >= rows.data.total} onClick={() => setOffset(offset + PAGE)}>
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
