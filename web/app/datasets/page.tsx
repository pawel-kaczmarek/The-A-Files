"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { Download, ExternalLink, FolderOpen, Loader2, Sparkles, Trash2, Upload } from "lucide-react";

import { Chip, EmptyState, ErrorNotice, LoadingLine, PageHeader, Section } from "@/components/common";
import { Modal } from "@/components/modal";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { api } from "@/lib/api";
import { useAsync, useCatalog, useInterval } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import type { Corpus, SubsetRule } from "@/lib/types";

const DOMAIN_ORDER = ["speech", "synthetic_speech", "music", "environmental"];

const DEFAULT_RULE: SubsetRule = {
  max_files: 50,
  target_sample_rate: 16000,
  min_duration_seconds: 2,
  max_duration_seconds: 20,
  excerpt_seconds: null,
  excerpt_offset_seconds: 0,
  seed: 0,
};

function NumberField({ label, value, onChange, step, placeholder }: { label: string; value: number | null; onChange: (value: number | null) => void; step?: number; placeholder?: string }) {
  return (
    <div className="space-y-1">
      <Label className="text-xs">{label}</Label>
      <Input
        type="number"
        step={step}
        placeholder={placeholder}
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value === "" ? null : Number(event.target.value))}
      />
    </div>
  );
}

function PrepareDialog({ corpus, onClose, onDone }: { corpus: Corpus; onClose: () => void; onDone: () => void }) {
  const { t } = useI18n();
  const music = corpus.domain === "music";
  const [rule, setRule] = useState<SubsetRule>({
    ...DEFAULT_RULE,
    ...(music ? { excerpt_seconds: 10, excerpt_offset_seconds: 30, max_duration_seconds: null, max_files: 20 } : {}),
  });
  const [name, setName] = useState("");
  const [source, setSource] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (key: keyof SubsetRule) => (value: number | null) => setRule({ ...rule, [key]: value });

  async function submit() {
    setBusy(true);
    setError(null);
    try {
      await api.prepareCorpus({ corpus_id: corpus.id, name: name || undefined, rule: { ...rule, max_files: rule.max_files || 50, seed: rule.seed ?? 0, min_duration_seconds: rule.min_duration_seconds ?? 0, excerpt_offset_seconds: rule.excerpt_offset_seconds ?? 0 }, source_path: source || null });
      onDone();
      onClose();
    } catch (reason) {
      setError((reason as Error).message);
      setBusy(false);
    }
  }

  return (
    <Modal
      title={t("datasets.prepareTitle", { corpus: corpus.name })}
      onClose={onClose}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button onClick={submit} disabled={busy || (!corpus.downloadable && !source)}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />} {t("datasets.prepare")}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <p className="text-xs leading-relaxed text-muted-foreground">{t("datasets.ruleHint")}</p>
        <div className="space-y-1">
          <Label className="text-xs">{t("datasets.nameLabel")}</Label>
          <Input value={name} placeholder={`${corpus.name} (${rule.max_files} ${t("common.files")}, seed ${rule.seed})`} onChange={(event) => setName(event.target.value)} />
        </div>
        <div className="grid gap-3 sm:grid-cols-3">
          <NumberField label={t("datasets.maxFiles")} value={rule.max_files} onChange={set("max_files")} />
          <NumberField label={t("datasets.targetRate")} value={rule.target_sample_rate} onChange={set("target_sample_rate")} placeholder="native" />
          <NumberField label={t("datasets.seed")} value={rule.seed} onChange={set("seed")} />
          <NumberField label={t("datasets.minDuration")} value={rule.min_duration_seconds} onChange={set("min_duration_seconds")} step={0.5} />
          <NumberField label={t("datasets.maxDuration")} value={rule.max_duration_seconds} onChange={set("max_duration_seconds")} step={0.5} />
          <div />
          <NumberField label={t("datasets.excerpt")} value={rule.excerpt_seconds} onChange={set("excerpt_seconds")} step={0.5} />
          <NumberField label={t("datasets.excerptOffset")} value={rule.excerpt_offset_seconds} onChange={set("excerpt_offset_seconds")} step={0.5} />
        </div>
        <div className="space-y-1">
          <Label className="text-xs">{t("datasets.sourcePath")}</Label>
          <Input className="font-mono text-xs" value={source} placeholder={corpus.downloadable ? corpus.download?.url : "C:/data/TIMIT"} onChange={(event) => setSource(event.target.value)} />
          <p className="text-[11px] text-muted-foreground">{t("datasets.sourcePathHint")}</p>
        </div>
        {error && <ErrorNotice error={error} />}
      </div>
    </Modal>
  );
}

function OwnMaterial({ onDone }: { onDone: () => void }) {
  const { t } = useI18n();
  const files = useRef<HTMLInputElement>(null);
  const [uploadName, setUploadName] = useState("");
  const [chosen, setChosen] = useState(0);
  const [localName, setLocalName] = useState("");
  const [localPath, setLocalPath] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run(name: string, action: () => Promise<unknown>) {
    setBusy(name);
    setError(null);
    try {
      await action();
      onDone();
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(null);
    }
  }

  return (
    <Section title={t("datasets.addOwn")}>
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Upload className="h-4 w-4 text-muted-foreground" /> {t("datasets.upload")}
          </div>
          <p className="text-xs text-muted-foreground">{t("datasets.uploadHint")}</p>
          <Input placeholder={t("datasets.nameLabel")} value={uploadName} onChange={(event) => setUploadName(event.target.value)} />
          <label className="flex cursor-pointer items-center gap-2 text-xs text-muted-foreground">
            <span className="inline-flex h-8 items-center rounded-md border bg-background px-3 font-medium text-foreground hover:bg-accent">{t("datasets.pickFiles")}</span>
            {chosen > 0 && t("common.selected", { count: chosen })}
            <input
              ref={files}
              type="file"
              multiple
              accept=".wav,.flac,.ogg"
              className="sr-only"
              onChange={(event) => setChosen(event.target.files?.length ?? 0)}
            />
          </label>
          <Button
            size="sm"
            variant="outline"
            disabled={busy !== null || chosen === 0}
            onClick={() => run("upload", () => api.uploadDataset(uploadName || "Uploaded audio", Array.from(files.current?.files ?? [])))}
          >
            {busy === "upload" && <Loader2 className="h-3.5 w-3.5 animate-spin" />} {t("datasets.create")}
          </Button>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium">
            <FolderOpen className="h-4 w-4 text-muted-foreground" /> {t("datasets.local")}
          </div>
          <p className="text-xs text-muted-foreground">{t("datasets.localHint")}</p>
          <Input placeholder={t("datasets.nameLabel")} value={localName} onChange={(event) => setLocalName(event.target.value)} />
          <Input className="font-mono text-xs" placeholder="C:/data/corpus" value={localPath} onChange={(event) => setLocalPath(event.target.value)} />
          <Button size="sm" variant="outline" disabled={busy !== null || !localPath} onClick={() => run("local", () => api.registerLocal({ name: localName || localPath, path: localPath }))}>
            {busy === "local" && <Loader2 className="h-3.5 w-3.5 animate-spin" />} {t("datasets.create")}
          </Button>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Sparkles className="h-4 w-4 text-muted-foreground" /> {t("datasets.synthetic")}
          </div>
          <p className="text-xs text-muted-foreground">{t("datasets.syntheticHint")}</p>
          <Button size="sm" variant="outline" disabled={busy !== null} onClick={() => run("synthetic", () => api.synthetic({ sample_rate: 16000, duration_seconds: 5, seed: 0 }))}>
            {busy === "synthetic" && <Loader2 className="h-3.5 w-3.5 animate-spin" />} {t("datasets.create")}
          </Button>
        </div>
      </div>
      {error && <ErrorNotice error={error} />}
    </Section>
  );
}

export default function DatasetsPage() {
  const { t, tOr, date } = useI18n();
  const catalog = useCatalog();
  const library = useAsync(() => api.datasets(), []);
  const [preparing, setPreparing] = useState<Corpus | null>(null);
  const busy = library.data?.some((dataset) => ["pending", "downloading", "preparing"].includes(dataset.status)) ?? false;
  useInterval(library.reload, 1500, busy);

  return (
    <>
      <PageHeader title={t("datasets.title")} subtitle={t("datasets.subtitle")} />
      {library.error && <ErrorNotice error={library.error} onRetry={library.reload} />}

      <div className="space-y-8">
        <Section title={t("datasets.library")}>
          {!library.data ? (
            <LoadingLine />
          ) : !library.data.length ? (
            <EmptyState>{t("common.empty")}</EmptyState>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-3 font-medium">{t("common.name")}</th>
                  <th className="py-2 pr-3 font-medium">{t("common.type")}</th>
                  <th className="py-2 pr-3 text-right font-medium">{t("datasets.filesTitle")}</th>
                  <th className="py-2 pr-3 text-right font-medium">{t("datasets.duration")}</th>
                  <th className="py-2 pr-3 font-medium">{t("common.status")}</th>
                  <th className="py-2 pr-3 text-right font-medium">{t("common.created")}</th>
                  <th className="py-2" />
                </tr>
              </thead>
              <tbody>
                {library.data.map((dataset) => (
                  <tr key={dataset.id} className="border-b last:border-0">
                    <td className="py-2.5 pr-3">
                      <Link href={`/datasets/${dataset.id}`} className="font-medium hover:underline">
                        {dataset.name}
                      </Link>
                      {dataset.license && <div className="text-xs text-muted-foreground">{dataset.license}</div>}
                    </td>
                    <td className="py-2.5 pr-3 text-xs">{tOr(`datasetKinds.${dataset.kind}`, dataset.kind)}</td>
                    <td className="num py-2.5 pr-3 text-right">{dataset.file_count || "–"}</td>
                    <td className="num py-2.5 pr-3 text-right text-xs">
                      {dataset.total_duration_seconds ? `${(dataset.total_duration_seconds / 60).toFixed(1)} ${t("common.minutes")}` : "–"}
                    </td>
                    <td className="w-48 py-2.5 pr-3">
                      <div className="text-xs" style={{ color: dataset.status === "failed" ? "var(--status-critical)" : undefined }} title={dataset.error ?? ""}>
                        {tOr(`datasetStatus.${dataset.status}`, dataset.status)}
                        {dataset.stage && dataset.status !== "ready" ? ` · ${dataset.stage}` : ""}
                      </div>
                      {["downloading", "preparing"].includes(dataset.status) && <Progress value={dataset.progress * 100} className="mt-1 h-1" />}
                    </td>
                    <td className="py-2.5 pr-3 text-right text-xs text-muted-foreground">{date(dataset.created_at)}</td>
                    <td className="py-2.5 text-right">
                      <button
                        type="button"
                        aria-label={t("common.delete")}
                        className="text-muted-foreground hover:text-destructive"
                        onClick={() => window.confirm(t("common.confirmDelete")) && api.deleteDataset(dataset.id).then(library.reload)}
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Section>

        <section>
          <div className="mb-3">
            <h2 className="text-base font-semibold">{t("datasets.corpora")}</h2>
            <p className="text-sm text-muted-foreground">{t("datasets.corporaHint")}</p>
          </div>
          {catalog.error && <ErrorNotice error={catalog.error} />}
          {!catalog.data ? (
            <LoadingLine />
          ) : (
            <div className="space-y-6">
              {DOMAIN_ORDER.filter((domain) => catalog.data!.corpora.some((corpus) => corpus.domain === domain)).map((domain) => (
                <div key={domain}>
                  <div className="eyebrow mb-2">{t(`domains.${domain}`)}</div>
                  <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                    {catalog.data!.corpora
                      .filter((corpus) => corpus.domain === domain)
                      .map((corpus) => (
                        <div key={corpus.id} className="flex flex-col rounded-lg border bg-card p-4">
                          <div className="flex items-start justify-between gap-2">
                            <div className="text-sm font-semibold">{corpus.name}</div>
                            <span className="num text-xs text-muted-foreground">{corpus.year}</span>
                          </div>
                          <p className="mt-1 flex-1 text-xs leading-relaxed text-muted-foreground">{corpus.description}</p>
                          <div className="mt-3 flex flex-wrap gap-1">
                            <Chip>{corpus.license}</Chip>
                            {corpus.native_sample_rate && <Chip>{t("datasets.native", { rate: corpus.native_sample_rate / 1000 })}</Chip>}
                            {corpus.download?.size_mb && <Chip>{t("datasets.size", { size: Math.round(corpus.download.size_mb).toLocaleString() })}</Chip>}
                            {corpus.language && <Chip>{corpus.language}</Chip>}
                            {corpus.tags.map((tag) => (
                              <Chip key={tag}>{tag}</Chip>
                            ))}
                          </div>
                          <p className="mt-3 text-[11px] italic leading-snug text-muted-foreground">{corpus.citation}</p>
                          <div className="mt-3 flex items-center justify-between gap-2">
                            <div className="flex items-center gap-3 text-xs">
                              {corpus.doi && (
                                <a href={`https://doi.org/${corpus.doi}`} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-primary">
                                  DOI <ExternalLink className="h-3 w-3" />
                                </a>
                              )}
                              {corpus.url && (
                                <a href={corpus.url} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-primary">
                                  web <ExternalLink className="h-3 w-3" />
                                </a>
                              )}
                            </div>
                            <Button size="sm" variant={corpus.downloadable ? "default" : "outline"} onClick={() => setPreparing(corpus)}>
                              {corpus.downloadable ? <Download className="h-3.5 w-3.5" /> : <FolderOpen className="h-3.5 w-3.5" />}
                              {corpus.downloadable ? t("datasets.prepare") : t("datasets.manual")}
                            </Button>
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <OwnMaterial onDone={library.reload} />
      </div>
      {preparing && <PrepareDialog corpus={preparing} onClose={() => setPreparing(null)} onDone={library.reload} />}
    </>
  );
}
