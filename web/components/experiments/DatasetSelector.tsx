"use client";

import { useRef, useState } from "react";
import { Loader2, Upload } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import type { DatasetInfo } from "@/lib/types";

export function DatasetSelector({
  datasets,
  value,
  onChange,
  fileLimit,
  onFileLimitChange,
  onDatasetsChanged,
}: {
  datasets: DatasetInfo[];
  value: string | null;
  onChange: (datasetId: string) => void;
  fileLimit: number | null;
  onFileLimitChange: (limit: number | null) => void;
  onDatasetsChanged: () => void;
}) {
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function upload(files: FileList | null) {
    if (!files || files.length === 0) return;
    setUploading(true);
    setUploadError(null);
    try {
      const uploaded = await api.uploadDataset("", [...files]);
      onDatasetsChanged();
      onChange(uploaded.dataset_name);
    } catch (err) {
      setUploadError((err as Error).message);
    } finally {
      setUploading(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3">
        {datasets.map((dataset) => (
          <label key={dataset.id} className="flex items-center gap-2 text-sm">
            <input
              type="radio"
              name="dataset"
              checked={value === dataset.id}
              onChange={() => onChange(dataset.id)}
            />
            {dataset.kind === "uploaded" ? dataset.label : dataset.id}
            <span className="text-muted-foreground">
              ({dataset.file_count} file{dataset.file_count === 1 ? "" : "s"}
              {dataset.kind === "uploaded" ? ", uploaded" : ""})
            </span>
          </label>
        ))}
      </div>
      <div className="flex flex-wrap items-end gap-4">
        <div className="space-y-1">
          <Label htmlFor="fileLimit">File limit</Label>
          <Input
            id="fileLimit"
            type="number"
            min={1}
            className="w-28"
            placeholder="all"
            value={fileLimit ?? ""}
            onChange={(event) =>
              onFileLimitChange(event.target.value === "" ? null : Number(event.target.value))
            }
          />
        </div>
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            type="file"
            accept=".wav,.flac,.ogg"
            multiple
            className="hidden"
            onChange={(event) => upload(event.target.files)}
          />
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={uploading}
            onClick={() => inputRef.current?.click()}
          >
            {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            Upload own sounds
          </Button>
          <span className="text-xs text-muted-foreground">WAV, FLAC or OGG</span>
        </div>
      </div>
      {uploadError && <p className="text-xs text-destructive">{uploadError}</p>}
    </div>
  );
}
